"""Candidate models. All take feature frames from `Preprocessor.transform`,
train on log(price) and return predictions in dollars."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from prepare import LINEAR_FEATURES, TREE_FEATURES

HYBRID_LINEAR_FEATURES = [c for c in LINEAR_FEATURES if c != "days_since_start"]

GBM_PARAMS = dict(learning_rate=0.05, max_iter=800, max_leaf_nodes=31,
                  min_samples_leaf=20, l2_regularization=0.0)


def _equipment(features: pd.DataFrame) -> pd.Series:
    cols = ["eq_dry_van", "eq_flatbed", "eq_reefer"]
    return features[cols].idxmax(axis=1)


class Baseline:
    """Median $/mile per equipment type x distance."""
    name = "Baseline ($/mile x distance)"

    def fit(self, X, log_y):
        rpm = np.exp(log_y) / X["distance"]
        self.rpm_ = rpm.groupby(_equipment(X)).median()
        return self

    def predict(self, X):
        return _equipment(X).map(self.rpm_).to_numpy() * X["distance"].to_numpy()


class Linear:
    """Linear regression on log(price): a power-law curve in dollars."""
    name = "Linear regression on log(price)"

    def fit(self, X, log_y):
        self.model_ = LinearRegression().fit(X[LINEAR_FEATURES], log_y)
        return self

    def predict_log(self, X):
        return self.model_.predict(X[LINEAR_FEATURES])

    def predict(self, X):
        return np.exp(self.predict_log(X))


class RandomForest:
    name = "Random forest"

    def __init__(self, random_state=0):
        self.random_state = random_state

    def fit(self, X, log_y):
        self.model_ = RandomForestRegressor(
            n_estimators=200, min_samples_leaf=3, n_jobs=-1,
            random_state=self.random_state,
        ).fit(X[TREE_FEATURES + ["days_since_start"]], log_y)
        return self

    def predict(self, X):
        return np.exp(self.model_.predict(X[TREE_FEATURES + ["days_since_start"]]))


class Boosting:
    name = "Gradient boosting"

    def __init__(self, random_state=0, **params):
        self.random_state = random_state
        self.params = {**GBM_PARAMS, **params}

    def fit(self, X, log_y):
        self.model_ = HistGradientBoostingRegressor(
            random_state=self.random_state, **self.params,
        ).fit(X[TREE_FEATURES + ["days_since_start"]], log_y)
        return self

    def predict(self, X):
        return np.exp(self.model_.predict(X[TREE_FEATURES + ["days_since_start"]]))


class Hybrid:
    """Three stages, each fitted on what the previous one left over:

    1. Linear model on log(price) WITHOUT a time trend: distance curve,
       equipment, market, weight, region.
    2. A straight-line time trend fitted on stage-1 residuals. It is fitted
       separately because market_index and time move together in the first
       half of the year; fitting them jointly mixes the two up and the trend
       extrapolates badly (3.67% vs 2.05% MAPE on the Jul-Aug fold).
    3. Gradient boosting on the remaining residuals (interactions). It gets
       no date-trend feature: trees cannot extrapolate, so the trend is
       left entirely to stage 2.
    """
    name = "Hybrid: linear + trend + boosting"

    def __init__(self, random_state=0, **params):
        self.random_state = random_state
        self.params = {**GBM_PARAMS, **params}

    def fit(self, X, log_y):
        log_y = np.asarray(log_y)
        self.linear_ = LinearRegression().fit(X[HYBRID_LINEAR_FEATURES], log_y)
        residual = log_y - self.linear_.predict(X[HYBRID_LINEAR_FEATURES])
        self.trend_slope_, self.trend_intercept_ = np.polyfit(X["days_since_start"], residual, 1)
        residual = residual - self._trend(X)
        self.trees_ = HistGradientBoostingRegressor(
            random_state=self.random_state, **self.params,
        ).fit(X[TREE_FEATURES], residual)
        return self

    def _trend(self, X):
        return self.trend_slope_ * X["days_since_start"].to_numpy() + self.trend_intercept_

    def predict(self, X):
        return np.exp(self.linear_.predict(X[HYBRID_LINEAR_FEATURES]) + self._trend(X)
                      + self.trees_.predict(X[TREE_FEATURES]))


class HybridWithLanes:
    """Hybrid plus a 4th stage: a per-lane price offset.

    Some lanes are consistently priced above or below what the hybrid
    predicts. Each lane's offset is the average of its residuals, shrunk
    toward zero by `lane_k` pseudo-loads (a lane with n loads keeps
    n / (n + lane_k) of its average), so lanes with little history barely
    move. Residuals come from 5-fold out-of-fold predictions, so the offset
    measures what the model misses on loads it has NOT seen, not its
    in-sample fit. Lanes without history (e.g. new cities) get offset 0.
    """
    name = "Hybrid + lane offsets"

    def __init__(self, lane_k=5, random_state=0, **params):
        self.lane_k = lane_k
        self.random_state = random_state
        self.params = params

    def _hybrid(self):
        return Hybrid(random_state=self.random_state, **self.params)

    def fit(self, X, log_y):
        log_y = np.asarray(log_y)
        out_of_fold = np.zeros(len(log_y))
        folds = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        for fit_idx, held_idx in folds.split(X):
            part = self._hybrid().fit(X.iloc[fit_idx], log_y[fit_idx])
            out_of_fold[held_idx] = log_y[held_idx] - np.log(part.predict(X.iloc[held_idx]))
        by_lane = pd.Series(out_of_fold).groupby(X["lane"].to_numpy())
        self.lane_offset_ = by_lane.sum() / (by_lane.count() + self.lane_k)
        self.hybrid_ = self._hybrid().fit(X, log_y)
        return self

    def predict(self, X):
        offset = X["lane"].map(self.lane_offset_).fillna(0.0).to_numpy()
        return self.hybrid_.predict(X) * np.exp(offset)


CANDIDATES = [Baseline, Linear, RandomForest, Boosting, Hybrid, HybridWithLanes]
