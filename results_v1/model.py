"""Candidate models. All take feature frames from `Preprocessor.transform`,
train on log(price) and return predictions in dollars."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

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


CANDIDATES = [Baseline, Linear, RandomForest, Boosting, Hybrid]
