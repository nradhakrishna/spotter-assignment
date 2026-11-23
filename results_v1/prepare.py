"""Data loading, cleaning and feature engineering.

Rules that learn nothing from the data (sign fix, date parsing) run anywhere.
Anything learned from data (median weight, the "expected price" used to find
bad labels) is fitted on the training part only, inside `Preprocessor.fit`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
TRAIN_FILE = "train-test.csv"
VALIDATION_FILE = "validation.csv"
TEMPLATE_FILE = "validation-predictions-template.csv"
DECEMBER_FILE = "december-chart-inputs.csv"

TARGET = "posted_rate"
START_DATE = pd.Timestamp("2025-01-01")
EQUIPMENT = ["Dry Van", "Flatbed", "Reefer"]

# US federal holidays 2025 (hard-coded to avoid an extra dependency).
US_HOLIDAYS_2025 = pd.to_datetime([
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-05-26", "2025-06-19",
    "2025-07-04", "2025-09-01", "2025-10-13", "2025-11-11", "2025-11-27",
    "2025-12-25",
])

# A label is treated as corrupted when it is this far from the expected price.
BAD_LABEL_LOW, BAD_LABEL_HIGH = 0.5, 2.0


def load(name: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    frame = pd.read_csv(Path(data_dir) / name)
    frame["date"] = pd.to_datetime(frame["date"])
    # Negative weights mirror the positive distribution exactly: a sign error.
    frame["weight"] = frame["weight"].abs()
    return frame


def city_coordinates(*frames: pd.DataFrame) -> pd.DataFrame:
    """One lat/lon per city, taken from pickup and delivery columns."""
    parts = []
    for frame in frames:
        for side in ("pickup", "delivery"):
            parts.append(frame[[side, f"{side}_lat", f"{side}_lon"]]
                         .set_axis(["city", "lat", "lon"], axis=1))
    return pd.concat(parts).groupby("city")[["lat", "lon"]].first()


def _expected_log_price_design(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack([
        np.ones(len(frame)),
        np.log(frame["distance"]),
        frame["equipment"].eq("Flatbed"),
        frame["equipment"].eq("Reefer"),
    ]).astype(float)


def find_bad_labels(frame: pd.DataFrame, coef: np.ndarray) -> pd.Series:
    """True where posted_rate is <0.5x or >2x the expected price."""
    ratio = frame[TARGET] / np.exp(_expected_log_price_design(frame) @ coef)
    return (ratio < BAD_LABEL_LOW) | (ratio > BAD_LABEL_HIGH)


def fit_expected_price(frame: pd.DataFrame, rounds: int = 3) -> np.ndarray:
    """log(price) ~ log(distance) + equipment, refitted without flagged rows."""
    keep = np.ones(len(frame), dtype=bool)
    X, y = _expected_log_price_design(frame), np.log(frame[TARGET].to_numpy())
    for _ in range(rounds):
        coef = np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]
        keep = ~find_bad_labels(frame, coef).to_numpy()
    return coef


class Preprocessor:
    """Fills gaps and builds features; learned values come from `fit` only."""

    def fit(self, train: pd.DataFrame) -> "Preprocessor":
        self.weight_median_ = float(train["weight"].median())
        self.market_median_ = float(train["market_index"].median())
        self.expected_price_coef_ = fit_expected_price(train)
        return self

    def clean_training_rows(self, train: pd.DataFrame) -> pd.DataFrame:
        """Drop corrupted labels. Only ever applied to training data."""
        return train.loc[~find_bad_labels(train, self.expected_price_coef_)]

    def fill_missing(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        frame["weight"] = frame["weight"].fillna(self.weight_median_)
        # market_index is a daily market level: use the same day's loads.
        day_mean = frame.groupby("date")["market_index"].transform("mean")
        frame["market_index"] = (frame["market_index"].fillna(day_mean)
                                 .fillna(self.market_median_))
        return frame

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = self.fill_missing(frame)
        f = pd.DataFrame(index=frame.index)
        for eq in EQUIPMENT:
            f[f"eq_{eq.lower().replace(' ', '_')}"] = frame["equipment"].eq(eq).astype(float)
        f["distance"] = frame["distance"]
        f["log_distance"] = np.log(frame["distance"])
        f["weight_10k"] = frame["weight"] / 10_000
        f["market_index"] = frame["market_index"]
        f["log_market_index"] = np.log(frame["market_index"])
        f["quote_signal"] = frame["quote_signal"]
        for col in ("pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"):
            f[col] = frame[col]
        f["delta_lat"] = frame["delivery_lat"] - frame["pickup_lat"]
        f["delta_lon"] = frame["delivery_lon"] - frame["pickup_lon"]
        f["days_since_start"] = (frame["date"] - START_DATE).dt.days
        f["day_of_week"] = frame["date"].dt.dayofweek
        f["is_holiday"] = frame["date"].isin(US_HOLIDAYS_2025).astype(float)
        return f


# Features for the linear part (smooth effects + the time trend it can extrapolate).
LINEAR_FEATURES = [
    "eq_flatbed", "eq_reefer", "log_distance", "weight_10k", "log_market_index",
    "quote_signal", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "days_since_start", "is_holiday",
]
# Features for the tree part (learns interactions in what the linear part misses).
TREE_FEATURES = [
    "eq_dry_van", "eq_flatbed", "eq_reefer", "distance", "weight_10k",
    "market_index", "quote_signal", "pickup_lat", "pickup_lon", "delivery_lat",
    "delivery_lon", "delta_lat", "delta_lon", "day_of_week",
]


def december_inputs(december: pd.DataFrame, validation: pd.DataFrame,
                    coords: pd.DataFrame) -> pd.DataFrame:
    """Add the columns the chart file lacks: coordinates and daily market values."""
    frame = december.copy()
    for side in ("pickup", "delivery"):
        frame[f"{side}_lat"] = frame[side].map(coords["lat"])
        frame[f"{side}_lon"] = frame[side].map(coords["lon"])
    daily = validation.groupby("date")[["market_index", "quote_signal"]].mean()
    frame = frame.join(daily, on="date")
    if frame[["pickup_lat", "delivery_lat", "market_index", "quote_signal"]].isna().any().any():
        raise ValueError("could not complete December inputs (unknown city or date)")
    return frame
