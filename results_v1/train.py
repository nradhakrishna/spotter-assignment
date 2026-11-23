"""Validate on time-based splits, compare models, tune the hybrid lightly.

Outputs (in reports/):
  model_comparison.csv  - every model on every check
  tuning.csv            - hybrid settings tried
  best_params.json      - settings used by predict.py
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from model import CANDIDATES, GBM_PARAMS, Hybrid
from prepare import TARGET, TRAIN_FILE, Preprocessor, find_bad_labels, load

REPORTS = Path(__file__).resolve().parent / "reports"

# Rolling time splits: always train on the past, test on the next two months.
FOLDS = {
    "Jan-Jun -> Jul-Aug": ("2025-07-01", "2025-09-01"),
    "Jan-Aug -> Sep-Oct": ("2025-09-01", "2025-11-01"),
}
N_UNSEEN_CITIES = 8  # validation has 8 cities never seen in training


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    error = predicted - actual
    return {
        "MAPE_%": float(np.mean(np.abs(error) / actual) * 100),
        "MAE_$": float(np.mean(np.abs(error))),
        "RMSE_$": float(np.sqrt(np.mean(error ** 2))),
    }


def score(model, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Fit preprocessing + model on `train`, score on `test` (all and clean rows)."""
    prep = Preprocessor().fit(train)
    train_clean = prep.clean_training_rows(train)
    model.fit(prep.transform(train_clean), np.log(train_clean[TARGET]))
    predicted = model.predict(prep.transform(test))
    actual = test[TARGET].to_numpy()
    # "Clean" test rows judged with the expected-price formula fitted on train.
    clean = ~find_bad_labels(test, prep.expected_price_coef_).to_numpy()
    out = {f"clean_{k}": v for k, v in metrics(actual[clean], predicted[clean]).items()}
    out.update({f"all_{k}": v for k, v in metrics(actual, predicted).items()})
    out["n_train"] = len(train_clean)
    out["n_test"] = len(test)
    return out


def splits(data: pd.DataFrame):
    for name, (start, end) in FOLDS.items():
        train = data[data["date"] < start]
        test = data[(data["date"] >= start) & (data["date"] < end)]
        yield name, train, test


def unseen_city_split(data: pd.DataFrame, seed: int = 0):
    """Last fold, but some cities are removed from training entirely."""
    start, end = FOLDS["Jan-Aug -> Sep-Oct"]
    cities = sorted(set(data["pickup"]) | set(data["delivery"]))
    held_out = set(np.random.default_rng(seed).choice(cities, N_UNSEEN_CITIES, replace=False))
    touches = data["pickup"].isin(held_out) | data["delivery"].isin(held_out)
    train = data[(data["date"] < start) & ~touches]
    test = data[(data["date"] >= start) & (data["date"] < end) & touches]
    return train, test


def compare_models(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    checks = list(splits(data)) + [("Unseen cities (Sep-Oct)", *unseen_city_split(data))]
    for check, train, test in checks:
        for cls in CANDIDATES:
            result = score(cls(), train, test)
            rows.append({"check": check, "model": cls.name, **result})
            print(f"  {check:24s} {cls.name:40s} clean MAPE {result['clean_MAPE_%']:5.2f}%  "
                  f"all MAPE {result['all_MAPE_%']:5.2f}%")
    return pd.DataFrame(rows)


def tune_hybrid(data: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    grid = {
        "max_leaf_nodes": [15, 31, 63],
        "min_samples_leaf": [20, 100],
        "max_iter": [400, 800],
    }
    rows = []
    folds = list(splits(data))
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid, values))
        mape = [score(Hybrid(**params), tr, te)["clean_MAPE_%"] for _, tr, te in folds]
        rows.append({**params, **{f"MAPE_{n}": m for (n, _, _), m in zip(folds, mape)},
                     "mean_MAPE_%": float(np.mean(mape))})
        print(f"  {params}  mean clean MAPE {np.mean(mape):.3f}%")
    table = pd.DataFrame(rows).sort_values("mean_MAPE_%")
    best = {k: int(v) for k, v in table.iloc[0][list(grid)].items()}
    return {**GBM_PARAMS, **best}, table


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    data = load(TRAIN_FILE)

    print("Comparing models on time-based checks...")
    comparison = compare_models(data)
    comparison.to_csv(REPORTS / "model_comparison.csv", index=False)

    print("\nTuning the hybrid (mean over both time folds)...")
    best, tuning = tune_hybrid(data)
    tuning.to_csv(REPORTS / "tuning.csv", index=False)
    (REPORTS / "best_params.json").write_text(json.dumps(best, indent=2))

    summary = comparison.pivot_table(index="model", columns="check", values="clean_MAPE_%")
    print("\nClean-row MAPE (%) by model and check:")
    print(summary.round(2).to_string())
    print(f"\nBest hybrid settings: {best}")
    print(f"Saved results to {REPORTS}/")


if __name__ == "__main__":
    main()
