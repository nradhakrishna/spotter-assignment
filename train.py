"""Validate on time-based checks, tune, then compare models with the tuned
settings, so the comparison table describes exactly what predict.py submits.

Order:
  1. Tune the hybrid's tree settings (mean error over all three checks).
  2. Tune the lane-offset shrinkage with those settings.
  3. Compare every model on every check (hybrids use the tuned settings).
  4. Choose the final model: the hybrid variant with the lowest mean error.

Outputs (in reports/):
  tuning.csv            - hybrid tree settings tried
  lane_tuning.csv       - lane shrinkage values tried
  model_comparison.csv  - every model on every check
  best_params.json      - final model + settings, read by predict.py
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from model import GBM_PARAMS, Baseline, Boosting, Hybrid, HybridWithLanes, Linear, RandomForest
from prepare import TARGET, TRAIN_FILE, Preprocessor, find_bad_labels, load

REPORTS = Path(__file__).resolve().parent / "reports"

# Rolling time splits: always train on the past, test on the next two months.
FOLDS = {
    "Jan-Jun -> Jul-Aug": ("2025-07-01", "2025-09-01"),
    "Jan-Aug -> Sep-Oct": ("2025-09-01", "2025-11-01"),
}
UNSEEN_CHECK = "Unseen cities (Sep-Oct)"
N_UNSEEN_CITIES = 8  # validation has 8 cities never seen in training

TREE_GRID = {
    "max_leaf_nodes": [15, 31, 63],
    "min_samples_leaf": [20, 100],
    "max_iter": [400, 800],
}
LANE_K_GRID = [2, 5, 10, 20]


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


def unseen_city_split(data: pd.DataFrame, seed: int = 0):
    """Last fold, but some cities are removed from training entirely."""
    start, end = FOLDS["Jan-Aug -> Sep-Oct"]
    cities = sorted(set(data["pickup"]) | set(data["delivery"]))
    held_out = set(np.random.default_rng(seed).choice(cities, N_UNSEEN_CITIES, replace=False))
    touches = data["pickup"].isin(held_out) | data["delivery"].isin(held_out)
    train = data[(data["date"] < start) & ~touches]
    test = data[(data["date"] >= start) & (data["date"] < end) & touches]
    return train, test


def checks(data: pd.DataFrame) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    out = []
    for name, (start, end) in FOLDS.items():
        out.append((name, data[data["date"] < start],
                    data[(data["date"] >= start) & (data["date"] < end)]))
    out.append((UNSEEN_CHECK, *unseen_city_split(data)))
    return out


def mean_clean_mape(make_model, all_checks) -> tuple[float, list[float]]:
    mape = [score(make_model(), tr, te)["clean_MAPE_%"] for _, tr, te in all_checks]
    return float(np.mean(mape)), mape


def tune_trees(all_checks) -> tuple[dict, pd.DataFrame]:
    rows = []
    for values in itertools.product(*TREE_GRID.values()):
        params = dict(zip(TREE_GRID, values))
        mean, per_check = mean_clean_mape(lambda: Hybrid(**params), all_checks)
        rows.append({**params, **{n: m for (n, _, _), m in zip(all_checks, per_check)},
                     "mean_MAPE_%": mean})
        print(f"  {params}  mean clean MAPE {mean:.3f}%")
    table = pd.DataFrame(rows).sort_values("mean_MAPE_%")
    best = {k: int(v) for k, v in table.iloc[0][list(TREE_GRID)].items()}
    return {**GBM_PARAMS, **best}, table


def tune_lane_k(tree_params: dict, all_checks) -> tuple[int, pd.DataFrame]:
    rows = []
    for k in LANE_K_GRID:
        mean, per_check = mean_clean_mape(lambda: HybridWithLanes(lane_k=k, **tree_params), all_checks)
        rows.append({"lane_k": k, **{n: m for (n, _, _), m in zip(all_checks, per_check)},
                     "mean_MAPE_%": mean})
        print(f"  lane_k={k:<3d} mean clean MAPE {mean:.3f}%")
    table = pd.DataFrame(rows).sort_values("mean_MAPE_%")
    return int(table.iloc[0]["lane_k"]), table


def compare_models(models: dict, all_checks) -> pd.DataFrame:
    rows = []
    for check, train, test in all_checks:
        for name, make_model in models.items():
            result = score(make_model(), train, test)
            rows.append({"check": check, "model": name, **result})
            print(f"  {check:24s} {name:36s} clean MAPE {result['clean_MAPE_%']:5.2f}%  "
                  f"all MAPE {result['all_MAPE_%']:5.2f}%")
    return pd.DataFrame(rows)


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    all_checks = checks(load(TRAIN_FILE))

    print("1. Tuning hybrid tree settings (mean over all three checks)...")
    tree_params, tuning = tune_trees(all_checks)
    tuning.to_csv(REPORTS / "tuning.csv", index=False)

    print("\n2. Tuning lane-offset shrinkage...")
    lane_k, lane_tuning = tune_lane_k(tree_params, all_checks)
    lane_tuning.to_csv(REPORTS / "lane_tuning.csv", index=False)

    print("\n3. Comparing models (hybrids with tuned settings)...")
    models = {
        Baseline.name: Baseline,
        Linear.name: Linear,
        RandomForest.name: RandomForest,
        Boosting.name: Boosting,
        Hybrid.name: lambda: Hybrid(**tree_params),
        HybridWithLanes.name: lambda: HybridWithLanes(lane_k=lane_k, **tree_params),
    }
    comparison = compare_models(models, all_checks)
    comparison.to_csv(REPORTS / "model_comparison.csv", index=False)

    mean_by_model = comparison.groupby("model")["clean_MAPE_%"].mean()
    final = min([Hybrid.name, HybridWithLanes.name], key=mean_by_model.get)
    best = {"final_model": "HybridWithLanes" if final == HybridWithLanes.name else "Hybrid",
            "lane_k": lane_k, "tree_params": tree_params}
    (REPORTS / "best_params.json").write_text(json.dumps(best, indent=2))

    summary = comparison.pivot_table(index="model", columns="check", values="clean_MAPE_%")
    summary["mean"] = summary.mean(axis=1)
    print("\nClean-row MAPE (%) by model and check:")
    print(summary.sort_values("mean").round(2).to_string())
    print(f"\nFinal model: {final}  ({best})")
    print(f"Saved results to {REPORTS}/")


if __name__ == "__main__":
    main()
