"""Retrain the chosen model on all of Jan-Oct and write both prediction files.

Outputs:
  validation_predictions.csv     - load_id,predicted_rate for all 12,000 loads
  december_chart_predictions.csv - december-chart-inputs.csv with predicted_rate filled
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from model import Hybrid, HybridWithLanes
from prepare import (DATA_DIR, DECEMBER_FILE, TARGET, TEMPLATE_FILE, TRAIN_FILE,
                     VALIDATION_FILE, Preprocessor, city_coordinates,
                     december_inputs, load)

ROOT = Path(__file__).resolve().parent
PARAMS_FILE = ROOT / "reports" / "best_params.json"
VALIDATION_OUT = ROOT / "validation_predictions.csv"
DECEMBER_OUT = ROOT / "december_chart_predictions.csv"


def main() -> None:
    train = load(TRAIN_FILE)
    validation = load(VALIDATION_FILE)
    if not PARAMS_FILE.exists():
        raise SystemExit("reports/best_params.json not found: run train.py first")
    best = json.loads(PARAMS_FILE.read_text())
    if best["final_model"] == "HybridWithLanes":
        model = HybridWithLanes(lane_k=best["lane_k"], **best["tree_params"])
    else:
        model = Hybrid(**best["tree_params"])
    print(f"Final model: {model.name}")

    # Learn preprocessing on all labelled data, drop corrupted labels, retrain.
    prep = Preprocessor().fit(train)
    train_clean = prep.clean_training_rows(train)
    print(f"Training on {len(train_clean):,} of {len(train):,} loads "
          f"({len(train) - len(train_clean)} corrupted labels removed)")
    model.fit(prep.transform(train_clean), np.log(train_clean[TARGET]))

    # Validation: every row gets a prediction, in the template's order.
    validation["predicted_rate"] = model.predict(prep.transform(validation))
    template = pd.read_csv(DATA_DIR / TEMPLATE_FILE)
    out = template[["load_id"]].merge(validation[["load_id", "predicted_rate"]],
                                      on="load_id", how="left", validate="one_to_one")
    if out["predicted_rate"].isna().any():
        raise SystemExit("some template load_ids are missing from validation.csv")
    out["predicted_rate"] = out["predicted_rate"].round(2)
    out.to_csv(VALIDATION_OUT, index=False)
    print(f"Wrote {VALIDATION_OUT.name}: {len(out):,} rows, "
          f"median ${out['predicted_rate'].median():,.0f}")

    # December chart: fixed load, only the date changes.
    december_raw = pd.read_csv(DATA_DIR / DECEMBER_FILE)
    december = december_inputs(load(DECEMBER_FILE), validation,
                               city_coordinates(train, validation))
    december_raw["predicted_rate"] = model.predict(prep.transform(december)).round(2)
    december_raw.to_csv(DECEMBER_OUT, index=False)
    rates = december_raw["predicted_rate"]
    print(f"Wrote {DECEMBER_OUT.name}: ${rates.min():,.2f} - ${rates.max():,.2f} "
          f"(mean ${rates.mean():,.2f})")


if __name__ == "__main__":
    main()
