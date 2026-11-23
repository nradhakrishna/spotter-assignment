# Freight Rate Prediction — Solution

## Run it

```bash
python -m pip install -r requirements.txt
python train.py      # time-based validation + model comparison + tuning (~1 min) -> reports/
python predict.py    # retrain on all Jan-Oct, write both prediction files
python score.py --predictions validation_predictions.csv --december-predictions december_chart_predictions.csv
```

The data files are read from this folder with their provided names
(`train-test.csv`, `validation.csv`, `validation-predictions-template.csv`,
`december-chart-inputs.csv`). `predict.py` writes the filled December file to
`december_chart_predictions.csv` so the original input file stays untouched.

## Code

| File | What it does |
|---|---|
| `prepare.py` | Loading, cleaning, feature engineering. `Preprocessor.fit` learns values (median weight, expected-price formula) from training data only. |
| `model.py` | The five candidate models, all trained on log(price). |
| `train.py` | Rolling time splits, unseen-city check, model comparison, light tuning. |
| `predict.py` | Final fit on Jan-Oct, writes `validation_predictions.csv` and `december_chart_predictions.csv`. |

## Approach

**Split by time, never randomly.** Validation is Nov-Dec, after all labelled
data, so every check trains on the past and tests on the following two months:
Jan-Jun -> Jul-Aug and Jan-Aug -> Sep-Oct. A random 80/20 split looked ~40%
more accurate (1.36% vs 2.21% MAPE) because neighbouring days leak into
training. A third check holds 8 random cities out of training entirely, since
1,447 validation rows involve 8 cities never seen in training.

**Cleaning.**
- Negative weights (292 train / 145 validation) mirror the positive
  distribution exactly: a sign error, fixed with the absolute value.
- Missing weight (random, unrelated to price): training median.
- Missing `market_index`: same-day average (within-day spread 0.025 vs 0.17
  overall); for validation it comes from validation's own rows for that day.
- Corrupted labels: 677 rows (1.4%) priced at 0.15-0.5x or 2-6x the expected
  price for their distance and equipment, with an empty gap between them and
  the normal 0.8-1.25x band. Removed from training only. Test scores are
  reported both on all rows and on clean rows.

**Features.** Distance (and log), equipment, weight, market index (and log),
quote signal, pickup/delivery lat/lon (southern lanes ~0.35% pricier per
degree, stable all year; generalises to unseen cities, unlike city names),
day of week, holiday flag, days since Jan 1. `load_id` and city names dropped.

**Model: linear + trend + gradient boosting, in three stages.**
1. Linear regression on log(price) without time: the power-law distance curve
   (price ~ distance^0.87), equipment, market, weight, region.
2. A straight-line time trend on stage-1 residuals: prices drift up ~6% Jan-Oct
   beyond what the market index explains. Fitted separately because market and
   time rise together early in the year; fitting them jointly mixed the two up
   (Jul-Aug MAPE 3.67% vs 2.05%).
3. Gradient boosting on what is left (interactions). No date-trend feature,
   because trees cannot extrapolate past the training period.

## Results (MAPE on clean test rows)

| Model | Jan-Jun -> Jul-Aug | Jan-Aug -> Sep-Oct | Unseen cities |
|---|---|---|---|
| Baseline ($/mile x distance) | 7.96% | 8.09% | 7.11% |
| Linear regression on log(price) | 3.92% | 2.46% | 2.84% |
| Random forest | 3.17% | 2.28% | 2.70% |
| Gradient boosting | 3.45% | 2.20% | 2.37% |
| **Hybrid: linear + trend + boosting** | **2.18%** | **1.98%** | **2.26%** |

On all test rows (including the ~1.4% corrupted labels no model can predict)
the hybrid scores 4.43% / 4.47% / 4.30%. Full tables are in `reports/`.

---

# Original assignment instructions

See `Freight_Rate_ML_Assessment.pdf` for the assessment instructions.

## What to do

1. Train and validate your model using `data/train_test.csv`.
2. Predict every load in `data/validation.csv`. Each load has a unique `load_id`.
3. Fill the matching `predicted_rate` values in `data/validation_predictions_template.csv` and save it as `validation_predictions.csv`.
4. Predict every row in `data/december_chart_inputs.csv` by filling its `predicted_rate` column.
5. Install the scorer requirements and run:

```bash
python -m pip install -r requirements.txt
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

The scorer validates both files and creates `scorer_results/candidate_december.png`.

## Submit

- GitHub repository containing your code, dependencies, and run instructions
- `validation_predictions.csv`
- PDF or DOCX report containing your validation, data split approach and `candidate_december.png`
- 2-3 minute Loom link
