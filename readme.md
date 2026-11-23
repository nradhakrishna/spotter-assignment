# Freight Rate Prediction — Solution

## Run it

```bash
python -m pip install -r requirements.txt
python train.py      # tuning, then time-based model comparison (~5 min) -> reports/
python predict.py    # retrain on all Jan-Oct, write both prediction files
python score.py --predictions validation_predictions.csv --december-predictions december_chart_predictions.csv
python make_report.py  # builds report.pdf from the saved results (needs Chromium/Chrome)
```

The data files are read from this folder with their provided names
(`train-test.csv`, `validation.csv`, `validation-predictions-template.csv`,
`december-chart-inputs.csv`). `predict.py` writes the filled December file to
`december_chart_predictions.csv` so the original input file stays untouched.

## Code

| File | What it does |
|---|---|
| `prepare.py` | Loading, cleaning, feature engineering. `Preprocessor.fit` learns values (median weight, expected-price formula) from training data only. |
| `model.py` | The six candidate models, all trained on log(price). |
| `train.py` | Tunes first, then compares all models with the tuned settings on rolling time splits and an unseen-city check, and picks the final model. |
| `predict.py` | Final fit on Jan-Oct, writes `validation_predictions.csv` and `december_chart_predictions.csv`. |
| `make_report.py` | Recomputes every number and figure from the data and `reports/`, writes `report.pdf`. |

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

**Model: linear + trend + gradient boosting + lane offsets, in four stages.**
1. Linear regression on log(price) without time: the power-law distance curve
   (price ~ distance^0.87), equipment, market, weight, region.
2. A straight-line time trend on stage-1 residuals: prices drift up ~6% Jan-Oct
   beyond what the market index explains. Fitted separately because market and
   time rise together early in the year; fitting them jointly mixed the two up
   (Jul-Aug MAPE 3.67% vs 2.05%).
3. Gradient boosting on what is left (interactions). No date-trend feature,
   because trees cannot extrapolate past the training period.
4. A per-lane offset. The model's errors clustered by lane (lane averages
   varied ~1.7x more than noise alone explains), so each lane gets the mean
   of its out-of-fold residuals, shrunk toward zero (n / (n + 5)) so thin
   lanes barely move. Offsets are small (90% between -0.9% and +0.9%); lanes
   with no history, e.g. new cities, get none.

**Tuning before comparing.** Tree settings and lane shrinkage are tuned on the
mean error over all three checks first; the comparison table below then uses
those tuned settings, so it describes exactly the submitted model.

## Results (MAPE on clean test rows)

| Model | Jan-Jun -> Jul-Aug | Jan-Aug -> Sep-Oct | Unseen cities | Mean |
|---|---|---|---|---|
| Baseline ($/mile x distance) | 7.96% | 8.09% | 7.11% | 7.72% |
| Linear regression on log(price) | 3.92% | 2.46% | 2.84% | 3.07% |
| Random forest | 3.17% | 2.28% | 2.70% | 2.72% |
| Gradient boosting | 3.45% | 2.20% | 2.37% | 2.67% |
| Hybrid: linear + trend + boosting | 2.07% | 1.98% | 2.21% | 2.09% |
| **Hybrid + lane offsets (final)** | **2.02%** | **1.93%** | **2.21%** | **2.05%** |

On all test rows (including the ~1.4% corrupted labels, which a classifier
could not tell apart from normal rows: AUC 0.50) the final model scores
4.27% / 4.42% / 4.24%. Full tables are in `reports/`.

**Tried and rejected:** averaging 5 random seeds (no gain), blending with plain
boosting (no gain), lagged market index features (much worse: confounded with
the trend, 3.5% mean).

**Known limitation:** the trend adds ~0.64% per month, so mid-December
predictions sit ~1% above a "drift stopped in October" scenario.

The first submitted version (three-stage hybrid, before these changes) is kept
in `results_v1/`.

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
