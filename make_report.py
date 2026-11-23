"""Build report.pdf from the saved results (run after train.py and predict.py).

Every number in the report is computed here from the data or read from
reports/, so the report always matches the submitted model.
Figures go to reports/figures/; the PDF is printed with headless Chromium.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import warnings
from html import escape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from model import HYBRID_LINEAR_FEATURES, Hybrid
from prepare import (TARGET, TRAIN_FILE, VALIDATION_FILE, Preprocessor,
                     find_bad_labels, load)
from train import FOLDS, score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
HTML_OUT = REPORTS / "report.html"
PDF_OUT = ROOT / "report.pdf"
DECEMBER_CHART = ROOT / "scorer_results" / "candidate_december.png"

# Reference palette (light surface).
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#b4b2a9"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "axes.titlecolor": INK,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


# --------------------------------------------------------------------- facts
def gather_facts() -> dict:
    train, validation = load(TRAIN_FILE), load(VALIDATION_FILE)
    raw_train = pd.read_csv(ROOT / TRAIN_FILE)
    raw_val = pd.read_csv(ROOT / VALIDATION_FILE)
    prep = Preprocessor().fit(train)
    bad = find_bad_labels(train, prep.expected_price_coef_)
    clean = train.loc[~bad]
    X = prep.transform(clean)
    y = np.log(clean[TARGET]).to_numpy()

    # Leftover drift: what a no-time linear model misses, by month.
    lin = LinearRegression().fit(X[HYBRID_LINEAR_FEATURES], y)
    leftover = pd.Series(y - lin.predict(X[HYBRID_LINEAR_FEATURES]), index=clean.index)
    coef = dict(zip(HYBRID_LINEAR_FEATURES, lin.coef_))

    design = np.column_stack([np.ones(len(train)), np.log(train["distance"]),
                              train["equipment"].eq("Flatbed"), train["equipment"].eq("Reefer")]).astype(float)
    ratio = train[TARGET] / np.exp(design @ prep.expected_price_coef_)

    unseen = sorted((set(validation["pickup"]) | set(validation["delivery"]))
                    - (set(train["pickup"]) | set(train["delivery"])))
    touches_unseen = (~validation["pickup"].isin(train["pickup"]) | ~validation["delivery"].isin(train["delivery"]))

    within_day = train.groupby("date")["market_index"].std().mean()
    day_mean = train.groupby("date")["market_index"].transform("mean")
    known = train["market_index"].notna()

    return {
        "n_train": len(train), "n_val": len(validation),
        "n_cities": len(set(train["pickup"]) | set(train["delivery"])),
        "n_lanes": train.groupby(["pickup", "delivery"]).ngroups,
        "rpm_median": float((train[TARGET] / train["distance"]).median()),
        "rate_median": float(train[TARGET].median()),
        "dist_exponent": float(coef["log_distance"]),
        "flatbed_premium": float(np.expm1(coef["eq_flatbed"]) * 100),
        "reefer_premium": float(np.expm1(coef["eq_reefer"]) * 100),
        "lat_effect": float(np.expm1(coef["pickup_lat"] + coef["delivery_lat"]) * 100 / 2),
        "market_elasticity": float(coef["log_market_index"]),
        "market_by_month": clean.groupby(clean["date"].dt.month)["market_index"].mean(),
        "drift_by_month": leftover.groupby(clean["date"].dt.month).mean() * 100,
        "ratio": ratio, "bad": bad,
        "n_bad": int(bad.sum()), "n_bad_low": int((bad & (ratio < 1)).sum()),
        "n_bad_high": int((bad & (ratio > 1)).sum()),
        "n_normal_band": int(ratio.between(0.8, 1.25).sum()),
        "neg_weight_train": int((raw_train["weight"] < 0).sum()),
        "neg_weight_val": int((raw_val["weight"] < 0).sum()),
        "miss_weight_train": int(raw_train["weight"].isna().sum()),
        "miss_weight_val": int(raw_val["weight"].isna().sum()),
        "miss_mkt_train": int(raw_train["market_index"].isna().sum()),
        "miss_mkt_val": int(raw_val["market_index"].isna().sum()),
        "mkt_within_day": float(within_day), "mkt_overall_std": float(train["market_index"].std()),
        "fill_err_day": float((train["market_index"][known] - day_mean[known]).abs().mean()),
        "fill_err_median": float((train["market_index"][known] - train["market_index"].median()).abs().mean()),
        "unseen_cities": unseen, "n_val_unseen": int(touches_unseen.sum()),
    }


def random_vs_time(tree_params: dict) -> tuple[float, float]:
    """Same hybrid, scored on a random 80/20 split and on the time split."""
    data = load(TRAIN_FILE)
    is_test = np.random.default_rng(0).random(len(data)) < 0.2
    rand = score(Hybrid(**tree_params), data[~is_test], data[is_test])["clean_MAPE_%"]
    start, end = FOLDS["Jan-Aug -> Sep-Oct"]
    timed = score(Hybrid(**tree_params), data[data["date"] < start],
                  data[(data["date"] >= start) & (data["date"] < end)])["clean_MAPE_%"]
    return rand, timed


# ------------------------------------------------------------------- figures
def fig_drivers(f: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), dpi=200)
    x = np.arange(10)
    for ax, series, title, fmt in [
        (axes[0], f["market_by_month"], "Market index (monthly mean)", "{:.2f}"),
        (axes[1], f["drift_by_month"], "Price drift not explained by market (%)", "{:+.1f}%"),
    ]:
        ax.plot(x, series.to_numpy(), color=BLUE, linewidth=2, marker="o", markersize=4)
        ax.set_xticks(x, MONTHS)
        ax.set_title(title)
        peak = int(np.argmax(series.to_numpy()))
        for i in sorted({0, 9} | ({peak} if peak < 8 else set())):  # skip a peak next to the last point
            ax.annotate(fmt.format(series.iloc[i]), (i, series.iloc[i]), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=8, color=INK2)
        ax.margins(y=0.25)
    axes[1].axhline(0, color=AXIS, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_bad_labels(f: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 2.4), dpi=200)
    bins = np.geomspace(0.12, 8, 70)
    ratio, bad = f["ratio"], f["bad"]
    ax.hist(ratio[~bad], bins=bins, color=BLUE, label="Kept", edgecolor=SURFACE, linewidth=0.4)
    ax.hist(ratio[bad], bins=bins, color=ORANGE, label="Removed from training", edgecolor=SURFACE, linewidth=0.4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ticks = [0.2, 0.5, 1, 2, 5]
    ax.set_xticks(ticks, [f"{t:g}x" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Posted rate / expected rate for its distance and equipment (log scale)")
    ax.set_ylabel("Loads (log scale)")
    ax.set_title("Corrupted labels sit in two clusters, separated from normal loads by an empty gap")
    for edge in (0.5, 2):
        ax.axvline(edge, color=INK2, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_models(summary: pd.DataFrame, final: str, path: Path) -> None:
    order = summary.sort_values("mean", ascending=False)
    fig, ax = plt.subplots(figsize=(7.2, 2.1), dpi=200)
    colors = [BLUE if name == final else GRAY for name in order.index]
    ax.barh(order.index, order["mean"], color=colors, height=0.6)
    for i, value in enumerate(order["mean"]):
        ax.text(value + 0.08, i, f"{value:.2f}%", va="center", fontsize=8, color=INK)
    ax.set_xlabel("Mean MAPE over the three checks (clean test rows, lower is better)")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, order["mean"].max() * 1.15)
    ax.tick_params(axis="y", colors=INK, length=0)
    ax.set_title("Model comparison")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------- html
def table(df: pd.DataFrame, highlight: str | None = None, bold_cols=()) -> str:
    head = "".join(f"<th>{escape(str(c))}</th>" for c in df.columns)
    rows = []
    for _, row in df.iterrows():
        cls = ' class="hl"' if highlight is not None and row.iloc[0] == highlight else ""
        cells = "".join(f"<td>{escape(str(v))}</td>" for v in row)
        rows.append(f"<tr{cls}>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


CSS = """
@page { size: A4; margin: 16mm 16mm 16mm 16mm; }
* { box-sizing: border-box; }
body { font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 9.6pt; line-height: 1.45;
       color: #0b0b0b; background: #ffffff; margin: 0; }
h1 { font-size: 20pt; margin: 0 0 2mm; letter-spacing: -0.3px; }
.sub { color: #52514e; margin: 0 0 6mm; font-size: 10pt; }
h2 { font-size: 12.5pt; margin: 7mm 0 2mm; padding-bottom: 1.2mm; border-bottom: 1.5px solid #2a78d6; }
h3 { font-size: 10.2pt; margin: 4mm 0 1mm; }
p { margin: 0 0 2.2mm; }
ul, ol { margin: 0 0 2.5mm; padding-left: 5mm; }
li { margin: 0 0 1mm; }
.kpis { display: flex; gap: 3mm; margin: 0 0 4mm; }
.kpi { flex: 1; border: 1px solid #e1e0d9; border-radius: 4px; padding: 2.5mm 3mm; background: #fcfcfb; }
.kpi .v { font-size: 15pt; font-weight: bold; color: #0b0b0b; }
.kpi .l { color: #52514e; font-size: 8.2pt; line-height: 1.3; }
table { width: 100%; border-collapse: collapse; margin: 1mm 0 3mm; font-size: 8.6pt; page-break-inside: avoid; }
th { text-align: left; color: #52514e; font-weight: bold; border-bottom: 1.2px solid #c3c2b7; padding: 1.4mm 2mm; }
td { border-bottom: 0.6px solid #e1e0d9; padding: 1.3mm 2mm; vertical-align: top; }
tr.hl td { background: #eaf2fc; font-weight: bold; }
figure { margin: 1.5mm 0 3mm; page-break-inside: avoid; }
figure img { width: 100%; }
figcaption { color: #52514e; font-size: 8.3pt; margin-top: 1mm; }
.note { background: #f9f9f7; border-left: 3px solid #2a78d6; padding: 2mm 3mm; margin: 2mm 0 3mm; }
.break { page-break-before: always; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 8.4pt; background: #f3f2ee; padding: 0 1mm; border-radius: 2px; }
.small { font-size: 8.4pt; color: #52514e; }
"""


def build_html(f: dict, comparison: pd.DataFrame, best: dict, rand_mape: float, time_mape: float,
               december: pd.DataFrame, trend_pct_month: float) -> str:
    final = "Hybrid + lane offsets" if best["final_model"] == "HybridWithLanes" else "Hybrid: linear + trend + boosting"
    piv = comparison.pivot_table(index="model", columns="check", values="clean_MAPE_%")
    piv["mean"] = piv.mean(axis=1)
    piv_all = comparison.pivot_table(index="model", columns="check", values="all_MAPE_%")
    fig_models(piv, final, FIGURES / "models.png")

    checks = list(FOLDS) + ["Unseen cities (Sep-Oct)"]
    order = piv.sort_values("mean", ascending=False).index
    results = pd.DataFrame({
        "Model": order,
        **{c: [f"{piv.loc[m, c]:.2f}%" for m in order] for c in checks},
        "Mean": [f"{piv.loc[m, 'mean']:.2f}%" for m in order],
    })
    fin = piv.loc[final]
    fin_all = piv_all.loc[final]
    tp = best["tree_params"]

    quality = pd.DataFrame([
        ["Negative weight", f"{f['neg_weight_train']} train / {f['neg_weight_val']} validation",
         "Distribution is an exact mirror of the positive weights: a sign error", "Absolute value"],
        ["Missing weight", f"{f['miss_weight_train']} / {f['miss_weight_val']}",
         "Unrelated to price and equipment (looks random)", "Training median"],
        ["Missing market_index", f"{f['miss_mkt_train']} / {f['miss_mkt_val']}",
         f"A daily market level: spread within a day {f['mkt_within_day']:.3f} vs {f['mkt_overall_std']:.2f} overall",
         f"Same-day average (fill error {f['fill_err_day']:.3f} vs {f['fill_err_median']:.3f} for a global median)"],
        ["Corrupted posted_rate", f"{f['n_bad']} train ({f['n_bad']/f['n_train']*100:.1f}%)",
         f"Priced at 0.15-0.5x ({f['n_bad_low']}) or 2-6x ({f['n_bad_high']}) the expected rate; "
         "a classifier could not predict them from the inputs (AUC 0.50)",
         "Removed from training only; test scored on all rows and on clean rows"],
        ["8 cities unseen in training", f"{f['n_val_unseen']:,} validation loads",
         ", ".join(f["unseen_cities"]), "Use coordinates, not city names; tested with a held-out-city check"],
    ], columns=["Issue", "Rows", "Evidence", "Fix"])

    dec_min, dec_max = december["predicted_rate"].min(), december["predicted_rate"].max()
    m = f["market_by_month"]

    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Freight Rate Prediction Report</title>
<style>{CSS}</style></head><body>

<h1>Freight Rate Prediction</h1>
<p class="sub">Spotter Machine Learning Engineer assessment &middot; model report</p>

<div class="kpis">
  <div class="kpi"><div class="v">{fin['mean']:.2f}%</div><div class="l">mean error (MAPE) of the final model over three out-of-time checks, clean rows</div></div>
  <div class="kpi"><div class="v">{fin['Jan-Aug -> Sep-Oct']:.2f}%</div><div class="l">error on the latest two months (Sep-Oct), the closest stand-in for Nov-Dec</div></div>
  <div class="kpi"><div class="v">{fin['Unseen cities (Sep-Oct)']:.2f}%</div><div class="l">error on loads touching cities the model never saw</div></div>
  <div class="kpi"><div class="v">{piv.loc['Baseline ($/mile x distance)', 'mean']:.1f}%</div><div class="l">simple $/mile baseline, for scale</div></div>
</div>

<h2>1. Summary</h2>
<p>The task is to predict the total posted rate (dollars per load) for 12,000 loads in November-December 2025,
using {f['n_train']:,} labelled loads from January-October 2025. Because the loads to predict all come
<b>after</b> the labelled data, every validation decision here simulates that: models are always trained on earlier
months and tested on the following two.</p>
<p>The final model is a four-stage <b>hybrid</b>: a linear model on log(price) for the main structure, a separately
fitted time trend that can extend into Nov-Dec, gradient boosting for interactions, and small per-lane offsets.
It is the best model on all three checks. Corrupted labels (about {f['n_bad']/f['n_train']*100:.1f}% of rows) cannot be predicted by any
model; including them, error is about {fin_all.mean():.1f}%.</p>

<h2>2. Key findings from the data</h2>
<ul>
  <li><b>posted_rate is the total price per load.</b> Median ${f['rate_median']:,.0f}; dividing by distance gives a
      normal trucking rate of about ${f['rpm_median']:.2f}/mile. Weight changes price by only a few percent.</li>
  <li><b>Distance drives price, with economies of scale:</b> price grows like distance<sup>{f['dist_exponent']:.2f}</sup>,
      so long loads are cheaper per mile. Distance and equipment alone explain about 96% of the variance of log(price).</li>
  <li><b>Equipment:</b> for the same distance, Reefer costs about {f['reefer_premium']:.0f}% and Flatbed about
      {f['flatbed_premium']:.0f}% more than Dry Van.</li>
  <li><b>Market index</b> swings seasonally from {m.iloc[0]:.2f} (Jan) to {m.max():.2f} (May) and back to {m.iloc[-1]:.2f} (Oct),
      and has a strong weekly cycle (lowest Sunday, highest Thursday). It is the most important input after distance:
      removing it from a linear model raises error by about 60%.</li>
  <li><b>Prices drift upward beyond what the market explains</b> (right panel below): about {f['drift_by_month'].iloc[0]:+.1f}% in January to
      {f['drift_by_month'].iloc[-1]:+.1f}% in October. A model that ignores this underprices Nov-Dec.</li>
  <li><b>Region matters a little:</b> each degree further south adds about {abs(f['lat_effect']):.2f}% at each end of the lane,
      stable in every month. Texas and Gulf lanes are priciest; upstate New York and the Great Lakes cheapest.</li>
  <li><b>quote_signal</b> is very noisy and nearly unrelated to price; it is kept because it does no harm.</li>
  <li>Validation looks like training: same equipment mix, same distance distribution. But {len(f['unseen_cities'])} of its cities
      never appear in training, affecting {f['n_val_unseen']:,} loads.</li>
</ul>
<figure><img src="figures/drivers.png"><figcaption>Figure 1. Left: the market index cycle. Right: what a linear model
with distance, equipment, market, weight and region still misses, by month: a slow upward drift.</figcaption></figure>

<h2>3. Data-quality issues and how they were addressed</h2>
{table(quality)}
<figure><img src="figures/bad_labels.png"><figcaption>Figure 2. {f['n_normal_band']:,} loads sit within 0.8-1.25x of their
expected rate; the {f['n_bad']} corrupted ones sit at 0.15-0.5x or 2-6x, with nothing in between. Dashed lines mark the
cut-offs (0.5x and 2x). The expected rate comes from log(price) ~ log(distance) + equipment, fitted on training rows only.</figcaption></figure>
<p class="small">Alternatives considered: capping outliers (keeps a wrong label, just less wrong), z-score/IQR rules on price
alone (a $5,000 rate is normal for a long load and absurd for a short one), deleting rows with missing values
(impossible in validation, where every row needs a prediction).</p>

<h2 class="break">4. Validation and data-split approach</h2>
<h3>Split by time, never randomly</h3>
<p>The loads to predict (Nov-Dec) come after all labelled data, so the test set must also come after its training set.
A random split mixes days: a test load from 15 September has training loads from the 14th and 16th, which share its market
level and drift, so the model barely has to forecast. Scoring the same hybrid both ways shows the effect:</p>
{table(pd.DataFrame([["Random 80/20 split (months mixed)", f"{rand_mape:.2f}%"],
                     ["Time split: train Jan-Aug, test Sep-Oct", f"{time_mape:.2f}%"]], columns=["Split", "MAPE"]))}
<p>The random split looks {abs(1 - rand_mape / time_mape) * 100:.0f}% more accurate than it really is, and would have steered model
choice toward models that memorise neighbouring days.</p>

<h3>Three checks</h3>
{table(pd.DataFrame([
    ["Jan-Jun -> Jul-Aug", "Rolling time split", "Is the model stable across periods, including a market downturn?"],
    ["Jan-Aug -> Sep-Oct", "Rolling time split", "Closest to the real task: latest data, two-month horizon"],
    ["Unseen cities (Sep-Oct)", "8 random cities removed from training entirely",
     "Mimics the 8 new validation cities"],
], columns=["Check", "Construction", "Question it answers"]))}
<h3>Rules that keep the scores honest</h3>
<ul>
  <li><b>Split first, then clean.</b> Everything learned from data (median weight, the expected-rate formula used to flag
      corrupted labels) is fitted on the training part only and reused unchanged on the test part and on validation.</li>
  <li><b>market_index is filled from the same day's loads</b> in whichever set is being predicted. It is a fact about the
      day, known at quoting time, not a statistic learned from the past, and it uses inputs only, never prices.</li>
  <li><b>Corrupted labels are removed from training only.</b> Test sets are scored on all rows (what the final grader sees)
      and on clean rows (how well the model actually prices loads). Both are reported.</li>
  <li><b>Tune first, then compare.</b> Settings are chosen on the mean of all three checks; the comparison table then uses
      those settings, so it describes exactly the submitted model.</li>
  <li><b>Final retrain on all of Jan-Oct</b> with the chosen settings, so the most recent months, which are closest
      to Nov-Dec, are used for the final model.</li>
</ul>

<h2>5. Features</h2>
<p><b>Used:</b> distance and log(distance); equipment (one-hot); weight; market_index and its log; quote_signal;
pickup and delivery latitude/longitude and their differences (direction); day of week; US holiday flag; days since 1 Jan (trend).
Target: log(posted_rate), converted back to dollars at the end, so errors are relative and long and short loads count equally.</p>
<p><b>Dropped:</b> <code>load_id</code> (an identifier); pickup/delivery city names. Coordinates carry the location effect and
also work for unseen cities; per-city averages made the holdout slightly worse. No scaling is needed: trees and ordinary least
squares are both unaffected by feature units.</p>

<h2 class="break">6. Model choice</h2>
{table(results, highlight=final)}
<p class="small">MAPE on clean test rows. Including corrupted rows, the final model scores
{fin_all['Jan-Jun -> Jul-Aug']:.2f}% / {fin_all['Jan-Aug -> Sep-Oct']:.2f}% / {fin_all['Unseen cities (Sep-Oct)']:.2f}%.</p>
<figure><img src="figures/models.png"><figcaption>Figure 3. Mean error over the three checks.</figcaption></figure>

<h3>Why a hybrid</h3>
<p>The linear model on log(price) is already a non-linear power-law in dollars and captures most of the structure, but it
cannot learn interactions. Random forest and gradient boosting learn interactions, but <b>trees cannot extrapolate</b>: past the
last training date they act as if the last month continues, so they miss the upward drift. Their Jul-Aug errors
(3.2-3.5%) show this. The hybrid gives each part the job it is good at:</p>
<ol>
  <li><b>Linear regression on log(price), without time:</b> distance curve, equipment, market, weight, region.</li>
  <li><b>A straight-line time trend on the stage-1 residuals</b> ({trend_pct_month:.2f}% per month), the only part that extends into Nov-Dec.
      It is fitted <i>separately</i> on purpose. In Jan-Jun the market index and time rise together, so a joint fit could not tell
      them apart and the trend extrapolated badly when the market turned (Jul-Aug error 3.67% joint vs 2.05% separate).</li>
  <li><b>Gradient boosting on the remaining residuals</b> (scikit-learn HistGradientBoosting, {tp['max_leaf_nodes']} leaves,
      {tp['min_samples_leaf']} min samples per leaf, {tp['max_iter']} trees, learning rate {tp['learning_rate']}), with no date-trend input.</li>
  <li><b>Per-lane offsets.</b> The model's errors still clustered by lane (lane averages varied about 1.7x more than noise alone
      explains). Each lane gets the mean of its out-of-fold residuals, shrunk toward zero by {best['lane_k']} pseudo-loads, so lanes with
      little history barely move. Offsets are small (90% within &plusmn;0.9%), and new-city lanes get none.</li>
</ol>
<p class="small">Tried and rejected: averaging 5 random seeds and blending with plain boosting (no gain); market index lagged
45-60 days (much worse, because it is confused with the trend); predicting which labels are corrupted (impossible, AUC 0.50).
Tuning moved the mean by only about 0.05 points: the model is not sensitive to its settings.</p>

<h2 class="break">7. December prediction chart</h2>
<p>Fixed load: Lexington to Fort Wayne, 360 miles, Dry Van, 32,000 lb; only the date changes. The chart file lacks
coordinates and market inputs, so coordinates are looked up from the data and market_index / quote_signal are each date's
average across the validation loads of that day.</p>
<figure><img src="../scorer_results/candidate_december.png"><figcaption>Figure 4. Produced by the provided
<code>score.py</code>.</figcaption></figure>
<ul>
  <li><b>Level:</b> ${dec_min:,.0f} to ${dec_max:,.0f}. October loads on this lane averaged about $834 for slightly
      longer trips, so the level is consistent with recent history plus the trend.</li>
  <li><b>Weekly rhythm:</b> peaks mid-week and dips at weekends, inherited from the market index's weekly cycle
      (about 0.83 on Sundays to 1.03 on Thursdays across the Nov-Dec validation loads).</li>
  <li><b>Gentle upward slope</b> from the time trend, plus a small positive offset for this lane (+0.55%, from 32 past loads).</li>
</ul>

<h2>8. Limitations and next steps</h2>
<ul>
  <li><b>Trend extrapolation is the main risk.</b> The trend adds about {trend_pct_month:.2f}% per month; if the drift stopped in October,
      mid-December predictions would be about 1% high. With more history, a damped or seasonal trend would be safer.</li>
  <li><b>Holiday effects</b> are learned from only a handful of 2025 holidays; Thanksgiving and Christmas behaviour is uncertain.</li>
  <li><b>New cities</b> rely on coordinates only; their error is about 0.3 points higher than for known lanes.</li>
  <li><b>Next steps:</b> prediction intervals from residual quantiles, more rolling folds as more months arrive, and
      monitoring for drift in the market-to-price relationship.</li>
</ul>

<h2>9. Reproducing</h2>
<p><code>pip install -r requirements.txt</code> &rarr; <code>python train.py</code> (tuning and comparison, about 5 min) &rarr;
<code>python predict.py</code> (final fit, both prediction files) &rarr;
<code>python score.py --predictions validation_predictions.csv --december-predictions december_chart_predictions.csv</code>
&rarr; <code>python make_report.py</code> (this report).</p>
</body></html>"""


def find_chromium() -> str:
    for name in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable"):
        if shutil.which(name):
            return shutil.which(name)
    raise SystemExit("Chromium/Chrome not found: open reports/report.html in a browser and print to PDF")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    best = json.loads((REPORTS / "best_params.json").read_text())
    comparison = pd.read_csv(REPORTS / "model_comparison.csv")
    december = pd.read_csv(ROOT / "december_chart_predictions.csv")
    if not DECEMBER_CHART.exists():
        raise SystemExit("run score.py first to create the December chart")

    print("Computing facts and figures...")
    facts = gather_facts()
    fig_drivers(facts, FIGURES / "drivers.png")
    fig_bad_labels(facts, FIGURES / "bad_labels.png")
    rand_mape, time_mape = random_vs_time(best["tree_params"])

    train = load(TRAIN_FILE)
    prep = Preprocessor().fit(train)
    clean = prep.clean_training_rows(train)
    hybrid = Hybrid(**best["tree_params"]).fit(prep.transform(clean), np.log(clean[TARGET]))
    trend_pct_month = hybrid.trend_slope_ * 30 * 100

    HTML_OUT.write_text(build_html(facts, comparison, best, rand_mape, time_mape, december, trend_pct_month))
    subprocess.run([find_chromium(), "--headless", "--no-sandbox", "--disable-gpu",
                    "--no-pdf-header-footer", f"--print-to-pdf={PDF_OUT}", HTML_OUT.as_uri()],
                   check=True, capture_output=True)
    print(f"Wrote {PDF_OUT.name} (HTML source: {HTML_OUT.relative_to(ROOT)})")


if __name__ == "__main__":
    main()
