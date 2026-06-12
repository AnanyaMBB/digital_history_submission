"""Backtested early-warning analysis for the four BoE crises.

Question: could pre-crisis BoE balance-sheet signals have warned of the
crises of 1857, 1866, 1890 and 1914 *before* their canonical trigger dates?

The design is strict on leakage:
- All rolling statistics use only past observations. A rolling mean/std for
  week t is computed from observations strictly before t (achieved via
  `.shift(1).expanding(...)` chains in pandas).
- The transparent early-warning index is converted to a percentile rank
  using only the expanding history of past index values.
- The Isolation-Forest backtest is fit *only* on data before
  (trigger - 12 weeks) AND with prior crisis acute windows (including the
  1847 panic, although it is not one of our four study crises, plus any of
  the four crises that precede the one being tested) removed from the
  training set so that crisis weeks do not contaminate the "normal"
  distribution the detector learns.

Outputs in `outputs/tables/`:
- early_warning_scores.csv           — weekly index + components
- early_warning_crisis_summary.csv   — per-crisis pre-trigger summary
- early_warning_placebo.csv          — placebo distribution + percentiles

Outputs in `outputs/figures/`:
- early_warning_timeline.png         — 1855-1916 timeline with crisis bars
- early_warning_event_time.png       — score in weeks-before-trigger

Important interpretive rules (followed throughout the paper too):
- We say the crisis "was visible in the Bank's own data before the trigger"
  when the pre-trigger score exceeds an explicit threshold relative to
  placebo. We do NOT say the model "predicted" the crisis.
- 1857 has the least pre-1857 training history (about 13 years of weekly
  observations once the 1847 acute window is excluded). It is reported
  with an explicit caveat.
"""
from __future__ import annotations
import sys
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from io_utils import read_balance_sheet, read_bank_rate  # noqa: E402
from crisis_windows import CRISES  # noqa: E402

PROC = ROOT / "data" / "processed"
TBL = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures"

# Acute windows to remove from training so they don't contaminate the
# "normal" distribution the Isolation Forest learns. Includes 1847 (which
# is in the source data but not one of our study crises) plus the four
# study crises.
PRIOR_ACUTE_WINDOWS = [
    (pd.Timestamp("1847-10-01"), pd.Timestamp("1847-12-31")),  # 1847 panic
] + [(c.acute_start, c.acute_end) for c in CRISES.values()]


# -----------------------------------------------------------------------------
# Feature construction — strictly past-only
# -----------------------------------------------------------------------------

def _pct_change(s: pd.Series, periods: int) -> pd.Series:
    return s.pct_change(periods=periods).replace([np.inf, -np.inf], np.nan)


def _diff(s: pd.Series, periods: int) -> pd.Series:
    return s.diff(periods=periods).replace([np.inf, -np.inf], np.nan)


def _past_z(s: pd.Series, min_periods: int = 52) -> pd.Series:
    """Z-score against an expanding window of strictly past observations.

    z(t) = (s(t) - mean(s[:t])) / std(s[:t]) using shift(1) to enforce no
    look-ahead.
    """
    past = s.shift(1)
    mu = past.expanding(min_periods=min_periods).mean()
    sd = past.expanding(min_periods=min_periods).std()
    z = (s - mu) / sd.replace(0, np.nan)
    return z


def _past_rolling_mean(s: pd.Series, window: int) -> pd.Series:
    return s.shift(1).rolling(window, min_periods=window // 2).mean()


def _expanding_pct_rank(s: pd.Series, min_periods: int = 52) -> pd.Series:
    """Percentile rank of s(t) within all prior values s[:t] (strictly before).

    Returns NaN until `min_periods` past observations exist.
    """
    out = pd.Series(np.nan, index=s.index, dtype=float)
    arr = s.values
    for i in range(len(arr)):
        if i < min_periods:
            continue
        past = arr[:i]
        past = past[~np.isnan(past)]
        if len(past) < min_periods:
            continue
        cur = arr[i]
        if np.isnan(cur):
            continue
        out.iloc[i] = float((past <= cur).mean() * 100.0)
    return out


def build_features(bs: pd.DataFrame, br_daily: pd.DataFrame) -> pd.DataFrame:
    df = bs.copy()
    df["bank_rate_aligned"] = br_daily.set_index("date")["bank_rate"].reindex(df.index, method="nearest")

    # Levels of interest
    levels = {
        "reserve": df["banking_reserve_notes_coin"],
        "crisis_lending": df["crisis_lending"],
        "reserve_ratio": df["reserve_ratio"],
        "lending_to_reserve": df["lending_to_reserve"],
        "bank_rate": df["bank_rate_aligned"],
    }

    feats = pd.DataFrame(index=df.index)

    # 4 and 8-week percentage changes (level series) / absolute changes (rate)
    for name, s in levels.items():
        if name == "bank_rate":
            feats[f"{name}_change_4w"] = _diff(s, 4)
            feats[f"{name}_change_8w"] = _diff(s, 8)
        else:
            feats[f"{name}_pct_change_4w"] = _pct_change(s, 4)
            feats[f"{name}_pct_change_8w"] = _pct_change(s, 8)

    # 52-week gaps (level minus 52w rolling past mean)
    for name in ("reserve_ratio", "crisis_lending", "lending_to_reserve"):
        feats[f"{name}_gap_52w"] = levels[name] - _past_rolling_mean(levels[name], 52)

    # Rolling z-scores (past-only)
    for name in ("reserve_ratio", "crisis_lending", "lending_to_reserve", "bank_rate"):
        feats[f"rolling_z_{name}"] = _past_z(levels[name], min_periods=52)

    return feats


# -----------------------------------------------------------------------------
# Transparent early-warning index
# -----------------------------------------------------------------------------

STRESS_COMPONENTS = [
    # (name, source_column, sign)
    # sign=+1 means "high values = stress"; -1 means "low values = stress"
    ("reserve_stress",            "rolling_z_reserve_ratio",        -1),
    ("lending_stress",            "rolling_z_crisis_lending",       +1),
    ("lending_to_reserve_stress", "rolling_z_lending_to_reserve",   +1),
    ("rate_stress",               "rolling_z_bank_rate",            +1),
    ("reserve_drawdown_stress",   "reserve_ratio_pct_change_8w",    -1),
    ("lending_surge_stress",      "crisis_lending_pct_change_8w",   +1),
]


def build_warning_index(feats: pd.DataFrame) -> pd.DataFrame:
    components = pd.DataFrame(index=feats.index)
    for name, col, sign in STRESS_COMPONENTS:
        if col not in feats.columns:
            continue
        components[name] = sign * feats[col]
    # Mean of *available* standardized components per week
    components["warning_index"] = components.mean(axis=1, skipna=True)
    # Percentile rank using strictly past observations
    components["warning_score"] = _expanding_pct_rank(components["warning_index"], min_periods=52)
    return components


# -----------------------------------------------------------------------------
# ML backtest — Isolation Forest with strict no-leakage training
# -----------------------------------------------------------------------------

ML_FEATURE_COLS = [
    "reserve_pct_change_4w", "reserve_pct_change_8w",
    "crisis_lending_pct_change_4w", "crisis_lending_pct_change_8w",
    "reserve_ratio_pct_change_4w", "reserve_ratio_pct_change_8w",
    "lending_to_reserve_pct_change_4w", "lending_to_reserve_pct_change_8w",
    "bank_rate_change_4w", "bank_rate_change_8w",
    "reserve_ratio_gap_52w", "crisis_lending_gap_52w", "lending_to_reserve_gap_52w",
]


def _exclude_acute(idx: pd.DatetimeIndex, windows: list[tuple[pd.Timestamp, pd.Timestamp]]) -> np.ndarray:
    mask = np.ones(len(idx), dtype=bool)
    for lo, hi in windows:
        mask &= ~((idx >= lo) & (idx <= hi))
    return mask


def run_ml_backtest(feats: pd.DataFrame, crisis_key: str) -> pd.DataFrame:
    c = CRISES[crisis_key]
    train_end = c.acute_peak - pd.Timedelta(weeks=12)
    test_start = c.acute_peak - pd.Timedelta(weeks=12)
    test_end = c.acute_peak - pd.Timedelta(days=1)

    cols = [c_ for c_ in ML_FEATURE_COLS if c_ in feats.columns]
    train_idx = feats.index < train_end
    train_idx &= _exclude_acute(feats.index, PRIOR_ACUTE_WINDOWS)
    train = feats.loc[train_idx, cols].dropna()

    if len(train) < 100:
        # Not enough training data — 1857 may fall here
        return pd.DataFrame(columns=["date", "iso_score", "iso_flag", "crisis"])

    model = IsolationForest(
        n_estimators=400, contamination=0.05, random_state=42,
        max_samples=min(512, len(train)),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(train.values)

    test = feats.loc[(feats.index >= test_start) & (feats.index <= test_end), cols].fillna(0.0)
    raw = -model.decision_function(test.values)
    # Normalise to [0,1] using the training-set raw scores so the test-window
    # scores have a meaningful reference. Anything > train-set max is >1 and
    # clipped to 1.
    train_raw = -model.decision_function(train.values)
    lo, hi = float(train_raw.min()), float(train_raw.max())
    scaled = np.clip((raw - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    flag = pd.Series(model.predict(test.values), index=test.index) == -1
    out = pd.DataFrame({
        "date": test.index,
        "iso_score": scaled,
        "iso_flag": flag.values,
        "crisis": crisis_key,
    })
    return out


# -----------------------------------------------------------------------------
# Crisis summary + placebo
# -----------------------------------------------------------------------------

def crisis_summary(comp: pd.DataFrame, ml_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for key, c in CRISES.items():
        trigger = c.acute_peak
        rows_for = {}
        for weeks in (12, 8, 4):
            start = trigger - pd.Timedelta(weeks=weeks)
            sub = comp.loc[start:trigger - pd.Timedelta(days=1)]
            scores = sub["warning_score"].dropna()
            if scores.empty:
                rows_for[f"max_warning_score_{weeks}w"] = np.nan
                rows_for[f"first_warning_date_{weeks}w"] = ""
                rows_for[f"weeks_before_trigger_first_warning_{weeks}w"] = np.nan
                continue
            rows_for[f"max_warning_score_{weeks}w"] = round(float(scores.max()), 1)
            # Threshold for "warning" = score >= 80 (percentile of historical distribution)
            above = scores[scores >= 80]
            if len(above):
                first = above.index.min()
                rows_for[f"first_warning_date_{weeks}w"] = first.date().isoformat()
                rows_for[f"weeks_before_trigger_first_warning_{weeks}w"] = \
                    int((trigger - first).days / 7)
            else:
                rows_for[f"first_warning_date_{weeks}w"] = ""
                rows_for[f"weeks_before_trigger_first_warning_{weeks}w"] = np.nan

        # ML
        ml_df = ml_dfs.get(key, pd.DataFrame())
        if not ml_df.empty:
            ml12 = ml_df
            rows_for["max_iso_score_12w"] = round(float(ml12["iso_score"].max()), 3)
            first_flag = ml12[ml12["iso_flag"]]["date"]
            if len(first_flag):
                rows_for["first_isolation_flag_date"] = first_flag.min().date().isoformat()
            else:
                rows_for["first_isolation_flag_date"] = ""
            rows_for["ml_status"] = "ok"
        else:
            rows_for["max_iso_score_12w"] = np.nan
            rows_for["first_isolation_flag_date"] = ""
            rows_for["ml_status"] = "insufficient_training_data"

        # Dominant features in 12w pre-trigger window
        start = trigger - pd.Timedelta(weeks=12)
        comps12 = comp.loc[start:trigger - pd.Timedelta(days=1)][[c_[0] for c_ in STRESS_COMPONENTS]]
        if not comps12.empty:
            means = comps12.mean().sort_values(ascending=False)
            rows_for["dominant_warning_features"] = ", ".join(
                f"{n}={round(float(v), 2)}" for n, v in means.head(3).items() if not np.isnan(v)
            )
        else:
            rows_for["dominant_warning_features"] = ""

        rows.append({
            "crisis": key, "trigger_date": trigger.date().isoformat(), **rows_for,
        })
    return pd.DataFrame(rows)


def placebo_test(comp: pd.DataFrame, n_samples: int = 1000,
                   window_weeks: int = 12, exclude_days: int = 365,
                   rng_seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(rng_seed)
    anchors = [c.acute_peak for c in CRISES.values()]
    t_min, t_max = comp.index.min(), comp.index.max()
    span_days = (t_max - t_min).days

    def _allowed(d: pd.Timestamp) -> bool:
        if d - pd.Timedelta(weeks=window_weeks) < t_min:
            return False
        if d + pd.Timedelta(days=1) > t_max:
            return False
        return all(abs((d - a).days) > exclude_days for a in anchors)

    samples = []
    attempts = 0
    while len(samples) < n_samples and attempts < n_samples * 50:
        attempts += 1
        offset = int(rng.integers(0, max(1, span_days)))
        d = t_min + pd.Timedelta(days=offset)
        if not _allowed(d):
            continue
        start = d - pd.Timedelta(weeks=window_weeks)
        sub = comp.loc[start:d - pd.Timedelta(days=1)]
        scores = sub["warning_score"].dropna()
        if scores.empty:
            continue
        samples.append({
            "center": d.date().isoformat(),
            "max_warning_score_12w": float(scores.max()),
            "weeks_above_80_12w": int((scores >= 80).sum()),
            "mean_score_12w": float(scores.mean()),
        })
    return pd.DataFrame(samples)


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------

def figure_timeline(comp: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.plot(comp.index, comp["warning_score"], color="#2c3e50", linewidth=0.7)
    ax.axhline(80, color="#c0392b", linewidth=0.7, linestyle="--", label="warning threshold (score=80)")
    for key, c in CRISES.items():
        win_start = c.acute_peak - pd.Timedelta(weeks=12)
        ax.axvspan(win_start, c.acute_peak, color="#f39c12", alpha=0.20)
        ax.axvline(c.acute_peak, color="#c0392b", linewidth=0.7, linestyle=":")
        ax.annotate(key, (c.acute_peak, 102), ha="center", fontsize=8, color="#7f8c8d")
    ax.set_ylim(0, 105)
    ax.set_xlim(pd.Timestamp("1855-01-01"), pd.Timestamp("1916-12-31"))
    ax.set_ylabel("Early-warning score (percentile of past history)")
    ax.set_title("Transparent early-warning score, 1855–1916 (pre-crisis 12-week windows shaded)")
    ax.grid(linestyle=":", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    plt.tight_layout()
    out = FIG / "early_warning_timeline.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")


def figure_event_time(comp: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    colors = {"1857": "#2980b9", "1866": "#16a085", "1890": "#f39c12", "1914": "#c0392b"}
    for key, c in CRISES.items():
        start = c.acute_peak - pd.Timedelta(weeks=12)
        sub = comp.loc[start:c.acute_peak - pd.Timedelta(days=1), "warning_score"].dropna()
        if sub.empty:
            continue
        weeks_from_trigger = (sub.index - c.acute_peak) / pd.Timedelta(weeks=1)
        ax.plot(weeks_from_trigger, sub.values, label=f"{c.name}",
                color=colors[key], linewidth=1.6)
    ax.axhline(80, color="#c0392b", linewidth=0.7, linestyle="--", label="warning threshold")
    ax.axvline(0, color="#444", linewidth=0.6)
    ax.set_xlim(-12, 0)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Weeks before canonical trigger")
    ax.set_ylabel("Early-warning score")
    ax.set_title("Pre-trigger early-warning score by crisis (12 weeks → trigger)")
    ax.grid(linestyle=":", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    plt.tight_layout()
    out = FIG / "early_warning_event_time.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    TBL.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    bs = read_balance_sheet(PROC / "boe_balance_sheet.parquet")
    bs = bs.loc["1844-09-01":"1919-12-31"]
    br = read_bank_rate(PROC / "bank_rate_daily.parquet")

    feats = build_features(bs, br)
    comp = build_warning_index(feats)

    # Write per-week scores
    full = comp.join(feats[ML_FEATURE_COLS], how="left")
    full.to_csv(TBL / "early_warning_scores.csv")
    print(f"Wrote {TBL / 'early_warning_scores.csv'} (rows={len(full):,})")

    # Per-crisis ML backtest
    ml_dfs: dict[str, pd.DataFrame] = {}
    for key in CRISES:
        ml_dfs[key] = run_ml_backtest(feats, key)
        status = "OK" if not ml_dfs[key].empty else "INSUFFICIENT_TRAINING"
        print(f"  ML backtest {key}: {status}  (n_test_rows={len(ml_dfs[key])})")

    summary = crisis_summary(comp, ml_dfs)
    summary.to_csv(TBL / "early_warning_crisis_summary.csv", index=False)
    print(f"\nWrote {TBL / 'early_warning_crisis_summary.csv'}")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(summary.to_string(index=False))

    # Placebo
    print("\nRunning placebo (1000 random non-crisis 12-week windows)…")
    plc = placebo_test(comp, n_samples=1000)
    rows = []
    for key, c in CRISES.items():
        r = summary[summary["crisis"] == key].iloc[0]
        cv = r["max_warning_score_12w"]
        # Also compute crisis's weeks-above-80 and mean-score in the 12w window
        start = c.acute_peak - pd.Timedelta(weeks=12)
        sub = comp.loc[start:c.acute_peak - pd.Timedelta(days=1)]["warning_score"].dropna()
        weeks_above_crisis = int((sub >= 80).sum()) if len(sub) else 0
        mean_score_crisis = float(sub.mean()) if len(sub) else np.nan

        if pd.isna(cv):
            rows.append({"crisis": key, "max_warning_score_12w": "n/a",
                         "placebo_mean": np.nan, "placebo_95th": np.nan,
                         "placebo_max": np.nan, "placebo_percentile_max": np.nan,
                         "weeks_above_80_crisis": weeks_above_crisis,
                         "placebo_weeks_above_80_mean": np.nan,
                         "placebo_weeks_above_80_95th": np.nan,
                         "placebo_percentile_weeks_above_80": np.nan,
                         "mean_score_crisis": mean_score_crisis,
                         "placebo_mean_score_mean": np.nan,
                         "placebo_percentile_mean_score": np.nan,
                         "false_positive_rate_max_at_80": np.nan,
                         "false_positive_rate_weeks_above_80_ge_6": np.nan})
            continue
        pv_max = plc["max_warning_score_12w"].values
        pv_weeks = plc["weeks_above_80_12w"].values
        pv_mean = plc["mean_score_12w"].values
        rows.append({
            "crisis": key,
            "max_warning_score_12w": cv,
            "placebo_mean": round(float(pv_max.mean()), 2),
            "placebo_95th": round(float(np.percentile(pv_max, 95)), 2),
            "placebo_max": round(float(pv_max.max()), 2),
            "placebo_percentile_max": round(float((pv_max <= cv).mean() * 100), 1),
            "weeks_above_80_crisis": weeks_above_crisis,
            "placebo_weeks_above_80_mean": round(float(pv_weeks.mean()), 2),
            "placebo_weeks_above_80_95th": int(np.percentile(pv_weeks, 95)),
            "placebo_percentile_weeks_above_80": round(float((pv_weeks <= weeks_above_crisis).mean() * 100), 1),
            "mean_score_crisis": round(mean_score_crisis, 2),
            "placebo_mean_score_mean": round(float(pv_mean.mean()), 2),
            "placebo_percentile_mean_score": round(float((pv_mean <= mean_score_crisis).mean() * 100), 1),
            "false_positive_rate_max_at_80": round(float((pv_max >= 80).mean() * 100), 1),
            "false_positive_rate_weeks_above_80_ge_6": round(float((pv_weeks >= 6).mean() * 100), 1),
        })
    plc_summary = pd.DataFrame(rows)
    plc_summary.to_csv(TBL / "early_warning_placebo.csv", index=False)
    print(f"Wrote {TBL / 'early_warning_placebo.csv'}  (n_placebo={len(plc)})")
    print(plc_summary.to_string(index=False))

    # Figures
    print("\nWriting figures:")
    figure_timeline(comp)
    figure_event_time(comp)


if __name__ == "__main__":
    main()
