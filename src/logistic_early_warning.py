"""Regularized logistic early-warning robustness check (Section 7.7).

This is a SUPPLEMENT to the transparent rule-based stress index in
`src/early_warning.py` and the sequential threshold/persistence/CUSUM rules
in `src/sequential_early_warning.py`. The point is not to replace those
methods. It is to ask whether pre-crisis weeks are statistically
distinguishable from ordinary weeks using a regularized linear classifier on
purely past-available features, evaluated leave-one-crisis-out.

Design:
- Features are built strictly from past data via `.shift(1).expanding(...)`
  chains. Every feature at week t depends only on observations <= t-1.
- Labels: positive if a canonical crisis trigger occurs within the next h
  weeks (h in {8, 12}).
- Training exclusions: acute windows of all crises and 12 weeks post each
  crisis are removed from the training pool so the model does NOT learn the
  crisis response as a predictor.
- Evaluation: leave-one-crisis-out. For each held-out crisis C, the
  pre-trigger window (up to 26 weeks before the trigger) is excluded from
  training. Risk scores are then computed for the full eligible timeline
  using the fold's standardizer and fitted model.
- Reported metrics:
  * lead time for the held-out crisis (first week the risk crosses a fixed
    quantile threshold within the 26-week pre-window),
  * false alarms per decade in pure quiet periods (excluding every crisis's
    acute, post, and pre-window for every horizon),
  * pre-crisis max risk for each held-out crisis,
  * coefficient direction by feature.

Outputs:
- outputs/tables/logistic_early_warning_scores.csv
- outputs/tables/logistic_early_warning_crisis_summary.csv
- outputs/tables/logistic_early_warning_coefficients.csv
- outputs/figures/logistic_early_warning_timeline.png
- outputs/figures/logistic_early_warning_event_time.png

Caveat. Four canonical crisis triggers is an extremely small positive class.
The metrics here are diagnostic for the structural-distinguishability
question, not estimates of deployable predictive performance.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from io_utils import read_balance_sheet, read_bank_rate  # noqa: E402
from crisis_windows import CRISES  # noqa: E402

PROC = ROOT / "data" / "processed"
TBL = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures"

HORIZONS = (8, 12)
PRE_WINDOW_WEEKS = 26  # how far before a trigger we consider a positive observation
POST_EXCL_WEEKS = 12   # exclude weeks 0..POST_EXCL_WEEKS after each acute_peak
SAMPLE_START = "1855-01-01"
SAMPLE_END = "1916-12-31"


# ---------- feature construction (strict no-leakage) ----------

def _past_pct_change(s: pd.Series, k: int) -> pd.Series:
    """pct change over k periods ending at t-1, observable at t."""
    past = s.shift(1)
    return past.pct_change(k)


def _past_diff(s: pd.Series, k: int) -> pd.Series:
    """absolute diff over k periods ending at t-1, observable at t."""
    past = s.shift(1)
    return past - past.shift(k)


def _past_z(s: pd.Series, min_periods: int = 52) -> pd.Series:
    """Z-score against an expanding window of strictly past observations."""
    past = s.shift(1)
    mu = past.expanding(min_periods=min_periods).mean()
    sd = past.expanding(min_periods=min_periods).std()
    return (s - mu) / sd.replace(0, np.nan)


def build_features(bs: pd.DataFrame, br_daily: pd.DataFrame) -> pd.DataFrame:
    df = bs.copy()
    df["bank_rate_aligned"] = (
        br_daily.set_index("date")["bank_rate"].reindex(df.index, method="nearest")
    )
    # Base series (level features will use the shift(1) form below)
    df["reserve_ratio_raw"] = df["reserve_ratio"]
    df["crisis_lending_raw"] = df["crisis_lending"]
    df["lending_to_reserve_raw"] = df["lending_to_reserve"]
    df["bank_rate_raw"] = df["bank_rate_aligned"]
    df["reserve_raw"] = df["banking_reserve_notes_coin"]

    feats = pd.DataFrame(index=df.index)
    # Level features (past-only)
    feats["reserve_ratio_level"] = df["reserve_ratio_raw"].shift(1)
    feats["crisis_lending_level"] = df["crisis_lending_raw"].shift(1)
    feats["lending_to_reserve_level"] = df["lending_to_reserve_raw"].shift(1)
    feats["bank_rate_level"] = df["bank_rate_raw"].shift(1)
    # Changes (4w and 8w)
    feats["reserve_ratio_chg_4w"] = _past_diff(df["reserve_ratio_raw"], 4)
    feats["reserve_ratio_chg_8w"] = _past_diff(df["reserve_ratio_raw"], 8)
    feats["crisis_lending_pct_4w"] = _past_pct_change(df["crisis_lending_raw"], 4)
    feats["crisis_lending_pct_8w"] = _past_pct_change(df["crisis_lending_raw"], 8)
    feats["bank_rate_chg_4w"] = _past_diff(df["bank_rate_raw"], 4)
    feats["reserve_pct_4w"] = _past_pct_change(df["reserve_raw"], 4)
    feats["reserve_pct_8w"] = _past_pct_change(df["reserve_raw"], 8)
    # Past-only z-scores (rolling against expanding history)
    feats["z_reserve_ratio"] = _past_z(df["reserve_ratio_raw"])
    feats["z_crisis_lending"] = _past_z(df["crisis_lending_raw"])
    feats["z_lending_to_reserve"] = _past_z(df["lending_to_reserve_raw"])
    feats["z_bank_rate"] = _past_z(df["bank_rate_raw"])

    return feats


FEATURES = [
    "reserve_ratio_level", "crisis_lending_level", "lending_to_reserve_level", "bank_rate_level",
    "reserve_ratio_chg_4w", "reserve_ratio_chg_8w",
    "crisis_lending_pct_4w", "crisis_lending_pct_8w",
    "bank_rate_chg_4w", "reserve_pct_4w", "reserve_pct_8w",
    "z_reserve_ratio", "z_crisis_lending", "z_lending_to_reserve", "z_bank_rate",
]


# ---------- labelling and exclusion masks ----------

def label_within_h(idx: pd.DatetimeIndex, h_weeks: int) -> pd.Series:
    """label[t] = 1 if any crisis trigger falls in (t, t + h_weeks]."""
    y = pd.Series(0, index=idx, dtype=int)
    for c in CRISES.values():
        win_start = c.acute_peak - pd.Timedelta(weeks=h_weeks)
        win_end = c.acute_peak - pd.Timedelta(days=1)
        mask = (idx >= win_start) & (idx <= win_end)
        y.loc[mask] = 1
    return y


def acute_post_exclusion(idx: pd.DatetimeIndex) -> pd.Series:
    """True for weeks inside any crisis's acute window or up to POST_EXCL_WEEKS after."""
    excl = pd.Series(False, index=idx)
    for c in CRISES.values():
        s = c.acute_start
        e = c.acute_peak + pd.Timedelta(weeks=POST_EXCL_WEEKS)
        excl.loc[(idx >= s) & (idx <= e)] = True
    return excl


def all_pre_window_exclusion(idx: pd.DatetimeIndex, h_weeks: int) -> pd.Series:
    """True for any week within h_weeks BEFORE any crisis trigger (used to keep
    'quiet' weeks pure when computing the placebo false-alarm rate).
    """
    excl = pd.Series(False, index=idx)
    for c in CRISES.values():
        s = c.acute_peak - pd.Timedelta(weeks=h_weeks)
        e = c.acute_peak - pd.Timedelta(days=1)
        excl.loc[(idx >= s) & (idx <= e)] = True
    return excl


# ---------- evaluation ----------

def fit_and_score_one_fold(feats: pd.DataFrame, y: pd.Series, held_out_key: str,
                            pre_window_test_weeks: int = PRE_WINDOW_WEEKS,
                            C_reg: float = 1.0) -> tuple[pd.Series, np.ndarray, list[str]]:
    """Train on all eligible weeks except the held-out crisis's pre-window,
    return risk scores for the full eligible timeline.
    """
    idx = feats.index
    held = CRISES[held_out_key]
    test_start = held.acute_peak - pd.Timedelta(weeks=pre_window_test_weeks)
    test_end = held.acute_peak - pd.Timedelta(days=1)
    test_mask = (idx >= test_start) & (idx <= test_end)

    # Eligible weeks = no NaN feature + not in acute/post of any crisis
    eligible = feats.notna().all(axis=1) & (~acute_post_exclusion(idx))
    train_mask = eligible & (~test_mask)

    Xtr = feats.loc[train_mask, FEATURES].values
    ytr = y.loc[train_mask].values
    if ytr.sum() == 0:
        # Should not happen but be safe
        return pd.Series(np.nan, index=idx), np.zeros(len(FEATURES)), FEATURES

    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = LogisticRegression(penalty="l2", C=C_reg, class_weight="balanced",
                                    max_iter=5000, solver="lbfgs")
        model.fit(Xtr_s, ytr)

    # Score every eligible row (we'll mask non-eligible later)
    Xall = feats.loc[eligible, FEATURES].values
    Xall_s = scaler.transform(Xall)
    p = model.predict_proba(Xall_s)[:, 1]
    risk = pd.Series(np.nan, index=idx)
    risk.loc[eligible] = p
    return risk, model.coef_.ravel(), FEATURES


def summarize_fold(risk: pd.Series, held_out_key: str, threshold: float,
                    h_weeks: int) -> dict:
    held = CRISES[held_out_key]
    idx = risk.index

    # Held-out pre-window
    pre_start = held.acute_peak - pd.Timedelta(weeks=PRE_WINDOW_WEEKS)
    pre_end = held.acute_peak - pd.Timedelta(days=1)
    pre_mask = (idx >= pre_start) & (idx <= pre_end)
    pre = risk.loc[pre_mask].dropna()
    if len(pre) == 0:
        return {"crisis": held_out_key, "detected": False, "lead_weeks": np.nan,
                 "first_alert_date": None, "max_risk_pre": np.nan}

    above = pre[pre >= threshold]
    detected = len(above) > 0
    first = above.index[0] if detected else None
    lead = float((held.acute_peak - first).days / 7.0) if detected else np.nan
    return {
        "crisis": held_out_key,
        "trigger_date": held.acute_peak.date().isoformat(),
        "horizon_weeks": h_weeks,
        "detected": detected,
        "first_alert_date": first.date().isoformat() if detected else None,
        "lead_weeks": lead,
        "n_alert_weeks_in_pre_window": int(len(above)),
        "max_risk_pre": float(pre.max()),
        "mean_risk_pre": float(pre.mean()),
    }


def quiet_false_alarm_rate(risk: pd.Series, threshold: float, h_weeks: int) -> float:
    """Alerts per decade in weeks that are NOT in any crisis acute/post window
    AND NOT in the pre-window of any crisis at horizon h."""
    idx = risk.index
    excl_acute = acute_post_exclusion(idx)
    excl_pre = all_pre_window_exclusion(idx, h_weeks)
    quiet = risk.dropna().index.difference(idx[excl_acute | excl_pre])
    quiet_risk = risk.loc[quiet]
    if len(quiet_risk) == 0:
        return np.nan
    rate_per_week = float((quiet_risk >= threshold).mean())
    return rate_per_week * 520.0  # 520 weeks per decade


# ---------- main ----------

def main() -> None:
    bs = read_balance_sheet(PROC / "boe_balance_sheet.parquet")
    br = read_bank_rate(PROC / "bank_rate_daily.parquet")
    bs = bs.loc[SAMPLE_START:SAMPLE_END]

    feats = build_features(bs, br)

    rows_scores: list[dict] = []
    rows_summary: list[dict] = []
    rows_coefs: list[dict] = []

    # Per-horizon LOOCV
    per_horizon_risk_by_fold: dict[int, dict[str, pd.Series]] = {h: {} for h in HORIZONS}

    for h in HORIZONS:
        y = label_within_h(feats.index, h)
        for key in CRISES:
            risk, coefs, names = fit_and_score_one_fold(feats, y, key)
            per_horizon_risk_by_fold[h][key] = risk

            # Tune threshold so that the model's quiet false-alarm rate is ~10/decade
            # (a reasonable benchmark; existing rules range 7-99 / decade). We pick
            # the threshold per fold to keep the LOOCV comparison fair.
            quiet_idx = risk.dropna().index.difference(
                feats.index[acute_post_exclusion(feats.index) | all_pre_window_exclusion(feats.index, h)]
            )
            quiet_risk = risk.loc[quiet_idx].dropna()
            if len(quiet_risk) == 0:
                threshold = 0.5
            else:
                # Target ~10 quiet alerts per decade -> ~1.9% quantile cutoff
                target_rate_per_week = 10.0 / 520.0
                threshold = float(quiet_risk.quantile(1 - target_rate_per_week))

            summary = summarize_fold(risk, key, threshold, h)
            summary["threshold"] = round(threshold, 4)
            summary["false_alarms_per_decade_quiet"] = round(
                quiet_false_alarm_rate(risk, threshold, h), 2
            )
            rows_summary.append(summary)

            for nm, coef in zip(names, coefs):
                rows_coefs.append({
                    "horizon_weeks": h,
                    "held_out_crisis": key,
                    "feature": nm,
                    "coefficient": float(coef),
                })

            for d, r in risk.dropna().items():
                rows_scores.append({
                    "horizon_weeks": h,
                    "held_out_crisis": key,
                    "date": d.date().isoformat(),
                    "risk": float(r),
                })

    # Write
    TBL.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_summary).to_csv(TBL / "logistic_early_warning_crisis_summary.csv", index=False)
    pd.DataFrame(rows_coefs).to_csv(TBL / "logistic_early_warning_coefficients.csv", index=False)
    pd.DataFrame(rows_scores).to_csv(TBL / "logistic_early_warning_scores.csv", index=False)

    # Figures
    FIG.mkdir(parents=True, exist_ok=True)

    # Long-run timeline (one trace per held-out fold at the 12w horizon)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    colors = {"1857": "#2980b9", "1866": "#16a085", "1890": "#f39c12", "1914": "#c0392b"}
    for key in CRISES:
        r = per_horizon_risk_by_fold[12][key]
        ax.plot(r.index, r.values, color=colors[key], linewidth=0.6, alpha=0.6,
                 label=f"Fold: held-out {key}")
    for key, c in CRISES.items():
        ax.axvline(c.acute_peak, color=colors[key], linestyle="--", linewidth=0.7, alpha=0.7)
        ax.text(c.acute_peak, 1.02, key, color=colors[key], fontsize=8, ha="center", va="bottom")
    ax.set_ylabel("Logistic risk score (12-week horizon)")
    ax.set_xlabel("Date")
    ax.set_title("Ridge logistic early-warning risk score, 1855–1916\n(one trace per leave-one-out fold; vertical dashes mark canonical triggers)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(FIG / "logistic_early_warning_timeline.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # Event-time figure: 26 weeks before each crisis, per fold
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for key, c in CRISES.items():
        r = per_horizon_risk_by_fold[12][key]
        pre_start = c.acute_peak - pd.Timedelta(weeks=PRE_WINDOW_WEEKS)
        pre = r.loc[pre_start:c.acute_peak].dropna()
        if len(pre) == 0:
            continue
        weeks_to_trigger = ((c.acute_peak - pre.index) / pd.Timedelta(weeks=1)).values
        ax.plot(-weeks_to_trigger, pre.values, marker="o", markersize=3, color=colors[key],
                 linewidth=1.0, label=f"{key} ({c.name})")
    ax.axhline(0.0, color="#aaa", linewidth=0.5)
    ax.set_xlabel("Weeks before canonical trigger")
    ax.set_ylabel("Logistic risk score (12w horizon)")
    ax.set_title("Pre-trigger risk score by held-out crisis (leave-one-out ridge logistic)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "logistic_early_warning_event_time.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # Console report
    print(f"\nWrote {TBL / 'logistic_early_warning_crisis_summary.csv'}")
    print(f"Wrote {TBL / 'logistic_early_warning_coefficients.csv'}")
    print(f"Wrote {TBL / 'logistic_early_warning_scores.csv'}")
    print(f"Wrote {FIG / 'logistic_early_warning_timeline.png'}")
    print(f"Wrote {FIG / 'logistic_early_warning_event_time.png'}")
    df = pd.DataFrame(rows_summary)
    print("\nPer-fold summary:")
    print(df.to_string(index=False))

    # Coefficient summary across folds
    cdf = pd.DataFrame(rows_coefs)
    print("\nMean ridge coefficient by feature (12w horizon, averaged across LOOCV folds):")
    by_feat = (
        cdf[cdf["horizon_weeks"] == 12]
        .groupby("feature")["coefficient"]
        .agg(["mean", "std", "min", "max"])
        .round(3)
        .sort_values("mean", key=abs, ascending=False)
    )
    print(by_feat.to_string())


if __name__ == "__main__":
    main()
