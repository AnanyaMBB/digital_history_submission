"""Sequential, real-time-style backtest of early-warning rules.

Complements `src/early_warning.py`. That earlier script asked "what is the
maximum warning score in the 12 weeks before each trigger?" — a visibility
test. This script asks something closer to a real-time backtest. At each
week t, using only information available before or at t, would a defined
alert rule have fired? And if so, did a canonical crisis trigger occur
within the next h weeks?

Important caveats up front. There are exactly four canonical crisis triggers
in the sample. Precision and recall computed against that denominator are
not statistically powered. The defensible comparison is the per-decade rate
of false alerts in non-crisis periods.

Design choices.
- All rolling statistics enforce no leakage via `.shift(1).expanding(...)`
  or `.shift(1).rolling(...)`.
- Three alert-rule families: threshold rules, persistence rules,
  EWMA/CUSUM rules.
- Three forecast horizons: 4, 8, 12 weeks.
- A seasonal-adjustment variant subtracts a past-only month-of-year mean
  and standard deviation from each stress component, then re-runs the
  warning index and rules.
- A sequential null counts how often each rule fires in non-crisis weeks.

Outputs in `outputs/tables/`:
- sequential_alert_rules.csv          — per-rule precision/recall/F1/lead-time
- sequential_crisis_detection.csv     — first-alert date and lead time per crisis
- sequential_false_alarms.csv         — false-alert rate per rule in quiet periods

Outputs in `outputs/figures/`:
- sequential_early_warning_event_time.png
- sequential_alert_timeline.png
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from io_utils import read_balance_sheet, read_bank_rate  # noqa: E402
from crisis_windows import CRISES  # noqa: E402

PROC = ROOT / "data" / "processed"
TBL = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures"

# Canonical crisis trigger dates
TRIGGERS = {key: c.acute_peak for key, c in CRISES.items()}
TRIGGER_DATES = sorted(TRIGGERS.values())


# -----------------------------------------------------------------------------
# Feature construction (strictly past-only)
# -----------------------------------------------------------------------------

def _past_z(s: pd.Series, min_periods: int = 52) -> pd.Series:
    past = s.shift(1)
    mu = past.expanding(min_periods=min_periods).mean()
    sd = past.expanding(min_periods=min_periods).std()
    return (s - mu) / sd.replace(0, np.nan)


def _past_seasonal_z(s: pd.Series, min_periods_per_month: int = 5) -> pd.Series:
    """Past-only month-of-year-normalised z-score.

    For each observation at week t, compute mean and std over PRIOR
    observations sharing the same calendar month, then z-score. Exploratory
    in early years because pre-1900 history is short.
    """
    out = pd.Series(np.nan, index=s.index, dtype=float)
    months = s.index.month
    values = s.values
    for i in range(len(s)):
        if i == 0:
            continue
        m = months[i]
        prior_mask = (months[:i] == m) & np.isfinite(values[:i])
        prior = values[:i][prior_mask]
        if len(prior) < min_periods_per_month:
            continue
        mu = prior.mean()
        sd = prior.std()
        if sd == 0 or not np.isfinite(sd):
            continue
        out.iloc[i] = (values[i] - mu) / sd
    return out


def build_features(bs: pd.DataFrame, br_daily: pd.DataFrame) -> pd.DataFrame:
    df = bs.copy()
    df["bank_rate_aligned"] = br_daily.set_index("date")["bank_rate"].reindex(df.index, method="nearest")
    levels = {
        "reserve_ratio": df["reserve_ratio"],
        "crisis_lending": df["crisis_lending"],
        "lending_to_reserve": df["lending_to_reserve"],
        "bank_rate": df["bank_rate_aligned"],
    }
    feats = pd.DataFrame(index=df.index)
    for name, s in levels.items():
        feats[f"z_{name}"] = _past_z(s, min_periods=52)
    # Stress signs: reserve_ratio falling = stress (negate); rest positive
    feats["stress_reserve"] = -feats["z_reserve_ratio"]
    feats["stress_lending"] = feats["z_crisis_lending"]
    feats["stress_lending_to_reserve"] = feats["z_lending_to_reserve"]
    feats["stress_rate"] = feats["z_bank_rate"]
    feats["stress_reserve_drawdown"] = -(levels["reserve_ratio"].pct_change(8))
    feats["stress_lending_surge"] = levels["crisis_lending"].pct_change(8)

    stress_cols = [
        "stress_reserve", "stress_lending", "stress_lending_to_reserve",
        "stress_rate", "stress_reserve_drawdown", "stress_lending_surge",
    ]
    feats["warning_index"] = feats[stress_cols].mean(axis=1, skipna=True)
    feats["warning_score"] = _expanding_pct_rank(feats["warning_index"], min_periods=52)

    # Seasonal-adjusted versions (exploratory)
    for name in ("reserve_ratio", "crisis_lending", "lending_to_reserve", "bank_rate"):
        feats[f"zS_{name}"] = _past_seasonal_z(levels[name])
    feats["stressS_reserve"] = -feats["zS_reserve_ratio"]
    feats["stressS_lending"] = feats["zS_crisis_lending"]
    feats["stressS_lending_to_reserve"] = feats["zS_lending_to_reserve"]
    feats["stressS_rate"] = feats["zS_bank_rate"]
    stressS = ["stressS_reserve", "stressS_lending",
                "stressS_lending_to_reserve", "stressS_rate"]
    feats["warning_index_seasonal"] = feats[stressS].mean(axis=1, skipna=True)
    feats["warning_score_seasonal"] = _expanding_pct_rank(
        feats["warning_index_seasonal"], min_periods=52
    )
    return feats


def _expanding_pct_rank(s: pd.Series, min_periods: int = 52) -> pd.Series:
    out = pd.Series(np.nan, index=s.index, dtype=float)
    arr = s.values
    for i in range(len(arr)):
        if i < min_periods:
            continue
        past = arr[:i]
        past = past[~np.isnan(past)]
        if len(past) < min_periods:
            continue
        if np.isnan(arr[i]):
            continue
        out.iloc[i] = float((past <= arr[i]).mean() * 100.0)
    return out


# -----------------------------------------------------------------------------
# Alert rules
# -----------------------------------------------------------------------------

def threshold_rule(score: pd.Series, threshold: float) -> pd.Series:
    return (score >= threshold).fillna(False)


def persistence_rule(score: pd.Series, k_of_n: tuple[int, int],
                       threshold: float = 80) -> pd.Series:
    """Fire if at least k of the previous n weeks were above threshold,
    looking strictly backward (the current week itself is included)."""
    k, n = k_of_n
    above = (score >= threshold).fillna(False).astype(int)
    rolled = above.rolling(n, min_periods=n).sum()
    return (rolled >= k).fillna(False)


def ewma_rule(stress: pd.Series, lam: float = 0.2,
                threshold_sigma: float = 2.0,
                min_history: int = 52) -> pd.Series:
    """EWMA on a stress series; fire when the EWMA exceeds threshold_sigma
    times the past-only standard deviation of the *EWMA itself*."""
    e = stress.ewm(alpha=lam, adjust=False).mean()
    past_std = e.shift(1).expanding(min_periods=min_history).std()
    past_mean = e.shift(1).expanding(min_periods=min_history).mean()
    upper = past_mean + threshold_sigma * past_std
    return (e > upper).fillna(False)


def cusum_rule(stress: pd.Series, k: float = 0.5, h: float = 5.0,
                 min_history: int = 52) -> pd.Series:
    """One-sided positive CUSUM on standardized stress.

    Standardize stress using past-only mean and std, then run S_t =
    max(0, S_{t-1} + (x_t - k)). Fire when S_t exceeds h.
    """
    z = stress.copy()
    # Standardize z using past expanding history
    past_mu = z.shift(1).expanding(min_periods=min_history).mean()
    past_sd = z.shift(1).expanding(min_periods=min_history).std()
    zn = (z - past_mu) / past_sd.replace(0, np.nan)
    out = pd.Series(False, index=z.index)
    S = 0.0
    fired = False
    for i, v in enumerate(zn.values):
        if not np.isfinite(v):
            S = 0.0
            continue
        S = max(0.0, S + v - k)
        if S >= h:
            out.iloc[i] = True
            fired = True
            S = 0.0  # reset after firing
        else:
            out.iloc[i] = False
    return out


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

HORIZONS = [4, 8, 12]


def label_horizon(idx: pd.DatetimeIndex, h_weeks: int) -> pd.Series:
    """Label week t as positive if any trigger date falls in (t, t+h_weeks]."""
    out = pd.Series(False, index=idx)
    horizon_td = pd.Timedelta(weeks=h_weeks)
    for d in idx:
        for trig in TRIGGER_DATES:
            if d < trig <= d + horizon_td:
                out.loc[d] = True
                break
    return out


def evaluate_rule(alert: pd.Series, idx: pd.DatetimeIndex,
                    h_weeks: int) -> dict:
    y = label_horizon(idx, h_weeks)
    a = alert.reindex(idx, fill_value=False)
    tp = int(((a) & (y)).sum())
    fp = int(((a) & (~y)).sum())
    fn = int(((~a) & (y)).sum())
    tn = int(((~a) & (~y)).sum())
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec and rec and (prec + rec) else float("nan")

    # False alerts per decade in NON-CRISIS periods (exclude ±365 days from triggers)
    exclude = pd.Series(False, index=idx)
    for trig in TRIGGER_DATES:
        exclude |= ((idx >= trig - pd.Timedelta(days=365)) &
                     (idx <= trig + pd.Timedelta(days=365)))
    quiet = ~exclude
    quiet_weeks = int(quiet.sum())
    quiet_alerts = int((a[quiet]).sum())
    fpr_per_decade = (quiet_alerts / quiet_weeks * 52 * 10) if quiet_weeks else float("nan")

    # Per-crisis first-alert lead time
    crisis_detection = {}
    for key, trig in TRIGGERS.items():
        pre_window = a[(a.index < trig) & (a.index >= trig - pd.Timedelta(weeks=h_weeks))]
        if pre_window.any():
            first = pre_window[pre_window].index.min()
            crisis_detection[key] = int((trig - first).days / 7)
        else:
            crisis_detection[key] = None

    detected = sum(1 for v in crisis_detection.values() if v is not None)
    lead_times = [v for v in crisis_detection.values() if v is not None]
    median_lead = float(np.median(lead_times)) if lead_times else float("nan")

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(prec, 4) if np.isfinite(prec) else None,
        "recall": round(rec, 4) if np.isfinite(rec) else None,
        "f1": round(f1, 4) if np.isfinite(f1) else None,
        "false_alerts_per_decade_quiet": round(fpr_per_decade, 2),
        "crises_detected": detected,
        "median_lead_weeks_detected": round(median_lead, 2) if np.isfinite(median_lead) else None,
        **{f"lead_{k}_weeks": v for k, v in crisis_detection.items()},
    }


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

    # Define the rule menu
    rule_defs = {
        # threshold rules
        "threshold_score>=80":  ("threshold", "warning_score", dict(threshold=80)),
        "threshold_score>=90":  ("threshold", "warning_score", dict(threshold=90)),
        "threshold_score>=95":  ("threshold", "warning_score", dict(threshold=95)),
        # persistence rules
        "persistence_2of4_80":  ("persistence", "warning_score", dict(k_of_n=(2, 4), threshold=80)),
        "persistence_3of6_80":  ("persistence", "warning_score", dict(k_of_n=(3, 6), threshold=80)),
        "persistence_6of12_80": ("persistence", "warning_score", dict(k_of_n=(6, 12), threshold=80)),
        # ewma / cusum (operate on raw warning_index, not the pct-rank score)
        "ewma_lam=0.2_2sigma":  ("ewma", "warning_index", dict(lam=0.2, threshold_sigma=2.0)),
        "ewma_lam=0.1_2sigma":  ("ewma", "warning_index", dict(lam=0.1, threshold_sigma=2.0)),
        "cusum_k=0.5_h=5":      ("cusum", "warning_index", dict(k=0.5, h=5.0)),
        "cusum_k=0.25_h=4":     ("cusum", "warning_index", dict(k=0.25, h=4.0)),
        # seasonal variants
        "threshold_seasonal>=80": ("threshold", "warning_score_seasonal", dict(threshold=80)),
        "persistence_3of6_seasonal_80": ("persistence", "warning_score_seasonal", dict(k_of_n=(3, 6), threshold=80)),
    }

    def _apply(rule_kind, series, params) -> pd.Series:
        if rule_kind == "threshold":
            return threshold_rule(feats[series], **params)
        if rule_kind == "persistence":
            return persistence_rule(feats[series], **params)
        if rule_kind == "ewma":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return ewma_rule(feats[series], **params)
        if rule_kind == "cusum":
            return cusum_rule(feats[series], **params)
        raise ValueError(rule_kind)

    rows = []
    for name, (kind, col, params) in rule_defs.items():
        alert = _apply(kind, col, params)
        for h in HORIZONS:
            stats = evaluate_rule(alert, feats.index, h)
            rows.append({"rule": name, "horizon_weeks": h, **stats})

    out = pd.DataFrame(rows)
    out.to_csv(TBL / "sequential_alert_rules.csv", index=False)
    print(f"Wrote {TBL / 'sequential_alert_rules.csv'}  rows={len(out)}")
    # Print a compact summary at horizon=12
    print("\nSummary at horizon=12 weeks (precision, recall, F1, false-alerts/decade, crises detected):")
    summary = out[out["horizon_weeks"] == 12][
        ["rule", "precision", "recall", "f1",
         "false_alerts_per_decade_quiet", "crises_detected",
         "median_lead_weeks_detected"]
    ]
    print(summary.to_string(index=False))

    # Per-crisis detection table (using the 12-week horizon canonical rule menu)
    detection_rows = []
    for name, (kind, col, params) in rule_defs.items():
        alert = _apply(kind, col, params)
        for key, trig in TRIGGERS.items():
            pre = alert[(alert.index < trig) &
                        (alert.index >= trig - pd.Timedelta(weeks=12))]
            if pre.any():
                first = pre[pre].index.min()
                detection_rows.append({
                    "rule": name, "crisis": key,
                    "trigger": trig.date().isoformat(),
                    "first_alert_date": first.date().isoformat(),
                    "lead_weeks": int((trig - first).days / 7),
                    "n_alert_weeks_in_pre_window": int(pre.sum()),
                })
            else:
                detection_rows.append({
                    "rule": name, "crisis": key,
                    "trigger": trig.date().isoformat(),
                    "first_alert_date": "",
                    "lead_weeks": None,
                    "n_alert_weeks_in_pre_window": 0,
                })
    detection_df = pd.DataFrame(detection_rows)
    detection_df.to_csv(TBL / "sequential_crisis_detection.csv", index=False)
    print(f"\nWrote {TBL / 'sequential_crisis_detection.csv'}")

    # False alarms in quiet periods (one row per rule)
    false_rows = []
    exclude = pd.Series(False, index=feats.index)
    for trig in TRIGGER_DATES:
        exclude |= ((feats.index >= trig - pd.Timedelta(days=365)) &
                     (feats.index <= trig + pd.Timedelta(days=365)))
    quiet_weeks = int((~exclude).sum())
    for name, (kind, col, params) in rule_defs.items():
        alert = _apply(kind, col, params)
        quiet = alert[~exclude]
        n_alerts = int(quiet.sum())
        false_rows.append({
            "rule": name,
            "quiet_weeks": quiet_weeks,
            "alerts_in_quiet": n_alerts,
            "alerts_per_decade": round(n_alerts / quiet_weeks * 52 * 10, 2),
            "alerts_per_year": round(n_alerts / quiet_weeks * 52, 3),
        })
    false_df = pd.DataFrame(false_rows)
    false_df.to_csv(TBL / "sequential_false_alarms.csv", index=False)
    print(f"Wrote {TBL / 'sequential_false_alarms.csv'}")
    print(false_df.to_string(index=False))

    # Figures
    _figure_event_time(feats)
    _figure_alert_timeline(feats, rule_defs, _apply)


def _figure_event_time(feats: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    colors = {"1857": "#2980b9", "1866": "#16a085",
              "1890": "#f39c12", "1914": "#c0392b"}
    for ax, score_col, title in [
        (axes[0], "warning_score", "Raw warning score"),
        (axes[1], "warning_score_seasonal", "Seasonally-adjusted warning score"),
    ]:
        for key, trig in TRIGGERS.items():
            start = trig - pd.Timedelta(weeks=12)
            sub = feats.loc[start:trig - pd.Timedelta(days=1), score_col].dropna()
            if sub.empty:
                continue
            weeks_from_trigger = (sub.index - trig) / pd.Timedelta(weeks=1)
            ax.plot(weeks_from_trigger, sub.values,
                    label=key, color=colors[key], linewidth=1.4)
        ax.axhline(80, color="#c0392b", linewidth=0.7, linestyle="--")
        ax.axvline(0, color="#444", linewidth=0.6)
        ax.set_xlim(-12, 0)
        ax.set_ylim(0, 105)
        ax.set_xlabel("Weeks before trigger")
        ax.set_title(title)
        ax.grid(linestyle=":", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Score")
    axes[0].legend(fontsize=8, frameon=False)
    plt.tight_layout()
    out = FIG / "sequential_early_warning_event_time.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def _figure_alert_timeline(feats: pd.DataFrame, rule_defs: dict, applier) -> None:
    # Show three representative rules' alert series across 1855-1916
    chosen = [
        "persistence_3of6_80",
        "ewma_lam=0.2_2sigma",
        "cusum_k=0.5_h=5",
    ]
    fig, axes = plt.subplots(len(chosen), 1, figsize=(11, 6), sharex=True)
    if len(chosen) == 1:
        axes = [axes]
    for ax, name in zip(axes, chosen):
        kind, col, params = rule_defs[name]
        alert = applier(kind, col, params)
        # Background: warning score (or warning index for cusum/ewma rules)
        ax.plot(feats.index, feats["warning_score"], color="#bbb", linewidth=0.5)
        # Alerts as red marks
        alert_dates = alert[alert].index
        ax.scatter(alert_dates, [100] * len(alert_dates),
                    marker="|", s=30, color="#c0392b", linewidth=0.7)
        for key, trig in TRIGGERS.items():
            ax.axvline(trig, color="#16a085", linewidth=0.6, linestyle=":")
            ax.annotate(key, (trig, 105), ha="center", fontsize=8, color="#444")
        ax.set_xlim(pd.Timestamp("1855-01-01"), pd.Timestamp("1916-12-31"))
        ax.set_ylim(0, 110)
        ax.set_ylabel(name, fontsize=8)
        ax.grid(linestyle=":", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Sequential alert timeline. Red ticks at top mark alert weeks.", fontsize=10, y=1.00)
    plt.tight_layout()
    out = FIG / "sequential_alert_timeline.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
