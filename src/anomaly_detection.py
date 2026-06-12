"""ML anomaly detection on BoE balance-sheet and Bank Rate series.

This complements the PELT change-point detection (`src/change_points.py`) by
asking a different question: not "where does the regime change" but "which
weeks look statistically unusual relative to surrounding history?"

Methods (kept transparent and reproducible — no deep learning):
1. **Robust z-score** via median + MAD on a 52-week rolling window, applied
   to the weekly percentage change of each level series. A week is flagged
   if |robust z| ≥ 3.5.
2. **Isolation Forest** (scikit-learn) on a multivariate feature matrix
   built from the same transformed series. Lower decision-function values
   = more anomalous; we report a per-week anomaly score in [0,1] normalised
   so 1 = most anomalous in the sample.

Outputs in `outputs/tables/`:
- anomaly_scores.csv             — per-week z-score, isolation score, flag
- anomaly_crisis_overlap.csv     — counts of anomaly weeks inside each
                                   crisis acute window, with the canonical
                                   trigger date for reference
- anomaly_placebo_comparison.csv — null distribution from 1,000 random
                                   non-crisis ±90-day windows

And one figure:
- outputs/figures/anomaly_timeline.png — 1855–1916 timeline with anomaly
  weeks marked, crisis windows shaded

Caveats. Anomaly detection is a cross-check, not proof of causality. The
Bank Rate signal in particular interacts with the change-point detector's
limitation noted in §7.2 of the paper.
"""
from __future__ import annotations
import sys
from pathlib import Path

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


def _robust_z(s: pd.Series, window: int = 52) -> pd.Series:
    """Rolling median + MAD z-score on a (weekly) percentage-change series."""
    med = s.rolling(window, min_periods=window // 2).median()
    mad = (s - med).abs().rolling(window, min_periods=window // 2).median()
    z = (s - med) / (1.4826 * mad.replace(0, np.nan))
    return z


def main() -> None:
    TBL.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    bs = read_balance_sheet(PROC / "boe_balance_sheet.parquet")
    bs = bs.loc["1855-01-01":"1916-12-31"].copy()

    # Daily Bank Rate → weekly mean to align with weekly balance sheet
    br = read_bank_rate(PROC / "bank_rate_daily.parquet").set_index("date").sort_index()
    br = br.loc["1855-01-01":"1916-12-31"]
    bs["bank_rate"] = br["bank_rate"].reindex(bs.index, method="nearest")

    # Transformed series (weekly pct change for levels; absolute change for the rate)
    feats = pd.DataFrame(index=bs.index)
    feats["d_crisis_lending_pct"] = bs["crisis_lending"].pct_change().clip(-1, 5)
    feats["d_reserve_pct"] = bs["banking_reserve_notes_coin"].pct_change().clip(-1, 5)
    feats["d_reserve_ratio_pct"] = bs["reserve_ratio"].pct_change().clip(-1, 5)
    feats["d_lending_to_reserve_pct"] = bs["lending_to_reserve"].pct_change().clip(-2, 10)
    feats["d_bank_rate"] = bs["bank_rate"].diff()

    # Robust z-scores per feature
    z = feats.apply(lambda c: _robust_z(c, window=52))
    z = z.replace([np.inf, -np.inf], np.nan)
    z_flag = (z.abs() >= 3.5).any(axis=1)

    # Isolation Forest on weeks with non-null features
    iso_feats = feats.fillna(0.0)
    model = IsolationForest(
        n_estimators=400, contamination=0.05, random_state=42,
        max_samples=min(512, len(iso_feats)),
    )
    model.fit(iso_feats.values)
    iso_score_raw = -model.decision_function(iso_feats.values)  # higher = more anomalous
    iso_score = (iso_score_raw - iso_score_raw.min()) / (iso_score_raw.max() - iso_score_raw.min() + 1e-9)
    iso_flag = pd.Series(model.predict(iso_feats.values), index=iso_feats.index) == -1

    out = pd.DataFrame({
        "date": bs.index,
        "crisis_lending": bs["crisis_lending"].values,
        "reserve": bs["banking_reserve_notes_coin"].values,
        "reserve_ratio": bs["reserve_ratio"].values,
        "bank_rate": bs["bank_rate"].values,
        "max_abs_z": z.abs().max(axis=1).values,
        "z_flag": z_flag.values,
        "isolation_score": iso_score,
        "isolation_flag": iso_flag.values,
    })
    # Which series triggered the z flag — compute manually to handle all-NA rows
    z_abs = z.abs()
    arr = z_abs.fillna(-np.inf).to_numpy()
    col_names = np.array(z_abs.columns)
    best_idx = np.argmax(arr, axis=1)
    has_valid = np.any(np.isfinite(arr) & (arr >= 0), axis=1)
    dominant = np.where(has_valid, col_names[best_idx], "")
    out["dominant_z_series"] = dominant
    out["anomaly_flag"] = out["z_flag"] | out["isolation_flag"]
    out.to_csv(TBL / "anomaly_scores.csv", index=False)
    print(f"Wrote {TBL / 'anomaly_scores.csv'}")
    print(f"  flagged weeks: {int(out['anomaly_flag'].sum())} of {len(out)} "
          f"({100 * out['anomaly_flag'].mean():.1f}% — Isolation Forest contamination=0.05)")

    # --- Per-crisis overlap ---
    crisis_rows = []
    for key, c in CRISES.items():
        win = (out["date"] >= c.acute_start) & (out["date"] <= c.acute_end)
        sub = out.loc[win]
        rec = {
            "crisis": key,
            "name": c.name,
            "trigger": c.acute_peak.date().isoformat(),
            "n_weeks_in_window": int(len(sub)),
            "n_z_flags": int(sub["z_flag"].sum()),
            "n_iso_flags": int(sub["isolation_flag"].sum()),
            "n_any_flag": int(sub["anomaly_flag"].sum()),
            "max_abs_z": round(float(sub["max_abs_z"].max()) if len(sub) else np.nan, 2),
            "max_iso_score": round(float(sub["isolation_score"].max()) if len(sub) else np.nan, 3),
        }
        if rec["n_any_flag"] > 0:
            flagged = sub[sub["anomaly_flag"]]
            rec["first_flagged_date"] = flagged["date"].min().date().isoformat()
            rec["last_flagged_date"] = flagged["date"].max().date().isoformat()
            rec["days_first_flag_to_trigger"] = (c.acute_peak - flagged["date"].min()).days
        else:
            rec["first_flagged_date"] = ""
            rec["last_flagged_date"] = ""
            rec["days_first_flag_to_trigger"] = np.nan
        crisis_rows.append(rec)
    crisis_df = pd.DataFrame(crisis_rows)
    crisis_df.to_csv(TBL / "anomaly_crisis_overlap.csv", index=False)
    print(f"\nWrote {TBL / 'anomaly_crisis_overlap.csv'}")
    print(crisis_df.to_string(index=False))

    # --- Placebo distribution ---
    # Same window length as the actual acute windows; centred on random non-crisis weeks
    rng = np.random.default_rng(42)
    anchors = [c.acute_peak for c in CRISES.values()]
    weeks = out["date"].values
    n_samples = 1000
    radius_days = 90
    placebo_records = []
    attempts = 0
    while len(placebo_records) < n_samples and attempts < n_samples * 50:
        attempts += 1
        i = rng.integers(0, len(weeks))
        center = pd.Timestamp(weeks[i])
        if any(abs((center - a).days) <= 365 for a in anchors):
            continue
        lo = center - pd.Timedelta(days=radius_days)
        hi = center + pd.Timedelta(days=radius_days)
        win = (out["date"] >= lo) & (out["date"] <= hi)
        sub = out.loc[win]
        placebo_records.append({
            "center": center.date().isoformat(),
            "n_z_flags": int(sub["z_flag"].sum()),
            "n_iso_flags": int(sub["isolation_flag"].sum()),
            "n_any_flag": int(sub["anomaly_flag"].sum()),
            "max_abs_z": float(sub["max_abs_z"].max()) if len(sub) else np.nan,
            "max_iso_score": float(sub["isolation_score"].max()) if len(sub) else np.nan,
        })
    placebo_df = pd.DataFrame(placebo_records)

    # Crisis vs placebo percentile comparison
    cmp_rows = []
    for key, c in CRISES.items():
        rec = crisis_df[crisis_df["crisis"] == key].iloc[0]
        for metric in ("n_z_flags", "n_iso_flags", "n_any_flag", "max_abs_z", "max_iso_score"):
            crisis_val = rec[metric]
            placebo_vals = placebo_df[metric].dropna().values
            if len(placebo_vals) == 0 or pd.isna(crisis_val):
                pct = np.nan
            else:
                pct = float((placebo_vals <= crisis_val).mean() * 100)
            cmp_rows.append({
                "crisis": key,
                "metric": metric,
                "crisis_value": round(float(crisis_val), 3) if not pd.isna(crisis_val) else "n/a",
                "placebo_mean": round(float(np.nanmean(placebo_vals)), 3) if len(placebo_vals) else "n/a",
                "placebo_95th": round(float(np.nanpercentile(placebo_vals, 95)), 3) if len(placebo_vals) else "n/a",
                "placebo_max": round(float(np.nanmax(placebo_vals)), 3) if len(placebo_vals) else "n/a",
                "percentile_of_crisis": round(pct, 1) if not pd.isna(pct) else "n/a",
            })
    cmp_df = pd.DataFrame(cmp_rows)
    cmp_df.to_csv(TBL / "anomaly_placebo_comparison.csv", index=False)
    print(f"\nWrote {TBL / 'anomaly_placebo_comparison.csv'}  (n_placebo={len(placebo_df)})")
    print(cmp_df.to_string(index=False))

    # --- Figure: timeline of isolation_score + flags ---
    fig, ax = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
    ax[0].plot(out["date"], out["isolation_score"], color="#7f8c8d", linewidth=0.8)
    ax[0].fill_between(out["date"], 0, out["isolation_score"],
                        where=out["anomaly_flag"], color="#c0392b", alpha=0.6, step="mid")
    ax[0].set_ylabel("Isolation score (0–1)")
    ax[0].set_title("Anomaly score timeline (Isolation Forest on weekly transforms)")
    ax[1].plot(out["date"], out["max_abs_z"], color="#34495e", linewidth=0.8)
    ax[1].axhline(3.5, color="#c0392b", linewidth=0.6, linestyle="--",
                   label="|z|=3.5 threshold")
    ax[1].set_ylabel("max |robust z| across series")
    ax[1].set_title("Robust z-score envelope (max across series)")
    ax[1].legend(fontsize=8, frameon=False)
    for axx in ax:
        for key, c in CRISES.items():
            axx.axvspan(c.acute_start, c.acute_end, color="#f39c12", alpha=0.18)
            axx.annotate(key, (c.acute_peak, axx.get_ylim()[1] * 0.92), ha="center",
                          fontsize=8, color="#7f8c8d")
        axx.grid(linestyle=":", alpha=0.4)
        axx.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out_fig = FIG / "anomaly_timeline.png"
    fig.savefig(out_fig, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_fig}")


if __name__ == "__main__":
    main()
