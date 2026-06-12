"""Sensitivity analysis: test whether headline conclusions survive reasonable
changes to the crisis-window definitions.

We vary (a) the acute window by ±2 weeks and ±1 month around the trigger,
and (b) the baseline window between 6 and 12 months. For each variant we
recompute the headline Tier-A LOLR-score components: scale ratio,
penalty-rate delta, days-to-first-rate-rise, days-to-peak-lending. The
output is a single CSV showing how each metric moves under each variant —
the test is whether the *qualitative* ordering across crises survives.

Output: outputs/tables/window_sensitivity.csv
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from io_utils import read_balance_sheet  # noqa: E402
from crisis_windows import CRISES  # noqa: E402

PROC = ROOT / "data" / "processed"
TBL = ROOT / "outputs" / "tables"


VARIANTS = [
    ("baseline_12m_acute", dict(baseline_months=12, acute_delta_weeks=0)),
    ("baseline_6m_acute",  dict(baseline_months=6,  acute_delta_weeks=0)),
    ("acute_plus_2w",      dict(baseline_months=12, acute_delta_weeks=2)),
    ("acute_minus_2w",     dict(baseline_months=12, acute_delta_weeks=-2)),
    ("acute_plus_1m",      dict(baseline_months=12, acute_delta_weeks=4)),
    ("acute_minus_1m",     dict(baseline_months=12, acute_delta_weeks=-4)),
]


def _metrics(bs: pd.DataFrame, c, baseline_months: int, acute_delta_weeks: int) -> dict:
    pre_start = c.acute_start - pd.Timedelta(days=baseline_months * 30)
    pre_end = c.acute_start - pd.Timedelta(days=1)
    acute_end = c.acute_end + pd.Timedelta(weeks=acute_delta_weeks)
    acute_start = c.acute_start - pd.Timedelta(weeks=max(0, acute_delta_weeks))
    if acute_delta_weeks < 0:
        acute_start = c.acute_start - pd.Timedelta(weeks=acute_delta_weeks)
    pre = bs.loc[pre_start:pre_end]
    acute = bs.loc[acute_start:acute_end]

    pre_lending = float(pre["crisis_lending"].mean()) if len(pre) else np.nan
    pre_rate = float(pre["bank_rate_weekly"].mean()) if len(pre) else np.nan
    peak_lending = float(acute["crisis_lending"].max()) if len(acute) else np.nan
    peak_rate = float(acute["bank_rate_weekly"].max()) if len(acute) else np.nan

    scale = peak_lending / pre_lending if pre_lending else np.nan
    penalty = peak_rate - pre_rate if not np.isnan(pre_rate) else np.nan

    peak_lend_date = acute["crisis_lending"].idxmax() if acute["crisis_lending"].notna().any() else pd.NaT
    days_peak = (peak_lend_date - c.acute_start).days if not pd.isna(peak_lend_date) else np.nan

    last_pre_rate = float(pre["bank_rate_weekly"].iloc[-1]) if len(pre) else np.nan
    rises = acute[acute["bank_rate_weekly"] > last_pre_rate] if not np.isnan(last_pre_rate) else pd.DataFrame()
    days_first_rate = (rises.index.min() - c.acute_start).days if len(rises) else np.nan

    return {
        "pre_avg_crisis_lending": round(pre_lending, 3),
        "pre_avg_bank_rate": round(pre_rate, 3),
        "acute_peak_crisis_lending": round(peak_lending, 3),
        "acute_peak_bank_rate": round(peak_rate, 3),
        "scale_ratio": round(scale, 3),
        "penalty_rate_delta": round(penalty, 3),
        "days_to_peak_lending": days_peak,
        "days_to_first_rate_rise": days_first_rate,
    }


def main() -> None:
    TBL.mkdir(parents=True, exist_ok=True)
    bs = read_balance_sheet(PROC / "boe_balance_sheet.parquet")

    rows = []
    for variant_name, params in VARIANTS:
        for key, c in CRISES.items():
            m = _metrics(bs, c, **params)
            rows.append({"variant": variant_name, "crisis": key, **m})

    df = pd.DataFrame(rows)
    out = TBL / "window_sensitivity.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out}")

    # Print compact pivot: how does scale_ratio move across variants?
    print("\nScale ratio across variants:")
    print(df.pivot(index="crisis", columns="variant", values="scale_ratio").to_string())
    print("\nPenalty-rate delta across variants:")
    print(df.pivot(index="crisis", columns="variant", values="penalty_rate_delta").to_string())


if __name__ == "__main__":
    main()
