"""Compute per-crisis Bagehot-test metrics and write a comparison table.

Metrics follow the project prompt:
- response speed
- scale of intervention
- penalty-rate test
- breadth / borrower concentration (HHI, top-5 share)
- stabilization (time to normalize)

Outputs:
- outputs/tables/crisis_metrics.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from crisis_windows import CRISES

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs" / "tables"


def _peak_date(s: pd.Series) -> pd.Timestamp:
    return s.idxmax() if s.notna().any() else pd.NaT


def balance_sheet_metrics(bs: pd.DataFrame, c) -> dict:
    """Compute response-speed / scale / penalty / stabilization metrics from the
    weekly balance sheet for one crisis."""
    pre = bs.loc[c.pre]
    acute = bs.loc[c.acute]
    post = bs.loc[c.post]

    out: dict[str, float | str | pd.Timestamp] = {"crisis": c.name}

    # Pre-crisis baseline averages
    out["pre_avg_crisis_lending"] = pre["crisis_lending"].mean()
    out["pre_avg_bank_rate"] = pre["bank_rate_weekly"].mean()
    out["pre_avg_reserve"] = pre["banking_reserve_notes_coin"].mean()
    out["pre_avg_reserve_ratio"] = pre["reserve_ratio"].mean()

    # Peak metrics during acute window
    out["acute_peak_crisis_lending"] = acute["crisis_lending"].max()
    out["acute_peak_bank_rate"] = acute["bank_rate_weekly"].max()
    out["acute_min_reserve"] = acute["banking_reserve_notes_coin"].min()
    out["acute_min_reserve_ratio"] = acute["reserve_ratio"].min()
    out["acute_peak_lending_to_reserve"] = acute["lending_to_reserve"].max()

    # Ratios — how big was the intervention?
    if out["pre_avg_crisis_lending"]:
        out["scale_ratio_lending"] = out["acute_peak_crisis_lending"] / out["pre_avg_crisis_lending"]
    out["scale_lending_share_of_bs"] = (
        acute["crisis_lending"].max() / acute["balance_sheet_size"].max()
        if acute["balance_sheet_size"].notna().any() else np.nan
    )
    out["max_reserve_loss_pct"] = (
        (out["pre_avg_reserve"] - out["acute_min_reserve"]) / out["pre_avg_reserve"]
        if out["pre_avg_reserve"] else np.nan
    )

    # Penalty-rate test: change in Bank Rate from pre to peak
    out["penalty_rate_delta"] = out["acute_peak_bank_rate"] - out["pre_avg_bank_rate"]

    # Response speed (days from acute_start to peak of crisis_lending and to first rate rise)
    peak_lend = _peak_date(acute["crisis_lending"])
    out["days_to_peak_lending"] = (peak_lend - c.acute_start).days if not pd.isna(peak_lend) else np.nan

    rate_rises = acute[acute["bank_rate_weekly"] > pre["bank_rate_weekly"].iloc[-1] if len(pre) else 0]
    if len(rate_rises):
        out["days_to_first_rate_rise"] = (rate_rises.index.min() - c.acute_start).days
    else:
        out["days_to_first_rate_rise"] = np.nan

    # Stabilization: days until reserve returns to pre-crisis average
    target_reserve = out["pre_avg_reserve"]
    after_peak = bs.loc[peak_lend:c.post_end] if not pd.isna(peak_lend) else pd.DataFrame()
    recover = after_peak[after_peak["banking_reserve_notes_coin"] >= target_reserve]
    out["days_reserve_recovery"] = (recover.index.min() - peak_lend).days if len(recover) else np.nan

    # Days until discount/advance volumes normalize (back to pre-crisis avg)
    after_peak_lending = after_peak[after_peak["crisis_lending"] <= out["pre_avg_crisis_lending"]] if len(after_peak) else pd.DataFrame()
    out["days_lending_normalize"] = (after_peak_lending.index.min() - peak_lend).days if len(after_peak_lending) else np.nan

    return out


def ledger_metrics(tx: pd.DataFrame, c, crisis_key: str) -> dict:
    """From transaction-level ledger, compute breadth / concentration metrics."""
    out: dict[str, float | str] = {}
    if crisis_key not in {"1857", "1866", "1914"}:
        # No ledger for 1890
        return {
            "n_transactions_acute": np.nan,
            "n_counterparties_acute": np.nan,
            "top5_share_acute": np.nan,
            "hhi_acute": np.nan,
            "share_to_discount_houses": np.nan,
            "share_to_commercial_banks": np.nan,
            "share_to_merchant_banks": np.nan,
            "share_to_merchants_other": np.nan,
            "share_rejected_value": np.nan,
        }

    df = tx[tx["crisis"] == crisis_key]
    df = df[(df["date"] >= c.acute_start) & (df["date"] <= c.acute_end)]
    out["n_transactions_acute"] = len(df)
    if df.empty:
        return out

    by_cp = df.groupby("counterparty_clean")["total_amount"].sum().sort_values(ascending=False)
    total = by_cp.sum()
    out["n_counterparties_acute"] = int(by_cp[by_cp > 0].size)
    out["top5_share_acute"] = float(by_cp.head(5).sum() / total) if total else np.nan
    shares = by_cp / total
    out["hhi_acute"] = float((shares ** 2).sum() * 10_000) if total else np.nan

    by_type = df.groupby("counterparty_type")["total_amount"].sum() / total if total else None
    if by_type is not None:
        out["share_to_discount_houses"] = float(by_type.get("discount_house", 0))
        out["share_to_commercial_banks"] = float(by_type.get("commercial_bank", 0))
        out["share_to_merchant_banks"] = float(by_type.get("merchant_bank", 0))
        out["share_to_merchants_other"] = float(by_type.get("merchant", 0) + by_type.get("other", 0))

    rejected = df["value_bills_rejected"].sum()
    brought = df["value_brought"].sum()
    out["share_rejected_value"] = float(rejected / brought) if brought else np.nan
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    from io_utils import read_balance_sheet, read_transactions
    bs = read_balance_sheet(PROC / "boe_balance_sheet.parquet")
    tx = read_transactions(PROC / "lolr_transactions.parquet")

    rows = []
    for key, c in CRISES.items():
        bsm = balance_sheet_metrics(bs, c)
        lm = ledger_metrics(tx, c, key)
        rows.append({"crisis_key": key, **bsm, **lm})

    df = pd.DataFrame(rows).set_index("crisis_key")
    # Round friendly columns for display
    floats = df.select_dtypes(include="float").columns
    df[floats] = df[floats].round(3)
    df.to_csv(OUT_DIR / "crisis_metrics.csv")
    print(f"Wrote {OUT_DIR / 'crisis_metrics.csv'}")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(df.transpose().to_string())


if __name__ == "__main__":
    main()
