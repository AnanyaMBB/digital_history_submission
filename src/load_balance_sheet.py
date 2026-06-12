"""Load the weekly BoE balance sheet from the LOLR historical dataset.

Produces a tidy weekly DataFrame indexed by date with consistent column names
and derived ratios. Written to data/processed/boe_balance_sheet.parquet.

The source sheets `A2a. Issue Department` and `A2b. Banking Department` start
on 7 September 1844 (a Saturday — weekly returns) and run through 1919.
Dates are recorded as DD/MM/YYYY.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LOLR = ROOT / "data" / "raw" / "boe_lolr" / "lolr-historical-dataset.xlsx"
OUT = ROOT / "data" / "processed" / "boe_balance_sheet.parquet"

ISSUE_COLS = {
    0: "date_raw",
    1: "issue_total_coin_bullion",
    2: "issue_gold",
    3: "issue_silver_bullion",
    4: "issue_silver_coin",
    5: "issue_total_govt_securities",
    6: "issue_govt_debt",
    7: "issue_other_govt_securities",
    8: "issue_other_securities",
    9: "issue_fiduciary_memo",
    10: "issue_total_assets",
    11: "notes_in_circulation",
    12: "notes_in_banking_dept",
    13: "total_notes_issued",
}

BANKING_COLS = {
    0: "date_raw",
    1: "banking_govt_securities",
    2: "banking_discounts_advances_other",
    3: "banking_discounts_total",
    4: "banking_discounts_london",
    5: "banking_discounts_country",
    6: "banking_discounts_total_check",
    7: "banking_advances_london",
    8: "banking_advances_country",
    9: "banking_advances_total",
    10: "banking_other_securities",
    11: "banking_reserve_notes_coin",
    12: "banking_reserve_notes",
    13: "banking_reserve_coin",
    14: "banking_total_assets",
    15: "banking_proprietary_and_rest",
    16: "banking_proprietary_capital",
    17: "banking_rest",
    18: "public_deposits",
    19: "other_deposits",
    20: "bankers_balances",
    21: "seven_day_and_other_bills",
    22: "banking_total_liabilities",
    27: "reserve_proportion",
    28: "bank_rate_weekly",
}


def _read_sheet(sheet: str, col_map: dict[int, str]) -> pd.DataFrame:
    raw = pd.read_excel(LOLR, sheet_name=sheet, header=None, engine="openpyxl")
    # Data rows start after the 7-row header. Find the first row whose col 0
    # parses as a date — that's our true start.
    def _is_date_like(v) -> bool:
        if isinstance(v, str) and "/" in v:
            return True
        if isinstance(v, (pd.Timestamp,)):
            return True
        return False

    first = next(i for i, v in enumerate(raw[0]) if _is_date_like(v))
    df = raw.iloc[first:].copy()
    keep = list(col_map.keys())
    df = df.loc[:, keep].rename(columns=col_map)
    # Parse date — mostly DD/MM/YYYY, occasionally Excel Timestamp objects
    df["date"] = pd.to_datetime(df["date_raw"], dayfirst=True, errors="coerce")
    df = df.drop(columns=["date_raw"]).dropna(subset=["date"])
    df = df.set_index("date").sort_index()
    # Coerce everything else to float
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load() -> pd.DataFrame:
    issue = _read_sheet("A2a. Issue Department", ISSUE_COLS)
    banking = _read_sheet("A2b. Banking Department", BANKING_COLS)
    df = issue.join(banking, how="outer")

    # Derived series — all in £m
    df["crisis_lending"] = df[["banking_discounts_total", "banking_advances_total"]].sum(axis=1, min_count=1)
    df["total_deposits"] = df[["public_deposits", "other_deposits"]].sum(axis=1, min_count=1)
    df["reserve_ratio"] = df["banking_reserve_notes_coin"] / df["total_deposits"]
    df["lending_to_reserve"] = df["crisis_lending"] / df["banking_reserve_notes_coin"]
    df["balance_sheet_size"] = df["banking_total_assets"]
    return df


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = load()
    df.to_parquet(OUT)
    print(f"Wrote {OUT}")
    print(f"  rows={len(df):,}  date range: {df.index.min().date()} – {df.index.max().date()}")
    print(f"  columns ({len(df.columns)}): {list(df.columns)}")
    print("\nSummary at end of 1856 (pre-1857 baseline):")
    print(df.loc["1856-11":"1856-12", ["banking_reserve_notes_coin", "crisis_lending", "bank_rate_weekly"]].tail())


if __name__ == "__main__":
    main()
