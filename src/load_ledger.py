"""Load transaction-level discount/advance ledgers for 1857, 1866, 1914.

Schema is slightly different across crises — this normalizes into a long
DataFrame with columns:
    date, crisis, counterparty_raw, counterparty_clean, drawing_office,
    rate, num_bills, value_bills_brought, value_bills_discounted (=loan_value),
    num_bills_rejected, value_bills_rejected, amount_advanced_bills,
    amount_advanced_securities, total_amount, transaction_type, remarks

Output: data/processed/lolr_transactions.parquet
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LOLR = ROOT / "data" / "raw" / "boe_lolr" / "lolr-historical-dataset.xlsx"
OUT = ROOT / "data" / "processed" / "lolr_transactions.parquet"

# Column index in source sheet → standardized name.
# 1857 source columns:  Date | NumBills | Rate | LoanValue | Counterparty |
#                       ValueBrought | DrawingOffice | NumRejected |
#                       AmtRejected | AmtAdvanced | Remarks
SCHEMA_1857 = {
    0: "date_raw", 1: "num_bills", 2: "rate", 3: "value_discounted",
    4: "counterparty_raw", 5: "value_brought", 6: "drawing_office",
    7: "num_bills_rejected", 8: "value_bills_rejected",
    9: "amount_advanced_bills", 10: "remarks",
}
# 1866 source columns:  Date | ValueBrought | Rate | LoanValue | Counterparty |
#                       DrawingOffice | NumBills | ValueRejected | NumRejected |
#                       AmtAdvanced | Remarks
SCHEMA_1866 = {
    0: "date_raw", 1: "value_brought", 2: "rate", 3: "value_discounted",
    4: "counterparty_raw", 5: "drawing_office", 6: "num_bills",
    7: "value_bills_rejected", 8: "num_bills_rejected",
    9: "amount_advanced_bills", 10: "remarks",
}
# 1914 source columns:  Date | Rate | LoanValue | NumBills | Counterparty |
#                       OnAccountFor | ValueBrought | NumRejected | ValueRejected |
#                       AmtAdvBills | AmtAdvSecurities
SCHEMA_1914 = {
    0: "date_raw", 1: "rate", 2: "value_discounted", 3: "num_bills",
    4: "counterparty_raw", 5: "on_account_for", 6: "value_brought",
    7: "num_bills_rejected", 8: "value_bills_rejected",
    9: "amount_advanced_bills", 10: "amount_advanced_securities",
}

SHEETS = {
    "1857": ("B2. 1857 ledger", SCHEMA_1857),
    "1866": ("B3. 1866 ledger", SCHEMA_1866),
    "1914": ("B3a. 1914 daily ledger", SCHEMA_1914),
}


def _clean_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = name.strip()
    s = re.sub(r"\s+", " ", s)
    # Common variants
    s = s.replace(" and ", " & ").replace(" And ", " & ")
    return s.title()


def _classify_counterparty(name: str) -> str:
    """Lightweight rule-based classifier; better than nothing pre-LLM step.

    Categories follow the prompt: commercial_bank / merchant_bank / bill_broker /
    discount_house / merchant / other / unknown.
    """
    if not name:
        return "unknown"
    n = name.lower()
    if "discount" in n or "discnt" in n:
        return "discount_house"
    if "bank" in n:
        # crude split: 'merchant bank' is rare; treat all 'bank' as commercial_bank
        return "commercial_bank"
    if any(k in n for k in ["broker", "brokers"]):
        return "bill_broker"
    if any(k in n for k in ["rothschild", "baring", "schroder", "huth", "bischoffsheim",
                              "goldschmidt", "morgan", "lazard", "gibbs", "kleinwort"]):
        return "merchant_bank"
    if any(k in n for k in [" & co", "ltd", "limited", "bros", "co ", " co.", " coy"]):
        return "merchant"
    return "other"


def _load_one(crisis: str) -> pd.DataFrame:
    sheet, schema = SHEETS[crisis]
    raw = pd.read_excel(LOLR, sheet_name=sheet, header=None, engine="openpyxl")
    # Data rows are those where col 0 parses as a date.
    keep_idx = []
    for i, v in enumerate(raw[0]):
        if isinstance(v, str) and re.match(r"\d{1,2}/\d{1,2}/\d{4}", v):
            keep_idx.append(i)
        elif isinstance(v, pd.Timestamp):
            keep_idx.append(i)
    df = raw.loc[keep_idx, list(schema.keys())].rename(columns=schema).copy()
    df["date"] = pd.to_datetime(df["date_raw"], dayfirst=True, errors="coerce")
    df = df.drop(columns=["date_raw"]).dropna(subset=["date"])

    # Numeric coercion
    num_cols = [c for c in df.columns if c not in {"counterparty_raw", "drawing_office",
                                                     "on_account_for", "remarks"}]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drawing office flag
    if "drawing_office" in df.columns:
        df["drawing_office"] = df["drawing_office"].astype(str).str.contains("D.O", na=False)
    else:
        df["drawing_office"] = False

    df["counterparty_raw"] = df["counterparty_raw"].astype(str)
    df["counterparty_clean"] = df["counterparty_raw"].map(_clean_name)
    df["counterparty_type"] = df["counterparty_clean"].map(_classify_counterparty)

    # Ensure all standard columns exist
    for c in ["value_brought", "value_discounted", "value_bills_rejected",
              "num_bills", "num_bills_rejected", "amount_advanced_bills",
              "amount_advanced_securities", "remarks", "on_account_for"]:
        if c not in df.columns:
            df[c] = np.nan if c not in {"remarks", "on_account_for"} else ""

    # Total cash flow per transaction: discounted bill value + any advance
    df["amount_advanced_total"] = df[["amount_advanced_bills", "amount_advanced_securities"]].sum(axis=1, min_count=1)
    df["total_amount"] = df[["value_discounted", "amount_advanced_total"]].sum(axis=1, min_count=1)

    df["transaction_type"] = np.where(
        df["amount_advanced_total"].fillna(0) > 0,
        np.where(df["value_discounted"].fillna(0) > 0, "discount+advance", "advance"),
        "discount",
    )
    df["crisis"] = crisis
    return df[[
        "date", "crisis", "counterparty_raw", "counterparty_clean", "counterparty_type",
        "drawing_office", "rate", "num_bills", "value_brought", "value_discounted",
        "num_bills_rejected", "value_bills_rejected",
        "amount_advanced_bills", "amount_advanced_securities", "amount_advanced_total",
        "total_amount", "transaction_type", "on_account_for", "remarks",
    ]]


def load() -> pd.DataFrame:
    frames = [_load_one(c) for c in SHEETS]
    df = pd.concat(frames, ignore_index=True).sort_values(["crisis", "date", "counterparty_clean"])
    return df.reset_index(drop=True)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = load()
    df.to_parquet(OUT)
    print(f"Wrote {OUT}")
    print(f"  rows={len(df):,}")
    print("  by crisis:")
    print(df.groupby("crisis").agg(
        n_rows=("date", "size"),
        n_dates=("date", "nunique"),
        n_counterparties=("counterparty_clean", "nunique"),
        total_value=("total_amount", "sum"),
        date_min=("date", "min"),
        date_max=("date", "max"),
    ))
    print("\n  counterparty_type distribution:")
    print(df.groupby(["crisis", "counterparty_type"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
