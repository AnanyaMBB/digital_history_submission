"""Load macro context: JST UK panel, daily Bank Rate, and annual UK headlines.

Outputs three parquet files in data/processed/:
- macro_jst_uk.parquet   — JST R6 restricted to UK, all years
- bank_rate_daily.parquet — Bank Rate as a daily series (constant between changes)
- macro_uk_annual.parquet — Millennium A1 headline series (annual UK)
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"


def load_jst_uk() -> pd.DataFrame:
    df = pd.read_stata(RAW / "jst" / "JSTdatasetR6.dta", convert_categoricals=False)
    uk = df[df["country"].astype(str).str.upper().isin(["UNITED KINGDOM", "UK"])].copy()
    if uk.empty:
        # fallback by iso code
        uk = df[df.get("iso", pd.Series([""] * len(df))).astype(str).str.upper() == "GBR"].copy()
    uk = uk.sort_values("year").reset_index(drop=True)
    return uk


def load_bank_rate_daily() -> pd.DataFrame:
    """From Millennium 'D1. Official Interest Rates'.

    Sheet is already daily: col 0 = date (DD/MM/YYYY string), col 2 = Bank Rate.
    """
    src = RAW / "boe_balance_sheet" / "millennium-of-macro-data-uk.xlsx"
    raw = pd.read_excel(src, sheet_name="D1. Official Interest Rates", header=None, engine="openpyxl")
    df = pd.DataFrame({
        "date": pd.to_datetime(raw[0], dayfirst=True, errors="coerce"),
        "bank_rate": pd.to_numeric(raw[2], errors="coerce"),
    }).dropna(subset=["date", "bank_rate"]).sort_values("date").drop_duplicates("date")
    return df.reset_index(drop=True)


def load_uk_annual() -> pd.DataFrame:
    """Annual UK headlines from Millennium A1 — keep year + nominal GDP + price index + Bank Rate."""
    src = RAW / "boe_balance_sheet" / "millennium-of-macro-data-uk.xlsx"
    # A1 has 4 header rows before data; first column is Year.
    raw = pd.read_excel(src, sheet_name="A1. Headline series", header=None, engine="openpyxl")
    # Header band rows 0..6; data starts at the first row where col 0 is a 4-digit int.
    first = next(i for i, v in enumerate(raw[0]) if isinstance(v, (int, float)) and 800 < v < 2100)
    df = raw.iloc[first:].copy()
    # We only need col 0 (year) and a handful of named series. Pull description row 3 to find columns.
    desc = raw.iloc[3]
    df = df.rename(columns={0: "year"})
    # Map of friendly name -> first matching column index
    def find_col(keyword: str) -> int | None:
        for j, v in desc.items():
            if isinstance(v, str) and keyword.lower() in v.lower():
                return j
        return None

    wanted = {
        "nominal_gdp": "Nominal UK GDP at market prices",
        "cpi": "Consumer price index",
        "bank_rate_annual": "Bank Rate",
        "consols_yield": "Consols",
        "broad_money": "Broad Money",
        "credit_total": "Credit",
        "boe_balance_sheet": "Bank of England Balance sheet",
    }
    keep = {"year": "year"}
    for friendly, key in wanted.items():
        col = find_col(key)
        if col is not None:
            keep[col] = friendly
            df[col] = pd.to_numeric(df[col], errors="coerce")
    out = df[list(keep.keys())].rename(columns=keep)
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["year"]).reset_index(drop=True)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jst = load_jst_uk()
    jst.to_parquet(OUT / "macro_jst_uk.parquet")
    print(f"Wrote macro_jst_uk: rows={len(jst):,} years={int(jst['year'].min())}–{int(jst['year'].max())}")

    rate = load_bank_rate_daily()
    rate.to_parquet(OUT / "bank_rate_daily.parquet")
    print(f"Wrote bank_rate_daily: rows={len(rate):,} dates={rate['date'].min().date()}–{rate['date'].max().date()}")

    ann = load_uk_annual()
    ann.to_parquet(OUT / "macro_uk_annual.parquet")
    print(f"Wrote macro_uk_annual: rows={len(ann):,} years={int(ann['year'].min())}–{int(ann['year'].max())}")
    print(f"  columns: {list(ann.columns)}")


if __name__ == "__main__":
    main()
