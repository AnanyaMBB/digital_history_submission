"""IO helpers — robust against pandas/pyarrow datetime round-trip quirks."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


def _coerce_datetime(s: pd.Series) -> pd.Series:
    """Convert an arbitrary date-column representation back to datetime64[ns]."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.astype("datetime64[ns]")
    if pd.api.types.is_integer_dtype(s):
        # Could be ns, us, ms, or s since epoch — try in plausibility order.
        # us is pandas 3.0's default for datetime64 parquet round-trip.
        # Score each candidate by fraction of years in [1750, 1950] (our project range).
        best, best_score = None, -1.0
        for unit in ("us", "ms", "s", "ns"):
            try:
                out = pd.to_datetime(s, unit=unit, errors="coerce")
            except (ValueError, OverflowError):
                continue
            yr = out.dt.year
            score = float(yr.between(1750, 1950).mean())
            if score > best_score:
                best, best_score = out, score
        if best is not None and best_score > 0.5:
            return best.astype("datetime64[ns]")
    return pd.to_datetime(s, errors="coerce", dayfirst=True).astype("datetime64[ns]")


def read_balance_sheet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df.index = _coerce_datetime(pd.Series(df.index)).values
    df.index.name = "date"
    return df.sort_index()


def read_transactions(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["date"] = _coerce_datetime(df["date"])
    return df.dropna(subset=["date"]).sort_values(["crisis", "date"])


def read_bank_rate(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["date"] = _coerce_datetime(df["date"])
    return df.sort_values("date")
