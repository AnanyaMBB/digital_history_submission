"""
v4_press_fetch.py — stream the British Library HMD newspapers corpus
(biglam/hmd_newspapers) and cache the articles inside each crisis window for the
paper_v4 study. Broad windows: ~6 months before to ~3 months after the public
crisis date, for all SEVEN crises (1847, 1857, 1866, 1873, 1890, 1907, 1914).

This both (a) probes HMD coverage for the new years (1873, 1907, 1914) and
(b) builds the cache the v4 text-mining pipeline reads. Uses date predicate
pushdown so only crisis-window rows are materialised.

Run:  ./.venv/bin/python src/v4_press_fetch.py
"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pds

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "press"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "hmd_v4_windows.parquet"
REPO = "biglam/hmd_newspapers"
COLS = ["date", "title", "location", "word_count", "ocr_quality_mean", "text"]

# Broad windows: ~6 months before to ~3 months after the public crisis date.
WINDOWS = {
    "1847": ("1847-04-01", "1848-01-31"),
    "1857": ("1857-05-01", "1858-02-28"),
    "1866": ("1865-11-01", "1866-08-31"),
    "1873": ("1873-05-01", "1874-02-28"),
    "1890": ("1890-05-01", "1891-02-28"),
    "1907": ("1907-04-01", "1908-01-31"),
    "1914": ("1914-02-01", "1914-11-30"),
}


def _filter():
    f = None
    for a, b in WINDOWS.values():
        clause = (pc.field("date") >= pa.scalar(datetime.fromisoformat(a))) & (
            pc.field("date") <= pa.scalar(datetime.fromisoformat(b + "T23:59:59")))
        f = clause if f is None else (f | clause)
    return f


def main():
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    shards = sorted(fs.glob(f"datasets/{REPO}/data/*.parquet"))
    print(f"[fetch] {len(shards)} shards; 7 crisis windows")
    filt = _filter()
    parts = []
    for i, path in enumerate(shards):
        try:
            tbl = pds.dataset(path, filesystem=fs, format="parquet").to_table(
                filter=filt, columns=COLS)
            if tbl.num_rows:
                parts.append(tbl.to_pandas())
            print(f"  shard {i:02d}: {tbl.num_rows:6d}")
        except Exception as e:  # noqa: BLE001
            print(f"  shard {i:02d}: ERROR {type(e).__name__}: {e}")
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLS)
    df["date"] = pd.to_datetime(df["date"])
    df.to_parquet(CACHE, index=False)
    print(f"[fetch] cached {len(df):,} articles -> {CACHE.name}\n")
    # per-crisis coverage
    print("=== HMD coverage per crisis window ===")
    for c, (a, b) in WINDOWS.items():
        s, e = pd.Timestamp(a), pd.Timestamp(b) + pd.Timedelta(hours=23)
        n = ((df["date"] >= s) & (df["date"] <= e)).sum()
        nq = ((df["date"] >= s) & (df["date"] <= e) &
              (df["ocr_quality_mean"] >= 0.70) & (df["word_count"] >= 40)).sum()
        print(f"  {c}: {n:6d} articles ({nq} with OCR>=0.70 & words>=40)")


if __name__ == "__main__":
    main()
