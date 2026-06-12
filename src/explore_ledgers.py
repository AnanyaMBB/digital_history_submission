"""Print header rows and a sample of each crisis ledger so we can document columns precisely."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LOLR = ROOT / "data" / "raw" / "boe_lolr" / "lolr-historical-dataset.xlsx"

SHEETS = ["B2. 1857 ledger", "B3. 1866 ledger", "B3a. 1914 daily ledger",
          "B5. 1857 daily metrics", "B6. 1866 daily metrics", "B6a. 1914 daily metrics",
          "A2a. Issue Department", "A2b. Banking Department"]

for s in SHEETS:
    print(f"\n========== {s} ==========")
    try:
        df = pd.read_excel(LOLR, sheet_name=s, header=None, nrows=15, engine="openpyxl")
        for i, row in df.iterrows():
            vals = [str(v) if pd.notna(v) else "" for v in row.tolist()]
            preview = " | ".join(f"[{j}]{v[:40]}" for j, v in enumerate(vals) if v)
            print(f"  row {i:>2}: {preview}")
    except Exception as e:
        print(f"  ERROR: {e}")
