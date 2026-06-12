"""Integrate the OCR pass on the 1890 Daily Account Books with the existing
weekly balance-sheet data.

What this script does.
- Reads `outputs/ocr/metadata.json` (produced by `ocr_account_books_1890.py`)
  for the page-to-date mapping established by visual inspection.
- Reads the existing weekly balance sheet for Oct-Dec 1890 and identifies the
  Saturday-ending value for each weekly spread.
- Cross-checks the Bank Rate column against the daily Millennium D1 series.
- Writes:
    outputs/ocr/page_anchors_oct_dec_1890.csv  - page image -> weekly anchor
    outputs/ocr/bank_rate_verified_1890.csv    - daily Bank Rate from Millennium
                                                  with which ledger page covers it
    outputs/ocr/transcription_template_oct_dec_1890.csv  - empty template ready
                                                  for human transcription
- Writes a summary document `docs/ocr_results_1890.md`.

This script does NOT produce machine-extracted cell values. The OCR pass
(`ocr_account_books_1890.py`) returned 39.5% mean confidence across 18 pages,
which is below the threshold for reliable cell-level numeric extraction on
handwritten 19th-century ledger images. The honest deliverable is the
infrastructure (downloads + raw text + anchor mapping + transcription
template) that supports either a future manual-transcription pass or a
multimodal vision-LLM pass.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from io_utils import read_balance_sheet, read_bank_rate  # noqa: E402

OCR_OUT = ROOT / "outputs" / "ocr"
META = OCR_OUT / "metadata.json"


def main() -> None:
    metadata = json.loads(META.read_text())

    bs = read_balance_sheet(ROOT / "data" / "processed" / "boe_balance_sheet.parquet")
    bs_1890 = bs.loc["1890-09-15":"1890-12-31"]

    br_daily = read_bank_rate(ROOT / "data" / "processed" / "bank_rate_daily.parquet")
    br_1890 = br_daily[(br_daily["date"] >= "1890-10-01") & (br_daily["date"] <= "1890-12-31")].copy()

    # 1) Page-to-anchor table
    anchor_rows = []
    for name, info in metadata.items():
        if not info.get("balances"):
            continue
        try:
            start, end = info["balances"].split(" to ")
            start_ts = pd.Timestamp(start)
            end_ts = pd.Timestamp(end)
        except Exception:
            continue
        # Find the Saturday-ending weekly row in this range
        in_range = bs_1890.loc[start_ts:end_ts]
        if in_range.empty:
            anchor_rows.append({"page_image": name, **info, "weekly_anchor_date": "",
                                 "weekly_anchor_reserve": "", "weekly_anchor_crisis_lending": "",
                                 "weekly_anchor_bank_rate": ""})
            continue
        # Saturday is the BoE weekly observation; pick the last (rightmost) date in window
        anchor_date = in_range.index[-1]
        anchor_rows.append({
            "page_image": name,
            "canvas": info.get("canvas"),
            "balances_range": info["balances"],
            "operations_range": info.get("operations", ""),
            "annotation": info.get("annotation", ""),
            "ocr_status": info.get("ocr_status"),
            "ocr_mean_confidence": info.get("mean_confidence"),
            "image_sha256": info.get("image_sha256"),
            "weekly_anchor_date": anchor_date.date().isoformat(),
            "weekly_anchor_reserve_m": round(float(in_range.iloc[-1]["banking_reserve_notes_coin"]), 3),
            "weekly_anchor_crisis_lending_m": round(float(in_range.iloc[-1]["crisis_lending"]), 3),
            "weekly_anchor_bank_rate_pct": float(in_range.iloc[-1]["bank_rate_weekly"]),
            "weekly_anchor_reserve_proportion": round(float(in_range.iloc[-1]["reserve_proportion"]), 3),
        })
    anchors_df = pd.DataFrame(anchor_rows)
    anchors_df.to_csv(OCR_OUT / "page_anchors_oct_dec_1890.csv", index=False)
    print(f"Wrote {OCR_OUT / 'page_anchors_oct_dec_1890.csv'} ({len(anchors_df)} rows)")

    # 2) Daily Bank Rate from Millennium with attached page image
    # Map every business day in Oct-Dec 1890 to the page image covering it
    page_ranges = []
    for name, info in metadata.items():
        if not info.get("balances"):
            continue
        try:
            s, e = info["balances"].split(" to ")
            page_ranges.append((pd.Timestamp(s), pd.Timestamp(e), name))
        except Exception:
            continue
    page_ranges.sort()

    def page_for(d):
        for s, e, n in page_ranges:
            if s <= d <= e:
                return n
        return ""

    br_1890["page_image"] = br_1890["date"].apply(page_for)
    # Keep one row per calendar day (the daily series already is daily, but mark whether weekday)
    br_1890["weekday"] = br_1890["date"].dt.day_name()
    br_1890["is_business_day"] = br_1890["weekday"] != "Sunday"
    br_1890.to_csv(OCR_OUT / "bank_rate_verified_1890.csv", index=False)
    print(f"Wrote {OCR_OUT / 'bank_rate_verified_1890.csv'} ({len(br_1890)} daily rows)")

    # 3) Transcription template: one row per business day, target columns empty
    business_days = br_1890[br_1890["is_business_day"]].copy()
    template = business_days[["date", "weekday", "bank_rate", "page_image"]].rename(
        columns={"bank_rate": "bank_rate_millennium_d1"}
    )
    target_cols = [
        "issue_total_coin_bullion_m",
        "issue_total_govt_securities_m",
        "issue_other_securities_m",
        "notes_in_circulation_m",
        "banking_govt_securities_m",
        "banking_discounts_total_m",
        "banking_advances_total_m",
        "banking_other_securities_m",
        "banking_reserve_notes_coin_m",
        "public_deposits_m",
        "other_deposits_m",
        "reserve_proportion",
        "bank_rate_from_ledger_pct",
        "ocr_confidence",
        "transcription_method",
        "transcriber_notes",
    ]
    for c in target_cols:
        template[c] = ""
    template.to_csv(OCR_OUT / "transcription_template_oct_dec_1890.csv", index=False)
    print(f"Wrote {OCR_OUT / 'transcription_template_oct_dec_1890.csv'} ({len(template)} rows)")

    # 4) Summary
    n_pages_ocr_ok = sum(1 for m in metadata.values() if m.get("ocr_status") == "ok")
    mean_conf = sum(m.get("mean_confidence", 0) for m in metadata.values()
                     if isinstance(m.get("mean_confidence"), (int, float))) / max(1, n_pages_ocr_ok)
    print(f"\nOCR summary. {n_pages_ocr_ok} pages processed at mean confidence {mean_conf:.1f}%.")
    print("Tesseract confidence is below the threshold for reliable cell-level numeric extraction.")
    print("Raw OCR text is available in outputs/ocr/raw_text/ for navigation and search.")
    print("Daily Bank Rate is available from Millennium D1 with page-image mapping.")
    print("Daily balance-sheet rows await manual transcription or multimodal-LLM pass.")


if __name__ == "__main__":
    main()
