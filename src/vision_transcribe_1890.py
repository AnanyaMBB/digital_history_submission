"""Vision-LLM assisted transcription of the 1890 BoE Daily Account Book pages.

What this does. For each downloaded weekly-spread image, the script sends
the high-resolution JPEG to a Claude vision model with a strict
JSON-output prompt that asks for daily balance-sheet values. Raw model
responses are saved per-page so the run is auditable. Parsed daily rows
are written to a single CSV.

Honesty rules. The prompt explicitly forbids the model from guessing
numbers or filling cells from external context. Cells the model cannot
read must come back as null. Confidence is recorded per day and per page.

Environment required.
- `ANTHROPIC_API_KEY` set in the environment
- `anthropic` Python SDK installed (>=0.50)
- A vision-capable Claude model. The script defaults to
  `claude-opus-4-7` but allows override via `--model`.

What this script will NOT do. It will not fabricate a result if the API is
unavailable. If `ANTHROPIC_API_KEY` is missing, the script prints the
required environment variable and exits without writing any output.

Page selection. By default the script processes only the 14 weekly-spread
pages (C1-38_063 through C1-38_076) and skips the cover, the
securities-register page, the mid-April layout reference, and the
end-of-year rate-history table. Use `--all-pages` to process all 18.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
PAGES_DIR = ROOT / "data" / "raw" / "account_books_1890" / "pages"
OCR_OUT = ROOT / "outputs" / "ocr"
RAW_DIR = OCR_OUT / "vision_raw"
TRANSCRIPTION_CSV = OCR_OUT / "vision_transcription_oct_dec_1890.csv"
NOTES_CSV = OCR_OUT / "vision_transcription_notes.csv"

# Subset of pages that are weekly-spread daily-accounts pages.
SPREAD_PAGES = [
    "C1-38_063", "C1-38_064", "C1-38_065", "C1-38_066", "C1-38_067",
    "C1-38_068", "C1-38_069", "C1-38_070", "C1-38_071", "C1-38_072",
    "C1-38_073", "C1-38_074", "C1-38_075", "C1-38_076",
]

PROMPT = """You are transcribing a Bank of England Daily Account Book page from 1890.
The page is a ruled double-page accounting ledger spread. The left page shows
BALANCES (stock variables at end of each day). The right page shows OPERATIONS
(flow variables for each day). The spread usually covers six days, Wednesday
through the following Tuesday.

EXTRACT ONLY NUMBERS YOU CAN READ WITH REASONABLE CONFIDENCE. Do not guess.
If a number is unclear, ambiguous, smudged, crossed-out, written over, or
not visible at the resolution you are seeing, return null and add a note.

Strict rules:
- Never infer a missing digit from context.
- Never use external knowledge of what 1890 BoE balance sheets "should" look like
  to fill in cells. Only return what is legibly written.
- Numbers in the BALANCES section are typically in pounds sterling, recorded
  in thousands. So a written value like "11,623" means £11,623,000 = £11.623m.
  Return the value AS WRITTEN (i.e. 11623 for "11,623"), then we will
  convert downstream. Same applies to all monetary values.
- The "Minimum Rate of Discount" / "Bank Rate" is shown as a percentage at the
  top-left of the BALANCES side. Return it as a percent (e.g. 5 for 5%, 6 for 6%).
- Reserve proportion is sometimes shown as a fraction like "33.5" meaning 33.5%.
  Return as a percent number.
- The page may have crossed-out corrections and red-ink rate-change annotations.
  These often indicate a rate change date. Note them in `page_notes`.

Distinguish carefully between the BALANCES side (left page) and OPERATIONS
side (right page). The BALANCES side has rows like "Bullion Total", "Notes
Total", "Public Deposits", "Discounts (Banking Dept)", "Advances", "Reserve",
"Total Assets". The OPERATIONS side has daily flow rows including
"Discounts (London)", "Discounts (Country)", "Total Discounted", "Advances
London", "Advances Country", "Total Advances", "Clearing House".

Return JSON ONLY, no prose, in exactly this schema:

{
  "page_image": "the filename label like C1-38_069",
  "date_range_visible": "human-readable range you see, e.g. '1890-11-12 to 1890-11-18'",
  "balances_columns_count": integer (number of date columns on Balances side),
  "operations_columns_count": integer (number of date columns on Operations side),
  "days": [
    {
      "date": "YYYY-MM-DD or null if you cannot determine",
      "day_of_week": "Wednesday|Thursday|Friday|Saturday|Monday|Tuesday|null",
      "banking_public_deposits": number_or_null,
      "banking_other_deposits": number_or_null,
      "banking_seven_day_bills": number_or_null,
      "banking_govt_securities": number_or_null,
      "banking_other_securities": number_or_null,
      "banking_discounts": number_or_null,
      "banking_advances": number_or_null,
      "banking_reserve_notes_coin": number_or_null,
      "banking_total_assets": number_or_null,
      "reserve_proportion": number_or_null,
      "issue_notes_total": number_or_null,
      "issue_bullion_total": number_or_null,
      "issue_govt_securities": number_or_null,
      "issue_other_securities": number_or_null,
      "ops_london_discounts": number_or_null,
      "ops_country_discounts": number_or_null,
      "ops_total_discounts": number_or_null,
      "ops_london_advances": number_or_null,
      "ops_country_advances": number_or_null,
      "ops_total_advances": number_or_null,
      "clearing_house_total": number_or_null,
      "minimum_discount_rate": number_or_null,
      "confidence": "high|medium|low",
      "notes": "free text, mention which cells you could not read"
    }
  ],
  "page_notes": "free text — any red-ink rate change annotations, header info, or layout caveats"
}

If you cannot orient the table at all (e.g. the image is the cover or a
summary page rather than a daily spread), return:

{ "page_image": "...", "page_notes": "not a daily spread", "days": [] }

Return JSON ONLY. No prose before or after.
"""


def encode_image(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode("ascii")


def transcribe_page(client, model: str, page_label: str, image_path: Path) -> dict[str, Any]:
    img_b64 = encode_image(image_path)
    resp = client.messages.create(
        model=model,
        max_tokens=8000,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                    {"type": "text",
                     "text": PROMPT + f"\n\nThis page is labelled `{page_label}`. Begin."},
                ],
            }
        ],
    )
    # Save the full raw response object as JSON for auditability
    raw = {
        "id": resp.id,
        "model": resp.model,
        "stop_reason": resp.stop_reason,
        "usage": {"input_tokens": resp.usage.input_tokens,
                   "output_tokens": resp.usage.output_tokens},
        "content_text": "".join(block.text for block in resp.content if block.type == "text"),
    }
    return raw


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_model_json(text: str) -> dict[str, Any]:
    """The model is asked to return JSON-only, but it sometimes wraps the JSON
    in code fences. Strip them and try to parse.
    """
    cleaned = text.strip()
    # Remove ```json ... ``` fences if present
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Fall back to extracting the largest {...} block
        m = _JSON_BLOCK_RE.search(cleaned)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Could not parse JSON from model output: {exc}") from exc
        raise


def flatten_rows(parsed: dict[str, Any], page_label: str) -> list[dict[str, Any]]:
    """Flatten the per-page JSON into one row per day."""
    rows = []
    for day in parsed.get("days", []) or []:
        rows.append({
            "page_image": page_label,
            "date_range_visible": parsed.get("date_range_visible", ""),
            **{k: v for k, v in day.items()},
            "balances_columns_count": parsed.get("balances_columns_count", ""),
            "operations_columns_count": parsed.get("operations_columns_count", ""),
            "page_notes": parsed.get("page_notes", ""),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="claude-opus-4-7",
                        help="Claude vision-capable model id")
    parser.add_argument("--all-pages", action="store_true",
                        help="Process all 18 images instead of the 14 weekly spreads")
    parser.add_argument("--only", help="Only process this single page label (e.g. C1-38_069)")
    parser.add_argument("--sleep", type=float, default=0.5,
                        help="Seconds to sleep between page requests")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR. ANTHROPIC_API_KEY is not set.")
        print("To run vision-LLM transcription, set:")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        print(f"and optionally:")
        print(f"  export ANTHROPIC_BASE_URL=...   (custom proxy if needed)")
        print(f"Then re-run: .venv/bin/python {__file__}")
        return 2

    try:
        from anthropic import Anthropic
    except ImportError:
        print("ERROR. The anthropic SDK is not installed.")
        print("Install with: .venv/bin/pip install anthropic")
        return 2

    client = Anthropic()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OCR_OUT.mkdir(parents=True, exist_ok=True)

    # Determine which pages to process
    if args.only:
        page_labels = [args.only]
    elif args.all_pages:
        page_labels = sorted(p.stem for p in PAGES_DIR.glob("C1-38_*.jpg"))
    else:
        page_labels = SPREAD_PAGES

    all_rows: list[dict[str, Any]] = []
    all_notes: list[dict[str, Any]] = []
    total_input_tokens = 0
    total_output_tokens = 0

    for label in page_labels:
        img_path = PAGES_DIR / f"{label}.jpg"
        if not img_path.exists():
            print(f"  SKIP {label}: image not found at {img_path}")
            continue
        out_raw = RAW_DIR / f"{label}.json"
        print(f"  Processing {label} ...", end=" ", flush=True)
        try:
            raw = transcribe_page(client, args.model, label, img_path)
        except Exception as exc:
            print(f"FAILED: {exc}")
            all_notes.append({"page_image": label, "status": "api_error", "error": str(exc)})
            time.sleep(args.sleep)
            continue
        out_raw.write_text(json.dumps(raw, indent=2))
        total_input_tokens += raw["usage"]["input_tokens"]
        total_output_tokens += raw["usage"]["output_tokens"]
        try:
            parsed = parse_model_json(raw["content_text"])
        except Exception as exc:
            print(f"PARSE_FAIL: {exc}")
            all_notes.append({"page_image": label, "status": "parse_error",
                              "error": str(exc), "raw_chars": len(raw["content_text"])})
            time.sleep(args.sleep)
            continue
        rows = flatten_rows(parsed, label)
        all_rows.extend(rows)
        all_notes.append({
            "page_image": label,
            "status": "ok",
            "n_days_returned": len(rows),
            "balances_columns_count": parsed.get("balances_columns_count", ""),
            "operations_columns_count": parsed.get("operations_columns_count", ""),
            "page_notes": parsed.get("page_notes", ""),
            "input_tokens": raw["usage"]["input_tokens"],
            "output_tokens": raw["usage"]["output_tokens"],
        })
        print(f"OK n_days={len(rows)} in={raw['usage']['input_tokens']} out={raw['usage']['output_tokens']}")
        time.sleep(args.sleep)

    # Write outputs
    import pandas as pd
    if all_rows:
        # Standardize column order
        cols_order = [
            "page_image", "date_range_visible", "date", "day_of_week",
            "banking_public_deposits", "banking_other_deposits",
            "banking_seven_day_bills", "banking_govt_securities",
            "banking_other_securities", "banking_discounts",
            "banking_advances", "banking_reserve_notes_coin",
            "banking_total_assets", "reserve_proportion",
            "issue_notes_total", "issue_bullion_total",
            "issue_govt_securities", "issue_other_securities",
            "ops_london_discounts", "ops_country_discounts",
            "ops_total_discounts", "ops_london_advances",
            "ops_country_advances", "ops_total_advances",
            "clearing_house_total", "minimum_discount_rate",
            "confidence", "notes",
            "balances_columns_count", "operations_columns_count",
            "page_notes",
        ]
        df = pd.DataFrame(all_rows)
        df = df.reindex(columns=[c for c in cols_order if c in df.columns] +
                         [c for c in df.columns if c not in cols_order])
        df.to_csv(TRANSCRIPTION_CSV, index=False)
        print(f"\nWrote {TRANSCRIPTION_CSV} ({len(df)} daily rows)")
    else:
        print("\nNo rows extracted.")
    pd.DataFrame(all_notes).to_csv(NOTES_CSV, index=False)
    print(f"Wrote {NOTES_CSV} ({len(all_notes)} page records)")

    print(f"\nTotal tokens: input={total_input_tokens:,} output={total_output_tokens:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
