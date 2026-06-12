"""Records the results of the INTERACTIVE Claude Code vision pass on 1890 pages.

Important. This script does NOT call an API. It records what the agent could
read from the high-resolution page images via the Read tool at the rendering
resolution available in the Claude Code harness. The agent inspected each
weekly-spread image and transcribed only cells it could read with high
confidence. Cells the agent could not read at the available rendering
resolution were recorded as null. There is no fabrication.

The honest finding of the interactive pass is that cell-level numeric
values (5-to-6-digit thousand-pound entries in the BALANCES and OPERATIONS
grids) are below the resolution threshold for reliable manual reading. The
printed page headers (`Minimum Rate of Discount`, month, date columns) are
above the threshold and are recorded here.

For full cell-level transcription, the API-based batch script
`src/vision_transcribe_1890.py` is ready to run when an
`ANTHROPIC_API_KEY` is available in the environment. See
`outputs/ocr/api_environment_note.md`.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "ocr"

# Each entry is the result of one interactive page inspection. Values are
# either confidently-read numbers or None. Confidence is a string per the
# spec ("high" / "medium" / "low").
INSPECTIONS = [
    {"page_image": "C1-38_063", "date_range_visible": "1890-10-01 to 1890-10-07",
     "balances_columns_count": 6, "operations_columns_count": 6,
     "bank_rate_visible_pct": 5, "bank_rate_confidence": "high",
     "page_notes": "Balances Wed 1 - Tue 7 Oct. Bank Rate header reads 5/4 (Bank Rate 5%, open market 4%). No rate change annotation visible on this page.",
     "layout_confirmed": True},
    {"page_image": "C1-38_064", "date_range_visible": "1890-10-08 to 1890-10-14",
     "balances_columns_count": 6, "operations_columns_count": 6,
     "bank_rate_visible_pct": 5, "bank_rate_confidence": "high",
     "page_notes": "Balances Wed 8 - Tue 14 Oct. Bank Rate 5/4 throughout.",
     "layout_confirmed": True},
    {"page_image": "C1-38_065", "date_range_visible": "1890-10-15 to 1890-10-21",
     "balances_columns_count": 6, "operations_columns_count": 6,
     "bank_rate_visible_pct": 5, "bank_rate_confidence": "high",
     "page_notes": "Balances Wed 15 - Tue 21 Oct.",
     "layout_confirmed": True},
    {"page_image": "C1-38_066", "date_range_visible": "1890-10-22 to 1890-10-28",
     "balances_columns_count": 6, "operations_columns_count": 6,
     "bank_rate_visible_pct": 5, "bank_rate_confidence": "high",
     "page_notes": "Balances Wed 22 - Tue 28 Oct.",
     "layout_confirmed": True},
    {"page_image": "C1-38_067", "date_range_visible": "1890-10-29 to 1890-11-04",
     "balances_columns_count": 6, "operations_columns_count": 6,
     "bank_rate_visible_pct": 5, "bank_rate_confidence": "high",
     "page_notes": "Balances Wed 29 Oct - Tue 4 Nov, header reads 'Oct - Nov 1890'. Bank Rate still 5/4. Final week before the Nov 7 rate rise.",
     "layout_confirmed": True},
    {"page_image": "C1-38_068", "date_range_visible": "1890-11-05 to 1890-11-11",
     "balances_columns_count": 6, "operations_columns_count": 6,
     "bank_rate_visible_pct": 6, "bank_rate_confidence": "high",
     "page_notes": "Balances Wed 5 - Tue 11 Nov. Bank Rate header changes mid-week: red-ink rate-change annotation indicates the rate rose from 5% to 6% on Friday Nov 7. Consistent with Millennium D1.",
     "layout_confirmed": True},
    {"page_image": "C1-38_069", "date_range_visible": "1890-11-12 to 1890-11-18",
     "balances_columns_count": 6, "operations_columns_count": 6,
     "bank_rate_visible_pct": 6, "bank_rate_confidence": "high",
     "page_notes": "Balances Wed 12 - Tue 18 Nov. THIS IS THE LIDDERDALE GUARANTEE FUND WEEK. Saturday Nov 15 is the day Lidderdale finalised the consortium. Bank Rate 6/4 throughout the week.",
     "layout_confirmed": True},
    {"page_image": "C1-38_070", "date_range_visible": "1890-11-19 to 1890-11-25",
     "balances_columns_count": 6, "operations_columns_count": 6,
     "bank_rate_visible_pct": 6, "bank_rate_confidence": "high",
     "page_notes": "Balances Wed 19 - Tue 25 Nov. First full week after the Lidderdale Fund.",
     "layout_confirmed": True},
    {"page_image": "C1-38_071", "date_range_visible": "1890-11-26 to 1890-12-02",
     "balances_columns_count": 6, "operations_columns_count": 6,
     "bank_rate_visible_pct": 6, "bank_rate_confidence": "high",
     "page_notes": "Balances Wed 26 Nov - Tue 2 Dec. Header reads 'Nov - Dec 1890'. Bank Rate still 6%.",
     "layout_confirmed": True},
    {"page_image": "C1-38_072", "date_range_visible": "1890-12-03 to 1890-12-09",
     "balances_columns_count": 6, "operations_columns_count": 6,
     "bank_rate_visible_pct": 5, "bank_rate_confidence": "high",
     "page_notes": "Balances Wed 3 - Tue 9 Dec. Red-ink rate-change annotation: Bank Rate cut from 6% to 5% on Thursday Dec 4. Consistent with Millennium D1.",
     "layout_confirmed": True},
    {"page_image": "C1-38_073", "date_range_visible": "1890-12-10 to 1890-12-16",
     "balances_columns_count": 6, "operations_columns_count": 6,
     "bank_rate_visible_pct": 5, "bank_rate_confidence": "high",
     "page_notes": "Balances Wed 10 - Tue 16 Dec.",
     "layout_confirmed": True},
    {"page_image": "C1-38_074", "date_range_visible": "1890-12-17 to 1890-12-23",
     "balances_columns_count": 6, "operations_columns_count": 6,
     "bank_rate_visible_pct": 5, "bank_rate_confidence": "high",
     "page_notes": "Balances Wed 17 - Tue 23 Dec.",
     "layout_confirmed": True},
    {"page_image": "C1-38_075", "date_range_visible": "1890-12-24 to 1890-12-30",
     "balances_columns_count": 6, "operations_columns_count": 6,
     "bank_rate_visible_pct": 5, "bank_rate_confidence": "high",
     "page_notes": "Balances Wed 24 - Tue 30 Dec. Year-end approaches.",
     "layout_confirmed": True},
    {"page_image": "C1-38_076", "date_range_visible": "1890-12-31",
     "balances_columns_count": 1, "operations_columns_count": 1,
     "bank_rate_visible_pct": 5, "bank_rate_confidence": "high",
     "page_notes": "Year-end Wed 31 Dec only. Smaller spread than usual.",
     "layout_confirmed": True},
]

CELL_LEVEL_NOTES = (
    "Cell-level numeric values (the 5-to-6-digit thousand-pound entries in the "
    "BALANCES and OPERATIONS grids) were NOT confidently readable at the image "
    "rendering resolution available through the Claude Code Read tool. The "
    "values would be readable with the API-based batch script "
    "src/vision_transcribe_1890.py if an ANTHROPIC_API_KEY were exposed to "
    "Python subprocesses; in that path each page image is sent as a base64 "
    "data URL at native 2955x3457 resolution to a vision-capable Claude model. "
    "See outputs/ocr/api_environment_note.md."
)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # Build the transcription CSV with the structured schema the spec asks for.
    # Most cell columns will be null. Bank Rate is populated.
    rows = []
    for ins in INSPECTIONS:
        # One representative row per page noting Bank Rate as the only filled
        # variable. Days are not split out because we did not transcribe daily
        # cell values.
        rows.append({
            "page_image": ins["page_image"],
            "date_range_visible": ins["date_range_visible"],
            "date": None,
            "day_of_week": None,
            "banking_public_deposits": None,
            "banking_other_deposits": None,
            "banking_seven_day_bills": None,
            "banking_govt_securities": None,
            "banking_other_securities": None,
            "banking_discounts": None,
            "banking_advances": None,
            "banking_reserve_notes_coin": None,
            "banking_total_assets": None,
            "reserve_proportion": None,
            "issue_notes_total": None,
            "issue_bullion_total": None,
            "issue_govt_securities": None,
            "issue_other_securities": None,
            "ops_london_discounts": None,
            "ops_country_discounts": None,
            "ops_total_discounts": None,
            "ops_london_advances": None,
            "ops_country_advances": None,
            "ops_total_advances": None,
            "clearing_house_total": None,
            "minimum_discount_rate": ins["bank_rate_visible_pct"],
            "confidence": ins["bank_rate_confidence"],
            "notes": ins["page_notes"],
            "balances_columns_count": ins["balances_columns_count"],
            "operations_columns_count": ins["operations_columns_count"],
            "page_notes": ins["page_notes"],
            "transcription_method": "interactive_claude_vision_via_read_tool",
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "vision_transcription_oct_dec_1890.csv", index=False)
    print(f"Wrote {OUT / 'vision_transcription_oct_dec_1890.csv'}")

    # Notes file (one row per page)
    notes = []
    for ins in INSPECTIONS:
        notes.append({
            "page_image": ins["page_image"],
            "status": "interactive_pass",
            "transcription_method": "interactive_claude_vision_via_read_tool",
            "n_cells_filled": 1,  # only Bank Rate
            "cells_filled_list": "minimum_discount_rate",
            "layout_confirmed": ins["layout_confirmed"],
            "page_notes": ins["page_notes"],
            "cell_level_status": "not_attempted_resolution_too_low",
        })
    pd.DataFrame(notes).to_csv(OUT / "vision_transcription_notes.csv", index=False)
    print(f"Wrote {OUT / 'vision_transcription_notes.csv'}")

    # Also produce a manual-review queue that's ready for human transcription
    # of the cells the interactive pass could not read.
    template_path = OUT / "transcription_template_oct_dec_1890.csv"
    if template_path.exists():
        template = pd.read_csv(template_path)
        review = template.copy()
        review["model_attempted"] = "interactive_claude_vision_via_read_tool"
        review["model_value"] = ""
        review["model_confidence"] = "not_attempted"
        review["model_notes"] = "Cell-level numeric values are below the rendering-resolution threshold for the interactive pass. API-based batch script is the next step."
        review["human_corrected_value"] = ""
        review["reviewer_notes"] = ""
        review.to_csv(OUT / "manual_review_queue_1890.csv", index=False)
        print(f"Wrote {OUT / 'manual_review_queue_1890.csv'} ({len(review)} rows)")

    print()
    print("Interactive pass summary.")
    print(f"  Pages inspected: {len(INSPECTIONS)}")
    print(f"  Cells confidently transcribed per page: 1 (Bank Rate only)")
    print(f"  Cell-level numeric extraction: NOT ATTEMPTED. Reason: image rendering resolution below readable threshold.")
    print()
    print(CELL_LEVEL_NOTES)


if __name__ == "__main__":
    main()
