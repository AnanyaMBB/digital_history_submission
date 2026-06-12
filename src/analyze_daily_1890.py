"""Build the cleaned, validated daily 1890 dataset and run the Lidderdale analysis.

The pipeline:
1. Load the raw vision-LLM transcription (`outputs/ocr/vision_transcription_oct_dec_1890.csv`).
2. Keep ONLY the variables that passed Wednesday-anchor validation (Total Assets,
   Issue Notes Total, Issue Bullion Total, Reserve at validated dates).
3. Override Bank Rate with Millennium D1 (authoritative source) wherever the
   model-read disagrees.
4. Compute the Lidderdale-fortnight summary and the rescue-vs-year-end separation.
5. Write the cleaned CSV with per-cell confidence/provenance, the summary table,
   and the analysis figure.

Outputs:
- outputs/ocr/daily_1890_cleaned.csv           # final validated daily dataset
- outputs/ocr/daily_1890_validation_summary.md # what's usable and why
- outputs/tables/daily_1890_lidderdale_summary.csv
- outputs/figures/daily_1890_lidderdale_cleaned.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from io_utils import read_balance_sheet, read_bank_rate  # noqa: E402

VISION_RAW = ROOT / "outputs" / "ocr" / "vision_transcription_oct_dec_1890.csv"
PARQUET = ROOT / "data" / "processed" / "boe_balance_sheet.parquet"
BANKRATE = ROOT / "data" / "processed" / "bank_rate_daily.parquet"
CLEAN_CSV = ROOT / "outputs" / "ocr" / "daily_1890_cleaned.csv"
VAL_SUMMARY = ROOT / "outputs" / "ocr" / "daily_1890_validation_summary.md"
LIDD_SUMMARY = ROOT / "outputs" / "tables" / "daily_1890_lidderdale_summary.csv"
LIDD_FIG = ROOT / "outputs" / "figures" / "daily_1890_lidderdale_cleaned.png"

# Variables that passed Wednesday-anchor validation at 1% tolerance.
# (See outputs/ocr/vision_validation_summary.md for the audit.)
VALIDATED_VARS = {
    "banking_total_assets": "Banking Dept Total Assets (£ thousand)",
    "banking_reserve_notes_coin": "Banking Dept Reserve in Notes and Coin (£ thousand)",
    "issue_notes_total": "Issue Dept Total Notes Issued (£ thousand)",
    "issue_bullion_total": "Issue Dept Total Bullion (£ thousand)",
    "reserve_proportion": "Reserve / Deposits ratio (%)",
}

# Variables we will NOT use as evidence (failed validation or low fill rate).
EXCLUDED_VARS = {
    "banking_govt_securities": "schema mismatch; 0/14 Wed-anchor pass",
    "banking_other_securities": "schema mismatch; 0/14 Wed-anchor pass",
    "banking_discounts": "42% fill rate; not anchor-tested",
    "banking_advances": "42% fill rate; not anchor-tested",
    "banking_public_deposits": "24% fill rate; not anchor-tested",
    "banking_other_deposits": "64% fill rate; not anchor-tested",
    "banking_seven_day_bills": "not anchor-tested",
    "ops_total_discounts": "65% fill rate; flow variable, not anchor-tested",
    "ops_total_advances": "66% fill rate; flow variable, not anchor-tested",
    "ops_london_discounts": "78% fill rate; flow variable, not anchor-tested",
    "ops_country_discounts": "74% fill rate; flow variable, not anchor-tested",
    "ops_london_advances": "75% fill rate; flow variable, not anchor-tested",
    "ops_country_advances": "81% fill rate; flow variable, not anchor-tested",
    "clearing_house_total": "81% fill rate; flow variable, not anchor-tested",
    "issue_govt_securities": "0% fill rate; model did not extract",
    "issue_other_securities": "0% fill rate; model did not extract",
}


def build_cleaned_daily() -> pd.DataFrame:
    """Load vision-LLM rows, keep only validated variables, override Bank Rate with D1."""
    raw = pd.read_csv(VISION_RAW)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Deduplicate: the source has two Wed rows for some dates because the page-spread
    # holds both end-of-week and start-of-next-week. Keep the row with the most non-null
    # validated variables.
    val_cols = list(VALIDATED_VARS.keys())
    def _score(row):
        return sum(pd.notna(row[c]) for c in val_cols if c in row)
    raw["_keep_score"] = raw.apply(_score, axis=1)
    raw = raw.sort_values(["date", "_keep_score"], ascending=[True, False])
    daily = raw.drop_duplicates("date", keep="first").drop(columns=["_keep_score"]).reset_index(drop=True)

    # Convert thousand-pound to millions for validated balance-sheet variables
    for col in ["banking_total_assets", "banking_reserve_notes_coin", "issue_notes_total", "issue_bullion_total"]:
        if col in daily.columns:
            daily[col + "_m"] = pd.to_numeric(daily[col], errors="coerce") / 1000.0
    if "reserve_proportion" in daily.columns:
        daily["reserve_proportion"] = pd.to_numeric(daily["reserve_proportion"], errors="coerce")

    # Authoritative Bank Rate from Millennium D1, overriding model-read
    bank_rate = read_bank_rate(BANKRATE).set_index("date").sort_index()
    daily["bank_rate_model"] = pd.to_numeric(daily.get("minimum_discount_rate"), errors="coerce")
    daily["bank_rate_d1"] = daily["date"].map(lambda d: float(bank_rate.loc[d, "bank_rate"]) if d in bank_rate.index else np.nan)
    daily["bank_rate"] = daily["bank_rate_d1"]  # authoritative
    daily["bank_rate_model_mismatch"] = (
        daily["bank_rate_model"].notna() & daily["bank_rate_d1"].notna()
        & (daily["bank_rate_model"] != daily["bank_rate_d1"])
    )

    # Per-row provenance / confidence
    daily["data_source"] = "vision-LLM (claude-opus-4-7) + Millennium D1 for Bank Rate"
    daily["validation_basis"] = "Wednesday weekly anchor against boe_balance_sheet.parquet at 1% tolerance"

    # Final columns: only validated + provenance
    keep = ["date", "day_of_week", "page_image",
            "bank_rate", "bank_rate_d1", "bank_rate_model", "bank_rate_model_mismatch",
            "banking_total_assets_m", "banking_reserve_notes_coin_m",
            "issue_notes_total_m", "issue_bullion_total_m",
            "reserve_proportion",
            "confidence", "notes",
            "data_source", "validation_basis"]
    keep = [c for c in keep if c in daily.columns]
    cleaned = daily[keep].copy()
    cleaned["date"] = cleaned["date"].dt.strftime("%Y-%m-%d")
    return cleaned


def lidderdale_summary(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Per-day Lidderdale-fortnight summary table + rescue-vs-year-end separation."""
    cleaned = cleaned.copy()
    cleaned["date"] = pd.to_datetime(cleaned["date"])
    ld = cleaned[(cleaned["date"] >= "1890-11-08") & (cleaned["date"] <= "1890-11-22")].sort_values("date")
    cols = ["date", "day_of_week", "bank_rate", "banking_reserve_notes_coin_m",
            "banking_total_assets_m", "reserve_proportion"]
    return ld[cols]


def rescue_vs_yearend(parquet_path: Path) -> pd.DataFrame:
    """Three anchor weeks for context: pre-rescue, rescue, year-end."""
    bs = read_balance_sheet(parquet_path)
    rows = []
    for d, label in [
        ("1890-11-05", "Pre-rescue (1890-11-05)"),
        ("1890-11-12", "Lidderdale week start (1890-11-12)"),
        ("1890-11-19", "Post-rescue first week (1890-11-19)"),
        ("1890-12-03", "Bank Rate cut week (1890-12-03)"),
        ("1890-12-31", "Year-end accounting (1890-12-31)"),
    ]:
        ts = pd.Timestamp(d)
        if ts in bs.index:
            r = bs.loc[ts]
            rows.append({
                "anchor_date": d,
                "label": label,
                "reserve_m": round(float(r["banking_reserve_notes_coin"]), 3),
                "crisis_lending_m": round(float(r["crisis_lending"]), 3),
                "discounts_m": round(float(r["banking_discounts_total"]), 3),
                "advances_m": round(float(r["banking_advances_total"]), 3),
                "bank_rate_pct": float(r.get("bank_rate_weekly", np.nan)),
                "reserve_proportion": round(float(r.get("reserve_proportion", np.nan)), 3),
            })
    return pd.DataFrame(rows)


def plot_cleaned_lidderdale(cleaned: pd.DataFrame) -> None:
    cleaned = cleaned.copy()
    cleaned["date"] = pd.to_datetime(cleaned["date"])
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)

    ax = axes[0]
    ax.plot(cleaned["date"], cleaned["banking_total_assets_m"], marker="o", markersize=3,
             color="#2c3e50", linewidth=1.2, label="Banking Dept Total Assets")
    ax.axvline(pd.Timestamp("1890-11-15"), color="#c0392b", linestyle="--", linewidth=0.8)
    ax.axvline(pd.Timestamp("1890-11-19"), color="#16a085", linestyle=":", linewidth=0.8)
    ax.text(pd.Timestamp("1890-11-15"), ax.get_ylim()[1] * 0.92, "Lidderdale\nGuarantee Fund\n(Sat Nov 15)",
             color="#c0392b", fontsize=8, ha="left", va="top")
    ax.text(pd.Timestamp("1890-11-19"), ax.get_ylim()[1] * 0.92, "Post-rescue\nWed weekly anchor\n(Wed Nov 19)",
             color="#16a085", fontsize=8, ha="left", va="top")
    ax.set_ylabel("£ million")
    ax.set_title("Daily Banking Dept Total Assets — 100% Wednesday-anchor validated\n(vision-LLM transcription of BoE Archive C1/38, Oct-Dec 1890)")
    ax.grid(linestyle=":", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", fontsize=8, frameon=False)

    ax = axes[1]
    ax.plot(cleaned["date"], cleaned["banking_reserve_notes_coin_m"], marker="o", markersize=3,
             color="#16a085", linewidth=1.2, label="Banking Dept Reserve (notes + coin)")
    ax.axvline(pd.Timestamp("1890-11-15"), color="#c0392b", linestyle="--", linewidth=0.8)
    ax.axvline(pd.Timestamp("1890-11-19"), color="#16a085", linestyle=":", linewidth=0.8)
    ax.set_ylabel("£ million")
    ax.set_title("Daily Banking Dept Reserve — 92% Wednesday-anchor validated")
    ax.grid(linestyle=":", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", fontsize=8, frameon=False)

    ax = axes[2]
    ax.step(cleaned["date"], cleaned["bank_rate"], where="post",
             color="#c0392b", linewidth=1.5, label="Bank Rate (authoritative: Millennium D1)")
    ax.set_ylabel("%")
    ax.set_xlabel("Date")
    ax.set_title("Bank Rate from Millennium D1 (vision-LLM read disagreed at 4 rate-change-week boundaries)")
    ax.grid(linestyle=":", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=2))  # Wednesdays
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    fig.tight_layout()
    LIDD_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(LIDD_FIG, dpi=170, bbox_inches="tight")
    plt.close(fig)


def write_validation_summary(cleaned: pd.DataFrame, anchors: pd.DataFrame) -> None:
    lines = ["# Daily 1890 validation summary (cleaned dataset)\n"]
    lines.append("**Source.** Vision-LLM batch transcription via `claude-opus-4-7` (Anthropic API) of 14 weekly-spread pages from BoE Archive item C1/38 (1890 Daily Accounts For Books), October-December 1890. Raw transcription in `outputs/ocr/vision_transcription_oct_dec_1890.csv`. This cleaned dataset is the validated subset suitable for citation.\n")

    lines.append("## Variables included (passed Wednesday-anchor validation)\n")
    lines.append("Validated against the `boe_balance_sheet.parquet` Wednesday weekly anchors at 1% absolute-percentage tolerance.\n")
    lines.append("| Variable | Wed-anchor pass | Daily rows | Notes |")
    lines.append("|---|---:|---:|---|")
    lines.append("| `banking_total_assets_m` | **14/14 (100%)** | 79 | Banking Dept Total Assets in £m |")
    lines.append("| `issue_notes_total_m` | **14/14 (100%)** | 79 | Issue Dept Total Notes Issued in £m |")
    lines.append("| `issue_bullion_total_m` | **14/14 (100%)** | 79 | Issue Dept Total Bullion in £m |")
    lines.append("| `banking_reserve_notes_coin_m` | **12/13 (92%)** | 73 | Banking Dept Reserve (notes + coin) in £m |")
    lines.append("| `reserve_proportion` | 11/14 reasonable | 79 | Reserve / Deposits ratio; one anomalous reading on Wed 1890-11-19 (model returned 26.0 vs parquet 35.8) |")
    lines.append("| `bank_rate` | **n/a** (sourced from Millennium D1) | 85 | Authoritative daily Bank Rate. Vision-LLM read disagreed at 4 dates around rate-change weeks (Nov 7-10, Dec 3); those are documented separately in `bank_rate_model_mismatch` |")
    lines.append("")

    lines.append("## Variables EXCLUDED (not usable as evidence)\n")
    for var, reason in EXCLUDED_VARS.items():
        lines.append(f"- `{var}` — {reason}")
    lines.append("")

    lines.append("## What this dataset can and cannot support\n")
    lines.append("**Can support.**")
    lines.append("- The daily Banking Dept Total Assets trajectory across October-December 1890 at 100% Wednesday-anchor validation.")
    lines.append("- The daily Banking Dept Reserve trajectory at 92% Wednesday-anchor validation (one Wednesday mismatch documented).")
    lines.append("- The Issue Department side (Notes Total, Bullion Total) at 100% Wednesday-anchor validation.")
    lines.append("- Comparison of the **Lidderdale week balance-sheet trajectory** against the **year-end accounting close** — both anchors are validated.")
    lines.append("")
    lines.append("**Cannot support.**")
    lines.append("- Daily Discounts or Advances (42% fill rate; not anchor-tested). The intra-week split between discounts and advances cannot be reported from this dataset.")
    lines.append("- Daily Public Deposits or Other Deposits (24% and 64% fill rate; not anchor-tested). The \"deposit-driven\" hypothesis about the 1890 rescue is **not** supported by validated daily data; it remains a hypothesis attributed to the secondary literature.")
    lines.append("- Daily Government Securities or Other Securities (both 0% Wed-anchor pass; schema-mismatch suspected). Investigating this mismatch is logged as Tier-1 follow-up work.")
    lines.append("")

    lines.append("## Anchor table — three key reference points\n")
    lines.append("From the existing weekly parquet, the structured Wednesday observations the vision data was validated against:\n")
    lines.append("| Anchor date | Reserve £m | Crisis lending £m | Discounts £m | Advances £m | Bank Rate | Reserve Prop |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, r in anchors.iterrows():
        lines.append(f"| {r['anchor_date']} ({r['label']}) | {r['reserve_m']} | {r['crisis_lending_m']} | {r['discounts_m']} | {r['advances_m']} | {r['bank_rate_pct']}% | {r['reserve_proportion']} |")
    lines.append("")
    lines.append("**Key separation.** The headline 1890 peak lending value of £38.86m reported in `crisis_metrics.csv` falls on the year-end accounting Wednesday 1890-12-31, not the Lidderdale rescue. The post-rescue weekly anchor on Wednesday 1890-11-19 shows lending at £33.56m. These are reported separately in the paper §4.3.\n")

    lines.append("## Manual / visual verification status\n")
    lines.append("A targeted visual re-verification pass was attempted on the Lidderdale fortnight (pages C1-38_068 through C1-38_071) using the Claude Code interactive `Read` tool. At the rendering resolution available to that tool, **no additional cell values could be confidently read beyond what the vision-LLM API already extracted**. This is a known limit of the interactive tool relative to native-resolution API access. No `manual_corrections_1890.csv` row was added because no corrections could be made with sufficient confidence. The empty manual-corrections file is retained at `outputs/ocr/manual_corrections_1890.csv` as a future-work scaffold.\n")

    VAL_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    VAL_SUMMARY.write_text("\n".join(lines))


def main() -> None:
    print("Building cleaned daily 1890 dataset...")
    cleaned = build_cleaned_daily()
    CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(CLEAN_CSV, index=False)
    print(f"  Wrote {CLEAN_CSV} ({len(cleaned)} daily rows)")

    print("\nComputing rescue-vs-year-end anchor table...")
    anchors = rescue_vs_yearend(PARQUET)
    LIDD_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    anchors.to_csv(LIDD_SUMMARY, index=False)
    print(f"  Wrote {LIDD_SUMMARY}")
    print(anchors.to_string(index=False))

    print("\nBuilding Lidderdale-fortnight figure...")
    plot_cleaned_lidderdale(cleaned)
    print(f"  Wrote {LIDD_FIG}")

    print("\nWriting validation summary...")
    write_validation_summary(cleaned, anchors)
    print(f"  Wrote {VAL_SUMMARY}")

    # Initialize empty manual-corrections file as a scaffold
    mc_path = ROOT / "outputs" / "ocr" / "manual_corrections_1890.csv"
    if not mc_path.exists():
        pd.DataFrame(columns=[
            "date","page_image","variable","model_value","corrected_value",
            "correction_status","reviewer_note","source_image_path",
        ]).to_csv(mc_path, index=False)
        print(f"  Wrote empty {mc_path} (scaffold for future manual review)")


if __name__ == "__main__":
    main()
