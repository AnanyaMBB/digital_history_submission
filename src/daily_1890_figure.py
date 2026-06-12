"""Daily-frequency 1890 figure produced from the validated vision-LLM
transcription.

Plots the daily balance-sheet trajectory through the Baring acute window
using the cells that pass Wednesday-anchor validation. The figure adds
intra-week resolution that the weekly Wednesday parquet cannot show.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

LIDDERDALE = pd.Timestamp("1890-11-15")  # Saturday Lidderdale Guarantee Fund finalised
RATE_RISE = pd.Timestamp("1890-11-07")    # Bank Rate 5% -> 6%
RATE_CUT = pd.Timestamp("1890-12-04")     # Bank Rate 6% -> 5%


def main() -> None:
    t = pd.read_csv(OUT / "ocr" / "daily_1890_validated.csv")
    t["date"] = pd.to_datetime(t["date"], errors="coerce")
    t = t.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ax_ta, ax_res, ax_br = axes

    # Panel 1: Total Assets daily
    sub = t.dropna(subset=["banking_total_assets_m"])
    ax_ta.plot(sub["date"], sub["banking_total_assets_m"], marker="o",
                color="#2c3e50", markersize=4, linewidth=1.2,
                label="Banking Dept Total Assets (daily, validated)")
    ax_ta.set_ylabel("Total Assets, £m")
    ax_ta.set_title("Daily 1890 balance sheet from vision-LLM transcription "
                     "(validated 100% against Wednesday weekly anchors)")
    ax_ta.legend(loc="lower right", fontsize=8, frameon=False)

    # Panel 2: Reserve daily — only the cells the model could read
    sub_res = t.dropna(subset=["banking_reserve_notes_coin_m"])
    ax_res.plot(sub_res["date"], sub_res["banking_reserve_notes_coin_m"],
                marker="s", color="#16a085", markersize=4, linewidth=1.2,
                label="Banking Dept Reserve (cells read confidently)")
    ax_res.set_ylabel("Reserve, £m")
    ax_res.legend(loc="lower right", fontsize=8, frameon=False)

    # Panel 3: Bank Rate daily from Millennium D1 (the model's transcription
    # has rate-change-week errors documented in the validation summary;
    # plot from Millennium for clarity)
    from io_utils import read_bank_rate
    br = read_bank_rate(ROOT / "data" / "processed" / "bank_rate_daily.parquet")
    br_win = br[(br["date"] >= "1890-10-01") & (br["date"] <= "1890-12-31")]
    ax_br.step(br_win["date"], br_win["bank_rate"], where="post",
                color="#c0392b", linewidth=1.3)
    ax_br.set_ylim(4, 7)
    ax_br.set_ylabel("Bank Rate, %")
    ax_br.set_xlabel("Date")

    # Crisis markers
    for ax in axes:
        ax.axvline(LIDDERDALE, color="#c0392b", linestyle="--", linewidth=0.8,
                    alpha=0.7)
        ax.axvline(RATE_RISE, color="#7f8c8d", linestyle=":", linewidth=0.7,
                    alpha=0.6)
        ax.axvline(RATE_CUT, color="#7f8c8d", linestyle=":", linewidth=0.7,
                    alpha=0.6)
        ax.grid(linestyle=":", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.WE))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    # Annotations on top panel
    ax_ta.annotate("Lidderdale\nGuarantee\nFund", xy=(LIDDERDALE, 56),
                    xytext=(LIDDERDALE + pd.Timedelta(days=2), 50),
                    fontsize=8, color="#c0392b",
                    arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.7))
    ax_ta.annotate("Bank Rate\n5→6%", xy=(RATE_RISE, 48),
                    xytext=(RATE_RISE - pd.Timedelta(days=12), 51),
                    fontsize=7.5, color="#555",
                    arrowprops=dict(arrowstyle="->", color="#555", lw=0.6))
    ax_ta.annotate("Bank Rate\n6→5%", xy=(RATE_CUT, 53),
                    xytext=(RATE_CUT + pd.Timedelta(days=2), 49),
                    fontsize=7.5, color="#555",
                    arrowprops=dict(arrowstyle="->", color="#555", lw=0.6))

    plt.tight_layout()
    out = OUT / "figures" / "daily_1890_lidderdale.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    main()
