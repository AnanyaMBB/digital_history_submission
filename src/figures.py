"""Crisis-window figures for the paper.

Produces:
- outputs/figures/crisis_small_multiples.png  — 4×3 grid:
    rows: 4 crises; cols: reserve, crisis_lending, bank_rate
- outputs/figures/lolr_score_chart.png — bar chart of LOLR-likeness scores
- outputs/figures/long_run_balance_sheet.png — long-run reserves + lending 1844-1919
- outputs/figures/concentration_comparison.png — top-5 share / HHI across crises
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from io_utils import read_balance_sheet, read_bank_rate  # noqa: E402
from crisis_windows import CRISES  # noqa: E402

PROC = ROOT / "data" / "processed"
TBL = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures"

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def small_multiples() -> None:
    bs = read_balance_sheet(PROC / "boe_balance_sheet.parquet")
    br = read_bank_rate(PROC / "bank_rate_daily.parquet").set_index("date")

    fig, axes = plt.subplots(4, 3, figsize=(11, 11), sharey=False)
    for i, (key, c) in enumerate(CRISES.items()):
        win = slice(c.pre_start, c.post_end)
        sub_bs = bs.loc[win]
        sub_br = br.loc[win]
        ax_r, ax_l, ax_b = axes[i]
        # Reserve
        ax_r.plot(sub_bs.index, sub_bs["banking_reserve_notes_coin"], color="#2c3e50")
        ax_r.axvspan(c.acute_start, c.acute_end, color="#e74c3c", alpha=0.10)
        ax_r.axvline(c.acute_peak, color="#c0392b", linewidth=0.8, linestyle="--")
        ax_r.set_title(f"{c.name} — Reserve (£m)")
        # Crisis lending
        ax_l.plot(sub_bs.index, sub_bs["crisis_lending"], color="#34495e")
        ax_l.axvspan(c.acute_start, c.acute_end, color="#e74c3c", alpha=0.10)
        ax_l.axvline(c.acute_peak, color="#c0392b", linewidth=0.8, linestyle="--")
        ax_l.set_title("Discounts + Advances (£m)")
        # Bank Rate (daily)
        ax_b.plot(sub_br.index, sub_br["bank_rate"], color="#16a085")
        ax_b.axvspan(c.acute_start, c.acute_end, color="#e74c3c", alpha=0.10)
        ax_b.axvline(c.acute_peak, color="#c0392b", linewidth=0.8, linestyle="--")
        ax_b.set_title("Bank Rate (%)")
        for ax in (ax_r, ax_l, ax_b):
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.tick_params(axis="x", labelrotation=30)
            ax.grid(linestyle=":", alpha=0.5)

    fig.suptitle("Bank of England crisis windows: reserve, lending, and Bank Rate",
                 fontsize=12, y=1.00)
    plt.tight_layout()
    out = FIG / "crisis_small_multiples.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")


def lolr_score_chart() -> None:
    df = pd.read_csv(TBL / "lolr_score.csv", index_col=0)
    if "tier_a_score" not in df.index:
        return
    def _toint(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return None
    tier_a = df.loc["tier_a_score"].apply(_toint)
    tier_b = df.loc["tier_b_score"].apply(_toint)
    crises = list(df.columns)
    # Replace None tier_a with 0 (shouldn't happen) so bar plotting works
    a_for_plot = [tier_a[c] if tier_a[c] is not None else 0 for c in crises]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = range(len(crises))
    width = 0.36
    a_vals = a_for_plot
    b_vals = [tier_b[c] if tier_b[c] is not None else 0 for c in crises]
    b_hatch = [tier_b[c] is None for c in crises]

    bars_a = ax.bar([i - width / 2 for i in x], a_vals, width,
                    color="#2980b9", label="Tier A: balance-sheet (max +4)")
    bars_b = []
    for i, (val, na) in enumerate(zip(b_vals, b_hatch)):
        bar = ax.bar(i + width / 2, val if not na else 0, width,
                     color="#16a085" if not na else "#bbb",
                     hatch="//" if na else None, edgecolor="#444",
                     label="Tier B: transaction-level (max +2)" if i == 0 else None)
        bars_b.append(bar)

    ax.set_xticks(list(x))
    ax.set_xticklabels(crises)
    ax.set_ylim(-2, 6.0)
    ax.axhline(0, color="#999", linewidth=0.7)
    ax.set_ylabel("LOLR-likeness score (component sum)")
    ax.set_title("LOLR-likeness score by tier",
                 pad=14)
    # Subtitle as text below title
    fig.text(0.5, 0.905,
             "Tier A applies to all four crises; Tier B requires transaction-level ledger data (n/a for 1890).",
             ha="center", fontsize=8, color="#555")

    # Value labels above each bar
    for rect, v in zip(bars_a, a_vals):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.12, str(v),
                ha="center", va="bottom", fontsize=9, color="#2c3e50")
    for bc, v, na in zip(bars_b, b_vals, b_hatch):
        rect = bc[0]
        label = "n/a" if na else str(v)
        ax.text(rect.get_x() + rect.get_width() / 2, (v if not na else 0) + 0.12,
                label, ha="center", va="bottom", fontsize=9, color="#2c3e50")

    ax.legend(loc="lower left", fontsize=8, frameon=False)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    out = FIG / "lolr_score_chart.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")


def long_run_balance_sheet() -> None:
    bs = read_balance_sheet(PROC / "boe_balance_sheet.parquet")
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(bs.index, bs["banking_reserve_notes_coin"], color="#2c3e50", linewidth=0.8)
    axes[0].set_title("Banking Department reserve, 1844–1919 (£m)")
    axes[1].plot(bs.index, bs["crisis_lending"], color="#c0392b", linewidth=0.8)
    axes[1].set_title("Crisis lending proxy: discounts + advances (£m)")
    for ax in axes:
        ax.grid(linestyle=":", alpha=0.5)
        for key, c in CRISES.items():
            ax.axvspan(c.acute_start, c.acute_end, color="#f39c12", alpha=0.15)
            ax.annotate(key, (c.acute_peak, ax.get_ylim()[1] * 0.92), ha="center",
                        fontsize=8, color="#7f8c8d")
    plt.tight_layout()
    out = FIG / "long_run_balance_sheet.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")


def concentration_comparison() -> None:
    netsum = pd.read_csv(TBL / "network_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].bar(netsum["crisis"].astype(str), netsum["top5_share"], color="#9b59b6")
    axes[0].set_ylim(0, 0.6)
    axes[0].set_title("Top-5 borrower share of acute-window lending")
    axes[0].set_ylabel("share")
    axes[0].axhline(0.30, color="#34495e", linestyle="--", linewidth=0.7,
                    label="0.30 threshold (broad market)")
    axes[0].legend(fontsize=8)
    axes[1].bar(netsum["crisis"].astype(str), netsum["hhi"], color="#e67e22")
    axes[1].set_title("HHI of acute-window lending")
    axes[1].set_ylabel("HHI (×10,000)")
    for ax in axes:
        ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    out = FIG / "concentration_comparison.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    print("Writing figures:")
    small_multiples()
    lolr_score_chart()
    long_run_balance_sheet()
    concentration_comparison()


if __name__ == "__main__":
    main()
