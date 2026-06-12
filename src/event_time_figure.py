"""Event-time normalized comparison of all four crises on one axis.

Index each crisis series to its pre-crisis baseline mean = 100, then plot
weeks from the acute trigger date. Lets the reader compare crisis shape
without being misled by the secular growth of the Bank's balance sheet
between 1857 and 1914.

Output: outputs/figures/event_time_comparison.png
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from io_utils import read_balance_sheet, read_bank_rate  # noqa: E402
from crisis_windows import CRISES  # noqa: E402

PROC = ROOT / "data" / "processed"
FIG = ROOT / "outputs" / "figures"

WEEKS_BEFORE = 26
WEEKS_AFTER = 52
SERIES = [
    ("crisis_lending", "Discounts + advances (indexed to baseline=100)"),
    ("banking_reserve_notes_coin", "Reserve (indexed)"),
    ("bank_rate_weekly", "Bank Rate (pp deviation from baseline)"),
]


def _slice_event_time(bs: pd.DataFrame, col: str, baseline: float,
                       trigger: pd.Timestamp, mode: str) -> pd.DataFrame:
    start = trigger - pd.Timedelta(weeks=WEEKS_BEFORE)
    end = trigger + pd.Timedelta(weeks=WEEKS_AFTER)
    sub = bs.loc[start:end, [col]].copy()
    sub["weeks_from_trigger"] = ((sub.index - trigger) / pd.Timedelta(weeks=1)).round().astype(int)
    if mode == "index":
        sub["value"] = sub[col] / baseline * 100.0
    elif mode == "pp_dev":
        sub["value"] = sub[col] - baseline
    else:
        sub["value"] = sub[col]
    return sub[["weeks_from_trigger", "value"]].reset_index(drop=True)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    bs = read_balance_sheet(PROC / "boe_balance_sheet.parquet")
    br = read_bank_rate(PROC / "bank_rate_daily.parquet").set_index("date").sort_index()
    # Make sure the weekly bank_rate column is populated even if NaN — use Millennium daily for fill
    bs["bank_rate_weekly"] = bs["bank_rate_weekly"].fillna(
        br["bank_rate"].reindex(bs.index, method="nearest")
    )

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True)
    crisis_colors = {"1857": "#2980b9", "1866": "#16a085",
                      "1890": "#f39c12", "1914": "#c0392b"}

    for j, (col, title) in enumerate(SERIES):
        ax = axes[j]
        ax.set_title(title, fontsize=10)
        for key, c in CRISES.items():
            pre = bs.loc[c.pre, col]
            baseline = pre.mean()
            mode = "index" if "index" in title.lower() else "pp_dev"
            df = _slice_event_time(bs, col, baseline, c.acute_peak, mode)
            if df.empty:
                continue
            ax.plot(df["weeks_from_trigger"], df["value"],
                    color=crisis_colors[key], linewidth=1.4,
                    label=f"{c.name}")
        ax.axvline(0, color="#444", linewidth=0.6)
        if "index" in title.lower():
            ax.axhline(100, color="#999", linewidth=0.5, linestyle=":")
        ax.set_xlabel("Weeks from canonical trigger")
        ax.grid(linestyle=":", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("Event-time comparison of four BoE crises (acute trigger = week 0)",
                  fontsize=12, y=1.02)
    plt.tight_layout()
    out = FIG / "event_time_comparison.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
