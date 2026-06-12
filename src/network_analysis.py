"""Bipartite borrower-BoE network analysis for 1857, 1866, 1914 acute windows.

For each crisis we:
  - aggregate transactions to counterparty totals
  - build a weighted star network (BoE → borrowers)
  - report HHI, top-5 share, Gini, n_counterparties
  - emit a horizontal lollipop chart of top-20 borrowers as a quick visual

Output: outputs/figures/network_top20_{crisis}.png, outputs/tables/network_summary.csv
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
from io_utils import read_transactions  # noqa: E402
from crisis_windows import CRISES  # noqa: E402

PROC = ROOT / "data" / "processed"
FIG_DIR = ROOT / "outputs" / "figures"
TBL_DIR = ROOT / "outputs" / "tables"


def _gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    x = x[x > 0]
    if x.size == 0:
        return float("nan")
    n = x.size
    cum = np.cumsum(x)
    return float((n + 1 - 2 * cum.sum() / x.sum()) / n)


def _summary_for(df: pd.DataFrame) -> dict:
    totals = df.groupby("counterparty_clean")["total_amount"].sum().sort_values(ascending=False)
    totals = totals[totals > 0]
    total = totals.sum()
    n = len(totals)
    shares = totals / total if total else totals * np.nan
    return {
        "n_counterparties": n,
        "total_value_m": float(total),
        "top1_share": float(shares.iloc[0]) if n else np.nan,
        "top5_share": float(shares.head(5).sum()) if n else np.nan,
        "top10_share": float(shares.head(10).sum()) if n else np.nan,
        "hhi": float((shares ** 2).sum() * 10_000) if n else np.nan,
        "gini": _gini(totals.values),
    }


def _plot_top20(df: pd.DataFrame, crisis: str, out_path: Path) -> None:
    totals = df.groupby("counterparty_clean")["total_amount"].sum().sort_values(ascending=False)
    top = totals.head(20).iloc[::-1]
    if top.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 7))
    y = np.arange(len(top))
    ax.hlines(y, 0, top.values, color="#444", linewidth=1.2)
    ax.scatter(top.values, y, s=40, color="#c0392b", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(top.index, fontsize=8)
    ax.set_xlabel("Total cash flow during acute window (£)")
    ax.set_title(f"Top 20 BoE counterparties — {crisis} acute window")
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TBL_DIR.mkdir(parents=True, exist_ok=True)
    tx = read_transactions(PROC / "lolr_transactions.parquet")

    rows = []
    for key in ("1857", "1866", "1914"):
        c = CRISES[key]
        sub = tx[(tx["crisis"] == key) & (tx["date"] >= c.acute_start) & (tx["date"] <= c.acute_end)]
        summary = _summary_for(sub)
        rows.append({"crisis": key, "name": c.name, **summary})
        _plot_top20(sub, c.name, FIG_DIR / f"network_top20_{key}.png")

    df = pd.DataFrame(rows)
    df.to_csv(TBL_DIR / "network_summary.csv", index=False)
    print(f"Wrote {TBL_DIR / 'network_summary.csv'}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
