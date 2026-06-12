"""Recompute borrower-network metrics using AI-resolved canonical names and types.

Compares against the original network_summary.csv (rule-based) and writes:
- outputs/tables/network_summary_ai.csv          – HHI, top-k, Gini after entity resolution
- outputs/tables/borrower_type_shares_ai.csv     – share by AI counterparty type per crisis
- outputs/tables/classification_comparison.csv   – old-vs-new side-by-side
- outputs/figures/concentration_comparison_ai.png
- outputs/figures/borrower_type_shares_ai.png
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
TBL = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures"


def _gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    x = x[x > 0]
    if x.size == 0:
        return float("nan")
    n = x.size
    cum = np.cumsum(x)
    return float((n + 1 - 2 * cum.sum() / x.sum()) / n)


def _summary(df: pd.DataFrame, key_col: str) -> dict:
    totals = df.groupby(key_col)["total_amount"].sum().sort_values(ascending=False)
    totals = totals[totals > 0]
    total = totals.sum()
    n = len(totals)
    shares = totals / total if total else totals * np.nan
    return {
        "n_entities": n,
        "total_value": float(total),
        "top1_share": float(shares.iloc[0]) if n else np.nan,
        "top5_share": float(shares.head(5).sum()) if n else np.nan,
        "top10_share": float(shares.head(10).sum()) if n else np.nan,
        "hhi": float((shares ** 2).sum() * 10_000) if n else np.nan,
        "gini": _gini(totals.values),
    }


def main() -> None:
    TBL.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    tx = read_transactions(PROC / "lolr_transactions.parquet")
    er = pd.read_csv(TBL / "counterparty_entity_resolution.csv")
    cls = pd.read_csv(TBL / "counterparty_ai_classification.csv")

    tx2 = tx.merge(er[["counterparty_clean", "canonical_counterparty"]],
                    on="counterparty_clean", how="left")
    tx2 = tx2.merge(cls[["canonical_counterparty", "ai_counterparty_type"]],
                    on="canonical_counterparty", how="left")

    # --- Network summary (post-AI) ---
    rows_ai = []
    rows_old = []
    for key, c in CRISES.items():
        if key not in {"1857", "1866", "1914"}:
            continue
        sub = tx2[(tx2["crisis"] == key) & (tx2["date"] >= c.acute_start) & (tx2["date"] <= c.acute_end)]
        rows_ai.append({"crisis": key, "name": c.name, **_summary(sub, "canonical_counterparty")})
        rows_old.append({"crisis": key, "name": c.name, **_summary(sub, "counterparty_clean")})
    df_ai = pd.DataFrame(rows_ai)
    df_old = pd.DataFrame(rows_old)
    df_ai.to_csv(TBL / "network_summary_ai.csv", index=False)
    print(f"Wrote {TBL / 'network_summary_ai.csv'}")
    print(df_ai.to_string(index=False))

    # --- Borrower-type shares (AI types) ---
    type_rows = []
    for key, c in CRISES.items():
        if key not in {"1857", "1866", "1914"}:
            continue
        sub = tx2[(tx2["crisis"] == key) & (tx2["date"] >= c.acute_start) & (tx2["date"] <= c.acute_end)]
        total = sub["total_amount"].sum()
        if total <= 0:
            continue
        by_type = sub.groupby("ai_counterparty_type")["total_amount"].sum() / total
        row = {"crisis": key, "name": c.name, "total_value_acute": float(total)}
        for t in ("commercial_bank", "merchant_bank", "bill_broker", "discount_house",
                  "merchant_trading_firm", "industrial_or_corporate",
                  "individual_or_partnership", "government_or_public_body",
                  "other", "unknown"):
            row[f"share_{t}"] = round(float(by_type.get(t, 0.0)), 4)
        type_rows.append(row)
    df_types = pd.DataFrame(type_rows)
    df_types.to_csv(TBL / "borrower_type_shares_ai.csv", index=False)
    print(f"\nWrote {TBL / 'borrower_type_shares_ai.csv'}")
    print(df_types.to_string(index=False))

    # --- Comparison: old vs new ---
    comp_rows = []
    for ai_row, old_row in zip(rows_ai, rows_old):
        comp_rows.append({
            "crisis": ai_row["crisis"],
            "n_entities_old": old_row["n_entities"],
            "n_entities_ai": ai_row["n_entities"],
            "n_merged": old_row["n_entities"] - ai_row["n_entities"],
            "top5_share_old": round(old_row["top5_share"], 3),
            "top5_share_ai": round(ai_row["top5_share"], 3),
            "hhi_old": round(old_row["hhi"], 1),
            "hhi_ai": round(ai_row["hhi"], 1),
        })
    # Type-share comparison: original rule-based merchant/other vs AI categories
    old_type_rows = []
    for key, c in CRISES.items():
        if key not in {"1857", "1866", "1914"}:
            continue
        sub = tx2[(tx2["crisis"] == key) & (tx2["date"] >= c.acute_start) & (tx2["date"] <= c.acute_end)]
        total = sub["total_amount"].sum()
        if total <= 0:
            continue
        by_type = sub.groupby("counterparty_type")["total_amount"].sum() / total
        old_type_rows.append({
            "crisis": key,
            "old_share_merchant_or_other": round(float(by_type.get("merchant", 0) + by_type.get("other", 0)), 4),
            "old_share_commercial_bank": round(float(by_type.get("commercial_bank", 0)), 4),
            "old_share_discount_house": round(float(by_type.get("discount_house", 0)), 4),
            "old_share_merchant_bank": round(float(by_type.get("merchant_bank", 0)), 4),
        })
    df_old_types = pd.DataFrame(old_type_rows)
    df_compare = pd.DataFrame(comp_rows).merge(df_old_types, on="crisis").merge(
        df_types[["crisis", "share_commercial_bank", "share_merchant_bank",
                  "share_bill_broker", "share_discount_house",
                  "share_merchant_trading_firm", "share_industrial_or_corporate",
                  "share_individual_or_partnership", "share_unknown"]],
        on="crisis",
    )
    df_compare.to_csv(TBL / "classification_comparison.csv", index=False)
    print(f"\nWrote {TBL / 'classification_comparison.csv'}")
    print(df_compare.to_string(index=False))

    # --- Figures ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(df_ai["crisis"].astype(str), df_ai["top5_share"], color="#9b59b6")
    axes[0].axhline(0.30, color="#34495e", linestyle="--", linewidth=0.7,
                     label="0.30 broad-market threshold")
    axes[0].set_ylim(0, 0.6)
    axes[0].set_title("Top-5 share of acute-window lending (AI canonical entities)")
    axes[0].set_ylabel("share")
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].bar(df_ai["crisis"].astype(str), df_ai["hhi"], color="#e67e22")
    axes[1].set_title("HHI of acute-window lending (AI canonical entities)")
    axes[1].set_ylabel("HHI (×10,000)")
    for ax in axes:
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out1 = FIG / "concentration_comparison_ai.png"
    fig.savefig(out1, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out1}")

    # Stacked-bar of borrower-type shares per crisis
    type_cols_order = ["commercial_bank", "merchant_bank", "bill_broker", "discount_house",
                       "merchant_trading_firm", "industrial_or_corporate",
                       "individual_or_partnership", "unknown", "other", "government_or_public_body"]
    colors = ["#2980b9", "#16a085", "#27ae60", "#f39c12", "#c0392b",
              "#8e44ad", "#7f8c8d", "#bdc3c7", "#95a5a6", "#34495e"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bottoms = np.zeros(len(df_types))
    x = np.arange(len(df_types))
    for col, color in zip(type_cols_order, colors):
        col_name = f"share_{col}"
        if col_name in df_types.columns:
            vals = df_types[col_name].values
            ax.bar(x, vals, bottom=bottoms, color=color, label=col.replace("_", " "))
            bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels(df_types["crisis"].astype(str))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Share of acute-window lending")
    ax.set_title("Borrower-type composition of acute-window lending (AI classification)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out2 = FIG / "borrower_type_shares_ai.png"
    fig.savefig(out2, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out2}")


if __name__ == "__main__":
    main()
