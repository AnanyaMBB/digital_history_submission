"""
v5_figures.py — figures for paper_v5 "The Information Club".
Reads outputs/tables/v5_*.csv + v4_clock_table.csv + v3_information_channel_scores.csv
+ data/processed/imm/imm_financial_monthly.csv. Writes outputs/figures/v5_*.png.

Figures:
  1. v5_info_chain.png   — the crisis information chain (inner circle -> public), concept
  2. v5_timeline.png     — per-crisis earliest market signal -> public rupture, by cable era
  3. v5_telegraph.png    — pre/post-cable lead comparison; 1857 vs 1907 transatlantic pair
  4. v5_recurrence.png   — inner-circle actor x crisis appearance matrix (cable divider)
  5. v5_market_price.png — Yale IMM financial-sector index around post-cable ruptures
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
TBL = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

PRE, POST, TRANS = "#C44E52", "#4C72B0", "#937860"   # pre / post / transition colours
CATCOL = {"discount_house": "#C44E52", "bill_broker": "#DD8452", "merchant_bank": "#4C72B0",
          "clearing_or_joint_stock_bank": "#55A868", "foreign_or_colonial_financial": "#8172B2",
          "industrial_or_commercial": "#937860", "unknown": "#C7C7C7"}
CATLAB = {"discount_house": "discount house", "bill_broker": "bill broker",
          "merchant_bank": "merchant bank", "clearing_or_joint_stock_bank": "clearing bank",
          "foreign_or_colonial_financial": "foreign/colonial", "unknown": "unknown"}
ERA = {"pre": PRE, "post": POST, "transition": TRANS}

clk = pd.read_csv(TBL / "v4_clock_table.csv"); clk["crisis"] = clk["crisis"].astype(str)
chain = pd.read_csv(TBL / "v5_information_chain.csv"); chain["crisis"] = chain["crisis"].astype(str)
ORDER = ["1847", "1857", "1866", "1873", "1890", "1907", "1914"]
chain = chain.set_index("crisis").reindex(ORDER).reset_index()


# ---------------------------------------------------------------- Fig 1 info chain
def fig_info_chain():
    fig, ax = plt.subplots(figsize=(13, 4.8)); ax.set_xlim(0, 14); ax.set_ylim(0, 5); ax.axis("off")
    ax.text(7, 4.65, "The crisis information chain: from the City's inner circle to the public",
            ha="center", fontsize=13, fontweight="bold")
    stages = [
        ("Inner circle", "merchant banks,\ndiscount houses,\nbill brokers", "#fdecea", "#C44E52"),
        ("Bank of England", "Bank Rate rises;\nintermediaries at\nthe discount window", "#fff6e6", "#D9A300"),
        ("Specialist press", "Bankers' Magazine,\nCity money columns", "#eef3fb", "#4C72B0"),
        ("Public press", "general newspapers;\ncoverage surges", "#eeeeee", "#555555"),
        ("Public rupture", "a failure; Stock\nExchange closes;\nHansard debate", "#eaf3ec", "#55A868"),
    ]
    x = 0.4
    for t, b, fc, ec in stages:
        ax.add_patch(FancyBboxPatch((x, 1.9), 2.35, 1.8, boxstyle="round,pad=0.06", fc=fc, ec=ec, lw=2))
        ax.text(x + 1.17, 3.25, t, ha="center", va="center", fontsize=10, fontweight="bold", color=ec)
        ax.text(x + 1.17, 2.4, b, ha="center", va="center", fontsize=7.6, color="#333")
        if x < 10:
            ax.annotate("", xy=(x + 3.05, 2.8), xytext=(x + 2.45, 2.8),
                        arrowprops=dict(arrowstyle="-|>", lw=2, color="#999"))
        x += 2.75
    ax.annotate("", xy=(12.1, 1.4), xytext=(0.4, 1.4),
                arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#888"))
    ax.text(6.2, 1.1, "lead time we measure = public rupture date  -  earliest market-facing signal date",
            ha="center", fontsize=9, color="#444")
    ax.text(7, 0.5, "Closed-club thesis: the same inner circle sits at the front, and the lead persists "
            "after the 1866 cable.\nEfficient-market thesis: the telegraph collapses the lag and the "
            "advantage fades. We measure visibility, never profit.",
            ha="center", fontsize=8.4, color="#444")
    plt.savefig(FIG / "v5_info_chain.png", dpi=150, bbox_inches="tight"); plt.close()
    print("wrote v5_info_chain.png")


# ---------------------------------------------------------------- Fig 2 timeline
def fig_timeline():
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    y = {c: i for i, c in enumerate(reversed(ORDER))}
    for _, r in chain.iterrows():
        c = r["crisis"]; yy = y[c]
        pub = pd.Timestamp(r["stage5_public_rupture"])
        early = pd.to_datetime(r["earliest_market_signal"], errors="coerce")
        col = ERA[r["cable_era"]]
        d = (early - pub).days
        ax.plot([d, 0], [yy, yy], color=col, lw=3, alpha=0.45, zorder=1)
        ax.scatter([d], [yy], s=240, marker="*", color=col, edgecolor="white", lw=1, zorder=5)
        ax.scatter([0], [yy], s=150, marker="s", color="#333", edgecolor="white", lw=1.2, zorder=5)
        ax.text(d, yy + 0.2, pd.to_datetime(early).strftime("%b %Y"), ha="center", fontsize=7.6, color=col)
        ax.text(6, yy, f"{c}  ({r['cable_era']}, {r['origin']})  lead {int(r['lead_earliest_to_public_days'])}d",
                va="center", fontsize=9, color=col, fontweight="bold")
    ax.axvline(0, color="#bbb", ls="--", lw=1)
    ax.text(0, len(ORDER) - 0.35, "public rupture", ha="center", fontsize=8.5, color="#333")
    ax.set_yticks(list(y.values())); ax.set_yticklabels([])
    ax.set_xlabel("Days relative to the public rupture (day 0)", fontsize=10)
    ax.set_ylim(-0.6, len(ORDER) - 0.2); ax.set_xlim(-310, 120)
    handles = [Line2D([0], [0], marker="*", color="w", markerfacecolor=PRE, markersize=15, label="pre-cable crisis"),
               Line2D([0], [0], marker="*", color="w", markerfacecolor=TRANS, markersize=15, label="1866 (cable year)"),
               Line2D([0], [0], marker="*", color="w", markerfacecolor=POST, markersize=15, label="post-cable crisis"),
               Line2D([0], [0], marker="s", color="w", markerfacecolor="#333", markersize=10, label="public rupture")]
    ax.legend(handles=handles, loc="lower left", fontsize=8.6, framealpha=0.95)
    ax.set_title("How early was the first market-facing signal? Star = earliest Bank Rate / ledger signal,\n"
                 "coloured by telegraph era. The advantage does not vanish after the 1866 cable.",
                 fontsize=11.5, pad=10)
    ax.grid(axis="x", alpha=0.2)
    plt.savefig(FIG / "v5_timeline.png", dpi=150, bbox_inches="tight"); plt.close()
    print("wrote v5_timeline.png")


# ---------------------------------------------------------------- Fig 3 telegraph test
def fig_telegraph():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.4, 1]})
    # left: all crises lead, grouped by era
    d = chain.copy()
    d["lead"] = clk.set_index("crisis").reindex(ORDER)["lead_time_days"].values
    d = d.sort_values("cable_era")
    order2 = ["1847", "1857", "1866", "1873", "1890", "1907", "1914"]
    d = d.set_index("crisis").reindex(order2).reset_index()
    yy = np.arange(len(d))
    cols = [ERA[e] for e in d["cable_era"]]
    axL.barh(yy, d["lead"], color=cols, alpha=0.8)
    for i, r in d.iterrows():
        axL.text(r["lead"] + 4, i, f"{int(r['lead'])}d", va="center", fontsize=8.5)
    axL.set_yticks(yy); axL.set_yticklabels([f"{r.crisis} ({r.cable_era[:4]})" for r in d.itertuples()], fontsize=9)
    axL.invert_yaxis(); axL.set_xlabel("insider-to-public lead (days, sustained measure)", fontsize=9.5)
    axL.axhline(2.5, color="#999", ls=":", lw=1)
    axL.text(axL.get_xlim()[1]*0.98, 2.5, " 1866 cable", ha="right", fontsize=8, color="#666", va="bottom")
    axL.set_title("Lead time by crisis and telegraph era", fontsize=10.5); axL.grid(axis="x", alpha=0.25)

    # right: the transatlantic pair 1857 vs 1907
    pair = pd.read_csv(TBL / "v5_transatlantic_pair.csv"); pair["crisis"] = pair["crisis"].astype(str)
    pair = pair.set_index("crisis").reindex(["1857", "1907"]).reset_index()
    axR.bar(["1857\n(pre-cable)", "1907\n(post-cable)"], pair["lead_time_days"],
            color=[PRE, POST], alpha=0.85, width=0.6)
    for i, v in enumerate(pair["lead_time_days"]):
        axR.text(i, v + 1.5, f"{int(v)} d", ha="center", fontsize=11, fontweight="bold")
    axR.annotate("", xy=(1, 64), xytext=(0, 27),
                 arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.6))
    axR.text(0.5, 50, "lead GREW,\nnot shrank", ha="center", fontsize=9, color="#333")
    axR.set_ylabel("insider-to-public lead (days)", fontsize=9.5)
    axR.set_title("The transatlantic test:\nboth US-origin panics", fontsize=10.5)
    axR.set_ylim(0, 85); axR.grid(axis="y", alpha=0.25)
    fig.suptitle("The telegraph test: did the 1866 cable shrink the insider-to-public lead? For the "
                 "transatlantic crises, no.", fontsize=12, y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(FIG / "v5_telegraph.png", dpi=150, bbox_inches="tight"); plt.close()
    print("wrote v5_telegraph.png")


# ---------------------------------------------------------------- Fig 4 recurrence
def fig_recurrence():
    ic = pd.read_csv(TBL / "v3_information_channel_scores.csv")
    top = ic.sort_values("information_channel_score", ascending=False).head(14).iloc[::-1]
    led_crises = ["1847", "1857", "1866", "1914"]  # only ledger crises have actor data
    xpos = {"1847": 0, "1857": 1, "1866": 2, "1914": 3.4}  # gap before 1914 = cable divider
    fig, ax = plt.subplots(figsize=(11, 7.2))
    for i, (_, r) in enumerate(top.iterrows()):
        appears = str(r["crises"]).split(",")
        col = CATCOL.get(r["actor_category"], "#999")
        xs = [xpos[c] for c in appears if c in xpos]
        if xs:
            ax.plot([min(xs), max(xs)], [i, i], color=col, lw=1.3, alpha=0.4, zorder=1)
        for c in appears:
            if c in xpos:
                ax.scatter([xpos[c]], [i], s=160, color=col, edgecolor="white", lw=1, zorder=4)
        ax.text(-0.25, i, r["counterparty"], ha="right", va="center", fontsize=8.4)
    ax.axvline(2.7, color="#333", ls="--", lw=1.4)
    ax.text(2.7, len(top) - 0.3, "1866 transatlantic cable", ha="center", fontsize=8.5, color="#333")
    ax.set_xticks(list(xpos.values())); ax.set_xticklabels(["1847", "1857", "1866", "1914"], fontsize=11)
    ax.set_yticks([]); ax.set_xlim(-3.4, 4.1); ax.set_ylim(-0.6, len(top) - 0.2)
    ax.set_xlabel("crisis (Bank-of-England ledger years only)", fontsize=10)
    cats = ["discount_house", "bill_broker", "merchant_bank"]
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=CATCOL[c], markersize=11,
                      label=CATLAB[c]) for c in cats]
    ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.95)
    ax.set_title("The recurring inner circle: which firms appear at the Bank's window across crises\n"
                 "(Frühling & Göschen and Stern Bros span the cable; no ledgers exist for 1873/1890/1907)",
                 fontsize=11, pad=10)
    plt.savefig(FIG / "v5_recurrence.png", dpi=150, bbox_inches="tight"); plt.close()
    print("wrote v5_recurrence.png")


# ---------------------------------------------------------------- Fig 5 market price
def fig_market_price():
    imm = pd.read_csv(ROOT / "data/processed/imm/imm_financial_monthly.csv")
    imm["date"] = pd.to_datetime(imm["date"])
    fin = imm[imm["sector"].str.contains("financ", case=False, na=False)]
    post = [("1873", "1873-11-07"), ("1890", "1890-11-15"), ("1907", "1907-10-22"), ("1914", "1914-07-31")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    for ax, (c, pub) in zip(axes, post):
        pub = pd.Timestamp(pub)
        w = fin[(fin["date"] >= pub - pd.Timedelta(days=400)) & (fin["date"] <= pub + pd.Timedelta(days=210))]
        if w.empty:
            ax.text(0.5, 0.5, "no IMM data", ha="center", transform=ax.transAxes); ax.set_title(c); continue
        ax.plot(w["date"], w["mean_return_index"], color=POST, lw=1.8)
        ax.axvline(pub, color="#333", ls="--", lw=1.5)
        ax.text(pub, ax.get_ylim()[1], " rupture", fontsize=7.5, va="top")
        ax.set_title(c, fontsize=12, fontweight="bold")
        ax.set_ylabel("financial-sector index" if c == "1873" else "")
        for lab in ax.get_xticklabels():
            lab.set_rotation(35); lab.set_ha("right"); lab.set_fontsize(7)
        ax.grid(alpha=0.25)
    fig.suptitle("Market-price proxy (Yale IMM, listed financials): the sector was still RISING into each "
                 "public rupture\n-- no sign of early repricing, so we make no claim of insider profit",
                 fontsize=11, y=1.03)
    plt.tight_layout()
    plt.savefig(FIG / "v5_market_price.png", dpi=150, bbox_inches="tight"); plt.close()
    print("wrote v5_market_price.png")


if __name__ == "__main__":
    fig_info_chain(); fig_timeline(); fig_telegraph(); fig_recurrence(); fig_market_price()
    print("done.")
