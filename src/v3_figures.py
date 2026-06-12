"""paper_v3 figures (revised, Stage 11). Reader-friendly visuals.

Fixes vs the first version:
- crisis columns are cast to str before filtering/reindexing (the int/str
  mismatch that produced empty three_clocks and concentration figures).
- Figure 2 is now a horizontal per-crisis timeline (ledger / official /
  parliamentary markers), not a sparse scatter.
- Figure 1 carries a worked example under each stage.
- Networks use top-8 borrowers, large labels, no clutter.
- The score figure is relabelled "intermediary access score" (no cognition).
- The 1890 figure is a qualitative coordination network (one file only).
"""
from __future__ import annotations
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
TBL = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location("v3mod", ROOT / "src" / "v3_information_flows.py")
v3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v3)

CATCOLOR = {
    "discount_house": "#C44E52", "bill_broker": "#DD8452", "merchant_bank": "#4C72B0",
    "clearing_or_joint_stock_bank": "#55A868", "foreign_or_colonial_financial": "#8172B2",
    "industrial_or_commercial": "#937860", "unknown": "#C7C7C7",
}
CATLABEL = {
    "discount_house": "discount house", "bill_broker": "bill broker",
    "merchant_bank": "merchant bank", "clearing_or_joint_stock_bank": "clearing / joint-stock bank",
    "foreign_or_colonial_financial": "foreign / colonial bank",
    "industrial_or_commercial": "industrial / commercial", "unknown": "unknown",
}


def load_tx():
    d47 = v3.load_1847(); doth = v3.load_others()
    tx = pd.concat([d47, doth], ignore_index=True)
    tx["crisis"] = tx["crisis"].astype(str)
    tx = tx[tx["counterparty_clean"].str.len() > 0].copy()
    tx["total_amount"] = pd.to_numeric(tx["total_amount"], errors="coerce").fillna(0.0)
    cats, _ = zip(*tx["counterparty_clean"].map(v3.classify))
    tx["actor_category"] = cats
    return tx


# ---------------------------------------------------------------- Fig 1
def fig_concept():
    fig, ax = plt.subplots(figsize=(13, 3.8)); ax.set_xlim(0, 13); ax.set_ylim(0, 4.3); ax.axis("off")
    steps = [
        ("1. Private stress", "a trade bill is no longer trusted;\na firm quietly needs cash",
         "e.g. Argentine bills sour, 1890", "#8172B2"),
        ("2. An intermediary moves", "a discount house or bill broker\ncomes to the Bank's window",
         "e.g. Overend, Gurney borrows, 1866", "#C44E52"),
        ("3. The Bank's ledger records it", "the loan is written down:\nwho, when, how much",
         "e.g. the 1857 discount ledger", "#4C72B0"),
        ("4. The crisis becomes public", "a failure, the newspapers,\nan emergency rule change, a debate",
         "e.g. 'The Panic in the City', 11 May 1866", "#55A868"),
    ]
    x = 0.3
    for i, (t, b, ex, c) in enumerate(steps):
        box = FancyBboxPatch((x, 1.25), 2.8, 2.0, boxstyle="round,pad=0.1",
                             linewidth=2, edgecolor=c, facecolor="white")
        ax.add_patch(box)
        ax.text(x + 1.4, 2.85, t, ha="center", va="center", fontsize=10.5, fontweight="bold", color=c)
        ax.text(x + 1.4, 2.15, b, ha="center", va="center", fontsize=8.3, color="#222")
        ax.text(x + 1.4, 1.5, ex, ha="center", va="center", fontsize=7.4, color="#777", style="italic")
        if i < 3:
            ax.annotate("", xy=(x + 3.42, 2.25), xytext=(x + 2.85, 2.25),
                        arrowprops=dict(arrowstyle="-|>", color="#444", linewidth=2))
        x += 3.18
    ax.text(6.5, 0.5, "This paper lines up stages 3 and 4 as 'clocks' to ask: did the ledger record stress before the crisis became public?",
            ha="center", va="center", fontsize=9, style="italic", color="#555")
    ax.set_title("From private stress to public panic: the four stages this paper traces",
                 fontsize=13, pad=6)
    plt.savefig(FIG / "v3_signal_to_panic_concept.png", dpi=150, bbox_inches="tight"); plt.close()
    print("wrote v3_signal_to_panic_concept.png")


# ---------------------------------------------------------------- Fig 2
def fig_four_clocks():
    """Four public-visibility clocks per crisis, in days relative to the
    official/legal emergency action (day 0): ledger peak, press-public
    (first article -> coverage surge, British Library HMD), official, and
    parliamentary (Hansard). 1914 has no HMD press coverage."""
    t = pd.read_csv(TBL / "v3_crisis_timeline.csv")
    t["crisis"] = t["crisis"].astype(str)
    press = pd.read_csv(TBL / "v3_press_public_markers.csv")
    press["crisis"] = press["crisis"].astype(str)
    pmap = {r["crisis"]: r for _, r in press.iterrows()}

    crises = ["1847", "1857", "1866", "1914"]
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    yld = {c: i for i, c in enumerate(reversed(crises))}
    RED, GOLD, BLACK, GREEN = "#C44E52", "#D9A300", "#333333", "#55A868"
    xmins, xmaxs = [], []
    for c in crises:
        row = t[t.crisis == c].iloc[0]
        off = pd.Timestamp(row["official_date"])
        led = pd.Timestamp(row["ledger_intermediary_peak_week"])
        par = pd.Timestamp(row["parliamentary_public_record_date"])
        led_d, par_d = (led - off).days, (par - off).days
        y = yld[c]
        xs = [led_d, 0, par_d]
        # press markers (none for 1914)
        pf_d = ps_d = None
        if c in pmap:
            pf = pd.Timestamp(pmap[c]["press_first_date"])
            ps = pd.Timestamp(pmap[c]["press_peak_week"])
            pf_d, ps_d = (pf - off).days, (ps - off).days
            xs += [pf_d, ps_d]
        lo, hi = min(xs) - 5, max(xs) + 5
        xmins.append(lo); xmaxs.append(hi)
        ax.plot([lo, hi], [y, y], color="#ececec", lw=1, zorder=0)
        spread = max(xs) - min(xs)
        tight = spread <= 6  # all clocks within ~a week -> one annotation
        # press span: first-mention (star) -> surge (large open circle that can
        # encircle a coincident ledger dot)
        if pf_d is not None:
            ax.plot([pf_d, ps_d], [y, y], color=GOLD, lw=2.4, alpha=0.40, zorder=1)
            ax.scatter([ps_d], [y], s=470, facecolor="none", edgecolor=GOLD,
                       linewidth=2.4, marker="o", zorder=4)
            ax.scatter([pf_d], [y], s=320, color=GOLD, marker="*", zorder=6,
                       edgecolor="white", linewidth=1.0)
            if not tight:
                ax.text(pf_d, y + 0.34, pf.strftime("%d %b"), ha="center", fontsize=7.8, color=GOLD)
                ax.text(ps_d, y - 0.40, "surge " + ps.strftime("%d %b"), ha="center",
                        va="top", fontsize=7.2, color=GOLD)
        # ledger dot drawn on top so it stays visible inside the surge ring
        ax.scatter([0], [y], s=250, color=BLACK, marker="s", zorder=5, edgecolor="white", linewidth=1.5)
        ax.scatter([par_d], [y], s=250, color=GREEN, marker="D", zorder=5, edgecolor="white", linewidth=1.5)
        ax.scatter([led_d], [y], s=235, color=RED, zorder=7, edgecolor="white", linewidth=1.5)
        if tight:
            # one combined annotation for crises where all clocks fire together
            alld = [pd.Timestamp(d) for d in [led, off, par] + (
                [pd.Timestamp(pmap[c]["press_first_date"]),
                 pd.Timestamp(pmap[c]["press_peak_week"])] if c in pmap else [])]
            lab = f"all clocks {min(alld):%d}–{max(alld):%d %b}"
            ax.text(max(xs) + 3, y, lab, ha="left", va="center", fontsize=8.2, color="#555")
        else:
            ax.text(led_d, y + 0.34, led.strftime("%d %b"), ha="center", fontsize=7.8, color=RED)
            ax.text(0, y + 0.18, off.strftime("%d %b"), ha="center", fontsize=7.8,
                    color=BLACK, fontweight="bold")
            # parliamentary label above its diamond (keeps it clear of the press-surge label)
            ax.text(par_d, y + 0.34, par.strftime("%d %b"), ha="center", va="bottom",
                    fontsize=7.8, color=GREEN)
        if c == "1914":
            ax.text(max(xs) + 7, y, "no HMD press coverage", va="center", ha="left",
                    fontsize=8, color="#999", style="italic")
    ax.axvline(0, color="#bbb", lw=1, ls="--", zorder=0)
    ax.set_yticks(list(yld.values())); ax.set_yticklabels(list(reversed(crises)), fontsize=13)
    ax.set_xlabel("Days relative to the Bank / government emergency action (day 0)", fontsize=10)
    ax.set_xlim(min(xmins) - 3, max(xmaxs) + 30)
    ax.set_ylim(-0.6, len(crises) - 0.4)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=RED, markersize=12,
               label="ledger clock: intermediary borrowing peak"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=GOLD, markersize=15,
               label="press-public clock: first crisis article (British Library HMD)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="w", markeredgecolor=GOLD,
               markeredgewidth=2, markersize=11, label="press coverage surge (peak week)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=BLACK, markersize=11,
               label="official / legal clock: emergency action (day 0)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=GREEN, markersize=11,
               label="parliamentary clock: Hansard debate"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8.6, framealpha=0.96)
    ax.set_title("Four public-visibility clocks per crisis: when did intermediary borrowing peak,\n"
                 "the press report it, the Bank act, and Parliament debate?",
                 fontsize=11.5, pad=10)
    ax.grid(axis="x", alpha=0.22)
    plt.savefig(FIG / "v3_four_clocks.png", dpi=150, bbox_inches="tight"); plt.close()
    print("wrote v3_four_clocks.png")


# ---------------------------------------------------------------- Fig 3
def fig_concentration():
    cz = pd.read_csv(TBL / "v3_lending_concentration.csv")
    cz["crisis"] = cz["crisis"].astype(str)
    full = cz[cz.window == "full"].set_index("crisis").reindex(["1847", "1857", "1866", "1914"])
    order = ["1847", "1857", "1866", "1914"]
    colors = ["#937860", "#C44E52", "#55A868", "#8172B2"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, title in [
        (axes[0], "top5_share", "Share taken by the 5 largest borrowers"),
        (axes[1], "share_intermediaries", "Share flowing through intermediaries\n(discount houses + bill brokers + merchant banks)")]:
        vals = (full[col] * 100).values
        bars = ax.bar(order, vals, color=colors)
        ax.set_title(title, fontsize=10.5); ax.set_ylabel("% of total crisis lending")
        ax.grid(axis="y", alpha=0.3); ax.set_ylim(0, max(vals) * 1.25)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02, f"{v:.0f}%",
                    ha="center", fontsize=10, fontweight="bold")
    fig.suptitle("Did emergency lending go broadly to many firms, or through a small circle?",
                 fontsize=12.5, y=1.01)
    plt.tight_layout(); plt.savefig(FIG / "v3_lending_concentration.png", dpi=150, bbox_inches="tight"); plt.close()
    print("wrote v3_lending_concentration.png")


# ---------------------------------------------------------------- Fig 5
def fig_scores():
    ic = pd.read_csv(TBL / "v3_information_channel_scores.csv")
    top = ic.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 6.2))
    colors = [CATCOLOR.get(c, "#999") for c in top["actor_category"]]
    ax.barh(top["counterparty"], top["information_channel_score"], color=colors)
    for i, (v, n) in enumerate(zip(top["information_channel_score"], top["n_crises"])):
        ax.text(v + 0.006, i, f"{v:.2f}  ({n} crises)", va="center", fontsize=8.5)
    ax.set_xlabel("Intermediary access score (behavioral, not cognitive)")
    ax.set_title("The recurring intermediary layer: top 'intermediary access' actors, 1847-1914", fontsize=12, pad=10)
    ax.set_xlim(0, max(top["information_channel_score"]) + 0.22)
    # inset explaining the score
    ax.text(0.98, 0.04,
            "score = early access + recurrence + cross-crisis\n+ scale + intermediary role + centrality\n(measures position in the flow of crisis credit,\nNOT knowledge or foresight)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#444",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f5f5f5", ec="#ccc"))
    seen = {c: CATCOLOR.get(c, "#999") for c in top["actor_category"]}
    handles = [plt.Rectangle((0, 0), 1, 1, color=col) for col in seen.values()]
    ax.legend(handles, [CATLABEL[c] for c in seen], loc="lower right", fontsize=8.5,
              bbox_to_anchor=(1.0, 0.28))
    plt.tight_layout(); plt.savefig(FIG / "v3_information_channel_scores.png", dpi=150, bbox_inches="tight"); plt.close()
    print("wrote v3_information_channel_scores.png")


# ---------------------------------------------------------------- Fig 4
def _network(ax, tx, crisis, topn=8):
    g = tx[tx["crisis"] == crisis]
    by = (g.groupby(["counterparty_clean", "actor_category"])["total_amount"].sum()
          .reset_index().sort_values("total_amount", ascending=False).head(topn))
    n = len(by)
    # wide limits so long firm names laid outside the ring are never clipped
    ax.set_xlim(-2.7, 2.7); ax.set_ylim(-1.9, 2.0); ax.axis("off")
    ax.scatter([0], [0], s=1500, color="#222", zorder=5)
    ax.text(0, 0, "Bank of\nEngland", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    amax = by["total_amount"].max()
    for i, (_, r) in enumerate(by.iterrows()):
        ang = 2 * np.pi * i / n + np.pi / 2
        x, y = np.cos(ang), np.sin(ang)
        w = 1.0 + 6.0 * (r["total_amount"] / amax)
        ax.plot([0, x], [0, y], color=CATCOLOR.get(r["actor_category"], "#ccc"), lw=w, alpha=0.6, zorder=1)
        s = 200 + 700 * (r["total_amount"] / amax)
        ax.scatter([x], [y], s=s, color=CATCOLOR.get(r["actor_category"], "#999"),
                   edgecolor="white", linewidth=1.5, zorder=4)
        nm = r["counterparty_clean"]; nm = nm if len(nm) <= 26 else nm[:25] + "…"
        # align the label outward from the node so it never sits under the circle
        if x > 0.3:
            lx, ha = x + 0.22, "left"
        elif x < -0.3:
            lx, ha = x - 0.22, "right"
        else:
            lx, ha = x, "center"
        ly = y + (0.20 if y > 0.3 else -0.20 if y < -0.3 else 0.0)
        ax.text(lx, ly, nm, ha=ha, va="center", fontsize=8.6, color="#222")
    ax.set_title(crisis, fontsize=15, fontweight="bold")


def fig_networks():
    tx = load_tx()
    for crisis in ["1857", "1866", "1914"]:
        fig, ax = plt.subplots(figsize=(8.2, 8.2))
        _network(ax, tx, crisis, topn=8)
        fig.suptitle(f"The 8 largest borrowers at the Bank's window, {crisis} (colour = firm type)",
                     fontsize=12, y=0.96)
        handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=CATCOLOR[c], markersize=11)
                   for c in ["discount_house", "bill_broker", "merchant_bank",
                             "clearing_or_joint_stock_bank", "foreign_or_colonial_financial"]]
        labels = [CATLABEL[c] for c in ["discount_house", "bill_broker", "merchant_bank",
                  "clearing_or_joint_stock_bank", "foreign_or_colonial_financial"]]
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8.5, frameon=False)
        plt.tight_layout(rect=[0, 0.06, 1, 1])
        plt.savefig(FIG / f"v3_network_{crisis}.png", dpi=150, bbox_inches="tight"); plt.close()
        print(f"wrote v3_network_{crisis}.png")


# ---------------------------------------------------------------- Fig 6
def fig_1890():
    fig, ax = plt.subplots(figsize=(11, 8)); ax.set_xlim(0, 12); ax.set_ylim(0, 10); ax.axis("off")
    nodes = {
        "Bank of England\n(Lidderdale)": (6, 8.4, "#4C72B0"),
        "Baring Brothers": (6, 4.8, "#C44E52"),
        "Treasury\n(Goschen)": (10, 7.3, "#55A868"),
        "Hambro": (2.4, 7.6, "#8172B2"),
        "Rothschilds\n(London + Paris)": (10.3, 4.4, "#CCB974"),
        "Banque de France": (10.3, 1.8, "#64B5CD"),
        "Russian state": (6, 1.4, "#D62728"),
        "Clearing-bank\nGuarantee Fund": (1.9, 1.9, "#9467BD"),
        "Argentine securities": (1.9, 4.6, "#8C564B"),
    }
    edges = [
        ("Hambro", "Bank of England\n(Lidderdale)", "8 Nov meeting"),
        ("Bank of England\n(Lidderdale)", "Treasury\n(Goschen)", "asks backing"),
        ("Bank of England\n(Lidderdale)", "Rothschilds\n(London + Paris)", "Paris gold"),
        ("Rothschilds\n(London + Paris)", "Banque de France", "£3m swap"),
        ("Rothschilds\n(London + Paris)", "Russian state", "£1.5m gold"),
        ("Bank of England\n(Lidderdale)", "Clearing-bank\nGuarantee Fund", "£17.1m fund"),
        ("Bank of England\n(Lidderdale)", "Baring Brothers", "£7.5m advance"),
        ("Argentine securities", "Baring Brothers", "toxic exposure"),
    ]
    for a, b, lbl in edges:
        x1, y1, _ = nodes[a]; x2, y2, _ = nodes[b]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#999", lw=1.2, shrinkA=34, shrinkB=34,
                                    connectionstyle="arc3,rad=0.08"))
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, lbl, fontsize=7, color="#555", style="italic",
                ha="center", bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
    for name, (x, y, c) in nodes.items():
        box = FancyBboxPatch((x - 1.1, y - 0.5), 2.2, 1.0, boxstyle="round,pad=0.08",
                             linewidth=2, edgecolor=c, facecolor="white")
        ax.add_patch(box)
        ax.text(x, y, name, ha="center", va="center", fontsize=8.5, fontweight="bold", color=c)
    ax.set_title("1890: a qualitative coordination network (no transaction ledger exists)\nReconstructed from White (2016), locally read. NOT a correspondence-text analysis.",
                 fontsize=11, pad=8)
    plt.savefig(FIG / "v3_network_1890.png", dpi=150, bbox_inches="tight"); plt.close()
    print("wrote v3_network_1890.png")


# ---------------------------------------------------------------- Fig 2b
def fig_press_coverage():
    """Monthly count of crisis-relevant HMD newspaper articles per crisis,
    showing when press coverage surged relative to the official action."""
    cov = pd.read_csv(TBL / "v3_press_coverage_monthly.csv")
    cov["crisis"] = cov["crisis"].astype(str)
    t = pd.read_csv(TBL / "v3_crisis_timeline.csv"); t["crisis"] = t["crisis"].astype(str)
    crises = ["1847", "1857", "1866"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for ax, c in zip(axes, crises):
        g = cov[cov.crisis == c].copy()
        g["mdt"] = pd.to_datetime(g["month"] + "-01")
        g = g.sort_values("mdt")
        ax.bar(g["mdt"], g["matching"], width=22, color="#D9A300", alpha=0.85,
               label="crisis articles")
        row = t[t.crisis == c].iloc[0]
        off = pd.Timestamp(row["official_date"])
        ax.axvline(off, color="#333", lw=1.6, ls="--")
        ax.text(off, ax.get_ylim()[1] * 0.92, " official\n action", fontsize=7.5,
                color="#333", va="top")
        ax.set_title(f"{c}", fontsize=13, fontweight="bold")
        ax.set_ylabel("crisis articles / month" if c == "1847" else "")
        for lab in ax.get_xticklabels():
            lab.set_rotation(35); lab.set_fontsize(7.5); lab.set_ha("right")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Press-public visibility: monthly count of crisis-relevant British Library HMD "
                 "newspaper articles\n(coverage surges at the public rupture; dashed line = "
                 "Bank/government emergency action)", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(FIG / "v3_press_coverage.png", dpi=150, bbox_inches="tight"); plt.close()
    print("wrote v3_press_coverage.png")


if __name__ == "__main__":
    fig_concept(); fig_four_clocks(); fig_press_coverage()
    fig_concentration(); fig_scores()
    fig_networks(); fig_1890()
    print("done.")
