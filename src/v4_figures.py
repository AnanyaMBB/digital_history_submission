"""
v4_figures.py — figures for paper_v4 "Before the Panic".
Reads outputs/tables/v4_*.csv + data/processed/bank_rate_daily.parquet +
outputs/tables/v4_press_coverage_monthly.csv. Writes outputs/figures/v4_*.png.

Figures:
  1. v4_concept.png            — "panic as information event": the three clocks + two traditions
  2. v4_three_clocks.png       — per crisis, specialist / public / official markers (days vs public=0)
  3. v4_lead_time.png          — lead-time by crisis, endogenous vs external (sustained + acute)
  4. v4_bank_rate_multiples.png— small-multiple Bank Rate paths with the three markers
  5. v4_text_coverage.png      — HMD monthly crisis-article share (1847/57/66/73) vs public date
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
TBL = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

ENDO = "#C44E52"   # endogenous = red
EXTO = "#4C72B0"   # external = blue
SPEC = "#D9A300"   # specialist (gold)
PUB = "#333333"    # public (black)
OFF = "#55A868"    # official (green)

clk = pd.read_csv(TBL / "v4_clock_table.csv")
clk["crisis"] = clk["crisis"].astype(str)
for col in ["specialist_signal_date", "specialist_acute_date", "public_visibility_date",
            "official_response_date"]:
    clk[col + "_dt"] = pd.to_datetime(clk[col], errors="coerce")
ORDER = ["1847", "1857", "1866", "1873", "1890", "1907", "1914"]
clk = clk.set_index("crisis").reindex(ORDER).reset_index()
clk["color"] = np.where(clk["origin"] == "endogenous", ENDO, EXTO)


# ---------------------------------------------------------------- Fig 1 concept
def fig_concept():
    fig, ax = plt.subplots(figsize=(12.5, 4.6)); ax.set_xlim(0, 13); ax.set_ylim(0, 5); ax.axis("off")
    ax.text(6.5, 4.6, "A panic before the panic: distress can be visible in market records "
            "before it becomes a public panic", ha="center", fontsize=12.5, fontweight="bold")
    stages = [
        ("1. Specialist /\nmarket signal", "Bank Rate rises;\nintermediaries borrow;\nCity press notes strain",
         "#fff7e6", SPEC),
        ("2. Public\nvisibility", "a famous failure;\nnewspaper coverage\nsurges; 'panic'", "#eeeeee", PUB),
        ("3. Official\nresponse", "Bank Charter Act\nsuspended; rescue;\nmoratorium", "#eaf3ec", OFF),
    ]
    x = 1.0
    for t, b, fc, ec in stages:
        ax.add_patch(FancyBboxPatch((x, 1.9), 3.1, 1.7, boxstyle="round,pad=0.08",
                                    fc=fc, ec=ec, lw=2))
        ax.text(x + 1.55, 3.2, t, ha="center", va="center", fontsize=10.5, fontweight="bold", color=ec)
        ax.text(x + 1.55, 2.4, b, ha="center", va="center", fontsize=8.2, color="#333")
        if x < 9:
            ax.annotate("", xy=(x + 4.05, 2.75), xytext=(x + 3.15, 2.75),
                        arrowprops=dict(arrowstyle="-|>", lw=2, color="#888"))
        x += 4.0
    ax.text(2.55, 1.55, "lead time", ha="center", fontsize=8.5, color=SPEC, style="italic")
    ax.annotate("", xy=(8.1, 1.5), xytext=(1.0, 1.5),
                arrowprops=dict(arrowstyle="-|>", lw=1.4, color=SPEC))
    ax.text(4.5, 1.2, "lead_time = public_visibility_date  -  specialist_signal_date",
            ha="center", fontsize=9, color=SPEC)
    ax.text(6.5, 0.55, "Tested: is the lead LONG when a crisis is born inside London's financial network "
            "(endogenous),\nand SHORT when it arrives from outside (imported / geopolitical)? "
            "We measure dated visibility, not private knowledge.",
            ha="center", fontsize=8.6, color="#444")
    plt.savefig(FIG / "v4_concept.png", dpi=150, bbox_inches="tight"); plt.close()
    print("wrote v4_concept.png")


# ---------------------------------------------------------------- Fig 2 three clocks
def fig_three_clocks():
    fig, ax = plt.subplots(figsize=(12.5, 6.4))
    y = {c: i for i, c in enumerate(reversed(ORDER))}
    for _, r in clk.iterrows():
        c = r["crisis"]; yy = y[c]
        pub = r["public_visibility_date_dt"]
        spec = (r["specialist_signal_date_dt"] - pub).days
        acute = (r["specialist_acute_date_dt"] - pub).days
        off = (r["official_response_date_dt"] - pub).days
        col = r["color"]
        ax.plot([min(spec, acute, 0, off) - 6, max(0, off) + 6], [yy, yy], color="#ececec", lw=1, zorder=0)
        # specialist sustained (gold star) and acute (gold open) connected
        ax.plot([spec, acute], [yy, yy], color=SPEC, lw=2.2, alpha=0.4, zorder=1)
        ax.scatter([spec], [yy], s=300, marker="*", color=SPEC, edgecolor="white", lw=1, zorder=6)
        if acute != spec:
            ax.scatter([acute], [yy], s=120, facecolor="white", edgecolor=SPEC, lw=2, zorder=6)
        ax.scatter([0], [yy], s=210, marker="s", color=PUB, edgecolor="white", lw=1.4, zorder=5)
        ax.scatter([off], [yy], s=210, marker="D", color=OFF, edgecolor="white", lw=1.4, zorder=5)
        ax.text(spec, yy + 0.22, f"{-r['lead_time_days']:+d}d", ha="center", fontsize=8, color=SPEC)
        if acute != spec:
            ax.text(acute, yy - 0.28, f"acute {-r['lead_time_acute_days']:+d}d", ha="center",
                    fontsize=6.8, color=SPEC, va="top")
        ax.text(max(0, off) + 9, yy, f"{c}  ({r['origin']})", va="center", fontsize=9.5,
                color=col, fontweight="bold")
    ax.axvline(0, color="#bbb", ls="--", lw=1)
    ax.set_yticks(list(y.values())); ax.set_yticklabels([])
    ax.set_xlabel("Days relative to PUBLIC VISIBILITY (the public rupture = day 0)", fontsize=10)
    ax.set_ylim(-0.6, len(ORDER) - 0.4)
    ax.set_xlim(-310, 95)
    handles = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor=SPEC, markersize=16,
               label="specialist signal: Bank Rate sustained defensive tightening (first measurable signal)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="w", markeredgecolor=SPEC,
               markeredgewidth=2, markersize=10, label="specialist signal: acute final run-up (stricter)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=PUB, markersize=11,
               label="public visibility: the public rupture (day 0)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=OFF, markersize=11,
               label="official response: suspension / rescue / moratorium / rate peak"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8.4, framealpha=0.96)
    ax.set_title("Three clocks per crisis: the market-facing signal led the public rupture in every case,\n"
                 "but by far longer in endogenous crises (red) than in external ones (blue)",
                 fontsize=11.5, pad=10)
    ax.grid(axis="x", alpha=0.2)
    plt.savefig(FIG / "v4_three_clocks.png", dpi=150, bbox_inches="tight"); plt.close()
    print("wrote v4_three_clocks.png")


# ---------------------------------------------------------------- Fig 3 lead time
def fig_lead_time():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    d = clk.sort_values("lead_time_days")
    yy = np.arange(len(d))
    for ax, col, title in [(axes[0], "lead_time_days",
                            "Sustained tightening start\n(first measurable market signal)"),
                           (axes[1], "lead_time_acute_days",
                            "Acute final run-up\n(stricter, crisis-specific)")]:
        ax.hlines(yy, 0, d[col], color=d["color"], lw=3, alpha=0.5)
        ax.scatter(d[col], yy, color=d["color"], s=90, zorder=5)
        for i, (_, r) in enumerate(d.iterrows()):
            ax.text(r[col] + 4, i, f"{int(r[col])}d", va="center", fontsize=8.5, color=r["color"])
        ax.set_yticks(yy); ax.set_yticklabels(d["crisis"], fontsize=11)
        ax.set_xlabel("lead time (days): public rupture − specialist signal", fontsize=9.5)
        ax.set_title(title, fontsize=10.5)
        ax.grid(axis="x", alpha=0.25); ax.set_xlim(0, max(310, d[col].max() + 30))
    # medians annotation
    endo = clk[clk.origin == "endogenous"]; ext = clk[clk.origin == "external"]
    axes[0].text(0.97, 0.05,
                 f"median lead  endogenous {endo['lead_time_days'].median():.0f}d  vs  "
                 f"external {ext['lead_time_days'].median():.0f}d",
                 transform=axes[0].transAxes, ha="right", fontsize=8.5, color="#333",
                 bbox=dict(boxstyle="round", fc="#fff", ec="#ccc"))
    axes[1].text(0.97, 0.05,
                 f"median lead  endogenous {endo['lead_time_acute_days'].median():.0f}d  vs  "
                 f"external {ext['lead_time_acute_days'].median():.0f}d\n"
                 f"(1866 collapses to 8d: the sudden endogenous case)",
                 transform=axes[1].transAxes, ha="right", fontsize=8.0, color="#333",
                 bbox=dict(boxstyle="round", fc="#fff", ec="#ccc"))
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=ENDO, markersize=11,
                      label="endogenous (born inside London finance): 1847, 1866, 1890"),
               Line2D([0], [0], marker="o", color="w", markerfacecolor=EXTO, markersize=11,
                      label="external / imported / geopolitical: 1857, 1873, 1907, 1914")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("How long was distress visible in market records before the public panic?",
                 fontsize=12.5, y=0.99)
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    plt.savefig(FIG / "v4_lead_time.png", dpi=150, bbox_inches="tight"); plt.close()
    print("wrote v4_lead_time.png")


# ---------------------------------------------------------------- Fig 4 bank-rate multiples
def fig_bank_rate_multiples():
    br = pd.read_parquet(ROOT / "data/processed/bank_rate_daily.parquet")
    br["date"] = pd.to_datetime(br["date"])
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.6))
    axes = axes.ravel()
    for ax, (_, r) in zip(axes, clk.iterrows()):
        pub = r["public_visibility_date_dt"]
        lo, hi = pub - pd.Timedelta(days=320), pub + pd.Timedelta(days=60)
        w = br[(br["date"] >= lo) & (br["date"] <= hi)]
        ax.plot(w["date"], w["bank_rate"], color=r["color"], lw=1.8, drawstyle="steps-post")
        ax.axvline(r["specialist_signal_date_dt"], color=SPEC, lw=1.6, ls="-")
        ax.axvline(pub, color=PUB, lw=1.4, ls="--")
        ax.axvline(r["official_response_date_dt"], color=OFF, lw=1.2, ls=":")
        ax.set_title(f"{r['crisis']}  ({r['origin']})  lead {int(r['lead_time_days'])}d",
                     fontsize=10, color=r["color"], fontweight="bold")
        ax.set_ylabel("Bank Rate %", fontsize=8)
        ax.tick_params(labelsize=7)
        for lab in ax.get_xticklabels():
            lab.set_rotation(30); lab.set_ha("right")
        ax.grid(alpha=0.2)
    axes[-1].axis("off")
    handles = [Line2D([0], [0], color=SPEC, lw=2, label="specialist signal (tightening start)"),
               Line2D([0], [0], color=PUB, lw=2, ls="--", label="public visibility (rupture)"),
               Line2D([0], [0], color=OFF, lw=2, ls=":", label="official response")]
    axes[-1].legend(handles=handles, loc="center", fontsize=10, frameon=False)
    fig.suptitle("Bank Rate in each crisis window: the defensive climb (gold) begins well before the public "
                 "rupture (black) in endogenous crises, and almost on top of it in 1914",
                 fontsize=12, y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(FIG / "v4_bank_rate_multiples.png", dpi=150, bbox_inches="tight"); plt.close()
    print("wrote v4_bank_rate_multiples.png")


# ---------------------------------------------------------------- Fig 5 text coverage
def fig_text_coverage():
    cov = pd.read_csv(TBL / "v4_press_coverage_monthly.csv")
    cov["crisis"] = cov["crisis"].astype(str)
    crises = [c for c in ["1847", "1857", "1866", "1873"] if c in cov["crisis"].unique()]
    fig, axes = plt.subplots(1, len(crises), figsize=(4 * len(crises), 4.0))
    if len(crises) == 1:
        axes = [axes]
    for ax, c in zip(axes, crises):
        g = cov[cov.crisis == c].copy()
        g["mdt"] = pd.to_datetime(g["month"] + "-01")
        g = g.sort_values("mdt")
        ax.bar(g["mdt"], g["share"] * 100, width=22, color=SPEC, alpha=0.85)
        pub = clk.loc[clk.crisis == c, "public_visibility_date_dt"].iloc[0]
        ax.axvline(pub, color=PUB, lw=1.6, ls="--")
        ax.text(pub, ax.get_ylim()[1] * 0.92, " public\n rupture", fontsize=7.3, va="top")
        ax.set_title(c, fontsize=12, fontweight="bold")
        ax.set_ylabel("% of articles crisis-related" if c == crises[0] else "")
        for lab in ax.get_xticklabels():
            lab.set_rotation(35); lab.set_ha("right"); lab.set_fontsize(7)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Public visibility, text-mined: share of HMD newspaper articles that are crisis-related "
                 "surges at the public rupture\n(HMD covers 1847/1857/1866; 1873 sparse; no HMD for 1890/1907/1914)",
                 fontsize=10.8)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(FIG / "v4_text_coverage.png", dpi=150, bbox_inches="tight"); plt.close()
    print("wrote v4_text_coverage.png")


if __name__ == "__main__":
    fig_concept(); fig_three_clocks(); fig_lead_time()
    fig_bank_rate_multiples(); fig_text_coverage()
    print("done.")
