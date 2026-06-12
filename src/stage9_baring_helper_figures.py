"""Stage 9 — three reader-friendly helper figures for paper_v2_baring.md.

Outputs:
- outputs/figures/baring_actor_network.png
- outputs/figures/baring_crisis_comparison_bars.png
- outputs/figures/baring_rescue_week_timeline.png

All values come from outputs/tables/crisis_metrics.csv and the White (2016)
chronology (text excerpt in references/white-2016-bank-underground-text.txt).
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np
import os

os.makedirs("outputs/figures", exist_ok=True)

# ----------------------------------------------------------------------
# Figure A. baring_actor_network.png
# Actor-network diagram showing the nine actors around Baring Brothers
# ----------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(13, 9))
ax.set_xlim(0, 12)
ax.set_ylim(0, 10)
ax.axis("off")

# Layout: Baring Brothers in the center; other actors around the rim
# (positions chosen for readability; not geographic)
nodes = {
    "Baring Brothers":     {"xy": (6.0, 5.0),  "color": "#C44E52",  "size": (2.8, 0.95), "label": "Baring Brothers\n(distressed merchant bank;\n£15.7m liabilities;\n£8.3m Argentine securities)"},
    "Bank of England":     {"xy": (6.0, 8.6),  "color": "#4C72B0",  "size": (2.8, 1.0),  "label": "Bank of England\n(Lidderdale, Governor;\nadvance £7.5m;\ncoordinator of rescue)"},
    "Treasury / Goschen":  {"xy": (10.0, 7.5), "color": "#55A868",  "size": (2.4, 0.9),  "label": "Treasury\n(Goschen, Chancellor;\ndeclined direct rescue;\noffered Chancellor's Letter)"},
    "Hambro":              {"xy": (2.5, 7.7),  "color": "#8172B2",  "size": (2.0, 0.7),  "label": "Everard Hambro\n(convened 8 Nov\npreview meeting)"},
    "Rothschilds":         {"xy": (10.5, 4.5), "color": "#CCB974",  "size": (2.4, 0.95), "label": "Rothschilds\n(Nathaniel in London,\nAlphonse in Paris;\ntrust channel for gold)"},
    "Banque de France":    {"xy": (10.5, 1.8), "color": "#64B5CD",  "size": (2.6, 0.9),  "label": "Banque de France\n(£3m gold-for-Treasury-\nbills swap)"},
    "Russian state":       {"xy": (6.0, 1.4),  "color": "#D62728",  "size": (2.5, 0.8),  "label": "Russian state\n(£1.5m gold\nexchange)"},
    "Clearing banks":      {"xy": (1.7, 1.8),  "color": "#9467BD",  "size": (2.6, 0.95), "label": "Clearing banks\n(syndicate subscribed\n£17.1m four-year\nGuarantee Fund)"},
    "Argentine borrowers": {"xy": (1.7, 4.7),  "color": "#8C564B",  "size": (2.5, 0.85), "label": "Argentine borrowers\n(securities; coup;\nharvest failure;\nsovereign-debt crisis)"},
}

# Edges: each tuple is (from, to, label, style)
edges = [
    ("Hambro", "Bank of England", "convenes 8 Nov", "solid"),
    ("Hambro", "Baring Brothers", "preview meeting", "solid"),
    ("Bank of England", "Treasury / Goschen", "asks for\nbacking", "solid"),
    ("Treasury / Goschen", "Bank of England", "offers Chancellor's\nLetter (refused)", "dashed"),
    ("Bank of England", "Rothschilds", "via Goschen,\nfor Paris gold", "solid"),
    ("Rothschilds", "Banque de France", "negotiates\n£3m swap", "solid"),
    ("Rothschilds", "Russian state", "negotiates\n£1.5m exchange", "solid"),
    ("Bank of England", "Clearing banks", "coordinates\n£17.1m Fund", "solid"),
    ("Bank of England", "Baring Brothers", "£7.5m advance +\nbad-bank oversight", "solid"),
    ("Clearing banks", "Baring Brothers", "guarantee\nshare", "solid"),
    ("Argentine borrowers", "Baring Brothers", "toxic\nsecurities\n(source of exposure)", "solid"),
]


def draw_node(ax, name):
    n = nodes[name]
    x, y = n["xy"]
    w, h = n["size"]
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.08",
        linewidth=1.8, edgecolor=n["color"], facecolor="white",
    )
    ax.add_patch(box)
    # bold first line, smaller body
    lines = n["label"].split("\n")
    top = lines[0]
    body = "\n".join(lines[1:])
    ax.text(x, y + h*0.18, top, ha="center", va="center", fontsize=10,
            fontweight="bold", color=n["color"])
    if body:
        ax.text(x, y - h*0.18, body, ha="center", va="center", fontsize=8,
                color="#222")


def draw_edge(ax, src, dst, label, style):
    sx, sy = nodes[src]["xy"]
    dx, dy = nodes[dst]["xy"]
    # Choose source/dst edge midpoints (rough)
    arrow = FancyArrowPatch(
        (sx, sy), (dx, dy),
        arrowstyle="->", mutation_scale=15,
        linewidth=1.0,
        color="#888" if style == "solid" else "#bbb",
        linestyle=style,
        connectionstyle="arc3,rad=0.08",
        shrinkA=42, shrinkB=42,
        zorder=0,
    )
    ax.add_patch(arrow)
    # Edge label at midpoint
    mx, my = (sx + dx) / 2, (sy + dy) / 2
    ax.text(mx, my, label, ha="center", va="center", fontsize=7,
            color="#555", style="italic",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85))


# Draw edges first (behind nodes), then nodes
for e in edges:
    draw_edge(ax, *e)
for name in nodes:
    draw_node(ax, name)

ax.set_title(
    "The 1890 Baring rescue as a network. Nine actors, coordinated by Lidderdale.",
    fontsize=13, pad=12,
)
ax.text(6, -0.4,
        "Solid arrows: actions taken. Dashed arrow: action offered but refused. Source: White (2016), locally read in full.",
        ha="center", va="center", fontsize=8.5, color="#555", style="italic")

plt.tight_layout()
plt.savefig("outputs/figures/baring_actor_network.png", dpi=160, bbox_inches="tight")
plt.close()
print("wrote outputs/figures/baring_actor_network.png")

# ----------------------------------------------------------------------
# Figure B. baring_crisis_comparison_bars.png
# 1890 vs 1857 / 1866 / 1914 on three metrics
# ----------------------------------------------------------------------

cm = pd.read_csv("outputs/tables/crisis_metrics.csv").set_index("crisis_key")

crises = [1857, 1866, 1890, 1914]
labels = ["1857", "1866", "1890 (Baring)", "1914"]
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

lend_to_res = [cm.loc[c, "acute_peak_lending_to_reserve"] for c in crises]
reserve_loss = [cm.loc[c, "max_reserve_loss_pct"] * 100 for c in crises]
penalty = [cm.loc[c, "penalty_rate_delta"] for c in crises]

fig, axes = plt.subplots(1, 3, figsize=(13, 5.2))

# Panel 1: lending-to-reserve peak
ax = axes[0]
bars = ax.bar(labels, lend_to_res, color=colors)
ax.set_title("Lending-to-reserve peak\n(higher = stronger discount-window response)", fontsize=10)
ax.set_ylabel("Ratio (peak crisis lending / acute min reserve)")
ax.grid(axis="y", alpha=0.3)
for b, v in zip(bars, lend_to_res):
    ax.text(b.get_x() + b.get_width()/2, v + 1, f"{v:.1f}",
            ha="center", va="bottom", fontsize=9)

# Panel 2: reserve drawdown %
ax = axes[1]
bars = ax.bar(labels, reserve_loss, color=colors)
ax.set_title("Reserve drawdown\n(% of pre-crisis baseline)", fontsize=10)
ax.set_ylabel("% reserve lost")
ax.grid(axis="y", alpha=0.3)
for b, v in zip(bars, reserve_loss):
    ax.text(b.get_x() + b.get_width()/2, v + 1.5, f"{v:.1f}%",
            ha="center", va="bottom", fontsize=9)

# Panel 3: Bank Rate rise (pp)
ax = axes[2]
bars = ax.bar(labels, penalty, color=colors)
ax.set_title("Bank Rate rise\n(percentage points above pre-crisis baseline)", fontsize=10)
ax.set_ylabel("Bank Rate change (pp)")
ax.grid(axis="y", alpha=0.3)
for b, v in zip(bars, penalty):
    ax.text(b.get_x() + b.get_width()/2, v + 0.15, f"+{v:.2f}",
            ha="center", va="bottom", fontsize=9)

fig.suptitle(
    "1890 sits at the muted end of every conventional Bank-of-England crisis measure",
    fontsize=12, y=1.02,
)
plt.tight_layout()
plt.savefig("outputs/figures/baring_crisis_comparison_bars.png", dpi=160, bbox_inches="tight")
plt.close()
print("wrote outputs/figures/baring_crisis_comparison_bars.png")

# ----------------------------------------------------------------------
# Figure C. baring_rescue_week_timeline.png
# Vertical timeline of 8-15 November 1890
# ----------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 7.5))
ax.set_xlim(0, 12)
ax.set_ylim(0, 11)
ax.axis("off")

# Days on a vertical line, top to bottom
events = [
    # (date, day_label, headline, body)
    ("Sat 8 Nov",  "Hambro convenes\npreview meeting",
     "Hambro brings two Barings partners and\nLidderdale together. Lidderdale sends note\nto Goschen asking him to come Monday.", "#8172B2"),
    ("Sun 9 Nov",  "Flight to quality",
     "Banks rumoured linked to Barings (Martin's)\nlose deposits. Private deposits at Bank rise.\nNew loans at Bank rise six-fold week-on-week.", "#9C9C9C"),
    ("Mon 10 Nov", "Goschen refuses, Lidderdale refuses",
     "Goschen: government will not interfere on behalf\nof insolvent house. Offers Chancellor's Letter.\nLidderdale REFUSES the letter; asks for Paris gold.", "#55A868"),
    ("Tue–Fri 11–14 Nov", "Gold from Paris and Russia",
     "Rothschilds work the trust channel.\nBanque de France £3m gold swap.\nRussia £1.5m gold exchange.", "#CCB974"),
    ("Fri 14 Nov", "Alphonse de Rothschild letter",
     "Compares Baring situation explicitly to\n1889 Comptoir d'Escompte rescue.\nProposes British guarantee syndicate structure.", "#64B5CD"),
    ("Sat 15 Nov", "Lifeboat announced",
     "Bank advance £7.5m. Four-year guarantee\nsyndicate £17.1m. Good-bank / bad-bank split.\nPartners surrender powers-of-attorney.", "#C44E52"),
    ("Wed 19 Nov", "Post-rescue balance sheet",
     "Total Assets £58.47m (up £9.59m since Sat 8 Nov).\nReserve £14.55m (rising). Bank Rate 6%.\nNo panic. Validated against weekly anchor.", "#4C72B0"),
]

n = len(events)
top, bot = 10.4, 0.7
ys = [top - i * (top - bot) / (n - 1) for i in range(n)]

# Vertical timeline line
ax.plot([3.5, 3.5], [bot - 0.3, top + 0.3], color="#444", linewidth=2)

for i, (date, headline, body, color) in enumerate(events):
    y = ys[i]
    # date label on the left
    ax.text(2.9, y, date, ha="right", va="center", fontsize=10,
            color=color, fontweight="bold")
    # dot on the line
    ax.scatter([3.5], [y], s=130, color=color, edgecolor="white", linewidth=2, zorder=5)
    # event card on the right
    box = FancyBboxPatch(
        (4.0, y - 0.5), 7.7, 1.0,
        boxstyle="round,pad=0.08",
        linewidth=1.4, edgecolor=color, facecolor="white",
    )
    ax.add_patch(box)
    ax.text(4.2, y + 0.27, headline, ha="left", va="center", fontsize=10,
            fontweight="bold", color=color)
    ax.text(4.2, y - 0.22, body, ha="left", va="center", fontsize=8.5, color="#222")

ax.set_title(
    "The rescue week, day by day. November 1890.",
    fontsize=13, pad=8,
)
ax.text(6.5, -0.1,
        "Chronology, named actors, and quotations sourced to White (2016, Bank Underground), locally read in full.",
        ha="center", va="center", fontsize=8.5, color="#555", style="italic")

plt.tight_layout()
plt.savefig("outputs/figures/baring_rescue_week_timeline.png", dpi=160, bbox_inches="tight")
plt.close()
print("wrote outputs/figures/baring_rescue_week_timeline.png")

print("done.")
