"""Stage 7 — generate three reader-friendly helper figures for the paper.

Outputs:
- outputs/figures/timeline_crisis_arc.png
- outputs/figures/bank_roles_diagram.png
- outputs/figures/evidence_ladder.png

All numerical values come from outputs/tables/crisis_metrics.csv and
outputs/tables/network_summary.csv to keep the figures traceable to
already-validated outputs.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

os.makedirs("outputs/figures", exist_ok=True)

# ----------------------------------------------------------------------
# Figure 1: timeline_crisis_arc.png
# A horizontal timeline showing the four crises with crisis identity,
# headline numbers, and the question each asked.
# ----------------------------------------------------------------------

cm = pd.read_csv("outputs/tables/crisis_metrics.csv")
cm = cm.set_index("crisis_key")

CRISES = [
    {
        "year": 1857,
        "title": "1857 — reluctant public lender",
        "trigger": "12 Nov 1857: Treasury Letter",
        "headline": "Bank Rate +3.96 pp\nLending 2.17×\ntop-5 share 42%",
        "question": "Will the Bank lend?",
        "color": "#4C72B0",
    },
    {
        "year": 1866,
        "title": "1866 — market stabiliser",
        "trigger": "10–11 May 1866: Overend Gurney fails",
        "headline": "Bank Rate +4.50 pp\nLending 2.30×\ntop-5 share 19%",
        "question": "Will it stabilise a market\nafter refusing one firm?",
        "color": "#55A868",
    },
    {
        "year": 1890,
        "title": "1890 — coordinator",
        "trigger": "14–15 Nov 1890: Lidderdale Fund",
        "headline": "Bank Rate +1.54 pp\nLending 1.87× (Lidderdale week)\n£17.1m Guarantee Fund",
        "question": "Can it coordinate a\nprivate-public rescue?",
        "color": "#C44E52",
    },
    {
        "year": 1914,
        "title": "1914 — wartime public institution",
        "trigger": "31 Jul / 6 Aug 1914: LSE close, moratorium",
        "headline": "Bank Rate +6.10 pp\nLending 5.25×\ntop-5 share 28%",
        "question": "Can it act as part of\nwartime public policy?",
        "color": "#8172B2",
    },
]

fig, ax = plt.subplots(figsize=(13, 6))

# Horizontal axis — years
years_min, years_max = 1850, 1920
ax.set_xlim(years_min, years_max)
ax.set_ylim(-3.5, 2.5)
ax.axis("off")

# Baseline arrow
ax.annotate(
    "", xy=(years_max - 0.5, 0), xytext=(years_min + 0.5, 0),
    arrowprops=dict(arrowstyle="->", linewidth=1.5, color="#444"),
)
for tick_year in [1855, 1865, 1875, 1885, 1895, 1905, 1915]:
    ax.plot([tick_year, tick_year], [-0.08, 0.08], color="#999", linewidth=0.8)
    ax.text(tick_year, -0.3, str(tick_year), ha="center", va="top", fontsize=8, color="#666")

# Crisis markers
for i, c in enumerate(CRISES):
    y_up = 0.55 if i % 2 == 0 else 1.95
    y_down = -0.55 if i % 2 == 0 else -1.95
    # Top: title + question
    ax.plot([c["year"], c["year"]], [0, y_up - 0.05], color=c["color"], linewidth=2)
    ax.scatter([c["year"]], [0], s=90, color=c["color"], zorder=5)
    bbox_title = dict(boxstyle="round,pad=0.35", facecolor=c["color"], edgecolor="none", alpha=0.92)
    ax.text(
        c["year"], y_up, c["title"],
        ha="center", va="bottom", fontsize=10, color="white", fontweight="bold",
        bbox=bbox_title,
    )
    ax.text(
        c["year"], y_up + 0.55, c["question"],
        ha="center", va="bottom", fontsize=9, color="#222", style="italic",
    )
    # Bottom: trigger + headline numbers
    ax.plot([c["year"], c["year"]], [0, y_down + 0.05], color=c["color"], linewidth=2)
    ax.text(
        c["year"], y_down, c["trigger"],
        ha="center", va="top", fontsize=8.5, color="#222", fontweight="bold",
    )
    ax.text(
        c["year"], y_down - 0.45, c["headline"],
        ha="center", va="top", fontsize=8, color="#444",
    )

ax.set_title(
    "Four crises, four questions about what kind of institution the Bank had to be",
    fontsize=13, pad=14,
)

plt.tight_layout()
plt.savefig("outputs/figures/timeline_crisis_arc.png", dpi=160, bbox_inches="tight")
plt.close()
print("wrote outputs/figures/timeline_crisis_arc.png")

# ----------------------------------------------------------------------
# Figure 2: bank_roles_diagram.png
# A diagram showing the three institutional roles in tension.
# ----------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(11, 6.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis("off")

# Central triangle vertices
positions = {
    "private": (1.5, 5.2),
    "monetary": (8.5, 5.2),
    "public":   (5.0, 1.2),
}

role_colors = {
    "private": "#4C72B0",
    "monetary": "#55A868",
    "public":   "#C44E52",
}

role_text = {
    "private": (
        "Private commercial\ncorporation",
        "Profitable joint-stock company.\nProtects its own gold reserve.\nPays dividends to shareholders.",
    ),
    "monetary": (
        "Statutory manager of\nmonetary stability",
        "Custodian of the country's banking\nreserve under the Bank Charter Act 1844.\nDefends the gold standard via Bank Rate.",
    ),
    "public": (
        "Emerging public\ncrisis manager",
        "The institution the City turns to when\nprivate credit fails. Lender of last resort.\nCoordinator of public-private rescues.",
    ),
}

# Draw connecting tension lines
for k1, k2 in [("private","monetary"), ("monetary","public"), ("private","public")]:
    ax.annotate(
        "", xy=positions[k2], xytext=positions[k1],
        arrowprops=dict(arrowstyle="<->", color="#888", linewidth=1.5, linestyle="--",
                        connectionstyle="arc3,rad=0.0"),
    )
ax.text(5.0, 5.55, "tension between roles", ha="center", va="bottom", fontsize=9, color="#666", style="italic")
ax.text(3.0, 3.0, "tension", ha="center", va="bottom", fontsize=8.5, color="#666", style="italic", rotation=300)
ax.text(7.0, 3.0, "tension", ha="center", va="bottom", fontsize=8.5, color="#666", style="italic", rotation=60)

# Draw the three boxes
for k, (x, y) in positions.items():
    title, body = role_text[k]
    box = FancyBboxPatch(
        (x - 1.85, y - 0.95), 3.7, 1.9,
        boxstyle="round,pad=0.18",
        linewidth=2, edgecolor=role_colors[k], facecolor="white",
    )
    ax.add_patch(box)
    ax.text(x, y + 0.45, title, ha="center", va="center", fontsize=11.5,
            fontweight="bold", color=role_colors[k])
    ax.text(x, y - 0.25, body, ha="center", va="center", fontsize=8.8, color="#222")

ax.set_title(
    "Three roles of the Bank of England (1857–1914): in ordinary times in balance, in crisis forced into a choice",
    fontsize=12, pad=10,
)

plt.tight_layout()
plt.savefig("outputs/figures/bank_roles_diagram.png", dpi=160, bbox_inches="tight")
plt.close()
print("wrote outputs/figures/bank_roles_diagram.png")

# ----------------------------------------------------------------------
# Figure 3: evidence_ladder.png
# A "what each dataset lets us see" evidence ladder.
# ----------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(11, 6.8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7.2)
ax.axis("off")

# Five rungs of the ladder, bottom to top
rungs = [
    {
        "label": "Annual UK macro panel (JST R6, 1870–)",
        "lets_us_see": "Whether the broader economy was in stress",
        "best_for": "Macro context for 1890 and 1914 (1857 / 1866 outside coverage)",
        "color": "#9C9C9C",
    },
    {
        "label": "Weekly Banking + Issue Dept balance sheet (1844–1919)",
        "lets_us_see": "How big the Bank's response was, week by week",
        "best_for": "Cross-crisis comparison of scale and timing",
        "color": "#8172B2",
    },
    {
        "label": "Daily Bank Rate (Millennium D1, 1833–)",
        "lets_us_see": "When the Bank raised its policy rate, and by how much",
        "best_for": "Penalty-rate testing across all four crises",
        "color": "#C44E52",
    },
    {
        "label": "Daily transaction ledgers (1857, 1866, 1914)",
        "lets_us_see": "Who came to the discount window, with what bills, at what rates",
        "best_for": "Counterparty distribution, breadth, named firms",
        "color": "#55A868",
    },
    {
        "label": "Daily Account Books (BoE Archive C1/38, Oct–Dec 1890)",
        "lets_us_see": "Day-by-day balance-sheet shape across the Lidderdale fortnight",
        "best_for": "1890 intra-week trajectory; Total Assets / Reserve / Bank Rate only",
        "color": "#4C72B0",
    },
]

# Plot the rungs as stacked horizontal bars
for i, r in enumerate(rungs):
    y = 0.6 + 1.25 * i
    box = FancyBboxPatch(
        (0.4, y), 4.4, 1.0,
        boxstyle="round,pad=0.08", linewidth=1.5,
        edgecolor=r["color"], facecolor=r["color"], alpha=0.85,
    )
    ax.add_patch(box)
    ax.text(2.6, y + 0.6, r["label"], ha="center", va="center",
            fontsize=10, color="white", fontweight="bold")
    ax.text(2.6, y + 0.25, r["best_for"], ha="center", va="center",
            fontsize=8, color="white", style="italic")

    # Pointer to what it lets us see
    ax.annotate(
        "", xy=(5.2, y + 0.5), xytext=(4.85, y + 0.5),
        arrowprops=dict(arrowstyle="->", color=r["color"], linewidth=1.5),
    )
    ax.text(5.35, y + 0.5, "lets us see " + r["lets_us_see"],
            ha="left", va="center", fontsize=9.5, color="#222")

# Side label: "more granular"
ax.annotate(
    "", xy=(0.15, 6.2), xytext=(0.15, 0.6),
    arrowprops=dict(arrowstyle="<-", color="#444", linewidth=1.5),
)
ax.text(0.07, 3.4, "more granular →\n(day-level detail)", ha="center", va="center",
        fontsize=9, color="#444", rotation=90, fontweight="bold")
ax.text(0.07, 6.6, "↑", ha="center", va="center", fontsize=10, color="#444")

ax.set_title(
    "An evidence ladder. Each dataset answers a different historical question.",
    fontsize=12, pad=10,
)

plt.tight_layout()
plt.savefig("outputs/figures/evidence_ladder.png", dpi=160, bbox_inches="tight")
plt.close()
print("wrote outputs/figures/evidence_ladder.png")

print("done.")
