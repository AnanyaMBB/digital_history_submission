"""
v6_access_network.py -- "Did the access network survive?" Rigorous version.

Traces the Bank of England's crisis-access network from Victorian bill brokers and
discount houses (1847-1914) through 1906, 1931, the long discount market, the
1973-75 lifeboat, and the 1976-1996 authorised money-market counterparties.

The careful claim this script supports: the access network survived by CHANGING
FORM. The names mostly changed, but the structure persisted; recognised
intermediaries remained the route through which the Bank managed money-market
stress until the 1997 reform. After 1914 the evidence supports a closed ACCESS
NETWORK more than a closed information club.

Most post-1914 tables are SOURCE-DERIVED structured data (every row sourced),
because the underlying archives are not downloadable. The one fully computed
series is the Bank's deposits placed with discount houses, 1986-1996 (BoE database
export). The Victorian baseline is reused from the project's earlier outputs.

Outputs: outputs/tables/v6_*.csv and outputs/figures/v6_*.png
Run    : ./.venv/bin/python src/v6_access_network.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
TBL = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures"
LIT = ROOT / "references" / "v6_lit"

MED, BROAD = "#C44E52", "#4C72B0"
GOLD, GREEN, PURPLE, GREY = "#D9A300", "#55A868", "#8172B2", "#888888"
CATCOL = {"bill broker": "#DD8452", "discount house": "#C44E52",
          "acceptance / merchant house": "#4C72B0", "clearing bank": "#55A868",
          "secondary bank": "#8172B2", "authorised counterparty": "#937860"}

PERIODS = ["1847-1914", "1906", "1931", "1830-1997 market", "1973-75", "1976-96", "1997 reform"]
PERIOD_LABEL = {
    "1847-1914": "1847-1914 Victorian crises", "1906": "1906 bill market",
    "1931": "1931 sterling crisis", "1830-1997 market": "1830-1997 discount-market structure",
    "1973-75": "1973-75 lifeboat", "1976-96": "1976-96 authorised counterparties",
    "1997 reform": "1997 reform",
}

# ---------------------------------------------------------------- 1. continuity index
# Six binary criteria scored from the read sources (see v6_source_log.md), plus a
# confidence weight. 1997 is the reform that broadens access (the structure breaks).
CRIT = ["named_institutions_visible", "formal_eligibility_membership",
        "bank_access_mediated", "crisis_support_via_intermediaries",
        "quantitative_evidence", "direct_crisis_evidence"]
CONTINUITY = {
    # period:            named eligib mediat crisis quant directcrisis  confidence
    "1847-1914":        [1,    0,     1,     1,     1,    1,            0.95],
    "1906":             [1,    1,     1,     0,     1,    0,            0.90],
    "1931":             [1,    1,     1,     1,     0,    1,            0.80],
    "1830-1997 market": [1,    1,     1,     1,     0,    0,            0.85],
    "1973-75":          [1,    1,     1,     1,     1,    1,            0.85],
    "1976-96":          [1,    1,     1,     0,     1,    0,            0.90],
    "1997 reform":      [0,    0,     0,     0,     0,    0,            0.70],
}


def build_continuity_index():
    rows = []
    for p in PERIODS:
        vals = CONTINUITY[p]
        rows.append(dict(period=PERIOD_LABEL[p],
                         **{c: vals[i] for i, c in enumerate(CRIT)},
                         continuity_score=sum(vals[:6]), confidence=vals[6]))
    df = pd.DataFrame(rows)
    df.to_csv(TBL / "v6_continuity_index.csv", index=False)
    # figure: heatmap of criteria x period + score bar
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw={"width_ratios": [3, 1]})
    M = np.array([CONTINUITY[p][:6] for p in PERIODS])
    ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(6)); ax.set_xticklabels(
        ["named\ninstitutions", "formal\neligibility", "access\nmediated",
         "crisis support\nvia intermediaries", "quantitative\nevidence", "direct\ncrisis evidence"],
        fontsize=8)
    ax.set_yticks(range(len(PERIODS))); ax.set_yticklabels([PERIOD_LABEL[p] for p in PERIODS], fontsize=8.5)
    for i in range(len(PERIODS)):
        for j in range(6):
            ax.text(j, i, "yes" if M[i, j] else "no", ha="center", va="center",
                    fontsize=7.5, color="#222")
    ax.set_title("Continuity index: six criteria scored per period", fontsize=10.5)
    scores = [sum(CONTINUITY[p][:6]) for p in PERIODS]
    cols = [BROAD if p == "1997 reform" else MED for p in PERIODS]
    axb.barh(range(len(PERIODS)), scores, color=cols, alpha=0.85)
    axb.set_yticks(range(len(PERIODS))); axb.set_yticklabels([])
    axb.invert_yaxis(); axb.set_xlim(0, 6.5); axb.set_xlabel("continuity score (0-6)", fontsize=9)
    for i, s in enumerate(scores):
        axb.text(s + 0.1, i, str(s), va="center", fontsize=9)
    axb.set_title("Score (1997: exclusivity ends)", fontsize=10)
    axb.grid(axis="x", alpha=0.25)
    fig.suptitle("The access structure scores high from 1847 to 1996; exclusive discount-house dealing "
                 "ends at the 1997 reform", fontsize=11.5, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(FIG / "v6_continuity_index.png", dpi=150, bbox_inches="tight"); plt.close()
    print("wrote v6_continuity_index.csv + .png")
    return df


# ---------------------------------------------------------------- 2. evidence strength
def build_evidence_strength():
    rows = [
        dict(section="Victorian crises (Part I)", evidence_type="quantitative transaction data (BoE ledgers)",
             granularity="firm-level", confidence="high"),
        dict(section="1906 bill market", evidence_type="quantitative structured data (23,493 bills)",
             granularity="firm-level", confidence="high"),
        dict(section="1931 sterling crisis", evidence_type="semi-quantitative archival reconstruction (BoE Archive)",
             granularity="institution-level, described", confidence="medium-high"),
        dict(section="1830-1997 discount market", evidence_type="qualitative institutional history (BoE 1967)",
             granularity="structural", confidence="medium"),
        dict(section="1973-75 lifeboat", evidence_type="official narrative + structured appendix (BoE 1978)",
             granularity="named institutions + totals", confidence="medium-high"),
        dict(section="1976-96 discount houses", evidence_type="quantitative aggregate time series (BoE database)",
             granularity="sector aggregate", confidence="high (sector, not firm)"),
        dict(section="named-institution recurrence", evidence_type="illustrative cross-source matching",
             granularity="a few firms", confidence="medium (illustrative)"),
    ]
    pd.DataFrame(rows).to_csv(TBL / "v6_evidence_strength.csv", index=False)
    print("wrote v6_evidence_strength.csv")


# ---------------------------------------------------------------- 3. 1906 concentration + funnel
def build_1906():
    # source-reported values (Accominotti, Lucena-Piquero & Ugolini 2025; arXiv:2103.01558)
    rows = [
        dict(stage="drawers (borrowers worldwide)", count=3554, note="originate the bill"),
        dict(stage="acceptors (acceptance houses)", count=1439, note="guarantee the bill; top 15 ~35% of accepting"),
        dict(stage="discounters (discount houses)", count=145, note="buy/hold the bill; top 15 >70% of discounting"),
        dict(stage="eligible houses at the Bank", count=np.nan, note="the Bank rediscounts only eligible discount houses (~2.47% of all 1906 bills)"),
    ]
    df = pd.DataFrame(rows)
    df.to_csv(TBL / "v6_1906_bill_market_concentration.csv", index=False)
    # access funnel figure
    stages = [("drawers\n(borrowers)", 3554), ("acceptors\n(acceptance houses)", 1439),
              ("discounters\n(discount houses)", 145), ("top-15 discounters\n(>70% share)", 22)]
    fig, ax = plt.subplots(figsize=(9.5, 5.2)); ax.set_xlim(-5, 5); ax.set_ylim(0, len(stages) + 0.5); ax.axis("off")
    maxc = stages[0][1]
    for i, (lab, c) in enumerate(stages):
        y = len(stages) - i
        half = 0.3 + 4.4 * (c / maxc) ** 0.5  # sqrt so 145 is still visible
        col = [MED, "#4C72B0", "#937860", GOLD][i]
        ax.add_patch(Polygon([(-half, y - 0.42), (half, y - 0.42), (half * 0.85, y + 0.42),
                              (-half * 0.85, y + 0.42)], closed=True, fc=col, ec="white", alpha=0.85))
        ax.text(0, y, f"{lab}", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        ax.text(half + 0.3, y, f"n = {c if i < 3 else '~22'}", ha="left", va="center", fontsize=9, color="#333")
        if i < len(stages) - 1:
            ax.annotate("", xy=(0, y - 0.55), xytext=(0, y - 0.42),
                        arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#777"))
    ax.text(0, 0.35, "and the Bank rediscounts only ELIGIBLE discount houses",
            ha="center", fontsize=8.6, color=MED, style="italic")
    ax.set_title("The 1906 access funnel: many borrowers, fewer acceptors, very few discounters\n"
                 "(of 23,493 rediscounted bills; Accominotti, Lucena-Piquero & Ugolini 2025)",
                 fontsize=11)
    plt.savefig(FIG / "v6_1906_access_funnel.png", dpi=150, bbox_inches="tight"); plt.close()
    print("wrote v6_1906_bill_market_concentration.csv + v6_1906_access_funnel.png")


# ---------------------------------------------------------------- 4. 1931 actor flow + network
def build_1931():
    rows = [
        dict(actor="Big Five clearing banks", actor_type="clearing bank",
             role="supply spare cash as call money", amount="", mechanism="call money to discount houses",
             source="Romer 2025, Sec. I"),
        dict(actor="discount houses", actor_type="discount house",
             role="hold bills; borrow call money", amount="", mechanism="rediscount bills at the Bank",
             source="Romer 2025, Sec. I"),
        dict(actor="acceptance houses", actor_type="acceptance house",
             role="guarantee bills", amount="", mechanism="frozen acceptances rediscounted by the Bank",
             source="Romer 2025"),
        dict(actor="Seccombe, Marshall & Campion", actor_type="Bank's bill agent",
             role="execute the Bank's market bill transactions", amount="", mechanism="open-market agent",
             source="Romer 2025, Fig. 2 p.1190"),
        dict(actor="Bank of England", actor_type="central bank",
             role="apex; open-market operations", amount="~£25m Treasury bills (16 Jul-4 Aug)",
             mechanism="buys Treasury bills from discount houses; raises bankers' deposits",
             source="Romer 2025, pp.1197-98"),
    ]
    pd.DataFrame(rows).to_csv(TBL / "v6_1931_actor_flow.csv", index=False)
    # flow network (left-to-right)
    fig, ax = plt.subplots(figsize=(12.5, 4.4)); ax.set_xlim(0, 13); ax.set_ylim(0, 4); ax.axis("off")
    nodes = [("Big Five\nclearing banks", 1.3, GREEN), ("discount\nhouses", 4.3, MED),
             ("acceptance\nhouses", 7.3, BROAD), ("Bank of\nEngland", 10.6, "#222")]
    for lab, x, col in nodes:
        ax.add_patch(FancyBboxPatch((x - 1.0, 1.7), 2.0, 1.0, boxstyle="round,pad=0.05", fc=col, ec=col, alpha=0.9))
        ax.text(x, 2.2, lab, ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    edges = [(2.3, 3.3, "call money"), (5.3, 6.3, "bills"), (8.3, 9.6, "rediscount /\nOMO")]
    for x0, x1, lab in edges:
        ax.annotate("", xy=(x1, 2.2), xytext=(x0, 2.2), arrowprops=dict(arrowstyle="-|>", lw=2, color="#777"))
        ax.text((x0 + x1) / 2, 2.55, lab, ha="center", fontsize=7.6, color="#555")
    ax.scatter([10.6], [3.4], s=10, alpha=0)
    ax.text(10.6, 3.2, "via agent Seccombe, Marshall & Campion", ha="center", fontsize=7.4, color="#555", style="italic")
    ax.annotate("", xy=(10.6, 2.75), xytext=(10.6, 3.05), arrowprops=dict(arrowstyle="-|>", lw=1, color="#aaa"))
    ax.text(6.5, 0.9, "The Bank added liquidity by buying about £25m of Treasury bills from the discount houses "
            "(open-market operations). It did not lend to firms or the public directly.",
            ha="center", fontsize=8.4, color="#444")
    ax.set_title("1931: the Bank's support flowed through the intermediary tier (source-derived from Römer 2025)",
                 fontsize=11)
    plt.savefig(FIG / "v6_1931_flow_network.png", dpi=150, bbox_inches="tight"); plt.close()
    print("wrote v6_1931_actor_flow.csv + v6_1931_flow_network.png")


# ---------------------------------------------------------------- 5. 1973-75 lifeboat network
def build_1973():
    rows = [
        dict(node="Bank of England", role="Control Committee chair; 10% of the risk", group="rescuer"),
        dict(node="English & Scottish clearing banks", role="Control Committee; ~90% of the risk", group="rescuer"),
        dict(node="Control Committee ('the lifeboat')", role="governs entry and support", group="structure"),
        dict(node="related banks", role="each monitors one rescued company", group="structure"),
        dict(node="London & County Securities", role="trigger case, Nov 1973", group="rescued"),
        dict(node="Stern Group", role="collapse over £100m, 1974", group="rescued"),
        dict(node="First National Finance Corp.", role="large finance company", group="rescued"),
        dict(node="United Dominions Trust", role="large finance company", group="rescued"),
        dict(node="Slater Walker", role="Bank's own-account rescue (1977)", group="rescued"),
        dict(node="(26 secondary banks in total)", role="vetted recipients", group="rescued"),
    ]
    pd.DataFrame(rows).to_csv(TBL / "v6_1973_lifeboat_network.csv", index=False)
    # star network: Control Committee centre; rescuers above; rescued below
    fig, ax = plt.subplots(figsize=(11.5, 6.0)); ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis("off")
    cc = (6, 5)
    ax.add_patch(FancyBboxPatch((cc[0] - 1.7, cc[1] - 0.5), 3.4, 1.0, boxstyle="round,pad=0.05", fc="#333", ec="#333"))
    ax.text(cc[0], cc[1], "Control Committee\n(the 'lifeboat')", ha="center", va="center", color="white",
            fontsize=9.5, fontweight="bold")
    rescuers = [("Bank of England\n(10% of risk)", 3.5, GREEN), ("Clearing banks\n(~90% of risk)", 8.5, GREEN)]
    for lab, x, col in rescuers:
        ax.add_patch(FancyBboxPatch((x - 1.4, 6.6), 2.8, 0.9, boxstyle="round,pad=0.04", fc=col, ec=col, alpha=0.9))
        ax.text(x, 7.05, lab, ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
        ax.plot([x, cc[0]], [6.6, cc[1] + 0.5], color="#999", lw=1.4)
    rescued = ["London & County\nSecurities", "Stern Group", "First National\nFinance", "United\nDominions Trust",
               "Slater Walker\n(own account)"]
    xs = np.linspace(1.5, 10.5, len(rescued))
    for x, lab in zip(xs, rescued):
        ax.add_patch(FancyBboxPatch((x - 1.0, 2.0), 2.0, 0.9, boxstyle="round,pad=0.04", fc=PURPLE, ec=PURPLE, alpha=0.85))
        ax.text(x, 2.45, lab, ha="center", va="center", color="white", fontsize=7.4)
        ax.plot([x, cc[0]], [2.9, cc[1] - 0.5], color="#ccc", lw=1.0)
    ax.text(6, 1.3, "26 secondary banks vetted in; entry required 'sufficient banking characteristics' and "
            "'a significant level of deposits from the public'.\nSelective, Bank-organised access, not an open public facility.",
            ha="center", fontsize=8.3, color="#444")
    ax.text(6, 7.8, "Rescuers (the recognised core)", ha="center", fontsize=8.5, color=GREEN)
    ax.text(6, 3.15, "Rescued outsiders (the 'secondary' banks)", ha="center", fontsize=8.5, color=PURPLE)
    ax.set_title("The 1973-75 lifeboat: a selective network rescue, peak support £1,285m (Mar 1975)\n"
                 "(source-derived from Bank of England Quarterly Bulletin 1978 Q2)", fontsize=11)
    plt.savefig(FIG / "v6_1973_lifeboat_network.png", dpi=150, bbox_inches="tight"); plt.close()
    print("wrote v6_1973_lifeboat_network.csv + v6_1973_lifeboat_network.png")


# ---------------------------------------------------------------- 6. RPMATJD deposits analysis
def load_dh_csv():
    raw = (LIT / "dh_balance_sheet_1976-1996.csv").read_text().splitlines()
    start = next(i for i, l in enumerate(raw) if l.startswith("DATE,"))
    df = pd.read_csv(LIT / "dh_balance_sheet_1976-1996.csv", skiprows=start)
    df["DATE"] = pd.to_datetime(df["DATE"], format="%d %b %Y")
    return df.sort_values("DATE").reset_index(drop=True)


def build_deposits_analysis():
    df = load_dh_csv()
    s = df["RPMATJD"].astype(float)
    n = len(s); nz = int((s > 0).sum())
    # linear trend (months since start)
    x = np.arange(n)
    slope = np.polyfit(x, s, 1)[0]  # £m per month
    last12 = s.iloc[-12:].mean(); prev = s.iloc[:-12].mean()
    spikes = df.assign(RPMATJD=s).nlargest(3, "RPMATJD")[["DATE", "RPMATJD"]]
    summ = dict(
        series="RPMATJD (Bank sterling deposits placed with discount houses, £m)",
        months=n, range=f"{df['DATE'].min().date()} to {df['DATE'].max().date()}",
        nonzero_months=nz, nonzero_share=round(nz / n, 2),
        mean=round(s.mean(), 1), median=round(s.median(), 1), max=round(s.max(), 1),
        start_value=round(s.iloc[0], 1), end_value=round(s.iloc[-1], 1),
        trend_per_month=round(slope, 2), cv=round(s.std() / s.mean(), 2),
        last12m_mean=round(last12, 1), earlier_mean=round(prev, 1),
        declines_before_1997=("yes" if last12 < prev else "no"),
        reads_as=("routine but volatile use of the discount houses, not a one-off; "
                  "still active in the mid-1990s"),
    )
    pd.DataFrame([summ]).to_csv(TBL / "v6_discount_house_deposits_summary.csv", index=False)
    # figure
    fig, ax = plt.subplots(figsize=(11.5, 4.4))
    ax.bar(df["DATE"], s, width=20, color=MED, alpha=0.8)
    ax.axhline(s.mean(), color="#333", lw=1.2, ls="--")
    ax.text(df["DATE"].iloc[2], s.mean() + 40, f"mean £{s.mean():.0f}m", fontsize=8, color="#333")
    for _, r in spikes.iterrows():
        ax.annotate(f"£{r['RPMATJD']:.0f}m", xy=(r["DATE"], r["RPMATJD"]), fontsize=7.5,
                    ha="center", va="bottom", color="#900")
    ax.axvline(pd.Timestamp("1997-03-03"), color=BROAD, lw=1.6, ls="--")
    ax.text(pd.Timestamp("1997-03-03"), s.max() * 0.85, " 1997 reform\n (broadening)", fontsize=7.6, color=BROAD)
    ax.set_ylabel("£m placed with\ndiscount houses", fontsize=9)
    ax.set_title(f"The club still in use: Bank deposits with the discount houses, monthly, "
                 f"{df['DATE'].min().year}-{df['DATE'].max().year}\n"
                 f"non-zero in {nz}/{n} months (share {nz/n:.0%}); mean £{s.mean():.0f}m, max £{s.max():.0f}m; "
                 f"routine use right up to the 1997 reform", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG / "v6_discount_house_deposits_analysis.png", dpi=150, bbox_inches="tight"); plt.close()
    print("wrote v6_discount_house_deposits_summary.csv + v6_discount_house_deposits_analysis.png")
    return summ


# ---------------------------------------------------------------- 7. named recurrence scoring
def build_named_recurrence():
    rows = [
        dict(institution="Union Discount Co.", periods="Victorian; 1906; 1914; LDMA 1945; 1996",
             n_periods=5, actor_type="discount house", persisted="same firm",
             source="v3 ledgers; Accominotti 2025; BoE database", confidence="high"),
        dict(institution="Seccombe, Marshall & Campion", periods="1931; 1996",
             n_periods=2, actor_type="discount house / Bank agent", persisted="same firm",
             source="Romer 2025; BoE database", confidence="high"),
        dict(institution="National Discount Co.", periods="Victorian; 1906",
             n_periods=2, actor_type="discount house", persisted="same firm",
             source="v3 ledgers; Accominotti 2025", confidence="high"),
        dict(institution="Overend, Gurney & Co.", periods="1847; 1857; 1866",
             n_periods=3, actor_type="discount house", persisted="same firm (failed 1866)",
             source="v3 ledgers", confidence="high"),
        dict(institution="Gerrard & National", periods="LDMA; 1996",
             n_periods=2, actor_type="discount house / authorised counterparty", persisted="same firm",
             source="BoE database", confidence="medium"),
        dict(institution="Barings; Rothschilds", periods="Victorian; 1906",
             n_periods=2, actor_type="acceptance / merchant house", persisted="same firms",
             source="v3 context; Accominotti 2025", confidence="high"),
        dict(institution="DISCOUNT HOUSE (actor type)", periods="every period 1830-1997",
             n_periods=7, actor_type="type", persisted="TYPE persisted (the robust claim)",
             source="all sources", confidence="high"),
    ]
    df = pd.DataFrame(rows)
    df.to_csv(TBL / "v6_named_recurrence_score.csv", index=False)
    # figure
    d = df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 5.0))
    cols = [GOLD if "TYPE" in p else (MED if "discount" in t else BROAD)
            for p, t in zip(d["persisted"], d["actor_type"])]
    ax.barh(range(len(d)), d["n_periods"], color=cols, alpha=0.85)
    for i, (_, r) in enumerate(d.iterrows()):
        ax.text(r["n_periods"] + 0.05, i, f"  {r['persisted']}", va="center", fontsize=8)
    ax.set_yticks(range(len(d))); ax.set_yticklabels(d["institution"], fontsize=8.6)
    ax.set_xlabel("number of periods the name appears", fontsize=9.5); ax.set_xlim(0, 9)
    ax.set_title("Named recurrence: a few firms carry across periods, but the ROBUST continuity is the\n"
                 "actor TYPE (gold). The named firms are illustrative, not proof the same firms ran everything.",
                 fontsize=10.5)
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG / "v6_named_recurrence_score.png", dpi=150, bbox_inches="tight"); plt.close()
    print("wrote v6_named_recurrence_score.csv + v6_named_recurrence_score.png")


# ---------------------------------------------------------------- 8. 1996 counterparty network
def build_1996_network():
    cps = ["Alexanders", "Cater Allen", "Clive Discount", "Gerrard & National",
           "King & Shaxson", "Seccombe Marshall\n& Campion", "Union Discount"]
    fig, ax = plt.subplots(figsize=(8.5, 7)); ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.6); ax.axis("off")
    ax.scatter([0], [0], s=2200, color="#222", zorder=5)
    ax.text(0, 0, "Bank of\nEngland", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
    n = len(cps)
    for i, name in enumerate(cps):
        ang = 2 * np.pi * i / n + np.pi / 2
        x, y = np.cos(ang), np.sin(ang)
        carry = name.startswith(("Union", "Seccombe", "Gerrard"))
        col = MED if carry else "#937860"
        ax.plot([0, x], [0, y], color=col, lw=2, alpha=0.5, zorder=1)
        ax.scatter([x], [y], s=420, color=col, edgecolor="white", linewidth=1.5, zorder=4)
        ax.text(x * 1.28, y * 1.24, name, ha="center", va="center", fontsize=8.2)
    ax.set_title("1996: the Bank operated through seven authorised money-market counterparties\n"
                 "(red = a name that also appears earlier in the network; the club, now a formal list)",
                 fontsize=10.5)
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=MED, markersize=11, label="name recurs from earlier periods"),
               Line2D([0], [0], marker="o", color="w", markerfacecolor="#937860", markersize=11, label="other authorised counterparty")]
    ax.legend(handles=handles, loc="lower center", fontsize=8, frameon=False, ncol=2)
    plt.savefig(FIG / "v6_1996_counterparty_network.png", dpi=150, bbox_inches="tight"); plt.close()
    print("wrote v6_1996_counterparty_network.png")


# ---------------------------------------------------------------- 9. curated reference tables
def build_reference_tables():
    timeline = [
        dict(period="1847 to 1914 Victorian crises", mechanism="discount window through recurring counterparties",
             access="mediated", confidence="high"),
        dict(period="1906 prewar bill market", mechanism="acceptance and discount screening; Bank rediscounts eligible houses only",
             access="mediated", confidence="high"),
        dict(period="1931 sterling crisis", mechanism="open-market Treasury-bill purchases through discount houses",
             access="mediated", confidence="high"),
        dict(period="1830 to 1997 discount-market structure", mechanism="last-resort lending and daily operations only through LDMA members",
             access="mediated", confidence="high"),
        dict(period="1973 to 1975 lifeboat", mechanism="selective Control-Committee rescue of vetted institutions",
             access="mediated and selective", confidence="high"),
        dict(period="1976 to 1996 authorised counterparties", mechanism="Bank operates through the recognised discount houses",
             access="mediated", confidence="high"),
        dict(period="1997 onward (epilogue)", mechanism="Bank broadens to a wide range of counterparties",
             access="broader", confidence="medium"),
    ]
    pd.DataFrame(timeline).to_csv(TBL / "v6_access_timeline.csv", index=False)
    preds = [
        dict(prediction="The 1866 cable shrinks the insider-to-public lead", closed_network="no_change_expected",
             open_efficient="lead shrinks", finding="transatlantic lead GREW, 1857 (25d) to 1907 (68d)",
             verdict="against democratization"),
        dict(prediction="Early access shows up as profit in listed prices", closed_network="-",
             open_efficient="prices fall before rupture", finding="listed financials ROSE into every rupture; no footprint",
             verdict="not supported; no profit claim"),
        dict(prediction="Support stays routed through recognised intermediaries after 1914",
             closed_network="yes", open_efficient="no", finding="true in 1906, 1931, 1973-75, 1976-96 (continuity index 4-6/6)",
             verdict="institutional continuity"),
        dict(prediction="The same firms run the whole period", closed_network="strong form", open_efficient="-",
             finding="only a few names recur (Union Discount, Seccombe Marshall & Campion); membership turns over",
             verdict="NOT supported (we do not claim sameness)"),
        dict(prediction="Access becomes broad and open", closed_network="no", open_efficient="yes",
             finding="not observed before the 1997 reform (continuity index drops to 0)",
             verdict="only after 1997"),
    ]
    pd.DataFrame(preds).to_csv(TBL / "v6_predictions.csv", index=False)
    print("wrote v6_access_timeline.csv + v6_predictions.csv")


# ---------------------------------------------------------------- 10. conceptual figures
def fig_long_timeline():
    fig, ax = plt.subplots(figsize=(13, 4.6)); ax.set_xlim(1840, 2005); ax.set_ylim(0, 3.2); ax.axis("off")
    ax.text(1922, 3.0, "The Bank of England's crisis-access network, 1847 to 1997: access stayed mediated",
            ha="center", fontsize=13, fontweight="bold")
    ax.annotate("", xy=(2002, 1.4), xytext=(1842, 1.4), arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#555"))
    bands = [(1847, 1914, "Victorian crises\nledgers"), (1906, 1914, "1906 bill market"),
             (1931, 1932, "1931 crisis"), (1973, 1975, "1973-75 lifeboat"),
             (1976, 1997, "authorised\ncounterparties")]
    for i, (a, b, lab) in enumerate(bands):
        y = 1.4 + (0.55 if i % 2 == 0 else -0.55)
        ax.plot([a, max(b, a + 1.2)], [1.4, 1.4], color=MED, lw=7, alpha=0.6, solid_capstyle="butt")
        ax.scatter([(a + b) / 2], [1.4], s=40, color=MED, zorder=5)
        ax.annotate(lab, xy=((a + b) / 2, 1.4), xytext=((a + b) / 2, y), ha="center",
                    fontsize=8.2, color="#333", va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.7, color="#bbb"))
    ax.axvline(1866, color="#999", ls=":", lw=1); ax.text(1866, 2.5, "1866 cable", fontsize=7.5, ha="center", color="#888")
    ax.axvline(1914, color="#999", ls=":", lw=1); ax.text(1914, 2.5, "WWI", fontsize=7.5, ha="center", color="#888")
    ax.axvline(1997, color=BROAD, ls="--", lw=1.6); ax.text(1997, 2.2, "3 Mar 1997:\nBank broadens\ncounterparties", fontsize=7.6, ha="center", color=BROAD)
    for yr in [1850, 1875, 1900, 1925, 1950, 1975, 2000]:
        ax.text(yr, 1.15, str(yr), ha="center", fontsize=8, color="#666")
    ax.text(1922, 0.4, "Access stayed mediated (red) through a recognised intermediary layer for about 150 years; "
            "it broadened only in 1997 (blue).", ha="center", fontsize=8.5, color="#444")
    plt.savefig(FIG / "v6_long_timeline.png", dpi=150, bbox_inches="tight"); plt.close()
    print("wrote v6_long_timeline.png")


def fig_actor_transformation():
    fig, ax = plt.subplots(figsize=(12.5, 5.2)); ax.set_xlim(0, 13); ax.set_ylim(0, 6); ax.axis("off")
    ax.text(6.5, 5.7, "One network, changing form: the actors transform but the Bank still works through them",
            ha="center", fontsize=12.5, fontweight="bold")
    ax.add_patch(FancyBboxPatch((5.2, 4.4), 2.6, 0.8, boxstyle="round,pad=0.06", fc="#222", ec="#222"))
    ax.text(6.5, 4.8, "Bank of England", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
    chain = [("bill brokers", "early 19th c.", CATCOL["bill broker"]),
             ("discount houses", "1830s-1997", CATCOL["discount house"]),
             ("acceptance /\nmerchant houses", "19th c.-1930s", CATCOL["acceptance / merchant house"]),
             ("clearing banks", "1900s-", CATCOL["clearing bank"]),
             ("authorised\ncounterparties", "1980s-96", CATCOL["authorised counterparty"])]
    x = 0.5
    for i, (lab, per, col) in enumerate(chain):
        ax.add_patch(FancyBboxPatch((x, 2.2), 2.1, 1.2, boxstyle="round,pad=0.05", fc=col, ec=col, alpha=0.85))
        ax.text(x + 1.05, 2.95, lab, ha="center", va="center", color="white", fontsize=8.8, fontweight="bold")
        ax.text(x + 1.05, 2.42, per, ha="center", va="center", color="white", fontsize=7.2)
        ax.plot([x + 1.05, 6.5], [3.4, 4.4], color="#bbb", lw=0.8, zorder=0)
        if i < len(chain) - 1:
            ax.annotate("", xy=(x + 2.45, 2.8), xytext=(x + 2.15, 2.8), arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#888"))
        x += 2.45
    ax.add_patch(FancyBboxPatch((4.2, 0.5), 4.6, 1.0, boxstyle="round,pad=0.05", fc="#f3eef7", ec=PURPLE))
    ax.text(6.5, 1.0, "secondary banks (1970s): the rescued OUTSIDERS,\nwho define the network by exclusion",
            ha="center", va="center", fontsize=8.2, color=PURPLE)
    plt.savefig(FIG / "v6_actor_transformation.png", dpi=150, bbox_inches="tight"); plt.close()
    print("wrote v6_actor_transformation.png")


def main():
    build_reference_tables()
    ci = build_continuity_index()
    fig_long_timeline(); fig_actor_transformation()
    build_evidence_strength()
    build_1906()
    build_1931()
    build_1973()
    summ = build_deposits_analysis()
    build_named_recurrence()
    build_1996_network()
    print("\n=== continuity index ===")
    print(ci[["period", "continuity_score", "confidence"]].to_string(index=False))
    print("\n=== RPMATJD deposits summary ===")
    for k, v in summ.items():
        print(f"  {k}: {v}")
    print("\ndone.")


if __name__ == "__main__":
    main()
