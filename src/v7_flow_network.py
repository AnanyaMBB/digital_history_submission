"""v7 — Information-flow sequence + rigorous access-network graph analysis.

This module does two things the earlier "network" code did not.

1. INFORMATION-FLOW SEQUENCE (per crisis).
   For each crisis we order the actors by the date stress first reached them and
   measure the day gap on each link. For the three transaction-ledger crises
   (1857, 1866, 1914) the order is taken from the ledger itself: the date each
   actor TYPE first appears at the Bank's window (from v3_lead_lag_results.csv).
   For the remaining crises the order comes from the dated information chain
   (origin -> Bank Rate signal -> specialist press -> public rupture) in
   v5_information_chain.csv. Every link is labelled with its time difference in
   days, which is exactly the "union bank -> next institution -> ..." sequence
   with a measured lead on each edge.

   Output: outputs/tables/v7_flow_sequence.csv
           website/assets/flow-data.js  (FLOW_DATA global, read by the site)

2. ACCESS NETWORK AS A GRAPH (graph-theoretic, the rigorous bit).
   The earlier network_analysis.py only reported concentration on a star graph
   (HHI, Gini, top-k). It never built an actual graph or measured centrality.
   Here we build the recurrence network the paper's title promises: a bipartite
   institution x crisis-period graph, drawn from the documented named-recurrence
   data (sourced in outputs/tables/v6_named_recurrence_score.csv and
   v6_named_institutions.csv). We project it onto an institution-institution
   co-appearance graph (two firms are linked when they sit at the Bank across the
   same period, weighted by the number of shared periods) and compute degree,
   weighted degree, eigenvector and betweenness centrality. The question the
   measures answer is concrete: which firms are structurally central to the
   access network across 150 years, and is that core the discount houses?

   Output: outputs/tables/v7_access_network_centrality.csv
           outputs/tables/v7_access_network_edges.csv
           outputs/figures/v7_access_network.png
           (network block appended into website/assets/flow-data.js)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TBL = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures"
WEB = ROOT / "website" / "assets"

# ----------------------------------------------------------------------------
# Display vocabulary
# ----------------------------------------------------------------------------
ACTOR_LABEL = {
    "discount_house": "Discount houses",
    "bill_broker": "Bill brokers",
    "merchant_bank": "Acceptance / merchant houses",
    "clearing_or_joint_stock_bank": "Clearing & joint-stock banks",
    "foreign_or_colonial_financial": "Foreign & colonial houses",
    "industrial_or_commercial": "Industrial / commercial firms",
}
ACTOR_TIER = {
    "discount_house": "core",
    "bill_broker": "core",
    "merchant_bank": "recognised",
    "clearing_or_joint_stock_bank": "recognised",
    "foreign_or_colonial_financial": "outer",
    "industrial_or_commercial": "outer",
}
# A light, sourced named exemplar for each actor type within a crisis.
EXEMPLAR = {
    ("1847", "discount_house"): "Overend, Gurney & Co.",
    ("1857", "discount_house"): "Overend, Gurney & Co.",
    ("1866", "discount_house"): "Overend, Gurney (fails 1866)",
    ("1866", "merchant_bank"): "Barings; Rothschilds",
    ("1914", "discount_house"): "Union Discount; National Discount",
    ("1914", "merchant_bank"): "Frühling & Göschen; Stern Bros",
    ("1914", "clearing_or_joint_stock_bank"): "Joint-stock banks",
}

LEDGER_CRISES = {"1847", "1857", "1866", "1914"}

CRISIS_META = {
    "1847": dict(label="Commercial crisis", origin="endogenous", origin_place="London"),
    "1857": dict(label="Imported panic", origin="external", origin_place="New York"),
    "1866": dict(label="The Overend panic", origin="endogenous", origin_place="London"),
    "1873": dict(label="Imported panic", origin="external", origin_place="New York"),
    "1890": dict(label="The Baring crisis", origin="endogenous", origin_place="London"),
    "1907": dict(label="Imported panic", origin="external", origin_place="New York"),
    "1914": dict(label="The war shock", origin="external", origin_place="Continent"),
}


def _fmt(d: pd.Timestamp) -> str:
    return d.strftime("%-d %b %Y")


# ----------------------------------------------------------------------------
# 1. Information-flow sequence
# ----------------------------------------------------------------------------
def build_flow_sequences() -> dict:
    lead = pd.read_csv(TBL / "v3_lead_lag_results.csv", parse_dates=["first_borrow_date"])
    chain = pd.read_csv(TBL / "v5_information_chain.csv")

    flows: dict[str, dict] = {}
    rows = []

    for cy in ["1847", "1857", "1866", "1873", "1890", "1907", "1914"]:
        meta = CRISIS_META[cy]
        steps = []
        crow = chain[chain["crisis"].astype(str) == cy]
        rupture = None
        if not crow.empty and pd.notna(crow.iloc[0].get("stage5_public_rupture")):
            try:
                rupture = pd.to_datetime(crow.iloc[0]["stage5_public_rupture"])
            except (ValueError, TypeError):
                rupture = None

        if cy in LEDGER_CRISES:
            sub = lead[(lead["crisis"].astype(str) == cy) & (lead["actor_category"] != "unknown")]
            sub = sub.dropna(subset=["first_borrow_date"]).sort_values("first_borrow_date")
            mode = "At the Bank's window — ledger order"
            for _, r in sub.iterrows():
                cat = r["actor_category"]
                steps.append(
                    dict(
                        actor=ACTOR_LABEL.get(cat, cat),
                        exemplar=EXEMPLAR.get((cy, cat), ""),
                        tier=ACTOR_TIER.get(cat, "outer"),
                        date=r["first_borrow_date"],
                        kind="window",
                    )
                )
        else:
            mode = "From signal to public — information chain"
            # Signal is the anchor; intermediate stages are kept only if they
            # genuinely fall between the signal and the public rupture, then
            # ordered by date. This drops stray pre-signal coverage and
            # post-rupture monthly issues so every link reads as a forward lead.
            signal = None
            if not crow.empty and pd.notna(crow.iloc[0].get("stage2_bank_rate_signal")):
                try:
                    signal = pd.to_datetime(crow.iloc[0]["stage2_bank_rate_signal"])
                except (ValueError, TypeError):
                    signal = None
            if signal is not None:
                steps.append(
                    dict(actor="Bank of England (Bank Rate)", exemplar="", tier="core", date=signal, kind="chain")
                )
                mids = [
                    ("stage3_specialist_press", "Specialist City press", "recognised"),
                    ("stage4_public_press_surge", "Public newspapers", "outer"),
                ]
                buf = []
                for col, lab, tier in mids:
                    val = crow.iloc[0].get(col)
                    if pd.notna(val) and str(val).strip() and "n/a" not in str(val):
                        try:
                            d = pd.to_datetime(val)
                        except (ValueError, TypeError):
                            continue
                        if rupture is not None and signal <= d <= rupture:
                            buf.append(dict(actor=lab, exemplar="", tier=tier, date=d, kind="chain"))
                buf.sort(key=lambda s: s["date"])
                steps.extend(buf)

        # terminal public-rupture node
        if rupture is not None:
            steps.append(
                dict(actor="The wider public", exemplar="public rupture", tier="public", date=rupture, kind="public")
            )

        # compute day gaps on each link
        prev = None
        clean = []
        for s in steps:
            d = s["date"]
            gap = None if prev is None else int((d - prev).days)
            clean.append(
                dict(
                    actor=s["actor"],
                    exemplar=s["exemplar"],
                    tier=s["tier"],
                    date=_fmt(d),
                    iso=d.strftime("%Y-%m-%d"),
                    gap=gap,
                    kind=s["kind"],
                )
            )
            rows.append(dict(crisis=cy, **{k: clean[-1][k] for k in ("actor", "exemplar", "tier", "iso", "gap")}))
            prev = d

        total = None
        if len(clean) >= 2:
            total = int(
                (pd.to_datetime(clean[-1]["iso"]) - pd.to_datetime(clean[0]["iso"])).days
            )
        flows[cy] = dict(
            year=cy,
            label=meta["label"],
            origin=meta["origin"],
            origin_place=meta["origin_place"],
            mode=mode,
            steps=clean,
            span_days=total,
        )

    pd.DataFrame(rows).to_csv(TBL / "v7_flow_sequence.csv", index=False)
    return flows


# ----------------------------------------------------------------------------
# 2. Access network as a graph
# ----------------------------------------------------------------------------
# Institution -> set of periods, drawn from the documented v6 recurrence tables.
# Each membership is sourced in outputs/tables/v6_named_institutions.csv and
# v6_named_recurrence_score.csv (Victorian = the 1847-66 ledgers).
INSTITUTIONS = {
    "Overend, Gurney & Co.": dict(type="discount house", periods=["1847", "1857", "1866"]),
    "Union Discount Co.": dict(type="discount house", periods=["1866", "1906", "1914", "1945", "1996"]),
    "National Discount Co.": dict(type="discount house", periods=["1866", "1906"]),
    "Seccombe, Marshall & Campion": dict(type="discount house", periods=["1931", "1996"]),
    "Gerrard & National": dict(type="discount house", periods=["1945", "1996"]),
    "Cater Allen": dict(type="discount house", periods=["1996"]),
    "King & Shaxson": dict(type="discount house", periods=["1996"]),
    "Alexanders Discount": dict(type="discount house", periods=["1996"]),
    "Clive Discount": dict(type="discount house", periods=["1996"]),
    "Barings": dict(type="acceptance house", periods=["1866", "1890", "1906"]),
    "Rothschilds": dict(type="acceptance house", periods=["1866", "1906"]),
    "Kleinwort & Co.": dict(type="acceptance house", periods=["1906"]),
    "Frühling & Göschen": dict(type="acceptance house", periods=["1866", "1914"]),
    "Stern Bros": dict(type="acceptance house", periods=["1866", "1914"]),
    "Big Five clearing banks": dict(type="clearing bank", periods=["1931", "1973"]),
    "London & County Securities": dict(type="secondary bank", periods=["1973"]),
    "First National Finance": dict(type="secondary bank", periods=["1973"]),
    "United Dominions Trust": dict(type="secondary bank", periods=["1973"]),
}

TYPE_TIER = {
    "discount house": "core",
    "acceptance house": "recognised",
    "clearing bank": "recognised",
    "secondary bank": "outer",
}


def build_access_network():
    # bipartite graph institution <-> period
    B = nx.Graph()
    periods = sorted({p for v in INSTITUTIONS.values() for p in v["periods"]})
    for p in periods:
        B.add_node(("period", p), bipartite="period")
    for name, v in INSTITUTIONS.items():
        B.add_node(("inst", name), bipartite="inst", itype=v["type"])
        for p in v["periods"]:
            B.add_edge(("inst", name), ("period", p))

    inst_nodes = [n for n, d in B.nodes(data=True) if d.get("bipartite") == "inst"]

    # project onto institution-institution co-appearance, weighted by shared periods
    G = nx.Graph()
    for n in inst_nodes:
        G.add_node(n[1], itype=B.nodes[n]["itype"], periods=len(INSTITUTIONS[n[1]]["periods"]))
    for i, a in enumerate(inst_nodes):
        for b in inst_nodes[i + 1 :]:
            shared = set(INSTITUTIONS[a[1]]["periods"]) & set(INSTITUTIONS[b[1]]["periods"])
            if shared:
                G.add_edge(a[1], b[1], weight=len(shared), periods=sorted(shared))

    deg = dict(G.degree())
    wdeg = dict(G.degree(weight="weight"))
    try:
        eig = nx.eigenvector_centrality_numpy(G, weight="weight")
    except Exception:
        eig = {n: float("nan") for n in G}
    btw = nx.betweenness_centrality(G, weight=None)

    rows = []
    for n in G.nodes():
        rows.append(
            dict(
                institution=n,
                actor_type=G.nodes[n]["itype"],
                n_periods=G.nodes[n]["periods"],
                degree=deg[n],
                weighted_degree=wdeg[n],
                eigenvector_centrality=round(eig[n], 4),
                betweenness_centrality=round(btw[n], 4),
            )
        )
    cent = pd.DataFrame(rows).sort_values(
        ["eigenvector_centrality", "n_periods"], ascending=False
    )
    cent.to_csv(TBL / "v7_access_network_centrality.csv", index=False)

    edge_rows = [
        dict(a=a, b=b, shared_periods=d["weight"], periods=";".join(d["periods"]))
        for a, b, d in G.edges(data=True)
    ]
    pd.DataFrame(edge_rows).to_csv(TBL / "v7_access_network_edges.csv", index=False)

    _draw_network(G, eig, cent)

    # JSON payload for the website
    type_color = {
        "discount house": "#1f6f54",
        "acceptance house": "#2c5f8a",
        "clearing bank": "#9a7d2e",
        "secondary bank": "#b04a3a",
    }
    pos = nx.spring_layout(G, weight="weight", seed=7, k=1.1)
    nodes = [
        dict(
            id=n,
            type=G.nodes[n]["itype"],
            tier=TYPE_TIER.get(G.nodes[n]["itype"], "outer"),
            periods=INSTITUTIONS[n]["periods"],
            n_periods=G.nodes[n]["periods"],
            eig=round(eig[n], 4),
            degree=deg[n],
            color=type_color.get(G.nodes[n]["itype"], "#777"),
            x=round(float(pos[n][0]), 3),
            y=round(float(pos[n][1]), 3),
        )
        for n in G.nodes()
    ]
    edges = [dict(a=a, b=b, w=d["weight"]) for a, b, d in G.edges(data=True)]
    summary = dict(
        n_institutions=G.number_of_nodes(),
        n_edges=G.number_of_edges(),
        density=round(nx.density(G), 3),
        periods=periods,
        most_central=cent.iloc[0]["institution"],
        most_central_eig=round(float(cent.iloc[0]["eigenvector_centrality"]), 3),
        core_share=round(
            sum(1 for n in G if G.nodes[n]["itype"] == "discount house") / G.number_of_nodes(), 2
        ),
    )
    return dict(nodes=nodes, edges=edges, summary=summary)


def _draw_network(G, eig, cent):
    type_color = {
        "discount house": "#1f6f54",
        "acceptance house": "#2c5f8a",
        "clearing bank": "#9a7d2e",
        "secondary bank": "#b04a3a",
    }
    pos = nx.spring_layout(G, weight="weight", seed=7, k=1.1)
    fig, ax = plt.subplots(figsize=(11, 8.5))
    for a, b, d in G.edges(data=True):
        x = [pos[a][0], pos[b][0]]
        y = [pos[a][1], pos[b][1]]
        ax.plot(x, y, color="#b9b2a4", lw=0.6 + 0.9 * d["weight"], alpha=0.55, zorder=1)
    for n in G.nodes():
        size = 220 + 2600 * eig[n]
        ax.scatter(
            pos[n][0], pos[n][1], s=size,
            color=type_color.get(G.nodes[n]["itype"], "#777"),
            edgecolors="white", linewidths=1.2, zorder=3,
        )
    for n in G.nodes():
        ax.text(
            pos[n][0], pos[n][1] - 0.085, n, ha="center", va="top",
            fontsize=7.5, color="#2b2b2b", zorder=4,
        )
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=10, label=t)
        for t, c in type_color.items()
    ]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=9, title="Actor type")
    top = cent.iloc[0]
    ax.set_title(
        "The access network as a graph: institutions linked when they sit at the Bank in a shared period\n"
        f"Node size = eigenvector centrality. Most central: {top['institution']} "
        f"({int(top['n_periods'])} periods). Edge width = shared periods.",
        fontsize=11, loc="left",
    )
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(FIG / "v7_access_network.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TBL.mkdir(parents=True, exist_ok=True)
    flows = build_flow_sequences()
    net = build_access_network()

    payload = "/* Generated by src/v7_flow_network.py — do not edit by hand. */\n"
    payload += "window.FLOW_DATA = " + json.dumps(flows, ensure_ascii=False, indent=2) + ";\n"
    payload += "window.ACCESS_NETWORK = " + json.dumps(net, ensure_ascii=False, indent=2) + ";\n"
    (WEB / "flow-data.js").write_text(payload, encoding="utf-8")

    print("Wrote outputs/tables/v7_flow_sequence.csv")
    print("Wrote outputs/tables/v7_access_network_centrality.csv")
    print("Wrote outputs/tables/v7_access_network_edges.csv")
    print("Wrote outputs/figures/v7_access_network.png")
    print("Wrote website/assets/flow-data.js")
    print()
    cent = pd.read_csv(TBL / "v7_access_network_centrality.csv")
    print("Top access-network institutions by eigenvector centrality:")
    print(cent.head(8).to_string(index=False))
    print()
    print("Network summary:", json.dumps(net["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
