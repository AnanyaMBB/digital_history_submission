"""paper_v3 (Information Flows) — analysis pipeline.

Builds the six v3_*.csv outputs from the Anson et al. (2017) transaction
ledgers for 1847, 1857, 1866, and 1914. A uniform keyword/curated classifier
is applied across ALL four crises so cross-crisis comparison is valid.

Source discipline:
- Transaction dates are stored in the processed parquet as microseconds since
  epoch (recovered with unit='us').
- 1847 is loaded directly from sheet 'B1. 1847 ledger' of the raw workbook.
- Counterparty types are EXPLORATORY (keyword/curated). They are lower-bound on
  the confident institutional categories. Flagged per row.
- "Public" / "official" anchor dates are literature-derived (see the dict
  below), NOT newspaper-OCR-derived; Chronicling America and the British
  newspaper archives were inaccessible (403 / subscription). This is documented
  in docs/source_notes/v3_newspaper_source_criticism.md.
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LOLR = ROOT / "data" / "raw" / "boe_lolr" / "lolr-historical-dataset.xlsx"
PARQUET = ROOT / "data" / "processed" / "lolr_transactions.parquet"
TBL = ROOT / "outputs" / "tables"
TBL.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Crisis anchor dates (literature-derived; see crisis_timeline.md, White 2016,
# Flandreau & Ugolini 2011, Bordo 1990). "official" = emergency action /
# major failure; "public" = the date the crisis was publicly unmistakable.
# ----------------------------------------------------------------------
ANCHORS = {
    "1847": {"official": "1847-10-25", "public": "1847-11-30",
             "label": "Bank Charter Act suspended (Treasury letter, ~23-25 Oct 1847)"},
    "1857": {"official": "1857-11-12", "public": "1857-12-04",
             "label": "Treasury Letter suspending the Bank Charter Act (12 Nov 1857)"},
    "1866": {"official": "1866-05-11", "public": "1866-05-11",
             "label": "Overend Gurney failure 10 May; Black Friday 11 May 1866"},
    "1914": {"official": "1914-08-06", "public": "1914-08-06",
             "label": "LSE closure 31 Jul; moratorium 6 Aug 1914"},
}

# Parliamentary (Hansard) public-record markers, RETRIEVED 2026 from
# api.parliament.uk/historic-hansard. These are dated public debates of each
# crisis -- a genuine public-record clock (parliamentary, NOT newspaper).
HANSARD = {
    "1847": {"date": "1847-11-30", "title": "Commercial Distress",
             "url": "https://api.parliament.uk/historic-hansard/commons/1847/nov/30"},
    "1857": {"date": "1857-12-04", "title": "Bank Issues Indemnity Bill",
             "url": "https://api.parliament.uk/historic-hansard/commons/1857/dec/04"},
    "1866": {"date": "1866-05-11", "title": "The Panic in the City; Suspension of the Bank Charter Act",
             "url": "https://api.parliament.uk/historic-hansard/commons/1866/may/11"},
    "1914": {"date": "1914-08-06", "title": "War in Europe; Currency and Bank Notes Bill",
             "url": "https://api.parliament.uk/historic-hansard/commons/1914/aug/06"},
}

# Press-public markers, produced by src/v3_press_clock.py from the British
# Library "Heritage Made Digital" newspapers dataset (biglam/hmd_newspapers).
# Loaded if present; HMD covers 1847/1857/1866/1890 but NOT 1914.
PRESS_CSV = ROOT / "outputs" / "tables" / "v3_press_public_markers.csv"


def load_press_markers():
    if not PRESS_CSV.exists():
        return {}
    p = pd.read_csv(PRESS_CSV, dtype=str)
    return {str(r["crisis"]): r.to_dict() for _, r in p.iterrows()}


# ----------------------------------------------------------------------
# Uniform counterparty classifier (curated + keyword). Applied to all crises.
# Categories: discount_house, bill_broker, merchant_bank,
#             clearing_or_joint_stock_bank, foreign_or_colonial_financial,
#             industrial_or_commercial, unknown
# ----------------------------------------------------------------------
DISCOUNT_HOUSES = [
    "overend gurney", "overend, gurney", "national discount", "london discount",
    "union discount", "consolidated discount", "joint stock discount",
    "discount corporation", "allen harvey", "alexanders discount", "hatimal discount",
    "general credit", "imperial discount",
]
BILL_BROKERS = [
    "alexander cunliffe", "alexanders cunliffe", "bruce wilkinson", "frith sands",
    "gillett", "sanderson", "smith st aubyn", "smith, st aubyn", "brightwen gillet",
]
MERCHANT_BANKS = [
    "baring", "rothschild", "hambro", "schroder", "kleinwort", "brown shipley",
    "huth", "bischoffsheim", "morgan", "glyn mills", "smith payne", "raphael",
    "antony gibbs", "antony gibb", "frühling", "fruhling", "goschen", "gibbs",
    "stern", "speyer", "seligman", "lazard", "wallace", "finlay",
]
# colonial / foreign banks: "bank of <place>" + chartered/colonial names
COLONIAL_PAT = re.compile(
    r"\b(bank of (australasia|hindustan|madras|montreal|new south wales|"
    r"ireland|scotland|egypt|india|africa|british)|chartered (bank|merchant bank)|"
    r"colonial|oriental bank|agra|mastermans|anglo[- ](egyptian|italian|"
    r"south american|austrian)|canadian bank|comptoir|credit lyonnais|"
    r"deutsche|dresdner|imperial ottoman|national bank of india|"
    r"standard bank|hongkong)\b", re.I)
# clearing / joint-stock banks
CLEARING_PAT = re.compile(
    r"\b(bank ltd|bank limited|bank, ltd|joint stock bank|"
    r"london (and|&) (county|westminster|provincial)|national provincial|"
    r"lloyds|lloyd's|barclay|midland|union bank|city bank|alliance bank|"
    r"capital (and|&) counties|martins|williams deacon|parr's|"
    r"london city|london joint stock|consolidated bank)\b", re.I)
INDUSTRIAL_PAT = re.compile(
    r"\b(railway|railroad|colliery|mills?|mining|steel|iron|gas (light|co)|"
    r"manufactur|spinning|cotton co|waterworks|dock co|steam|brewery)\b", re.I)


def classify(name: str) -> tuple[str, float]:
    if not isinstance(name, str) or not name.strip():
        return "unknown", 0.0
    n = name.lower()
    for k in DISCOUNT_HOUSES:
        if k in n:
            return "discount_house", 0.9
    for k in BILL_BROKERS:
        if k in n:
            return "bill_broker", 0.85
    for k in MERCHANT_BANKS:
        if k in n:
            return "merchant_bank", 0.85
    if COLONIAL_PAT.search(n):
        return "foreign_or_colonial_financial", 0.7
    if CLEARING_PAT.search(n):
        return "clearing_or_joint_stock_bank", 0.7
    if INDUSTRIAL_PAT.search(n):
        return "industrial_or_commercial", 0.6
    # bare bank mention
    if re.search(r"\bbank\b", n):
        return "clearing_or_joint_stock_bank", 0.4
    return "unknown", 0.2


INTERMEDIARY_CATS = {"discount_house", "bill_broker", "merchant_bank"}


def _clean(name) -> str:
    if not isinstance(name, str):
        return ""
    s = re.sub(r"\s+", " ", name).strip()
    return s


# ----------------------------------------------------------------------
# Load 1847 from raw sheet B1 (header row index 3, data from row 4).
# Columns: 0 date | 1 drawing office | 2 value brought | 3 rate-from |
#          4 rate-to | 5 loan value | 6 counterparty | 7 num bills |
#          8 value rejected | 9 num rejected | 10 remarks
# ----------------------------------------------------------------------
def load_1847() -> pd.DataFrame:
    raw = pd.read_excel(LOLR, sheet_name="B1. 1847 ledger", header=None, skiprows=4)
    raw = raw.iloc[:, :11]
    raw.columns = ["date_raw", "drawing_office", "value_brought", "rate_from",
                   "rate_to", "value_discounted", "counterparty_raw", "num_bills",
                   "value_bills_rejected", "num_bills_rejected", "remarks"]
    raw = raw[raw["date_raw"].notna() & raw["counterparty_raw"].notna()].copy()
    # The sheet mixes full month names ("1 January 1847") with abbreviations
    # ("2 Aug 1847", "13 Sept 1847"). Normalise, then parse per-element so
    # pandas does not lock onto a single inferred format and drop the rest.
    s = raw["date_raw"].astype(str).str.replace("Sept", "Sep", regex=False)
    raw["date"] = pd.to_datetime(s, errors="coerce", dayfirst=True, format="mixed")
    raw = raw[raw["date"].notna()]
    raw["crisis"] = "1847"
    raw["counterparty_clean"] = raw["counterparty_raw"].map(_clean)
    raw["rate"] = pd.to_numeric(raw["rate_from"], errors="coerce")
    raw["value_discounted"] = pd.to_numeric(raw["value_discounted"], errors="coerce")
    raw["total_amount"] = raw["value_discounted"]
    return raw[["date", "crisis", "counterparty_clean", "rate",
                "value_discounted", "total_amount"]]


def load_others() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET)
    df["date"] = pd.to_datetime(df["date"], unit="us")
    df["counterparty_clean"] = df["counterparty_clean"].fillna(
        df["counterparty_raw"]).map(_clean)
    # total_amount: prefer value_discounted, else total_amount
    df["total_amount"] = pd.to_numeric(df["total_amount"], errors="coerce").fillna(
        pd.to_numeric(df["value_discounted"], errors="coerce"))
    return df[["date", "crisis", "counterparty_clean", "rate",
               "value_discounted", "total_amount"]]


def main():
    d47 = load_1847()
    doth = load_others()
    tx = pd.concat([d47, doth], ignore_index=True)
    tx = tx[tx["counterparty_clean"].str.len() > 0].copy()
    tx["total_amount"] = pd.to_numeric(tx["total_amount"], errors="coerce").fillna(0.0)
    # classify
    cats, confs = zip(*tx["counterparty_clean"].map(classify))
    tx["actor_category"] = cats
    tx["class_confidence"] = confs
    print("rows per crisis:", tx.groupby("crisis").size().to_dict())

    # ---- v3_counterparty_classification.csv (one row per canonical name/crisis) ----
    cc = (tx.groupby(["counterparty_clean", "actor_category", "class_confidence"])
          .agg(n_transactions=("total_amount", "size"),
               total_amount=("total_amount", "sum"),
               crises=("crisis", lambda s: ",".join(sorted(s.unique()))))
          .reset_index()
          .sort_values("total_amount", ascending=False))
    cc.to_csv(TBL / "v3_counterparty_classification.csv", index=False)

    # ---- v3_lending_concentration.csv (per crisis, pre vs acute) ----
    rows = []
    for crisis, g in tx.groupby("crisis"):
        off = pd.Timestamp(ANCHORS[crisis]["official"])
        for window, sub in [("full", g),
                            ("pre_official", g[g["date"] < off]),
                            ("acute_official_on", g[g["date"] >= off])]:
            if len(sub) == 0:
                continue
            by = sub.groupby("counterparty_clean")["total_amount"].sum().sort_values(ascending=False)
            tot = by.sum()
            if tot <= 0:
                continue
            shares = (by / tot).values
            hhi = float((shares ** 2).sum() * 10000)
            # gini
            x = np.sort(by.values)
            nn = len(x)
            gini = float((2 * np.arange(1, nn + 1) - nn - 1).dot(x) / (nn * x.sum())) if x.sum() > 0 else np.nan
            catsh = sub.groupby("actor_category")["total_amount"].sum() / tot
            rows.append({
                "crisis": crisis, "window": window, "n_transactions": len(sub),
                "n_counterparties": int(by.size),
                "top5_share": round(float(shares[:5].sum()), 4),
                "top10_share": round(float(shares[:10].sum()), 4),
                "gini": round(gini, 4), "hhi": round(hhi, 1),
                "share_discount_house": round(float(catsh.get("discount_house", 0)), 4),
                "share_bill_broker": round(float(catsh.get("bill_broker", 0)), 4),
                "share_merchant_bank": round(float(catsh.get("merchant_bank", 0)), 4),
                "share_intermediaries": round(float(
                    catsh.get("discount_house", 0) + catsh.get("bill_broker", 0)
                    + catsh.get("merchant_bank", 0)), 4),
                "share_clearing_jointstock": round(float(catsh.get("clearing_or_joint_stock_bank", 0)), 4),
                "share_foreign_colonial": round(float(catsh.get("foreign_or_colonial_financial", 0)), 4),
            })
    pd.DataFrame(rows).to_csv(TBL / "v3_lending_concentration.csv", index=False)

    # ---- v3_lead_lag_results.csv (per crisis, per actor category) ----
    rows = []
    for crisis, g in tx.groupby("crisis"):
        off = pd.Timestamp(ANCHORS[crisis]["official"])
        pub = pd.Timestamp(ANCHORS[crisis]["public"])
        for cat, sub in g.groupby("actor_category"):
            first = sub["date"].min()
            med = sub["date"].median()
            wk = sub.set_index("date")["total_amount"].resample("W").sum()
            peak = wk.idxmax() if len(wk) and wk.max() > 0 else pd.NaT
            rows.append({
                "crisis": crisis, "actor_category": cat,
                "n_transactions": len(sub),
                "first_borrow_date": first.date().isoformat(),
                "median_borrow_date": med.date().isoformat() if pd.notna(med) else "",
                "peak_week": peak.date().isoformat() if pd.notna(peak) else "",
                "days_first_vs_official": int((first - off).days),
                "days_median_vs_official": int((med - off).days) if pd.notna(med) else None,
                "days_peak_vs_public": int((peak - pub).days) if pd.notna(peak) else None,
            })
    lead = pd.DataFrame(rows)
    # earliest actor category flag per crisis (intermediaries only, by median date)
    lead["earliest_intermediary_flag"] = ""
    for crisis in lead["crisis"].unique():
        m = lead[(lead["crisis"] == crisis)
                 & (lead["actor_category"].isin(INTERMEDIARY_CATS))
                 & (lead["n_transactions"] >= 3)]
        if len(m):
            idx = m["median_borrow_date"].idxmin()
            lead.loc[idx, "earliest_intermediary_flag"] = "earliest_intermediary"
    lead.to_csv(TBL / "v3_lead_lag_results.csv", index=False)

    # ---- v3_information_channel_scores.csv (per counterparty) ----
    rows = []
    amt_q75 = tx.groupby("crisis")["total_amount"].quantile(0.75).to_dict()
    for name, g in tx.groupby("counterparty_clean"):
        cat = g["actor_category"].mode().iloc[0]
        crises = sorted(g["crisis"].unique())
        early = 0
        for crisis, gg in g.groupby("crisis"):
            off = pd.Timestamp(ANCHORS[crisis]["official"])
            if (gg["date"] < off).any():
                early = 1
        per_crisis_tot = g.groupby("crisis")["total_amount"].sum()
        scale = int(any(per_crisis_tot.get(c, 0) > amt_q75.get(c, np.inf)
                        for c in per_crisis_tot.index))
        recurrence = len(g)
        cross = len(crises)
        intermediary = int(cat in INTERMEDIARY_CATS)
        total = float(g["total_amount"].sum())
        rows.append({
            "counterparty": name, "actor_category": cat,
            "crises": ",".join(crises), "n_crises": cross,
            "n_transactions": recurrence, "total_amount": round(total, 1),
            "early_access": early, "top_quartile_scale": scale,
            "intermediary_role": intermediary,
        })
    ic = pd.DataFrame(rows)
    # centrality = normalized total amount (weighted degree to the Bank)
    ic["centrality_norm"] = (ic["total_amount"] / ic["total_amount"].max()).round(4)
    # recurrence normalized
    ic["recurrence_norm"] = (ic["n_transactions"] / ic["n_transactions"].max()).round(4)
    ic["cross_crisis_norm"] = ((ic["n_crises"] - 1) / 3.0).round(4)
    ic["information_channel_score"] = (
        0.20 * ic["early_access"] + 0.15 * ic["recurrence_norm"]
        + 0.15 * ic["cross_crisis_norm"] + 0.15 * ic["top_quartile_scale"]
        + 0.15 * ic["intermediary_role"] + 0.20 * ic["centrality_norm"]).round(4)
    ic = ic.sort_values("information_channel_score", ascending=False)
    ic.to_csv(TBL / "v3_information_channel_scores.csv", index=False)

    # ---- v3_crisis_timeline.csv (the four clocks) ----
    press = load_press_markers()
    rows = []
    for crisis in ["1847", "1857", "1866", "1890", "1914"]:
        a = ANCHORS.get(crisis)
        pm = press.get(crisis, {})
        press_date = pm.get("press_first_date", "")
        press_src = (f"{pm.get('newspaper','')} ({pm.get('location','')})"
                     if pm else "")
        if crisis == "1890":
            rows.append({
                "crisis": "1890",
                "ledger_window": "no transaction ledger (Anson 2017 has none for 1890)",
                "ledger_first_intermediary_signal": "n/a (daily account books only)",
                "ledger_intermediary_peak_week": "n/a",
                "official_date": "1890-11-15",
                "official_event": "Lidderdale Guarantee Fund announced (public lifeboat)",
                "press_public_record_date": press_date,
                "press_source": press_src,
                "parliamentary_public_record_date": "",
                "parliamentary_debate_title": "(not searched: rescue concluded before Parliament debated)",
                "hansard_url": "",
                "note": "daily account books show steady multi-day Total-Assets build-up 8-19 Nov; rescue kept out of public view until 15 Nov (White 2016)",
            })
            continue
        g = tx[tx["crisis"] == crisis]
        off = pd.Timestamp(a["official"])
        inter = g[g["actor_category"].isin(INTERMEDIARY_CATS)]
        # first intermediary borrowing week with >=2 intermediary transactions
        wk = (inter.set_index("date")
              .groupby(pd.Grouper(freq="W")).size())
        sig = wk[wk >= 2]
        first_sig = sig.index.min().date().isoformat() if len(sig) else (
            inter["date"].min().date().isoformat() if len(inter) else "n/a")
        # intermediary peak week (volume) = the meaningful ledger clock
        inter_pk = (inter.set_index("date")["total_amount"].resample("W").sum())
        inter_peak = inter_pk.idxmax().date().isoformat() if len(inter_pk) and inter_pk.max() > 0 else "n/a"
        h = HANSARD.get(crisis, {})
        rows.append({
            "crisis": crisis,
            "ledger_window": f"{g['date'].min().date()} to {g['date'].max().date()}",
            "ledger_first_intermediary_signal": first_sig,
            "ledger_intermediary_peak_week": inter_peak,
            "official_date": a["official"],
            "official_event": a["label"],
            "press_public_record_date": press_date,
            "press_source": press_src,
            "parliamentary_public_record_date": h.get("date", ""),
            "parliamentary_debate_title": h.get("title", ""),
            "hansard_url": h.get("url", ""),
            "note": "ledger first signal = first week with >=2 intermediary transactions; press date = first relevant HMD newspaper article (lower bound); parliamentary date = Hansard debate (retrieved)",
        })
    pd.DataFrame(rows).to_csv(TBL / "v3_crisis_timeline.csv", index=False)

    # ---- v3_public_awareness_timeline.csv ----
    # Two marker types: OFFICIAL (emergency action / failure, standard chronology)
    # and PARLIAMENTARY (Hansard debate, RETRIEVED from api.parliament.uk).
    # Newspaper-OCR markers are NOT included (archives inaccessible; see
    # v3_newspaper_source_criticism.md).
    pub_rows = [
        {"crisis": "1847", "marker": "Bank Charter Act suspended", "date": "1847-10-25",
         "marker_type": "official", "retrieved": "no (standard chronology)",
         "source": "crisis_timeline.md", "note": "Treasury letter ~23-25 Oct 1847"},
        {"crisis": "1847", "marker": "Commons debate: 'Commercial Distress'", "date": "1847-11-30",
         "marker_type": "parliamentary_record", "retrieved": "YES (Hansard)",
         "source": "api.parliament.uk/historic-hansard/commons/1847/nov/30",
         "note": "longest debate of the day (44,702 words); Parliament recalled after the autumn suspension"},
        {"crisis": "1857", "marker": "Treasury Letter / Bank Charter Act suspension", "date": "1857-11-12",
         "marker_type": "official", "retrieved": "no (standard chronology)",
         "source": "crisis_timeline.md", "note": "official emergency action"},
        {"crisis": "1857", "marker": "Commons debate: 'Bank Issues Indemnity Bill'", "date": "1857-12-04",
         "marker_type": "parliamentary_record", "retrieved": "YES (Hansard)",
         "source": "api.parliament.uk/historic-hansard/commons/1857/dec/04",
         "note": "dominated the sitting (35,067 words); Parliament recalled 3 Dec 1857"},
        {"crisis": "1866", "marker": "Overend Gurney suspends payment 3:30pm", "date": "1866-05-10",
         "marker_type": "official", "retrieved": "no (locally read literature)",
         "source": "Flandreau & Ugolini 2011 pp.14-15", "note": "the central public failure event"},
        {"crisis": "1866", "marker": "Commons debate: 'The Panic in the City' + 'Suspension of the Bank Charter Act'", "date": "1866-05-11",
         "marker_type": "parliamentary_record", "retrieved": "YES (Hansard)",
         "source": "api.parliament.uk/historic-hansard/commons/1866/may/11",
         "note": "Parliament was already sitting, so it debated the panic the same day as Black Friday"},
        {"crisis": "1890", "marker": "Barings rescue (lifeboat) announced publicly", "date": "1890-11-15",
         "marker_type": "official", "retrieved": "no (locally read literature)",
         "source": "White 2016", "note": "crisis kept out of public view 8-14 Nov; no transaction ledger exists"},
        {"crisis": "1914", "marker": "Moratorium operative; extended bank holiday", "date": "1914-08-06",
         "marker_type": "official", "retrieved": "no (standard chronology)",
         "source": "crisis_timeline.md", "note": "public emergency"},
        {"crisis": "1914", "marker": "Commons debate: 'War in Europe' + 'Currency and Bank Notes Bill'", "date": "1914-08-06",
         "marker_type": "parliamentary_record", "retrieved": "YES (Hansard)",
         "source": "api.parliament.uk/historic-hansard/commons/1914/aug/06",
         "note": "Parliament already sitting; debated the war-finance emergency the same day"},
    ]
    # PRESS-PUBLIC markers (retrieved from the British Library HMD newspapers
    # dataset via src/v3_press_clock.py). Inserted if available. HMD covers
    # 1847/1857/1866/1890 but NOT 1914.
    for crisis, pm in load_press_markers().items():
        pub_rows.append({
            "crisis": crisis,
            "marker": f"First relevant HMD newspaper article ({pm.get('newspaper','')})",
            "date": pm.get("press_first_date", ""),
            "marker_type": "press_record",
            "retrieved": "YES (British Library HMD newspapers, Hugging Face)",
            "source": f"{pm.get('newspaper','')} [{pm.get('location','')}], OCR conf {pm.get('ocr_quality_mean','')}; anchors: {pm.get('matched_anchors','')}",
            "note": (f"lower bound: HMD is a curated subset of titles; "
                     f"{pm.get('n_matching_articles','?')}/{pm.get('n_articles_in_window','?')} "
                     f"window articles matched the crisis keyword cluster"),
        })
    pd.DataFrame(pub_rows).to_csv(TBL / "v3_public_awareness_timeline.csv", index=False)

    print("wrote v3 CSVs to", TBL)
    # quick print of the key timeline + concentration
    print("\n=== crisis timeline (3 clocks) ===")
    print(pd.read_csv(TBL / "v3_crisis_timeline.csv")[
        ["crisis", "ledger_intermediary_peak_week", "official_date",
         "parliamentary_public_record_date"]].to_string(index=False))
    print("\n=== concentration (full window) ===")
    cz = pd.read_csv(TBL / "v3_lending_concentration.csv")
    print(cz[cz.window == "full"][
        ["crisis", "top5_share", "hhi", "share_intermediaries"]].to_string(index=False))


if __name__ == "__main__":
    main()
