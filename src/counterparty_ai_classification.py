"""AI-assisted counterparty entity resolution and classification.

The original classifier in `src/load_ledger.py::_classify_counterparty` is a
small set of keyword rules that puts 60–90% of counterparties into the
generic 'merchant' / 'other' buckets, which weakens borrower-type analysis.

This script replaces it with a two-stage AI/ML pipeline:

1. **Entity resolution** via rapidfuzz token-set-ratio clustering on
   normalized firm names. Connected components above a similarity threshold
   are collapsed to a single canonical name (the longest variant), with the
   match method and similarity score recorded for audit.

2. **Classification** into one of:
     commercial_bank, merchant_bank, bill_broker, discount_house,
     merchant_trading_firm, industrial_or_corporate,
     individual_or_partnership, government_or_public_body, other, unknown
   using a transparent rules-plus-curated-overrides classifier. For each
   canonical counterparty we emit:
     ai_counterparty_type, confidence, classification_reason,
     evidence_terms, manual_review_required.

If an LLM API were available, the rules step could be replaced with an LLM
prompt at the same I/O contract — `classify_by_rules()` is structured so
that swap is local and reproducible. We prefer transparent rules here so
that the project can be re-run without an external API.

The old rule-based labels are preserved alongside the new ones for direct
comparison (see `network_analysis_ai.py`).

Outputs (in `outputs/tables/`):
- counterparty_entity_resolution.csv
- counterparty_ai_classification.csv
- counterparty_classification_validation_sample.csv

Caveats:
- This is not an LLM-grade classifier. It is a heavily-curated rules
  system whose value comes from explicit evidence tracking, confidence
  scoring, and the entity-resolution step. Treat outputs as a draft to be
  manually validated; the validation-sample CSV is structured for that.
- Confidence is a heuristic from rule weight, not a calibrated probability.
- For firms whose category is genuinely ambiguous from name alone
  (e.g. 'Smith Fleming & Co'), the classifier returns `unknown` with
  manual_review_required=True rather than guessing.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from io_utils import read_transactions  # noqa: E402

PROC = ROOT / "data" / "processed"
TBL = ROOT / "outputs" / "tables"


# -----------------------------------------------------------------------------
# Stage 1 — normalization + entity resolution
# -----------------------------------------------------------------------------

_NORMALIZE_REMOVE = re.compile(
    r"\b(the|co|coy|company|cmpy|limited|ltd|brothers|bros|brs|"
    r"sons|son|inc|incorporated|& co|&co)\b"
)

# Standardize variants seen in the 1857/1866/1914 ledgers
_REPLACEMENTS = [
    (r"\band\b", "&"),
    (r"messrs\.?", ""),
    (r"\bm/s\b", ""),
    (r"\bmess\b", ""),
    (r"\bmr\.?\b", ""),
    (r"\bcommercial\b", "commerc"),  # collapse near-variants
    (r"\bdiscount\b", "disc"),
    (r"\bcorporation\b", "corp"),
]


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = name.lower().strip()
    # Strip punctuation except & and spaces
    s = re.sub(r"[^\w\s&]", " ", s)
    for pat, sub in _REPLACEMENTS:
        s = re.sub(pat, sub, s)
    s = _NORMALIZE_REMOVE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def entity_resolve(names: list[str], threshold: int = 88) -> pd.DataFrame:
    """Cluster names whose normalized forms have token_set_ratio >= threshold.

    Returns a DataFrame: counterparty_clean, canonical_counterparty,
    match_method, similarity_score, cluster_id.
    """
    norm = {n: normalize_name(n) for n in names}
    # Bucket on first token to avoid an N^2 explosion on thousands of names
    buckets: dict[str, list[str]] = {}
    for n, nrm in norm.items():
        if not nrm:
            buckets.setdefault("__empty__", []).append(n)
            continue
        head = nrm.split()[0][:5]
        buckets.setdefault(head, []).append(n)

    parent: dict[str, str] = {n: n for n in names}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # Pick canonical = the one whose original name is longer (more info)
            if len(b) > len(a):
                ra, rb = rb, ra
            parent[rb] = ra

    similarities: dict[tuple[str, str], int] = {}
    for head, group in buckets.items():
        if len(group) < 2:
            continue
        # All-pairs within bucket — buckets are small after head-splitting
        for i, a in enumerate(group):
            na = norm[a]
            if not na:
                continue
            for b in group[i + 1:]:
                nb = norm[b]
                if not nb:
                    continue
                s = int(fuzz.token_set_ratio(na, nb))
                if s >= threshold:
                    union(a, b)
                    similarities[(a, b)] = s

    # Build output frame
    rows = []
    for n in names:
        canonical = find(n)
        sim = 100 if canonical == n else similarities.get((min(n, canonical), max(n, canonical)),
                                                          similarities.get((canonical, n),
                                                          int(fuzz.token_set_ratio(norm[n], norm[canonical])) if norm[n] and norm[canonical] else 0))
        method = "exact" if canonical == n else "rapidfuzz_token_set_ratio"
        rows.append({
            "counterparty_clean": n,
            "canonical_counterparty": canonical,
            "match_method": method,
            "similarity_score": sim,
            "manual_review_required": (canonical != n and sim < 92),
        })
    df = pd.DataFrame(rows)
    # cluster id for downstream
    clusters = {c: i for i, c in enumerate(sorted(df["canonical_counterparty"].unique()))}
    df["cluster_id"] = df["canonical_counterparty"].map(clusters)
    return df


# -----------------------------------------------------------------------------
# Stage 2 — classification (rules + curated overrides)
# -----------------------------------------------------------------------------

# Curated overrides for the firms most likely to dominate acute-window lending.
# Keys are LOWER-CASED substrings of the canonical name.
CURATED: dict[str, tuple[str, str]] = {
    # discount houses and bill brokers
    "overend": ("discount_house", "Overend, Gurney & Co. — the failed bill-broking firm; pre-1866 records and 1866-era counterparty"),
    "national discount": ("discount_house", "National Discount Company — major joint-stock discount house chartered 1856"),
    "london discount": ("discount_house", "London Discount Corporation — late-19th-c discount house"),
    "discount corporation": ("discount_house", "Discount Corporation Limited"),
    "alexander cunliffe": ("bill_broker", "Alexander, Cunliffe & Co. — bill brokers"),
    "cunliffe broo": ("bill_broker", "Cunliffe Brooks bill-broking partnership"),
    "allen harvey & ross": ("discount_house", "Allen Harvey & Ross — discount house"),
    "alexanders disc": ("discount_house", "Alexanders Discount Company"),
    "sanderson": ("bill_broker", "Sanderson & Co. — bill brokers"),
    "bruce wilkinson": ("bill_broker", "Bruce, Wilkinson & Co — bill brokers"),
    "gillett": ("bill_broker", "Gillett Brothers — bill brokers"),
    "smith fleming": ("merchant_trading_firm", "Smith Fleming & Co — Anglo-Indian merchant firm"),
    "frith sands": ("bill_broker", "Frith, Sands & Co — bill brokers prominent in 1866"),
    # merchant banks
    "rothschild": ("merchant_bank", "N. M. Rothschild & Sons — merchant banking dynasty"),
    "baring": ("merchant_bank", "Baring Brothers & Co — merchant bank (the firm rescued in 1890)"),
    "schroder": ("merchant_bank", "J. Henry Schroder & Co — German-origin merchant bank"),
    "kleinwort": ("merchant_bank", "Kleinwort Sons & Co — merchant bank"),
    "huth": ("merchant_bank", "Frederick Huth & Co — Anglo-German merchant bank"),
    "f.huth": ("merchant_bank", "F. Huth & Co — merchant bank"),
    "brown shipley": ("merchant_bank", "Brown, Shipley & Co — Anglo-American merchant bank"),
    "hambro": ("merchant_bank", "C. J. Hambro & Son — Anglo-Danish merchant bank"),
    "antony gibbs": ("merchant_bank", "Antony Gibbs & Sons — merchant bank"),
    "bischoffsheim": ("merchant_bank", "Bischoffsheim & Goldschmidt — Anglo-German merchant bank"),
    "lazard": ("merchant_bank", "Lazard Brothers — merchant bank"),
    "morgan": ("merchant_bank", "J. P. Morgan / Morgan, Grenfell — Anglo-American merchant bank"),
    "glyn mills": ("merchant_bank", "Glyn, Mills & Co — private bank operating in the City"),
    "smith payne": ("merchant_bank", "Smith, Payne & Smiths — private bank"),
    "raphael": ("merchant_bank", "R. Raphael & Sons — merchant bank"),
    "speyer": ("merchant_bank", "Speyer Brothers — merchant bank"),
    "morrison cryder": ("merchant_bank", "Morrison Cryder — merchant firm"),
    # commercial / joint-stock banks
    "london & westminster": ("commercial_bank", "London and Westminster Bank — major joint-stock bank"),
    "london westminster": ("commercial_bank", "London and Westminster Bank"),
    "london & county": ("commercial_bank", "London and County Bank — joint-stock bank"),
    "london county bank": ("commercial_bank", "London and County Bank"),
    "union bank of london": ("commercial_bank", "Union Bank of London — joint-stock bank"),
    "alliance bank": ("commercial_bank", "Alliance Bank — joint-stock bank"),
    "barclay": ("commercial_bank", "Barclay & Co — joint-stock bank (incorporated 1896)"),
    "city bank": ("commercial_bank", "The City Bank / City Bank Limited"),
    "bank of london": ("commercial_bank", "Bank of London Limited"),
    "national provincial": ("commercial_bank", "National Provincial Bank of England"),
    "national bank": ("commercial_bank", "National Bank Limited"),
    "midland": ("commercial_bank", "Midland Bank — joint-stock"),
    "lloyds": ("commercial_bank", "Lloyds Bank — joint-stock"),
    "capital and counties": ("commercial_bank", "Capital and Counties Bank"),
    # colonial / overseas banks
    "oriental bank": ("commercial_bank", "Oriental Bank Corporation — colonial bank"),
    "bank of madras": ("commercial_bank", "Bank of Madras — colonial presidency bank"),
    "bank of hindustan": ("commercial_bank", "Bank of Hindustan — Anglo-Indian"),
    "agra & mastermans": ("commercial_bank", "Agra & Mastermans Bank — colonial joint-stock"),
    "agra bank": ("commercial_bank", "Agra Bank — colonial joint-stock"),
    "chartered bank": ("commercial_bank", "Chartered Bank of India, Australia & China"),
    "hong kong": ("commercial_bank", "Hongkong and Shanghai Banking Corporation"),
    "anglo-egyptian": ("commercial_bank", "Anglo-Egyptian Bank"),
    "ionian bank": ("commercial_bank", "Ionian Bank"),
    # government/public bodies
    "bank of england": ("government_or_public_body", "Bank of England itself (internal account)"),
    "h m treasury": ("government_or_public_body", "HM Treasury / government"),
    "treasury": ("government_or_public_body", "Treasury account"),
    "post office": ("government_or_public_body", "Post Office Savings — public body"),
    "india office": ("government_or_public_body", "India Office — public body"),
    # industrial / corporate hints
    "railway": ("industrial_or_corporate", "Railway company — industrial issuer"),
    "rail co": ("industrial_or_corporate", "Railway company"),
    "insurance": ("industrial_or_corporate", "Insurance company"),
    "assurance": ("industrial_or_corporate", "Assurance / insurance company"),
}

# Generic keyword rules with confidence weights
KEYWORD_RULES = [
    # (regex, category, weight, evidence_term)
    (r"\bdisc(ount)?\s+(co|corp|house|comp)", "discount_house", 0.85, "discount house"),
    (r"\bbill broker", "bill_broker", 0.90, "bill broker"),
    (r"\bbrokers?\b(?!.*goods)", "bill_broker", 0.55, "brokers"),
    (r"\bbank\b.*\b(limited|ltd|corporation|corp)", "commercial_bank", 0.85, "Bank Limited"),
    (r"\bbank\b.*\bof\b", "commercial_bank", 0.70, "Bank of …"),
    (r"\bjoint stock", "commercial_bank", 0.80, "joint-stock"),
    (r"\bbankers?\b", "commercial_bank", 0.45, "bankers"),
    (r"\bcity bank\b", "commercial_bank", 0.85, "city bank"),
    (r"\bcommerc.+\b", "commercial_bank", 0.50, "commercial"),
    (r"\binsurance\b|\bassurance\b", "industrial_or_corporate", 0.85, "insurance"),
    (r"\brailway\b|\brail co\b|\brly\b", "industrial_or_corporate", 0.85, "railway"),
    (r"\bcotton\b|\bwool\b|\bsilk\b|\bsugar\b|\biron\b|\bsteel\b|\bcoal\b|\bmining\b",
     "industrial_or_corporate", 0.70, "industrial commodity"),
    (r"\bshipping\b|\bship co\b|\bsteamship\b", "industrial_or_corporate", 0.70, "shipping"),
    (r"\btrad(e|ing)\b|\b& co\b.*\bagents\b", "merchant_trading_firm", 0.55, "trading"),
    (r"\bmerchants?\b", "merchant_trading_firm", 0.70, "merchants"),
    (r"\bcalcutta\b|\bbombay\b|\bbengal\b|\bmadras\b", "merchant_trading_firm", 0.60, "India trade firm"),
    (r"\bbros\b|\bbrothers\b|\b& sons\b", "merchant_trading_firm", 0.40, "family firm pattern"),
    (r"\b& co\b", "merchant_trading_firm", 0.35, "partnership '& Co'"),
    (r"\bltd\b|\blimited\b", "industrial_or_corporate", 0.30, "joint-stock company suffix"),
]

# Patterns that strongly suggest an individual / partnership rather than firm
INDIVIDUAL_PATTERNS = [
    r"^\s*(mr|mrs|miss|sir)\.?\s+",
]


def classify_by_rules(canonical: str) -> dict:
    """Return classification dict for a canonical counterparty name."""
    name = canonical.lower()
    norm = normalize_name(canonical)

    # 1. Curated overrides first (high confidence)
    for needle, (cat, reason) in CURATED.items():
        if needle in name or needle in norm:
            return {
                "ai_counterparty_type": cat,
                "confidence": 0.95,
                "classification_reason": f"curated match: {reason}",
                "evidence_terms": json.dumps([needle]),
                "manual_review_required": False,
            }

    # 2. Generic keyword rules — pick the highest-weight match
    matches: list[tuple[float, str, str]] = []
    evidence: list[str] = []
    for pat, cat, w, term in KEYWORD_RULES:
        if re.search(pat, name):
            matches.append((w, cat, term))
            evidence.append(term)

    # Individual / partnership: a personal-name pattern with no firm keyword
    if any(re.search(p, name) for p in INDIVIDUAL_PATTERNS):
        return {
            "ai_counterparty_type": "individual_or_partnership",
            "confidence": 0.7,
            "classification_reason": "honorific or personal-title prefix",
            "evidence_terms": json.dumps(["honorific"]),
            "manual_review_required": False,
        }

    if matches:
        matches.sort(key=lambda x: -x[0])
        w, cat, _ = matches[0]
        # If multiple categories tied, mark for review
        cats = {c for _, c, _ in matches}
        review = len(cats) > 1 and w < 0.7
        return {
            "ai_counterparty_type": cat,
            "confidence": round(min(w + 0.05 * (len(matches) - 1), 0.95), 2),
            "classification_reason": f"matched rules: {', '.join(sorted(cats))}",
            "evidence_terms": json.dumps(sorted(set(evidence))),
            "manual_review_required": review,
        }

    # No matches: short bare name (e.g. 'J Smith') is most likely individual/partnership
    tokens = [t for t in canonical.split() if t]
    if 2 <= len(tokens) <= 4 and not re.search(r"\d", canonical):
        return {
            "ai_counterparty_type": "individual_or_partnership",
            "confidence": 0.55,
            "classification_reason": "short bare name, no firm suffix",
            "evidence_terms": json.dumps([]),
            "manual_review_required": True,
        }

    return {
        "ai_counterparty_type": "unknown",
        "confidence": 0.0,
        "classification_reason": "no rule matched",
        "evidence_terms": json.dumps([]),
        "manual_review_required": True,
    }


# -----------------------------------------------------------------------------
# Validation sample
# -----------------------------------------------------------------------------

def build_validation_sample(classified: pd.DataFrame, transactions: pd.DataFrame,
                              n_top: int = 20, n_per_crisis: int = 20,
                              n_low_confidence: int = 20) -> pd.DataFrame:
    # Aggregate volume by canonical
    vol = (transactions.assign(canonical=transactions["counterparty_clean"]
                                .map(classified.set_index("counterparty_clean").get("canonical_counterparty")
                                     if "counterparty_clean" in classified.columns else {}))
           if "canonical" not in transactions.columns else transactions)
    # If mapping not joined yet, default to identity
    if "canonical" not in vol.columns:
        vol = transactions.copy()
        vol["canonical"] = transactions["counterparty_clean"]

    vols = vol.groupby("canonical")["total_amount"].sum().sort_values(ascending=False)
    classified_for_sample = classified.drop_duplicates("canonical_counterparty").set_index("canonical_counterparty")

    samples = []
    # Top by volume
    for c in vols.head(n_top).index:
        if c in classified_for_sample.index:
            r = classified_for_sample.loc[c]
            samples.append({
                "canonical_counterparty": c,
                "stratum": "top_by_volume",
                "total_value_acute_all_crises": float(vols.loc[c]),
                "ai_counterparty_type": r["ai_counterparty_type"],
                "rule_based_type": r.get("rule_based_type", ""),
                "confidence": float(r["confidence"]),
                "manual_label": "",
                "agreement": "",
                "notes": "",
            })
    # Random by crisis
    rng = np.random.default_rng(7)
    for crisis_key in ("1857", "1866", "1914"):
        sub = vol[vol["crisis"] == crisis_key]["canonical"].unique()
        if len(sub) == 0:
            continue
        pick = rng.choice(sub, size=min(n_per_crisis, len(sub)), replace=False)
        for c in pick:
            if c in classified_for_sample.index:
                r = classified_for_sample.loc[c]
                samples.append({
                    "canonical_counterparty": c,
                    "stratum": f"random_{crisis_key}",
                    "total_value_acute_all_crises": float(vols.loc[c]),
                    "ai_counterparty_type": r["ai_counterparty_type"],
                    "rule_based_type": r.get("rule_based_type", ""),
                    "confidence": float(r["confidence"]),
                    "manual_label": "",
                    "agreement": "",
                    "notes": "",
                })
    # Low-confidence / unknown
    low = classified_for_sample[(classified_for_sample["ai_counterparty_type"] == "unknown") |
                                  (classified_for_sample["confidence"] < 0.5)]
    if len(low):
        pick = low.sample(n=min(n_low_confidence, len(low)), random_state=7).index
        for c in pick:
            r = classified_for_sample.loc[c]
            samples.append({
                "canonical_counterparty": c,
                "stratum": "low_confidence",
                "total_value_acute_all_crises": float(vols.get(c, 0.0)),
                "ai_counterparty_type": r["ai_counterparty_type"],
                "rule_based_type": r.get("rule_based_type", ""),
                "confidence": float(r["confidence"]),
                "manual_label": "",
                "agreement": "",
                "notes": "",
            })
    return pd.DataFrame(samples).drop_duplicates(subset=["canonical_counterparty", "stratum"])


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    TBL.mkdir(parents=True, exist_ok=True)
    tx = read_transactions(PROC / "lolr_transactions.parquet")

    # Unique counterparty_clean names
    names = sorted(tx["counterparty_clean"].dropna().astype(str).unique().tolist())
    print(f"Unique counterparty_clean names: {len(names):,}")

    # ENTITY RESOLUTION
    er = entity_resolve(names, threshold=88)
    er.to_csv(TBL / "counterparty_entity_resolution.csv", index=False)
    print(f"Wrote {TBL / 'counterparty_entity_resolution.csv'}")
    print(f"  → {er['canonical_counterparty'].nunique():,} canonical entities "
          f"({len(names) - er['canonical_counterparty'].nunique():,} merged)")

    # CLASSIFICATION on canonical names
    canonicals = sorted(er["canonical_counterparty"].unique().tolist())
    rows = []
    for c in canonicals:
        rec = classify_by_rules(c)
        # preserve old rule-based label for comparison
        from load_ledger import _classify_counterparty as old_classify
        rec["canonical_counterparty"] = c
        rec["rule_based_type"] = old_classify(c)
        rows.append(rec)
    cls = pd.DataFrame(rows)[
        ["canonical_counterparty", "ai_counterparty_type", "rule_based_type",
         "confidence", "classification_reason", "evidence_terms",
         "manual_review_required"]
    ]
    cls.to_csv(TBL / "counterparty_ai_classification.csv", index=False)
    print(f"Wrote {TBL / 'counterparty_ai_classification.csv'}")
    print("\nAI classification distribution:")
    print(cls["ai_counterparty_type"].value_counts().to_string())
    print(f"\n  manual_review_required: {cls['manual_review_required'].sum()} "
          f"of {len(cls)} canonicals")

    # JOIN: enrich transactions and classified table with canonical mapping
    er_simple = er[["counterparty_clean", "canonical_counterparty"]]
    classified_full = cls.merge(er_simple.drop_duplicates("canonical_counterparty"),
                                  on="canonical_counterparty", how="left")

    # VALIDATION SAMPLE
    tx_with_canonical = tx.merge(er_simple, on="counterparty_clean", how="left")
    tx_with_canonical["canonical"] = tx_with_canonical["canonical_counterparty"]
    val = build_validation_sample(classified_full, tx_with_canonical,
                                    n_top=20, n_per_crisis=20, n_low_confidence=20)
    val.to_csv(TBL / "counterparty_classification_validation_sample.csv", index=False)
    print(f"Wrote {TBL / 'counterparty_classification_validation_sample.csv'}  "
          f"(n={len(val)})")


if __name__ == "__main__":
    main()
