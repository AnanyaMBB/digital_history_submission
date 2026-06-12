"""Heuristic 'LOLR-likeness' score, redesigned per code review.

Per reviewer feedback, the score is split into two tiers so that crises with
no transaction-level ledger (here: 1890) are not penalised by mechanical
zeros on components they cannot be evaluated on. 1890 is explicitly marked
n/a on Tier B components.

Tier A – Balance-sheet test (applies to all 4 crises)
  + Lending expanded ≥ 2× baseline
  + Penalty rate ≥ +2pp (tightened from +1pp; 1890's +1.5pp no longer auto-passes)
  + Fast rate response (first rate rise ≤ 14 days into acute window)
  + Fast lending peak (peak lending ≤ 60 days into acute window)
  − Penalty rate near-zero (≤ +1pp) — anti-Bagehot signal of muted rate response

Tier B – Transaction-level test (1857, 1866, 1914 only — n/a for 1890)
  + Low rejection rate (≤ 10% by value) — note: low rejection rate is *not*
    direct evidence of collateral quality; it reflects what the Bank actually
    accepted. Renamed from "collateral screened" per review.
  + Broad market support (top-5 share ≤ 30%)
  − Highly concentrated (top-5 share > 50%)
  − Private-rescue tilt (share to merchant banks > 20%)

Each tier has a max of +4 (Tier A) and +2 (Tier B) and a min of −1 each.
The combined score is reported only for crises with Tier-B data; for 1890
we report Tier A separately.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "tables"

TIER_A_COMPONENTS = [
    "A1: + expanded ≥2×",
    "A2: + penalty rate ≥+2pp",
    "A3: + fast rate rise (≤14d)",
    "A4: + fast lending peak (≤60d)",
    "A5: − penalty rate ≤+1pp (muted)",
]
TIER_B_COMPONENTS = [
    "B1: + low rejection rate (≤10%)",
    "B2: + broad market support (top-5 ≤30%)",
    "B3: − highly concentrated (top-5 >50%)",
    "B4: − private-rescue tilt (mb share >20%)",
]


def score_tier_a(r: pd.Series) -> dict:
    s = {}
    s["A1: + expanded ≥2×"] = int(r.get("scale_ratio_lending", 0) >= 2)
    s["A2: + penalty rate ≥+2pp"] = int(r.get("penalty_rate_delta", 0) >= 2)
    s["A3: + fast rate rise (≤14d)"] = int(r.get("days_to_first_rate_rise", 999) <= 14)
    s["A4: + fast lending peak (≤60d)"] = int(r.get("days_to_peak_lending", 999) <= 60)
    s["A5: − penalty rate ≤+1pp (muted)"] = -int(r.get("penalty_rate_delta", 99) <= 1)
    s["tier_a_score"] = sum(s.values())
    return s


def score_tier_b(r: pd.Series, has_ledger: bool) -> dict:
    s: dict[str, object] = {}
    if not has_ledger:
        for k in TIER_B_COMPONENTS:
            s[k] = "n/a"
        s["tier_b_score"] = "n/a"
        return s
    rej = r.get("share_rejected_value")
    top5 = r.get("top5_share_acute")
    mb = r.get("share_to_merchant_banks")
    s["B1: + low rejection rate (≤10%)"] = int(pd.notna(rej) and rej <= 0.10)
    s["B2: + broad market support (top-5 ≤30%)"] = int(pd.notna(top5) and top5 <= 0.30)
    s["B3: − highly concentrated (top-5 >50%)"] = -int(pd.notna(top5) and top5 > 0.50)
    s["B4: − private-rescue tilt (mb share >20%)"] = -int(pd.notna(mb) and mb > 0.20)
    s["tier_b_score"] = sum(v for v in s.values() if isinstance(v, int))
    return s


def main() -> None:
    metrics = pd.read_csv(OUT_DIR / "crisis_metrics.csv", dtype={"crisis_key": str}).set_index("crisis_key")
    rows: dict[str, dict] = {}
    for key, r in metrics.iterrows():
        has_ledger = str(key) in {"1857", "1866", "1914"}
        rows[key] = {**score_tier_a(r), **score_tier_b(r, has_ledger)}
        if has_ledger:
            rows[key]["combined_score"] = rows[key]["tier_a_score"] + rows[key]["tier_b_score"]
        else:
            rows[key]["combined_score"] = "n/a"
    table = pd.DataFrame(rows)
    # Reorder rows
    order = TIER_A_COMPONENTS + ["tier_a_score"] + TIER_B_COMPONENTS + ["tier_b_score", "combined_score"]
    table = table.reindex(order)
    table.to_csv(OUT_DIR / "lolr_score.csv")
    print(f"Wrote {OUT_DIR / 'lolr_score.csv'}")
    print(table.to_string())
    print("\nMax: Tier A +4 / Tier B +2 / Combined +6. Min: Tier A −1 / Tier B −2.")
    print("1890 is reported on Tier A only — Tier B requires transaction-level data.")


if __name__ == "__main__":
    main()
