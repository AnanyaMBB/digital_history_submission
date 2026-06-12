"""
v5_information_club.py — "The Information Club": reconstruct the crisis
information chain (inner circle -> specialist -> public), test whether a stable
inner circle of intermediaries recurs at the front of it, and test whether the
1866 transatlantic cable shrank the insider-to-public lead.

Reuses prior outputs (no re-streaming):
  outputs/tables/v4_clock_table.csv          (specialist Bank-Rate / public / official + leads + origin)
  outputs/tables/v3_information_channel_scores.csv  (recurring actors + which crises)
  outputs/tables/v3_crisis_timeline.csv      (ledger first intermediary signal)
  data/processed/lolr_transactions.parquet   (ledger, for pre-rupture positioning)
  data/processed/imm/imm_financial_monthly.csv (Yale IMM financial-sector index, market-price proxy)

Discipline: we measure dated VISIBILITY / POSITIONING, never private knowledge or profit.

Outputs: outputs/tables/v5_*.csv
Run    : ./.venv/bin/python src/v5_information_club.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TBL = ROOT / "outputs" / "tables"

# 1866 transatlantic cable opened 27/28 July 1866. Cable era classification:
CABLE = {"1847": "pre", "1857": "pre", "1866": "transition",
         "1873": "post", "1890": "post", "1907": "post", "1914": "post"}
TRANSATLANTIC = {"1857", "1907"}  # the cleanest US-origin comparison pair


def load_clock():
    c = pd.read_csv(TBL / "v4_clock_table.csv")
    c["crisis"] = c["crisis"].astype(str)
    c["cable_era"] = c["crisis"].map(CABLE)
    return c


# ---------------------------------------------------------------- information chain
def information_chain(clk):
    """Order the stages of crisis visibility per crisis, with dates + stage gaps."""
    v3 = pd.read_csv(TBL / "v3_crisis_timeline.csv"); v3["crisis"] = v3["crisis"].astype(str)
    v3i = v3.set_index("crisis")
    rows = []
    for _, r in clk.iterrows():
        c = r["crisis"]
        ledger = ""
        if c in v3i.index:
            ledger = str(v3i.loc[c].get("ledger_first_intermediary_signal", "") or "")
            if ledger in ("n/a", "nan"):
                ledger = ""
        pub = pd.Timestamp(r["public_visibility_date"])
        spec = pd.to_datetime(r["specialist_signal_date"], errors="coerce")
        bm = pd.to_datetime(r.get("bankers_magazine_report", ""), errors="coerce")
        hmd = r.get("hmd_surge_month", "")
        hmd_dt = pd.to_datetime(hmd + "-15", errors="coerce") if isinstance(hmd, str) and hmd else pd.NaT
        led = pd.to_datetime(ledger, errors="coerce")
        # earliest known inner-circle/market signal
        early = min([d for d in [spec, led] if pd.notna(d)], default=pd.NaT)
        rows.append(dict(
            crisis=c, cable_era=r["cable_era"], origin=r["origin"],
            stage1_innercircle_ledger=ledger,
            stage2_bank_rate_signal=r["specialist_signal_date"],
            stage3_specialist_press=(bm.date().isoformat() if pd.notna(bm) else ""),
            stage4_public_press_surge=(hmd_dt.date().isoformat() if pd.notna(hmd_dt) else ""),
            stage5_public_rupture=r["public_visibility_date"],
            stage6_official_response=r["official_response_date"],
            earliest_market_signal=(early.date().isoformat() if pd.notna(early) else ""),
            lead_earliest_to_public_days=((pub - early).days if pd.notna(early) else np.nan),
            confidence=("high" if r["hmd_window_articles"] > 0 and c in {"1857", "1866"} else
                        ("medium" if (r["hmd_window_articles"] > 0 or pd.notna(bm)) else "low")),
        ))
    out = pd.DataFrame(rows)
    out.to_csv(TBL / "v5_information_chain.csv", index=False)
    return out


# ---------------------------------------------------------------- inner-circle recurrence
def inner_circle(clk):
    ic = pd.read_csv(TBL / "v3_information_channel_scores.csv")
    ic["crises_list"] = ic["crises"].astype(str).str.split(",")
    def era_set(lst):
        return {CABLE.get(x.strip(), "?") for x in lst}
    ic["eras"] = ic["crises_list"].apply(era_set)
    ic["spans_cable"] = ic["eras"].apply(
        lambda s: ("yes" if ({"pre"} & s or {"transition"} & s) and ("post" in s) else "no"))
    keep = ["counterparty", "actor_category", "crises", "n_crises", "n_transactions",
            "total_amount", "information_channel_score", "spans_cable"]
    rec = ic[keep].sort_values("information_channel_score", ascending=False).reset_index(drop=True)
    rec.to_csv(TBL / "v5_inner_circle_recurrence.csv", index=False)

    # "early access" summary from v3 (computed with correct ledger dates; the
    # processed transactions parquet has a known date-encoding bug, so we reuse
    # v3's early_access flag rather than recomputing positioning from bad dates).
    INT = {"discount_house", "bill_broker", "merchant_bank"}
    rows = []
    for n in (10, 20, 30):
        top = ic.sort_values("information_channel_score", ascending=False).head(n)
        ti = top[top["actor_category"].isin(INT)]
        rows.append(dict(top_n=n, intermediaries=len(ti),
                         with_early_access=int(ti["early_access"].sum()),
                         recurring_2plus_crises=int((ti["n_crises"] >= 2).sum()),
                         spans_cable=int((ti["crises"].astype(str).str.contains("1914") &
                                          ti["crises"].astype(str).str.contains(
                                              "1847|1857|1866")).sum())))
    pos = pd.DataFrame(rows)
    pos.to_csv(TBL / "v5_innercircle_earlyaccess.csv", index=False)
    return rec, pos


# ---------------------------------------------------------------- telegraph test
def telegraph_test(clk):
    rows = []
    for era_name, members in [("pre-cable (1847,1857)", ["1847", "1857"]),
                              ("transition (1866)", ["1866"]),
                              ("post-cable (1873,1890,1907,1914)", ["1873", "1890", "1907", "1914"])]:
        sub = clk[clk["crisis"].isin(members)]
        rows.append(dict(group=era_name, n=len(sub),
                         median_lead_sustained=sub["lead_time_days"].median(),
                         mean_lead_sustained=round(sub["lead_time_days"].mean(), 1),
                         median_lead_acute=sub["lead_time_acute_days"].median()))
    # also by origin x era
    for era in ["pre", "post"]:
        for orig in ["endogenous", "external"]:
            sub = clk[(clk["cable_era"] == era) & (clk["origin"] == orig)]
            if len(sub):
                rows.append(dict(group=f"{era}-cable x {orig}", n=len(sub),
                                 median_lead_sustained=sub["lead_time_days"].median(),
                                 mean_lead_sustained=round(sub["lead_time_days"].mean(), 1),
                                 median_lead_acute=sub["lead_time_acute_days"].median()))
    tt = pd.DataFrame(rows)
    tt.to_csv(TBL / "v5_telegraph_test.csv", index=False)

    # the clean transatlantic pair 1857 vs 1907
    pair = clk[clk["crisis"].isin(TRANSATLANTIC)][
        ["crisis", "cable_era", "lead_time_days", "lead_time_acute_days"]]
    pair.to_csv(TBL / "v5_transatlantic_pair.csv", index=False)
    return tt, pair


# ---------------------------------------------------------------- market-price proxy
def market_price_proxy(clk):
    imm = pd.read_csv(ROOT / "data/processed/imm/imm_financial_monthly.csv")
    imm["date"] = pd.to_datetime(imm["date"])
    fin = imm[imm["sector"].str.contains("financ", case=False, na=False)].copy()
    rows = []
    for _, r in clk.iterrows():
        c = r["crisis"]
        pub = pd.Timestamp(r["public_visibility_date"])
        if fin["date"].min() > pub or fin.empty:
            rows.append(dict(crisis=c, imm_available="no", note="IMM begins 1869"))
            continue
        pre = fin[(fin["date"] >= pub - pd.Timedelta(days=190)) & (fin["date"] <= pub)]
        if pre.empty:
            rows.append(dict(crisis=c, imm_available="no"))
            continue
        idx0 = pre.iloc[0]["mean_return_index"]
        idx1 = pre.iloc[-1]["mean_return_index"]
        run = (idx1 / idx0 - 1) * 100 if idx0 else np.nan
        # share of pre-rupture months with a declining sector
        decl = pre["share_declining"].mean()
        rows.append(dict(crisis=c, imm_available="yes",
                         months_before=len(pre),
                         fin_index_6mo_before=round(idx0, 1), fin_index_at_rupture=round(idx1, 1),
                         pre_rupture_return_pct=round(run, 1),
                         avg_share_declining=round(float(decl), 2) if pd.notna(decl) else np.nan,
                         reads_as=("financial sector already falling before the public rupture"
                                   if run < -1 else "no clear pre-rupture decline")))
    mp = pd.DataFrame(rows)
    mp.to_csv(TBL / "v5_market_price_proxy.csv", index=False)
    return mp


# ---------------------------------------------------------------- predictions table
def predictions():
    rows = [
        dict(prediction="Insider-to-public lead persists after 1866",
             closed_club="yes", efficient_market="no",
             finding="External-crisis lead did NOT shrink: 1857 (pre) 25d vs 1907 (post) 68d",
             verdict="supports closed club"),
        dict(prediction="Same inner-circle actors recur at the front",
             closed_club="yes", efficient_market="no",
             finding="Fruhling & Goschen (1857,1866,1914) & Stern Bros (1847,1914) span the cable; "
                     "discount/merchant houses dominate top access scores",
             verdict="supports closed club (limited: only 1914 post-cable ledger)"),
        dict(prediction="Specialist/market signals still precede public signals post-cable",
             closed_club="yes", efficient_market="no",
             finding="Bank-Rate signal precedes the public rupture in all post-cable crises (1-142d)",
             verdict="supports closed club"),
        dict(prediction="London-NY PRICE gap collapses after the cable",
             closed_club="no", efficient_market="yes",
             finding="Hoag (2006) / Richmond Fed: cable cut the lag from ~10 days to near zero",
             verdict="supports efficient market (price channel only)"),
        dict(prediction="Early signals become diffuse, not tied to a stable circle",
             closed_club="no", efficient_market="yes",
             finding="Not observed in the ledgers: the same intermediary layer recurs",
             verdict="not supported"),
    ]
    pd.DataFrame(rows).to_csv(TBL / "v5_predictions.csv", index=False)
    return pd.DataFrame(rows)


def main():
    clk = load_clock()
    chain = information_chain(clk)
    rec, pos = inner_circle(clk)
    tt, pair = telegraph_test(clk)
    mp = market_price_proxy(clk)
    preds = predictions()

    pd.set_option("display.width", 220)
    print("=== information chain (earliest market signal -> public) ===")
    print(chain[["crisis", "cable_era", "origin", "earliest_market_signal",
                 "stage5_public_rupture", "lead_earliest_to_public_days", "confidence"]].to_string(index=False))
    print("\n=== telegraph test ===")
    print(tt.to_string(index=False))
    print("\n=== transatlantic pair 1857 vs 1907 ===")
    print(pair.to_string(index=False))
    print("\n=== inner-circle recurrence (top 10) ===")
    print(rec.head(10)[["counterparty", "actor_category", "crises", "n_crises", "spans_cable",
                        "information_channel_score"]].to_string(index=False))
    print("\n=== inner-circle early access (from v3, correct dates) ===")
    print(pos.to_string(index=False))
    print("\n=== market-price proxy (Yale IMM) ===")
    print(mp.to_string(index=False))


if __name__ == "__main__":
    main()
