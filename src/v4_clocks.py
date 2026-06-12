"""
v4_clocks.py — "Before the Panic": three dated clocks per crisis and the
specialist-to-public lead time, for British financial crises 1847-1914.

This does NOT measure what anyone privately "knew." It measures dated VISIBILITY
signals and the gap between them:

  lead_time = public_visibility_date - specialist_signal_date

Three clocks per crisis:
  1. SPECIALIST / informed-observer clock  (market-facing records, available early)
       - Bank Rate first sustained defensive increase in the run-up (ALL 7 crises,
         daily Millennium series): the earliest dated market-facing reaction.
       - Bank of England ledger first intermediary borrowing signal (4 ledger
         crises; from v3 timeline) as a second specialist marker.
       - (Bankers' Magazine specialist press: see source assessment; partial.)
  2. PUBLIC VISIBILITY clock  (broad public record)
       - documented public-rupture / failure date, corroborated by the HMD
         newspaper coverage SURGE where the corpus covers the year.
  3. OFFICIAL RESPONSE clock  (decisive institutional action)
       - Bank Charter Act suspension / Treasury Letter / moratorium / rescue,
         or the Bank Rate crisis peak where there was no suspension (1873, 1907).

Crisis origin classification (tested hypothesis): crises born INSIDE London's
financial network (endogenous) should show a longer specialist-to-public lead
than crises arriving from OUTSIDE (imported / geopolitical).

Inputs : data/processed/bank_rate_daily.parquet
         outputs/tables/v3_crisis_timeline.csv      (ledger peaks, official, parliamentary)
         outputs/tables/v3_press_public_markers.csv (HMD press first/surge, 1847/57/66)
         outputs/press/hmd_v4_windows.parquet       (HMD broad windows, all 7; optional)
Outputs: outputs/tables/v4_*.csv
Run    : ./.venv/bin/python src/v4_clocks.py
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TBL = ROOT / "outputs" / "tables"
PRESS = ROOT / "outputs" / "press"
TBL.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Crisis metadata. Dates are well-sourced public/official markers (see
# crisis_timeline.md, Bordo 1990, Flandreau & Ugolini 2011, White 2016,
# Roberts 2013, and standard chronologies). Origin per the paper's spec.
# ----------------------------------------------------------------------
CRISES = {
    "1847": dict(origin="endogenous", label="Commercial / railway-credit distress",
                 public="1847-10-18", public_what="Bank failures peak (Royal Bank of Liverpool, 18 Oct); commercial distress",
                 official="1847-10-25", official_what="Treasury Letter suspends Bank Charter Act"),
    "1857": dict(origin="external", label="Transatlantic (US) panic transmitted to Britain",
                 public="1857-11-09", public_what="Western Bank of Scotland fails (9 Nov); UK panic",
                 official="1857-11-12", official_what="Treasury Letter suspends Bank Charter Act"),
    "1866": dict(origin="endogenous", label="Overend, Gurney / discount-market crisis",
                 public="1866-05-11", public_what="Black Friday panic (Overend fails 10 May 3:30pm)",
                 official="1866-05-11", official_what="Treasury Letter suspends Bank Charter Act"),
    "1873": dict(origin="external", label="International market shock / long-depression onset",
                 public="1873-11-07", public_what="Money-market crisis; Bank Rate driven to 9%",
                 official="1873-11-07", official_what="Bank Rate peak 9% (no suspension)"),
    "1890": dict(origin="endogenous", label="Barings / elite merchant-bank sovereign-debt exposure",
                 public="1890-11-15", public_what="Baring rescue / Guarantee Fund made public",
                 official="1890-11-15", official_what="Lidderdale Guarantee Fund announced (rescue)"),
    "1907": dict(origin="external", label="US-centred panic with international transmission",
                 public="1907-10-22", public_what="US panic (Knickerbocker, 22 Oct); transmitted to London",
                 official="1907-11-07", official_what="Bank Rate raised to 7% (defence of reserve)"),
    "1914": dict(origin="external", label="War / geopolitical shock",
                 public="1914-07-31", public_what="London Stock Exchange closes (31 Jul)",
                 official="1914-08-06", official_what="Moratorium operative; Currency and Bank Notes Act"),
}
ENDOGENOUS = {"1847", "1866", "1890"}

# Specialist City-press markers, READ from The Bankers' Magazine (London),
# `sim_` monthly issues on Internet Archive (verified quotes; see
# docs/source_notes/v4_specialist_press_assessment.md). The magazine is monthly
# and compiled at month-start, so for mid-month crises (1866, 1890) the first
# full report lands in the FOLLOWING issue -> a built-in monthly lag. It is
# therefore specialist-press CONFIRMATION (a lower bound on specialist awareness),
# not an early-warning lead; the early signal is the daily Bank Rate.
BANKERS_MAGAZINE = {
    "1857": dict(issue="1857-12", report_date="1857-12-01",
                 quote="\"the crisis of 1857 ... has not reached its culminating point\"; "
                       "notes Bank Rate 8-9% on 5 Nov and \"The suspension of the Bank Act\""),
    "1866": dict(issue="1866-06", report_date="1866-06-01",
                 quote="\"THE PANIC OF 1866 ... it reached its culminating point on the afternoon "
                       "of the 10th ... Overend, Gurney and Co., Limited, were ... compelled to close their doors\""),
    "1890": dict(issue="1890-12", report_date="1890-12-01",
                 quote="\"THE BARING CRISIS ... On Saturday, November 8th, the affairs of Messrs. "
                       "Baring Brothers & Co. were disclosed to the directors of the Bank of England\""),
}


# ---------------------------------------------------------------- Bank Rate clock
def bank_rate_signal(br: pd.DataFrame, public: pd.Timestamp):
    """First SUSTAINED Bank-Rate increase in the run-up to the public marker.

    Returns (signal_date, baseline_rate, peak_rate, peak_date). The signal is the
    earliest increase from which the rate climbs (net) to the crisis peak without
    falling back to the pre-crisis baseline -- i.e. the start of the defensive
    tightening that the market could observe well before the public rupture.
    """
    lo, hi = public - pd.Timedelta(days=300), public + pd.Timedelta(days=45)
    w = br[(br["date"] >= lo) & (br["date"] <= hi)].reset_index(drop=True)
    if w.empty:
        return (pd.NaT, np.nan, np.nan, pd.NaT)
    # crisis peak around the public marker
    near = w[(w["date"] >= public - pd.Timedelta(days=90)) & (w["date"] <= hi)]
    peak_rate = near["bank_rate"].max()
    peak_date = near.loc[near["bank_rate"].idxmax(), "date"]
    # calm baseline ~6-10 months before
    base_w = w[w["date"] <= public - pd.Timedelta(days=150)]
    baseline = base_w["bank_rate"].min() if not base_w.empty else w["bank_rate"].min()
    # step changes up to the peak
    seg = w[w["date"] <= peak_date].reset_index(drop=True)
    seg["chg"] = seg["bank_rate"].diff()
    chg = seg[seg["chg"].fillna(0) != 0].reset_index(drop=True)  # change-points only

    # (a) SUSTAINED tightening start: first increase above the calm baseline from
    #     which the rate never falls back below the immediately-prior level.
    sustained = pd.NaT
    for i in range(len(chg)):
        if chg.loc[i, "chg"] > 0 and chg.loc[i, "bank_rate"] > baseline:
            prior = chg.loc[i, "bank_rate"] - chg.loc[i, "chg"]
            after = seg[seg["date"] >= chg.loc[i, "date"]]["bank_rate"].min()
            if after >= prior - 1e-9:
                sustained = chg.loc[i, "date"]
                break

    # (b) ACUTE final run-up start: start of the last monotone non-decreasing
    #     climb to the peak (the first increase AFTER the last rate cut before peak).
    acute = pd.NaT
    last_cut = None
    for i in range(len(chg)):
        if chg.loc[i, "chg"] < 0:
            last_cut = chg.loc[i, "date"]
    after = chg[chg["chg"] > 0]
    if last_cut is not None:
        after = after[after["date"] > last_cut]
    if len(after):
        acute = after["date"].min()

    return (sustained, acute, float(baseline), float(peak_rate), peak_date)


# ---------------------------------------------------------------- HMD press clock
KEY_FAMILIES = {
    "panic_distress": r"\b(?:panic|commercial distress|commercial crisis|monetary crisis|money panic)\b",
    "money_market": r"\b(?:money market|discount market|bill brokers?|discount house)\b",
    "stress_terms": r"\b(?:suspension|failure|stoppage|stopp\w+|pressure|distrust|scarcity|reserve|"
                    r"drain of gold|run upon|crisis)\b",
    "firms_events": r"\b(?:overend|gurney|baring|barings|bank charter act|american panic|"
                    r"railway panic|knickerbocker|black friday)\b",
}
MIN_OCR, MIN_WORDS = 0.70, 40


def hmd_press_clocks(public_by_crisis):
    """Return per-crisis HMD press signals from the broad-window cache, if present:
    first sustained crisis mention, coverage-surge week, peak week, and monthly counts.
    Coverage is measured as the share of window articles matching any crisis family."""
    cache = PRESS / "hmd_v4_windows.parquet"
    out, monthly = {}, []
    if not cache.exists():
        return out, pd.DataFrame()
    df = pd.read_parquet(cache)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["ocr_quality_mean"] >= MIN_OCR) & (df["word_count"] >= MIN_WORDS)].copy()
    low = df["text"].str.lower()
    fam_hit = pd.Series(False, index=df.index)
    for rx in KEY_FAMILIES.values():
        fam_hit = fam_hit | low.str.contains(rx, regex=True, na=False)
    df["crisis_hit"] = fam_hit
    for c, (a, b) in public_by_crisis.items():
        s, e = pd.Timestamp(a), pd.Timestamp(b)
        win = df[(df["date"] >= s) & (df["date"] <= e)].copy()
        if win.empty:
            out[c] = dict(n_articles=0)
            continue
        win["month"] = win["date"].dt.to_period("M").astype(str)
        g = win.groupby("month").agg(articles=("text", "size"),
                                     crisis_articles=("crisis_hit", "sum")).reset_index()
        g["share"] = g["crisis_articles"] / g["articles"].clip(lower=1)
        g.insert(0, "crisis", c)
        monthly.append(g)
        hits = win[win["crisis_hit"]].sort_values("date")
        first = hits["date"].min() if len(hits) else pd.NaT
        # surge = month with max crisis-article share (>=3 articles base)
        gg = g[g["articles"] >= 3]
        surge_month = gg.loc[gg["share"].idxmax(), "month"] if len(gg) else (
            g.loc[g["crisis_articles"].idxmax(), "month"] if len(g) else None)
        peak_month = g.loc[g["crisis_articles"].idxmax(), "month"] if len(g) else None
        out[c] = dict(n_articles=int(len(win)),
                      n_crisis_articles=int(win["crisis_hit"].sum()),
                      first_mention=(first.date().isoformat() if pd.notna(first) else ""),
                      surge_month=surge_month, peak_month=peak_month)
    return out, (pd.concat(monthly, ignore_index=True) if monthly else pd.DataFrame())


def month_mid(m):
    return pd.Timestamp(m + "-15") if m else pd.NaT


def main():
    br = pd.read_parquet(ROOT / "data/processed/bank_rate_daily.parquet")
    br["date"] = pd.to_datetime(br["date"])
    # v3 timeline (ledger first signal / peak) where available
    v3 = pd.read_csv(TBL / "v3_crisis_timeline.csv")
    v3["crisis"] = v3["crisis"].astype(str)
    v3i = v3.set_index("crisis")
    press_windows = {c: (CRISES[c]["public"][:4] + "-01-01"
                         if False else _win(c)) for c in CRISES}
    hmd, monthly = hmd_press_clocks(press_windows)

    rows = []
    for c, m in CRISES.items():
        public = pd.Timestamp(m["public"])
        official = pd.Timestamp(m["official"])
        sustained, acute, base, peak, peak_date = bank_rate_signal(br, public)
        # ledger first intermediary signal (a second specialist marker, where available)
        ledger_sig = ""
        if c in v3i.index:
            v = v3i.loc[c]
            ledger_sig = str(v.get("ledger_first_intermediary_signal", "") or "")
            if ledger_sig in ("n/a", "nan"):
                ledger_sig = ""
        # PRIMARY specialist signal = sustained defensive tightening start
        # (the "first measurable market-facing signal"); ACUTE = stricter robustness.
        specialist = sustained
        h = hmd.get(c, {})
        bm = BANKERS_MAGAZINE.get(c, {})
        lead = (public - specialist).days if pd.notna(specialist) else np.nan
        lead_acute = (public - acute).days if pd.notna(acute) else np.nan
        rows.append(dict(
            crisis=c, origin=m["origin"], crisis_label=m["label"],
            specialist_signal_date=(specialist.date().isoformat() if pd.notna(specialist) else ""),
            specialist_acute_date=(acute.date().isoformat() if pd.notna(acute) else ""),
            specialist_ledger_date=ledger_sig,
            bank_rate_baseline=base, bank_rate_peak=peak,
            bank_rate_peak_date=(peak_date.date().isoformat() if pd.notna(peak_date) else ""),
            public_visibility_date=public.date().isoformat(), public_marker=m["public_what"],
            hmd_first_mention=h.get("first_mention", ""),
            hmd_surge_month=h.get("surge_month") or "",
            hmd_window_articles=h.get("n_articles", 0),
            bankers_magazine_report=bm.get("report_date", ""),
            official_response_date=official.date().isoformat(), official_marker=m["official_what"],
            lead_time_days=lead, lead_time_acute_days=lead_acute,
        ))
    clk = pd.DataFrame(rows)
    clk.to_csv(TBL / "v4_clock_table.csv", index=False)

    # lead-time summary by origin (both measures)
    summ = (clk.groupby("origin").agg(
                n=("lead_time_days", "count"),
                lead_sustained_mean=("lead_time_days", "mean"),
                lead_sustained_median=("lead_time_days", "median"),
                lead_acute_mean=("lead_time_acute_days", "mean"),
                lead_acute_median=("lead_time_acute_days", "median")).reset_index())
    summ.to_csv(TBL / "v4_lead_time_summary.csv", index=False)

    # crisis classification table
    cls = clk[["crisis", "origin", "crisis_label"]].copy()
    cls.to_csv(TBL / "v4_crisis_classification.csv", index=False)

    # specialist City-press markers (Bankers' Magazine, read locally)
    bm_rows = [dict(crisis=c, source="The Bankers' Magazine (London)", issue=v["issue"],
                    report_date=v["report_date"], verified_quote=v["quote"],
                    note="monthly, compiled at month-start -> lower bound / lagging for mid-month crises")
               for c, v in BANKERS_MAGAZINE.items()]
    pd.DataFrame(bm_rows).to_csv(TBL / "v4_specialist_press_markers.csv", index=False)

    # source-availability matrix (what evidence exists per crisis)
    avail = []
    for c in CRISES:
        avail.append(dict(
            crisis=c, origin=CRISES[c]["origin"],
            bank_rate_daily="yes",
            boe_ledger=("yes" if c in {"1847", "1857", "1866", "1914"} else "no"),
            hmd_general_press=("yes" if hmd.get(c, {}).get("n_articles", 0) > 0 else "no"),
            bankers_magazine=("yes" if c in BANKERS_MAGAZINE else "no"),
            hansard=("yes" if c in {"1847", "1857", "1866", "1914"} else "no"),
        ))
    pd.DataFrame(avail).to_csv(TBL / "v4_source_availability.csv", index=False)

    if not monthly.empty:
        monthly.to_csv(TBL / "v4_press_coverage_monthly.csv", index=False)

    # ---- console report ----
    pd.set_option("display.width", 220)
    print("=== v4 clock table ===")
    print(clk[["crisis", "origin", "specialist_signal_date", "specialist_acute_date",
               "public_visibility_date", "official_response_date",
               "lead_time_days", "lead_time_acute_days", "bank_rate_baseline",
               "bank_rate_peak", "hmd_window_articles"]].to_string(index=False))
    print("\n=== lead-time by origin (sustained = first measurable signal; acute = final run-up) ===")
    print(summ.to_string(index=False))
    print("\nSUSTAINED  endogenous:",
          sorted(clk[clk.origin == "endogenous"]["lead_time_days"].tolist()),
          "| external:", sorted(clk[clk.origin == "external"]["lead_time_days"].tolist()))
    print("ACUTE      endogenous:",
          sorted(clk[clk.origin == "endogenous"]["lead_time_acute_days"].tolist()),
          "| external:", sorted(clk[clk.origin == "external"]["lead_time_acute_days"].tolist()))


def _win(c):
    # broad press window matching v4_press_fetch
    W = {"1847": ("1847-04-01", "1848-01-31"), "1857": ("1857-05-01", "1858-02-28"),
         "1866": ("1865-11-01", "1866-08-31"), "1873": ("1873-05-01", "1874-02-28"),
         "1890": ("1890-05-01", "1891-02-28"), "1907": ("1907-04-01", "1908-01-31"),
         "1914": ("1914-02-01", "1914-11-30")}
    return W[c]


if __name__ == "__main__":
    main()
