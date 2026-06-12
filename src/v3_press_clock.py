"""
v3_press_clock.py
=================
Build a PRESS-PUBLIC visibility clock for the information-flows paper (paper_v3)
from the British Library "Heritage Made Digital" (HMD) newspapers dataset on
Hugging Face (biglam/hmd_newspapers).

The HMD corpus is an open, article-level set of 19th-century UK newspapers with
OCR text, publication date, title/location, and per-article OCR quality. We use
it to date the *first relevant newspaper coverage* of each crisis whose window it
covers (1847, 1857, 1866, 1890). It does NOT cover 1914, so 1914 keeps its
Hansard / official markers only.

Method (transparent and reproducible):
  1. Stream each of the 29 parquet shards, using a date predicate so only rows
     inside the four acute crisis windows are materialised (text is pulled only
     for those rows). Cache the result to a local parquet so keyword iteration
     does not re-stream 9.7 GB.
  2. For each crisis, keep articles inside its acute window with OCR quality
     >= MIN_OCR and word_count >= MIN_WORDS, then match a crisis-specific
     keyword cluster (named firms + crisis context) against the lower-cased text.
  3. The PRESS-PUBLIC MARKER is the earliest dated matching article. We record
     its date, newspaper, location, OCR quality, the matched anchors, and a
     snippet, plus coverage counts so the surge is visible.

Important framing (kept in the paper): this measures *public visibility* through
dated press records, not "public knowledge". HMD is a curated subset of titles,
so a press marker is a LOWER BOUND: the true first mention may be earlier in
un-digitised papers. Where HMD has no coverage for a window, we say so and fall
back to the parliamentary / official markers.

Run:  ./.venv/bin/python src/v3_press_clock.py
"""
from __future__ import annotations
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pds

ROOT = Path(__file__).resolve().parents[1]
OUT_TABLES = ROOT / "outputs" / "tables"
OUT_PRESS = ROOT / "outputs" / "press"
OUT_PRESS.mkdir(parents=True, exist_ok=True)
CACHE = OUT_PRESS / "hmd_crisis_windows.parquet"

REPO = "biglam/hmd_newspapers"
COLS = ["date", "title", "location", "word_count", "ocr_quality_mean", "text"]

MIN_OCR = 0.70      # drop low-confidence OCR
MIN_WORDS = 40      # drop fragments

# Acute crisis windows: aligned to the SAME episode the ledger / official /
# parliamentary clocks refer to (so "before/after" is a like-for-like comparison).
WINDOWS = {
    "1847": ("1847-09-01", "1848-01-31"),  # autumn panic; official suspension 25 Oct
    "1857": ("1857-08-01", "1858-02-28"),  # US trigger Aug; official suspension 12 Nov
    "1866": ("1866-04-01", "1866-08-31"),  # Overend fails 10 May; official 11 May
    "1890": ("1890-10-01", "1891-02-28"),  # Baring rescue public 15 Nov
}

# Crisis-ACUTE anchor clauses. An article "matches" if ANY clause matches.
# A clause is either:
#   - a regex string  -> matches if the pattern is found, OR
#   - a tuple of regex strings -> matches only if ALL patterns are present
#     in the article (used to require a named firm AND crisis language, so
#     routine pre-crisis mentions -- share listings, policy debates, market
#     columns -- do not count as crisis coverage).
# Anchors are deliberately tight: generic terms ("bank charter act",
# "money market", a bare firm name) are excluded because they appear routinely
# before the panic. Every resulting first-mention snippet is verified by hand.
# NB: each HMD "article" is in practice a full OCR'd newspaper PAGE (often
# thousands of words), so co-occurrence ANYWHERE on a page is meaningless. We
# therefore use either (a) distinctive exact crisis phrases that essentially do
# not occur outside crisis coverage, or (b) WITHIN-SENTENCE proximity regexes
# ([^.]{0,N} keeps the match inside one sentence). Generic words (bare "panic",
# "money market", a bare firm name) are excluded. Every first-mention snippet is
# verified by hand before it enters the paper.
ANCHORS = {
    "1847": [
        ("commercial distress", r"commercial distress"),
        ("commercial crisis", r"(?:great |general )?commercial crisis"),
        ("bank (charter) act suspended (same sentence)",
         r"suspen\w*[^.]{0,50}bank (?:charter )?act|bank (?:charter )?act[^.]{0,50}suspen"),
        ("royal bank of liverpool fails (same sentence)",
         r"royal bank of liverpool[^.]{0,60}(?:fail|stopp|suspen|clos)"
         r"|(?:fail|stopp|suspen|clos)\w*[^.]{0,60}royal bank of liverpool"),
    ],
    "1857": [
        ("western bank of scotland fails (same sentence)",
         r"western bank of scotland[^.]{0,80}(?:fail|stopp|suspen|clos|payment)"
         r"|(?:fail|stopp|suspen)\w*[^.]{0,80}western bank of scotland"),
        ("borough bank of liverpool", r"borough bank of liverpool"),
        ("american panic/crisis", r"american (?:panic|monetary crisis|commercial crisis)"),
        ("bank (charter) act suspended (same sentence)",
         r"suspen\w*[^.]{0,50}bank (?:charter )?act|bank (?:charter )?act[^.]{0,50}suspen"),
    ],
    "1866": [
        # Overend, Gurney within ONE sentence of acute failure language
        ("overend gurney + failure (same sentence)",
         r"overend[,. ]*(?:and |& )?gurney[^.]{0,120}"
         r"(?:suspend|suspension|stopp|fail|gazette|wind|liquidat|smash|crash)"
         r"|(?:suspend|suspension|stopp|fail|gazette|wind|liquidat|smash|crash)\w*"
         r"[^.]{0,120}overend[,. ]*(?:and |& )?gurney"),
        ("black friday", r"black friday"),
        ("bank (charter) act suspended (same sentence)",
         r"suspen\w*[^.]{0,50}bank (?:charter )?act|bank (?:charter )?act[^.]{0,50}suspen"),
    ],
    # 1890 retained for completeness; HMD has no coverage in this window, so it
    # yields no marker (the paper keeps official markers for 1890).
    "1890": [
        ("baring + crisis (same sentence)",
         r"baring[s']*[^.]{0,100}(?:argentine|argentina|guarantee fund|liabilit|"
         r"embarrass|rescue|difficult|suspen)"),
        ("guarantee fund", r"guarantee fund"),
    ],
}


def _date_filter():
    f = None
    for a, b in WINDOWS.values():
        clause = (pc.field("date") >= pa.scalar(datetime.fromisoformat(a))) & (
            pc.field("date") <= pa.scalar(datetime.fromisoformat(b + "T23:59:59"))
        )
        f = clause if f is None else (f | clause)
    return f


def fetch_windows() -> pd.DataFrame:
    """Stream all shards, pull only crisis-window rows, cache to parquet."""
    if CACHE.exists():
        print(f"[fetch] using cache {CACHE.name}")
        return pd.read_parquet(CACHE)

    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    shards = sorted(fs.glob(f"datasets/{REPO}/data/*.parquet"))
    print(f"[fetch] {len(shards)} shards; pulling 4 crisis windows from each")
    filt = _date_filter()
    parts = []
    for i, path in enumerate(shards):
        try:
            tbl = pds.dataset(path, filesystem=fs, format="parquet").to_table(
                filter=filt, columns=COLS
            )
            if tbl.num_rows:
                parts.append(tbl.to_pandas())
            print(f"  shard {i:02d}: {tbl.num_rows:6d} window rows")
        except Exception as e:  # noqa: BLE001  network / read resilience
            print(f"  shard {i:02d}: ERROR {type(e).__name__}: {e}")
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLS)
    df["date"] = pd.to_datetime(df["date"])
    df.to_parquet(CACHE, index=False)
    print(f"[fetch] cached {len(df):,} window articles -> {CACHE.name}")
    return df


def _clause_hit(low: pd.Series, clause):
    """Boolean Series: which articles satisfy this clause (str=any, tuple=all)."""
    if isinstance(clause, (tuple, list)):
        m = pd.Series(True, index=low.index)
        for rx in clause:
            m = m & low.str.contains(rx, regex=True, na=False)
        return m
    return low.str.contains(clause, regex=True, na=False)


def _clause_fires(text_low: str, clause) -> bool:
    if isinstance(clause, (tuple, list)):
        return all(re.search(rx, text_low) for rx in clause)
    return bool(re.search(clause, text_low))


def _first_pos(text_low: str, clause):
    rxs = clause if isinstance(clause, (tuple, list)) else [clause]
    positions = [m.start() for rx in rxs if (m := re.search(rx, text_low))]
    return min(positions) if positions else 0


def match_crisis(df: pd.DataFrame, crisis: str):
    a, b = WINDOWS[crisis]
    start, end = pd.Timestamp(a), pd.Timestamp(b) + pd.Timedelta(hours=23, minutes=59)
    win = df[(df["date"] >= start) & (df["date"] <= end)].copy()
    win = win[(win["ocr_quality_mean"] >= MIN_OCR) & (win["word_count"] >= MIN_WORDS)]
    n_window = len(win)
    if n_window == 0:
        return None, [], dict(crisis=crisis, n_window=0, n_match=0)

    low = win["text"].str.lower()
    mask = pd.Series(False, index=win.index)
    for _, clause in ANCHORS[crisis]:
        mask = mask | _clause_hit(low, clause)
    hits = win[mask].sort_values("date")
    n_match = len(hits)
    if n_match == 0:
        return None, [], dict(crisis=crisis, n_window=n_window, n_match=0)

    def describe(row):
        tl = row["text"].lower()
        fired = [lab for lab, cl in ANCHORS[crisis] if _clause_fires(tl, cl)]
        pos = min((_first_pos(tl, cl) for lab, cl in ANCHORS[crisis]
                   if _clause_fires(tl, cl)), default=0)
        snip = re.sub(r"\s+", " ",
                      row["text"][max(0, pos - 90): pos + 170].replace("\n", " ")).strip()
        return fired, snip

    # earliest 3 matches (for hand verification)
    earliest = []
    for _, r in hits.head(3).iterrows():
        fired, snip = describe(r)
        earliest.append(dict(date=r["date"].date().isoformat(),
                             paper=r["title"], anchors="; ".join(fired), snippet=snip))

    first = hits.iloc[0]
    fired, snip = describe(first)
    # press "saturation" peak: ISO week with the most matching articles
    hits_wk = hits.set_index("date").resample("W").size()
    peak_week = hits_wk.idxmax().date().isoformat() if len(hits_wk) else ""

    marker = dict(
        crisis=crisis,
        press_first_date=first["date"].date().isoformat(),
        newspaper=first["title"],
        location=first["location"],
        ocr_quality_mean=round(float(first["ocr_quality_mean"]), 3),
        word_count=int(first["word_count"]),
        matched_anchors="; ".join(fired),
        snippet=snip,
        press_peak_week=peak_week,
        n_articles_in_window=n_window,
        n_matching_articles=n_match,
    )
    return marker, earliest, dict(crisis=crisis, n_window=n_window, n_match=n_match)


def monthly_coverage(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for crisis in WINDOWS:
        a, b = WINDOWS[crisis]
        start, end = pd.Timestamp(a), pd.Timestamp(b) + pd.Timedelta(hours=23, minutes=59)
        win = df[(df["date"] >= start) & (df["date"] <= end)].copy()
        win = win[(win["ocr_quality_mean"] >= MIN_OCR) & (win["word_count"] >= MIN_WORDS)]
        if win.empty:
            continue
        low = win["text"].str.lower()
        mask = pd.Series(False, index=win.index)
        for _, clause in ANCHORS[crisis]:
            mask = mask | _clause_hit(low, clause)
        win["match"] = mask
        win["month"] = win["date"].dt.to_period("M").astype(str)
        g = win.groupby("month").agg(
            articles=("text", "size"), matching=("match", "sum")
        ).reset_index()
        g.insert(0, "crisis", crisis)
        rows.append(g)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main():
    df = fetch_windows()
    print(f"\n[match] {len(df):,} cached window articles "
          f"(OCR>={MIN_OCR}, words>={MIN_WORDS})")
    markers, summary = [], []
    for crisis in WINDOWS:
        m, earliest, s = match_crisis(df, crisis)
        summary.append(s)
        if m:
            markers.append(m)
            print(f"\n  {crisis}: PRESS marker {m['press_first_date']} "
                  f"(peak week {m['press_peak_week']}; "
                  f"{m['n_matching_articles']}/{m['n_articles_in_window']} match) "
                  f"-- {m['newspaper'][:45]} [{m['location'][:25]}]")
            print(f"        anchors: {m['matched_anchors']}")
            for e in earliest:
                print(f"        [{e['date']}] {e['anchors']}")
                print(f"            …{e['snippet'][:140]}…")
        else:
            print(f"\n  {crisis}: NO press coverage in HMD window "
                  f"(window articles={s['n_window']})")

    mdf = pd.DataFrame(markers)
    mdf.to_csv(OUT_TABLES / "v3_press_public_markers.csv", index=False)
    cov = monthly_coverage(df)
    cov.to_csv(OUT_TABLES / "v3_press_coverage_monthly.csv", index=False)
    print("\n[write] outputs/tables/v3_press_public_markers.csv")
    print("[write] outputs/tables/v3_press_coverage_monthly.csv")
    print("\nCoverage summary:")
    for s in summary:
        print(f"  {s['crisis']}: window={s['n_window']:5d}  matching={s['n_match']:4d}")


if __name__ == "__main__":
    main()
