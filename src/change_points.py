"""Change-point detection on key BoE series, cross-checked against crisis dates,
with a placebo distribution from random non-crisis windows.

Uses PELT (ruptures) on:
- weekly crisis_lending (discounts + advances)
- weekly banking_reserve_notes_coin
- weekly reserve_ratio
- daily bank_rate (Millennium D1)

The placebo test addresses a reviewer concern: counting change-points within
±90 days of *known* crisis dates is not independent discovery. The placebo
samples random 181-day windows from periods at least 365 days from any known
crisis, runs the same detector, and reports the distribution of CP counts.
We then check whether the crisis-window CP counts sit in the upper tail of
the placebo distribution.

Outputs:
- outputs/tables/change_points.csv          – every detected change-point
- outputs/tables/change_point_placebo.csv   – placebo distribution
"""
from __future__ import annotations
import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import ruptures as rpt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from io_utils import read_balance_sheet, read_bank_rate  # noqa: E402
from crisis_windows import CRISES  # noqa: E402

PROC = ROOT / "data" / "processed"
TBL_DIR = ROOT / "outputs" / "tables"

# Known crisis trigger dates as anchor points
ANCHORS = {key: c.acute_peak for key, c in CRISES.items()}


def _detect(series: pd.Series, penalty: float, model: str = "rbf") -> list[pd.Timestamp]:
    s = series.dropna()
    if len(s) < 30:
        return []
    arr = s.values.reshape(-1, 1)
    algo = rpt.Pelt(model=model).fit(arr)
    breaks = algo.predict(pen=penalty)
    # breaks contains indices of *next* segment starts; last is len(s)
    idx_dates = [s.index[i - 1] for i in breaks if 0 < i <= len(s)]
    return idx_dates


def _nearest_anchor(d: pd.Timestamp) -> tuple[str, int]:
    diffs = {k: abs((d - v).days) for k, v in ANCHORS.items()}
    k = min(diffs, key=diffs.get)
    return k, diffs[k]


def _placebo(all_cps: dict[str, list[pd.Timestamp]], time_range: tuple[pd.Timestamp, pd.Timestamp],
             n_samples: int = 1000, window_radius_days: int = 90,
             exclude_radius_days: int = 365, rng_seed: int = 42) -> pd.DataFrame:
    """Sample n_samples random center-dates from non-crisis periods, then count
    how many of the *already-detected* full-series change-points fall within
    ±window_radius_days of each center. This mirrors the crisis-window
    construction exactly and yields a null distribution.
    """
    import numpy as np
    rng = np.random.default_rng(rng_seed)
    t_min, t_max = time_range
    forbidden = [(d - pd.Timedelta(days=exclude_radius_days),
                  d + pd.Timedelta(days=exclude_radius_days)) for d in ANCHORS.values()]

    def _allowed(center: pd.Timestamp) -> bool:
        if center - pd.Timedelta(days=window_radius_days) < t_min:
            return False
        if center + pd.Timedelta(days=window_radius_days) > t_max:
            return False
        return all(not (lo <= center <= hi) for lo, hi in forbidden)

    span_days = (t_max - t_min).days
    samples: list[dict] = []
    attempts = 0
    while len(samples) < n_samples and attempts < n_samples * 50:
        attempts += 1
        offset = int(rng.integers(0, max(1, span_days)))
        center = t_min + pd.Timedelta(days=offset)
        if not _allowed(center):
            continue
        lo = center - pd.Timedelta(days=window_radius_days)
        hi = center + pd.Timedelta(days=window_radius_days)
        rec: dict = {"center": center.date().isoformat()}
        for name, cps in all_cps.items():
            rec[name] = sum(1 for d in cps if lo <= d <= hi)
        samples.append(rec)
    return pd.DataFrame(samples)


def main() -> None:
    TBL_DIR.mkdir(parents=True, exist_ok=True)
    bs = read_balance_sheet(PROC / "boe_balance_sheet.parquet")
    # Restrict to the long-run window covering all 4 crises
    bs = bs.loc["1855-01-01":"1916-12-31"]

    series = {
        "crisis_lending_weekly": bs["crisis_lending"],
        "reserve_weekly": bs["banking_reserve_notes_coin"],
        "reserve_ratio_weekly": bs["reserve_ratio"],
    }
    br = read_bank_rate(PROC / "bank_rate_daily.parquet").set_index("date").sort_index()
    br = br.loc["1855-01-01":"1916-12-31"]
    series["bank_rate_daily"] = br["bank_rate"]

    penalties = {
        "crisis_lending_weekly": 30,
        "reserve_weekly": 20,
        "reserve_ratio_weekly": 0.05,
        "bank_rate_daily": 5,
    }

    import numpy as np  # local to keep imports tidy
    all_cps: dict[str, list[pd.Timestamp]] = {}
    rows = []
    for name, s in series.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cps = _detect(s, penalties[name])
        all_cps[name] = cps
        for d in cps:
            crisis, gap = _nearest_anchor(d)
            rows.append({
                "series": name,
                "date": d.date().isoformat(),
                "nearest_crisis": crisis,
                "days_to_crisis_peak": gap,
            })

    df = pd.DataFrame(rows).sort_values(["series", "date"])
    df.to_csv(TBL_DIR / "change_points.csv", index=False)
    print(f"Wrote {TBL_DIR / 'change_points.csv'}")

    # Crisis-window CP counts
    crisis_summary = (
        df[df["days_to_crisis_peak"] <= 90]
        .groupby(["nearest_crisis", "series"])
        .size()
        .unstack(fill_value=0)
    )
    print("\nChange-points within ±90 days of each crisis (per series):")
    print(crisis_summary.to_string())

    # Placebo: same-radius windows centered on non-crisis dates
    print("\nRunning placebo (1000 random non-crisis ±90-day windows)…")
    t_min = min(s.index.min() for s in series.values())
    t_max = max(s.index.max() for s in series.values())
    placebo_df = _placebo(all_cps, (t_min, t_max), n_samples=1000)
    placebo_df.to_csv(TBL_DIR / "change_point_placebo.csv", index=False)
    print(f"  collected {len(placebo_df)} placebo windows")

    # Compare crisis CP counts to placebo distribution
    placebo_stats = []
    for name in series:
        placebo_vals = placebo_df[name].values
        for key in CRISES:
            cnt = int(crisis_summary.loc[key, name]) if key in crisis_summary.index else 0
            pct = float((placebo_vals <= cnt).mean() * 100)
            placebo_stats.append({
                "series": name,
                "crisis": key,
                "crisis_cp_count": cnt,
                "placebo_mean": round(float(placebo_vals.mean()), 2),
                "placebo_95th": round(float(np.percentile(placebo_vals, 95)), 1),
                "placebo_max": int(placebo_vals.max()),
                "placebo_percentile_of_crisis": round(pct, 1),
            })
    stats_df = pd.DataFrame(placebo_stats)
    stats_df.to_csv(TBL_DIR / "change_point_vs_placebo.csv", index=False)
    print("\nCrisis-window CP counts vs placebo percentile:")
    print(stats_df.to_string(index=False))


if __name__ == "__main__":
    main()
