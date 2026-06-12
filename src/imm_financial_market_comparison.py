"""Compare ordinary listed financial institutions with the Bank's crisis role.

The Yale Investor's Monthly Manual (IMM) can show how ordinary listed financial
institutions were priced or quoted in securities markets. It cannot show their
daily balance sheets, internal lending decisions, or central-bank-like
instruments. That limitation is exactly why the comparison is useful: it
separates market actors from the institution whose own records contain reserve,
Bank Rate, and emergency-liquidity choices.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/yale_imm/Totaldata.zip"
PROC_DIR = ROOT / "data/processed/imm"
TABLE_DIR = ROOT / "outputs/tables"
FIG_DIR = ROOT / "outputs/figures"

STOCK_TYPES = {"Common Stock", "Preferred Stock"}
EVENTS = {
    "1890 Baring Crisis": pd.Timestamp("1890-11-30"),
    "1914 WWI outbreak": pd.Timestamp("1914-08-31"),
}


def load_imm_prices() -> pd.DataFrame:
    usecols = [
        "NewID",
        "Name",
        "Year",
        "Month",
        "Industry",
        "Type",
        "PriceMonthLate",
    ]
    with zipfile.ZipFile(RAW) as archive:
        with archive.open("Totaldata.csv") as handle:
            df = pd.read_csv(handle, usecols=usecols, low_memory=False)

    df = df.rename(
        columns={
            "NewID": "security_id",
            "Name": "name",
            "Year": "year",
            "Month": "month",
            "Industry": "industry",
            "Type": "type",
            "PriceMonthLate": "price",
        }
    )
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df[
        df["type"].isin(STOCK_TYPES)
        & df["industry"].isin([2, 3])
        & df["price"].gt(0)
        & df["year"].between(1869, 1919)
    ].copy()
    df["date"] = pd.to_datetime(
        {"year": df["year"], "month": df["month"], "day": 1}
    ) + pd.offsets.MonthEnd(0)
    df["sector"] = np.where(
        df["industry"].eq(2),
        "Listed financial institutions",
        "Listed non-financial corporates",
    )

    df = (
        df.groupby(["security_id", "date", "sector"], as_index=False)
        .agg(name=("name", "first"), type=("type", "first"), price=("price", "median"))
        .sort_values(["security_id", "date"])
    )
    df["prev_price"] = df.groupby("security_id")["price"].shift(1)
    df["return"] = df["price"] / df["prev_price"] - 1
    df.loc[~df["return"].between(-0.80, 2.00), "return"] = np.nan
    return df


def monthly_panel(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df.groupby(["date", "sector"], as_index=False)
        .agg(n_quoted=("security_id", "nunique"))
        .sort_values(["sector", "date"])
    )
    returns = (
        df.dropna(subset=["return"])
        .groupby(["date", "sector"], as_index=False)
        .agg(
            median_return=("return", "median"),
            mean_return=("return", "mean"),
            share_declining=("return", lambda s: float((s < 0).mean())),
            n_return_securities=("security_id", "nunique"),
        )
        .sort_values(["sector", "date"])
    )
    monthly = counts.merge(returns, on=["date", "sector"], how="left")
    monthly["log_mean_return"] = np.log1p(monthly["mean_return"].fillna(0))
    monthly["mean_return_index"] = (
        monthly.groupby("sector")["log_mean_return"].cumsum().pipe(np.exp) * 100
    )
    return monthly


def event_window(monthly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    windows: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for event_name, event_month in EVENTS.items():
        start = event_month - pd.DateOffset(months=12)
        end = event_month + pd.DateOffset(months=12)
        event_panel = monthly[monthly["date"].between(start, end)].copy()
        for sector, sector_panel in event_panel.groupby("sector"):
            full_dates = pd.DataFrame(
                {"date": pd.date_range(start, end, freq="ME"), "sector": sector}
            )
            sector_panel = (
                full_dates.merge(sector_panel, on=["date", "sector"], how="left")
                .sort_values("date")
                .copy()
            )
            sector_panel["n_quoted"] = sector_panel["n_quoted"].fillna(0)
            sector_panel["mean_return_index"] = sector_panel[
                "mean_return_index"
            ].ffill()
            base = sector_panel.loc[
                sector_panel["date"].eq(start), "mean_return_index"
            ]
            count_base = sector_panel.loc[sector_panel["date"].eq(start), "n_quoted"]
            if base.empty:
                base = sector_panel["mean_return_index"].iloc[:1]
            if count_base.empty:
                count_base = sector_panel["n_quoted"].iloc[:1]

            sector_panel["event"] = event_name
            sector_panel["months_from_event"] = (
                (sector_panel["date"].dt.year - event_month.year) * 12
                + (sector_panel["date"].dt.month - event_month.month)
            )
            sector_panel["indexed_to_pre_event"] = (
                sector_panel["mean_return_index"] / base.iloc[0] * 100
            )
            sector_panel["quoted_count_index"] = (
                sector_panel["n_quoted"] / count_base.iloc[0] * 100
            )
            windows.append(sector_panel)

            event_to_2 = sector_panel[
                sector_panel["months_from_event"].between(0, 2)
            ]["mean_return"].dropna()
            prior_to_0 = sector_panel[
                sector_panel["months_from_event"].between(-2, 0)
            ]["mean_return"].dropna()
            rolling_peak = sector_panel["indexed_to_pre_event"].cummax()
            drawdown = sector_panel["indexed_to_pre_event"] / rolling_peak - 1
            event_row = sector_panel[sector_panel["months_from_event"].eq(0)]
            summaries.append(
                {
                    "event": event_name,
                    "sector": sector,
                    "event_month": event_month.strftime("%Y-%m"),
                    "n_securities_event_month": int(event_row["n_quoted"].iloc[0])
                    if not event_row.empty
                    else np.nan,
                    "event_month_quote_count_index": float(
                        event_row["quoted_count_index"].iloc[0]
                    )
                    if not event_row.empty
                    else np.nan,
                    "event_month_mean_return": float(event_row["mean_return"].iloc[0])
                    if not event_row.empty
                    and pd.notna(event_row["mean_return"].iloc[0])
                    else np.nan,
                    "event_month_share_declining": float(
                        event_row["share_declining"].iloc[0]
                    )
                    if not event_row.empty
                    and pd.notna(event_row["share_declining"].iloc[0])
                    else np.nan,
                    "three_month_cumulative_return_event_to_plus2": float(
                        np.expm1(np.log1p(event_to_2).sum())
                    )
                    if len(event_to_2)
                    else np.nan,
                    "three_month_cumulative_return_minus2_to_event": float(
                        np.expm1(np.log1p(prior_to_0).sum())
                    )
                    if len(prior_to_0)
                    else np.nan,
                    "max_drawdown_minus12_to_plus12": float(drawdown.min()),
                    "window_min_index_pre_event_100": float(
                        sector_panel["indexed_to_pre_event"].min()
                    ),
                    "window_max_index_pre_event_100": float(
                        sector_panel["indexed_to_pre_event"].max()
                    ),
                }
            )
    return pd.concat(windows, ignore_index=True), pd.DataFrame(summaries)


def plot_windows(windows: pd.DataFrame, monthly: pd.DataFrame) -> None:
    colors = {
        "Listed financial institutions": "#165A64",
        "Listed non-financial corporates": "#A64E2A",
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    panel_1890 = windows[windows["event"].eq("1890 Baring Crisis")]
    for sector, sector_panel in panel_1890.groupby("sector"):
        sector_panel = sector_panel.sort_values("months_from_event")
        axes[0].plot(
            sector_panel["months_from_event"],
            sector_panel["indexed_to_pre_event"],
            marker="o",
            linewidth=2.2,
            markersize=3.5,
            color=colors[sector],
            label=sector,
        )
    axes[0].axvline(0, color="#2F2F2F", linestyle="--", linewidth=1)
    axes[0].axhline(100, color="#8A8A8A", linestyle=":", linewidth=1)
    axes[0].set_title("1890: ordinary listed firms")
    axes[0].set_xlabel("Months from Baring rescue month")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].set_xlim(-12, 12)

    annual = (
        monthly[monthly["date"].dt.year.between(1888, 1916)]
        .assign(year=lambda d: d["date"].dt.year)
        .groupby(["year", "sector"], as_index=False)["n_quoted"]
        .sum()
    )
    for sector, sector_panel in annual.groupby("sector"):
        axes[1].plot(
            sector_panel["year"],
            sector_panel["n_quoted"],
            marker="o",
            linewidth=2.2,
            markersize=3.5,
            color=colors[sector],
            label=sector,
        )
    axes[1].axvline(1914, color="#2F2F2F", linestyle="--", linewidth=1)
    axes[1].set_title("1914: IMM stock-price source gap")
    axes[1].set_xlabel("Year")
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].set_xlim(1888, 1916)
    axes[1].annotate(
        "No usable stock-price panel\nfor the crisis month",
        xy=(1914, 0),
        xytext=(1908.5, annual["n_quoted"].max() * 0.45),
        arrowprops={"arrowstyle": "->", "color": "#444444"},
        fontsize=9,
        color="#333333",
    )
    axes[0].set_ylabel("Mean-return price index, 12 months before crisis = 100")
    axes[1].set_ylabel("Valid late-month stock quotes per year")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "Ordinary listed financial institutions as market actors and source limits",
        y=0.98,
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.93])
    fig.savefig(FIG_DIR / "imm_financial_market_comparison.png", dpi=220)
    plt.close(fig)


def write_notes(summary: pd.DataFrame) -> None:
    notes = ROOT / "docs/imm_comparison_notes.md"
    rows = summary.copy()
    pct_cols = [
        "event_month_quote_count_index",
        "event_month_mean_return",
        "event_month_share_declining",
        "three_month_cumulative_return_event_to_plus2",
        "max_drawdown_minus12_to_plus12",
    ]
    rows["event_month_quote_count_index"] = rows[
        "event_month_quote_count_index"
    ].round(1)
    for col in pct_cols[1:]:
        rows[col] = (rows[col] * 100).round(2)
    table = rows[
        [
            "event",
            "sector",
            "n_securities_event_month",
            "event_month_quote_count_index",
            "event_month_mean_return",
            "event_month_share_declining",
            "three_month_cumulative_return_event_to_plus2",
            "max_drawdown_minus12_to_plus12",
        ]
    ].to_markdown(index=False)
    notes.write_text(
        "# Yale Investor's Monthly Manual comparison notes\n\n"
        "This supplementary layer compares the Bank of England evidence with "
        "ordinary listed financial institutions in the Yale Investor's Monthly "
        "Manual (IMM). It is a market-price and quote-availability comparison, "
        "not a balance-sheet or transaction-ledger comparison.\n\n"
        "## Method\n\n"
        "- Source: `data/raw/yale_imm/Totaldata.zip`, cleaned security-level IMM file.\n"
        "- Securities used: common and preferred stock only.\n"
        "- Comparison groups: `Industry == 2` as listed financial institutions "
        "(banks, finance, and insurance in the raw IMM classification); "
        "`Industry == 3` as listed non-financial corporate securities.\n"
        "- Return measure: month-to-month change in `PriceMonthLate`, summarised "
        "as equal-weighted mean returns. The table also records the share of "
        "securities with negative returns. Median returns are often zero because "
        "many historical quotes are stale from month to month.\n"
        "- Quote-availability measure: number of listed securities with a valid "
        "`PriceMonthLate`, indexed to twelve months before the crisis. This is "
        "especially important in 1914, when stock-exchange closure removes the "
        "ordinary quotation stream.\n"
        "- Event windows: 12 months before to 12 months after November 1890 "
        "and August 1914.\n"
        "- Outlier rule: one-month returns outside -80% to +200% are excluded.\n\n"
        "## Summary table\n\n"
        + table
        + "\n\n## Interpretation\n\n"
        "The IMM comparison is useful because it gives the paper an external "
        "view of ordinary financial institutions. These firms appear as priced "
        "or quoted market actors. The Bank of England appears in a different "
        "source form: its own balance sheet, Bank Rate, reserve, and lending "
        "ledgers. That source difference is historically meaningful. It supports "
        "the paper's claim that the Bank was not simply another bank in the "
        "crisis ecology; it was the institution whose decisions changed the "
        "operating conditions of the market itself.\n\n"
        "The 1890 window is the usable substantive comparison: listed financial "
        "institutions show a modest event-month mean return of -0.90%, while "
        "listed non-financial corporates fall -1.24%. The result does not show "
        "ordinary banks acting as public stabilisers; it shows them being priced "
        "inside the market that the Bank was trying to stabilise. The 1914 window "
        "is not usable as a stock-price comparison because the IMM panel has a "
        "severe quotation gap around the crisis month; it is retained as source "
        "criticism rather than evidence about ordinary-bank behaviour.\n\n"
        "## Limits\n\n"
        "The IMM does not prove that ordinary banks lacked crisis agency. It "
        "does not provide their daily balance sheets, discount-window decisions, "
        "or internal correspondence. It can support a limited comparison only: "
        "ordinary banks can be observed as priced or quoted market actors, while "
        "the Bank of England can be observed as the rate-setting and "
        "reserve-managing institution. The comparison applies substantively to "
        "1890 only. The IMM begins in 1869, so it cannot compare the 1857 or "
        "1866 cases; the stock-price panel also has a severe gap around 1914, "
        "so the 1914 result is source criticism rather than substantive market "
        "comparison.\n",
        encoding="utf-8",
    )


def main() -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    prices = load_imm_prices()
    monthly = monthly_panel(prices)
    windows, summary = event_window(monthly)

    monthly.to_csv(PROC_DIR / "imm_financial_monthly.csv", index=False)
    windows.to_csv(TABLE_DIR / "imm_financial_market_windows.csv", index=False)
    summary.to_csv(TABLE_DIR / "imm_crisis_market_summary.csv", index=False)
    plot_windows(windows, monthly)
    write_notes(summary)

    print(f"Wrote {PROC_DIR / 'imm_financial_monthly.csv'}")
    print(f"Wrote {TABLE_DIR / 'imm_financial_market_windows.csv'}")
    print(f"Wrote {TABLE_DIR / 'imm_crisis_market_summary.csv'}")
    print(f"Wrote {FIG_DIR / 'imm_financial_market_comparison.png'}")


if __name__ == "__main__":
    main()
