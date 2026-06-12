"""Place each crisis in broader UK macro-financial context using JST R6.

JST covers the UK starting in 1870, so 1857 and 1866 are *outside JST coverage*
and are reported as such — JST cannot speak to those crises and we do not pretend
otherwise. The analysis here is substantive only for 1890 and 1914.

Outputs
- outputs/tables/jst_crisis_context.csv  — crisis-year levels and 1-yr changes
- outputs/figures/jst_macro_context.png  — UK macro 1870-1914 with crisis lines

Important caveat: JST is annual. It contextualizes the macro-financial
environment in which a crisis happened; it does not identify the Bank's
intervention timing.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
TBL = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures"

CRISIS_YEARS = {"1857": 1857, "1866": 1866, "1890": 1890, "1914": 1914}
JST_COVERAGE_START = 1870


def _yoy(df: pd.DataFrame, year: int, col: str) -> float:
    prev = df[df["year"] == year - 1]
    curr = df[df["year"] == year]
    if prev.empty or curr.empty:
        return np.nan
    p, c = float(prev[col].iloc[0]), float(curr[col].iloc[0])
    if not np.isfinite(p) or not np.isfinite(c) or p == 0:
        return np.nan
    return (c / p) - 1.0


def _diff(df: pd.DataFrame, year: int, col: str) -> float:
    prev = df[df["year"] == year - 1]
    curr = df[df["year"] == year]
    if prev.empty or curr.empty:
        return np.nan
    p, c = float(prev[col].iloc[0]), float(curr[col].iloc[0])
    if not np.isfinite(p) or not np.isfinite(c):
        return np.nan
    return c - p


def context_table(jst: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, year in CRISIS_YEARS.items():
        rec: dict = {"crisis": label, "year": year}
        if year < JST_COVERAGE_START:
            rec["jst_coverage"] = "outside coverage (JST UK begins 1870)"
            rec["crisisJST_flag"] = "n/a"
            rec["stir_pct"] = "n/a"
            rec["ltrate_pct"] = "n/a"
            rec["bill_rate_pct"] = "n/a"
            rec["tloans_gbpbn"] = "n/a"
            rec["money_gbpbn"] = "n/a"
            rec["cpi_index"] = "n/a"
            rec["gdp_gbpbn"] = "n/a"
            rec["rgdp_per_capita_usd"] = "n/a"
            rec["tloans_yoy"] = "n/a"
            rec["money_yoy"] = "n/a"
            rec["gdp_yoy"] = "n/a"
            rec["cpi_yoy"] = "n/a"
            rec["stir_change_pp"] = "n/a"
            rec["ltrate_change_pp"] = "n/a"
        else:
            row = jst[jst["year"] == year]
            if row.empty:
                rec["jst_coverage"] = "missing"
                rows.append(rec)
                continue
            r = row.iloc[0]
            rec["jst_coverage"] = "covered"
            rec["crisisJST_flag"] = int(r["crisisJST"]) if pd.notna(r.get("crisisJST")) else "n/a"
            rec["stir_pct"] = round(float(r["stir"]), 2) if pd.notna(r.get("stir")) else "n/a"
            rec["ltrate_pct"] = round(float(r["ltrate"]), 2) if pd.notna(r.get("ltrate")) else "n/a"
            rec["bill_rate_pct"] = round(float(r["bill_rate"]) * 100, 2) if pd.notna(r.get("bill_rate")) else "n/a"
            rec["tloans_gbpbn"] = round(float(r["tloans"]), 3) if pd.notna(r.get("tloans")) else "n/a"
            rec["money_gbpbn"] = round(float(r["money"]), 3) if pd.notna(r.get("money")) else "n/a"
            rec["cpi_index"] = round(float(r["cpi"]), 2) if pd.notna(r.get("cpi")) else "n/a"
            rec["gdp_gbpbn"] = round(float(r["gdp"]), 3) if pd.notna(r.get("gdp")) else "n/a"
            rec["rgdp_per_capita_usd"] = round(float(r["rgdpmad"]), 0) if pd.notna(r.get("rgdpmad")) else "n/a"
            rec["tloans_yoy"] = round(_yoy(jst, year, "tloans") * 100, 2)
            rec["money_yoy"] = round(_yoy(jst, year, "money") * 100, 2)
            rec["gdp_yoy"] = round(_yoy(jst, year, "gdp") * 100, 2)
            rec["cpi_yoy"] = round(_yoy(jst, year, "cpi") * 100, 2)
            rec["stir_change_pp"] = round(_diff(jst, year, "stir"), 2)
            rec["ltrate_change_pp"] = round(_diff(jst, year, "ltrate"), 2)
        rows.append(rec)
    return pd.DataFrame(rows)


def macro_figure(jst: pd.DataFrame) -> None:
    win = jst[(jst["year"] >= 1870) & (jst["year"] <= 1920)].copy()
    win = win.sort_values("year").reset_index(drop=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    (axA, axB), (axC, axD) = axes

    # Panel A: interest rates
    axA.plot(win["year"], win["stir"], color="#2980b9", label="Short-term rate (stir)")
    axA.plot(win["year"], win["ltrate"], color="#c0392b", label="Long-term rate (ltrate)")
    axA.set_title("UK interest rates (%)")
    axA.legend(fontsize=8, frameon=False)

    # Panel B: total loans level + yoy change
    axB.bar(win["year"], win["tloans"].pct_change() * 100, color="#9b59b6",
            alpha=0.7, label="tloans YoY %")
    axB.axhline(0, color="#666", linewidth=0.6)
    axB.set_title("Total loans / credit — YoY growth (%)")
    axB.legend(fontsize=8, frameon=False)

    # Panel C: broad money level + yoy change
    axC.bar(win["year"], win["money"].pct_change() * 100, color="#16a085",
            alpha=0.7, label="money YoY %")
    axC.axhline(0, color="#666", linewidth=0.6)
    axC.set_title("Broad money — YoY growth (%)")
    axC.legend(fontsize=8, frameon=False)

    # Panel D: CPI YoY (inflation) + GDP YoY
    axD.plot(win["year"], win["cpi"].pct_change() * 100, color="#e67e22", label="CPI YoY %")
    axD.plot(win["year"], win["gdp"].pct_change() * 100, color="#34495e", label="Nominal GDP YoY %")
    axD.axhline(0, color="#666", linewidth=0.6)
    axD.set_title("Inflation and nominal GDP growth (%)")
    axD.legend(fontsize=8, frameon=False)

    # Crisis markers (only 1890 and 1914 — within JST coverage)
    for ax in (axA, axB, axC, axD):
        for y, color, label in [(1890, "#c0392b", "1890 Baring"),
                                  (1914, "#16a085", "1914 WWI")]:
            ax.axvline(y, color=color, linestyle="--", linewidth=0.9, alpha=0.7)
        ax.grid(linestyle=":", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)

    # Mark JST crisis flag (==1) years with shaded dot on Panel A
    flagged = win[win["crisisJST"] == 1]
    for _, r in flagged.iterrows():
        axA.scatter(r["year"], r["stir"], s=60, color="#f1c40f", edgecolor="#7f8c8d",
                    zorder=4, label="JST crisis flag" if r["year"] == flagged["year"].min() else None)
    axA.legend(fontsize=8, frameon=False)

    fig.suptitle("UK macro-financial context, 1870-1920 (Jordà-Schularick-Taylor R6)",
                 fontsize=12, y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIG / "jst_macro_context.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")


def main() -> None:
    TBL.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    jst = pd.read_parquet(PROC / "macro_jst_uk.parquet").sort_values("year").reset_index(drop=True)

    table = context_table(jst)
    table.to_csv(TBL / "jst_crisis_context.csv", index=False)
    print(f"Wrote {TBL / 'jst_crisis_context.csv'}")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(table.to_string(index=False))

    print("\nWriting figure:")
    macro_figure(jst)

    # Print plain-English summary lines for paper use
    print("\nJST crisis-year summary (UK only):")
    for _, r in table.iterrows():
        if r["jst_coverage"] == "covered":
            print(f"  {r['crisis']}: crisisJST={r['crisisJST_flag']}, "
                  f"stir={r['stir_pct']}%, ltrate={r['ltrate_pct']}%, "
                  f"tloans YoY={r['tloans_yoy']}%, money YoY={r['money_yoy']}%, "
                  f"GDP YoY={r['gdp_yoy']}%, CPI YoY={r['cpi_yoy']}%")
        else:
            print(f"  {r['crisis']}: {r['jst_coverage']}")


if __name__ == "__main__":
    main()
