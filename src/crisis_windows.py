"""Single source of truth for crisis event windows.

Dates follow secondary literature (Bordo 1990; Flandreau & Ugolini 2011;
Anson et al. 2017; White 2016) plus the timeline in `docs/crisis_timeline.md`.

Each crisis defines:
    pre_start  – beginning of baseline window
    acute_start, acute_peak, acute_end  – acute window with the canonical "trigger" date
    post_end   – end of recovery window
"""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class Crisis:
    name: str
    pre_start: pd.Timestamp
    acute_start: pd.Timestamp
    acute_peak: pd.Timestamp
    acute_end: pd.Timestamp
    post_end: pd.Timestamp
    trigger_label: str

    @property
    def pre(self) -> slice:
        return slice(self.pre_start, self.acute_start - pd.Timedelta(days=1))

    @property
    def acute(self) -> slice:
        return slice(self.acute_start, self.acute_end)

    @property
    def post(self) -> slice:
        return slice(self.acute_end + pd.Timedelta(days=1), self.post_end)


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


CRISES: dict[str, Crisis] = {
    "1857": Crisis(
        name="1857 Panic",
        pre_start=_ts("1856-10-01"),
        acute_start=_ts("1857-10-01"),
        acute_peak=_ts("1857-11-12"),   # Treasury Letter suspending Bank Charter Act
        acute_end=_ts("1857-12-31"),
        post_end=_ts("1858-12-31"),
        trigger_label="1857-11-12 Treasury Letter; suspension of Bank Charter Act",
    ),
    "1866": Crisis(
        name="1866 Overend Gurney",
        pre_start=_ts("1865-05-01"),
        acute_start=_ts("1866-05-01"),
        acute_peak=_ts("1866-05-11"),   # Black Friday — failure of Overend, Gurney & Co
        acute_end=_ts("1866-06-30"),
        post_end=_ts("1867-06-30"),
        trigger_label="1866-05-11 'Black Friday' — Overend, Gurney & Co. fails",
    ),
    "1890": Crisis(
        name="1890 Baring Crisis",
        pre_start=_ts("1889-11-01"),
        acute_start=_ts("1890-11-01"),
        acute_peak=_ts("1890-11-15"),   # Lidderdale Guarantee Fund organized
        acute_end=_ts("1890-12-31"),
        post_end=_ts("1891-12-31"),
        trigger_label="1890-11-15 Lidderdale Guarantee Fund for Barings",
    ),
    "1914": Crisis(
        name="1914 WWI outbreak",
        pre_start=_ts("1913-08-01"),
        acute_start=_ts("1914-07-27"),
        acute_peak=_ts("1914-08-06"),   # Moratorium and extended bank holiday
        acute_end=_ts("1914-09-30"),
        post_end=_ts("1915-06-30"),
        trigger_label="1914-08-06 Moratorium during WWI mobilisation",
    ),
}
