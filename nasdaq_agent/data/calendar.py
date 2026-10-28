"""Which trading day does "today" mean?

Weekends, US holidays and pre-market runs all map to the most recent session.
During regular hours the current session is used and flagged as intraday.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from functools import lru_cache
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd

NY = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class MarketSession:
    trade_date: date
    is_intraday: bool
    previous_date: date

    def as_dict(self) -> dict:
        return {
            "trade_date": self.trade_date.isoformat(),
            "is_intraday": self.is_intraday,
            "previous_date": self.previous_date.isoformat(),
        }


@lru_cache(maxsize=1)
def _cal() -> xcals.ExchangeCalendar:
    # NASDAQ follows the NYSE holiday calendar.
    return xcals.get_calendar("XNYS")


def market_session(now: datetime | None = None) -> MarketSession:
    cal = _cal()
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    today = pd.Timestamp(now.astimezone(NY).date())
    ts = pd.Timestamp(now)

    if cal.is_session(today) and ts >= cal.session_open(today):
        session, intraday = today, ts < cal.session_close(today)
    else:
        session, intraday = cal.date_to_session(today, direction="previous"), False
        if session == today:  # a session day, but before the open
            session = cal.previous_session(today)
    return MarketSession(
        trade_date=session.date(),
        is_intraday=bool(intraday),
        previous_date=cal.previous_session(session).date(),
    )


def sessions_back(end: date, count: int) -> date:
    """Calendar date of the session ``count`` sessions before ``end``."""
    cal = _cal()
    return cal.session_offset(pd.Timestamp(end), -count).date()


def to_ny_date(epoch_s: float) -> date:
    return datetime.fromtimestamp(epoch_s, tz=timezone.utc).astimezone(NY).date()
