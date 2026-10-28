"""Publication dates shown and matched using the US market's calendar zone."""
import pandas as pd


def news_date(published) -> str | None:
    """An unzoned timestamp cannot establish a New York session date."""
    if not isinstance(published, str) or not published:
        return None
    try:
        stamp = pd.Timestamp(published)
        return stamp.tz_convert('America/New_York').date().isoformat() if stamp.tzinfo else None
    except (ValueError, TypeError, OverflowError):
        return None
