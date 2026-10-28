"""Google News RSS search, used as the fallback headline source (personal / non-commercial use)."""

from __future__ import annotations

import email.utils
import xml.etree.ElementTree as ET
from typing import Any

from nasdaq_agent.infra import http
from nasdaq_agent.infra.errors import ProviderError

NAME = "google_news"
URL = "https://news.google.com/rss/search"


def news(symbol: str, company: str = "", count: int = 10) -> list[dict[str, Any]]:
    query = f'"{symbol}" stock' + (f' OR "{company}"' if company else "")
    response = http.get(URL, params={"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"})
    try:
        root = ET.fromstring(response.content)
    except ET.ParseError as exc:
        raise ProviderError(NAME, f"invalid RSS: {exc}") from exc
    out = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        source = item.find("source")
        publisher = source.text.strip() if source is not None and source.text else ""
        if publisher and title.endswith(f" - {publisher}"):
            title = title[: -len(publisher) - 3]
        pub = item.findtext("pubDate")
        published = email.utils.parsedate_to_datetime(pub).isoformat() if pub else None
        out.append({"title": title, "publisher": publisher, "published": published, "link": item.findtext("link")})
        if len(out) >= count:
            break
    return out
