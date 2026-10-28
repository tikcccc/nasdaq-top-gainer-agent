# Recorded historical windows

Six explicit stock/date episodes were downloaded from Yahoo's public chart
endpoint. `manifest.json` stores the request URL, retrieval timestamp, SHA-256,
daily bars, events and independent arithmetic gold. The 18 original JSON
responses are retained in `raw/`. `record_historical.py` refuses to overwrite an
existing destination. Tests verify checksums and decode parity before replay.

These are retrospective prices retrieved in 2026, **not point-in-time market
snapshots**. Later corrections and split adjustments can be present. `close` is
split-adjusted and not dividend-adjusted; `adjclose` and current metadata never
enter replay. Splits must not be applied twice. META's historical ticker was FB;
this dataset uses Yahoo's current META identifier. Benchmarks use the same dates.

| Episode | End date | Coverage |
|---|---|---|
| AAPL | 2024-05-03 | Earnings window |
| NVDA | 2024-05-23 | Earnings window |
| NVDA | 2024-06-11 | Ten-for-one split and cash dividend in the window |
| META | 2022-02-03 | Declining earnings window, renamed ticker |
| AMZN | 2022-04-29 | Declining earnings window, later split-adjusted history |
| MSFT | 2024-11-29 | Thanksgiving holiday and 18:00 UTC early close; prior dividend |

News titles are curated from linked issuer releases/filings. Only their dates
are established, so replay uses conservative end-of-day UTC timestamps. A title
does not establish a catalyst's effect on the share price or the contents of the
full release. An empty curated list does not prove no news existed.

No historical NASDAQ-wide screener was recorded. **Ranking accuracy is N/A.**
Live replay seeds a named candidate and uses the production graph from the
history step onward with an explicit historical task prompt. It is a
post-selection analysis test, not an end-to-end historical winner test. The
supplied daily quote is derived from the same recorded closes, so its cross-check
is not independent-source corroboration. Secondary unrecorded providers are
disabled; email is always dry-run.

```powershell
.venv/Scripts/python.exe scripts/eval_historical.py --output outputs/eval/historical-contracts.json
.venv/Scripts/python.exe scripts/eval_historical.py --mode live --provider mimo --output outputs/eval/historical-mimo.json
```

This small collection does not cover halts, delistings, real IPO gaps, or a
second independently recorded market-data provider. Those remain additional
coverage, alongside independently labelled natural-language reports.
