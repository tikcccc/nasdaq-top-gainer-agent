# Evidence-constrained reports (default)

New reports use `evidence-v2`. The agent still selects tools, fetches data,
writes/repairs Python, chooses relevant news and decides which verified facts lead
the report. Backend code writes every sentence; the sending tool has no free-text
narrative or risk explanation field.

```json
{
  "selected_news_ids": ["n1"],
  "selected_risk_evidence_ids": [],
  "include_data_limitations": true,
  "key_findings": [{"kind": "final_session_share"}, {"kind": "news", "ref": "n1"}, {"kind": "vs_sp500"}]
}
```

## Key findings

`key_findings` is an ordered list of 1–3 kinds: `window_return`, `final_session_share`,
`vs_sp500`, `vs_nasdaq`, `volatility`, `drawdown`, `volume_spike` (trade-date volume
against the earlier sessions, computed from the hash-verified dataset), `dividend` (a
dividend that went ex in the window per corporate-action data, against the close before
its ex-date; otherwise the one per-share amount a selected headline states, attributed to
and citing that headline, against the last close before it was published), `news` (with
`ref`), `prior_shock` and `data_limitation`. A headline amount is read only from a dividend headline that
states a single amount and does not suspend, cut or deny the payout. They open the body under "Key findings". This is where the model's judgement shows: which facts matter most and in
what order. It does not control wording:

- Each sentence, including direction words such as "above" or "below", is rendered
  from verified data. A featured fact moves to the top instead of being repeated, so
  the word budget is unchanged.
- Unselected secondary metrics are omitted from the prose. The report contains one
  summary and at most three distinct findings. Mandatory limitations reserve a
  finding slot and are never discarded to make room for another metric.
- When the ranking, filter eligibility or price check is unconfirmed, or the data is
  intraday, a `data_limitation` finding is required; otherwise the report is rejected.
- Duplicates, unknown kinds, a `ref` on a non-news kind, stale news IDs and kinds the
  data cannot support (for example a missing benchmark) are rejected with a hint.
- The "In brief" line is the day's move plus the most important fact the first finding
  does not state: the next finding's fact, else the price path, a dividend, heavy volume
  or the benchmarks. It never repeats finding 1.
- When both indexes moved less than a tenth of a wide stock move, the two benchmark
  comparisons become one sentence, whether featured or in the body.
- Without a choice the report starts with the final-session contribution, then a
  dividend, selected news or unusually heavy volume when available. Required
  limitations take precedence over a lower-priority optional finding.

`prior_shock` covers the largest absolute earlier **single-session** return when
it is at least 20%. It excludes the final session and is not triggered by a
cumulative multi-day loss. It reserves a finding slot even when the model has
selected three other findings, without displacing a material limitation. The
sentence states that this report has not verified its cause. A fetched headline
is matched only if it belongs to the current stock/run and its timezone-aware
publication timestamp converts to that session's `America/New_York` calendar
date. A matching date does not establish causation; a missing match does not
claim that no public announcement exists. Quotes, news cards and source dates
use that same New York date. Ambiguous unzoned timestamps are not matched.

Screener market caps lack a dated, split-reconciled share-count basis and are
omitted from the email. Their raw value and `market_cap_status=unverified` stay
in saved report facts. High turnover alone never marks a market cap as stale.

The contract rejects unknown fields, malformed/duplicate IDs, unrelated or stale
headlines and stale analysis/sentiment. Each selected source must belong to the
current stock and trade date. The graph guard and the sending tool independently
perform validation. Recipients, stock and analysis window are not model-chosen
report arguments.

The email shows the stock identity and ranking scope, one summary, the KPI cards,
and at most three findings. It does not append a second paragraph containing all
unselected metrics. Daily moves and charts remain below the short body. The method
note is two lines: screen filters and the vendor/date range/price-return basis.
A recorded split adds at most one short adjustment/volume note. Snapshot capture
times, raw leaders, excluded stocks and detailed split checks stay in the run
record, not in the email. Material ranking, proxy and price limitations still
appear in the findings.

Generic news-coverage and calculation-scope disclaimers are omitted. The
`include_data_limitations` input is retained for compatibility; it does not add a
generic note or hide material limitations. There is no minimum body length and
no extra sentences are added to meet a word count.

Source disclosure IDs select headlines for quotation; they do not authorize the
agent to infer a risk, event outcome, investor motive or price cause. The same
source is quoted once even if selected in both lists. No generic risk bullet is
invented when there is no disclosure to quote.

The rendered body must be at most 250 words under the default settings, with no minimum. Full
quotations are added in selected order only when they fit. Overlong or formatted
titles remain in the separate source list; `quoted_ids` and `omitted_ids` record
the distinction and tool warnings explain omissions. Code does not shorten a
quotation in a way that could remove a negation or condition. Material data
limitations are never hidden by `include_data_limitations=false`.

Headline sentiment is shown separately as a model estimate of the fetched items.
The default report does not display inferred catalysts or dilution probabilities
as factual disclosures. Its pre-send check does not require a semantic model;
sentiment can still use Jev or the configured LLM. A failed sentiment assessment
must not be replaced with a fabricated score.

The older free-prose checker is retained as `check_legacy_narrative` for historical
evaluation scripts and regression fixtures only. The production graph and tool
do not call it. Previous free-prose blind scores belong to that earlier contract;
they are not accuracy scores for this renderer. Archived reports and review
packets remain unchanged. Resending an archived email retains its saved content.

## Deployment and replay checks

`GET /api/config` returns `report_renderer` (contract version and SHA-256 of
reporting code, templates and the reviewed announcement catalogue). New report
facts, the HTML `report-renderer` meta tag and MIME `X-Report-Renderer` /
`X-Report-Fingerprint` headers record that same identity. A runtime match alone
does not upgrade an existing archive: manual resend reports the identity of the
saved MIME message, or `archived-unknown` when it predates these headers.

Run `python -m scripts.replay_report --run run.json --series series.json --out
outputs/review` to regenerate a saved API snapshot locally without network,
model or SMTP calls. It binds new CSV hashes after reserialising the series,
checks the analysis against the original snapshot, and saves an unsent HTML,
MIME message, provenance and `acceptance.json`. This tests rendering on the
recorded data; it does not establish that the live server has been redeployed.

This contract controls what the model can assert in a new report. It does not
certify upstream quote/ranking accuracy, the truth of source headlines, corporate
identity resolution, or independent human acceptance. Source claims remain
attributed claims. Numerical, provenance, orchestration and external delivery
checks retain their separate meanings.

## What the reader sees

- One day move everywhere (header, subject, summary, findings): the verified price
  history's. The screener's figure only ranked the stock; a gap of 5 points or more is
  a data limitation, a smaller one is logged.
- `reporting/facts.py` builds the shared `ReportFacts` snapshot. Prices, volumes,
  return metrics, tables and both chart formats use the same trimmed frames.
  Header prices and volumes no longer come from the screener. Market cap remains
  explicitly labelled as a screener figure. The rendered facts are saved in the
  run record under `email.report_facts`, including detailed selection and split
  diagnostics that are deliberately absent from the short email.
- The source list contains only citations used by the body, compact method note
  or visible news cards. Generic profile/filing links and the excluded leader are
  not appended automatically. A reviewed primary announcement is matched by
  symbol, effective date and split ratio from `data/primary_announcements.json`;
  it receives a data reference from `d8` onwards. An unmatched event retains its
  explicit price-data attribution. A primary document confirms the announced
  event, not whether a price vendor adjusted its data.
- Excess return is the difference between the stock and benchmark returns, shown
  in percentage points (`pp`). Volume uses one rounding rule in cards and prose,
  and names the earlier-session averaging period, excluding the final session.
- Subject, cards, prose and chart labels share the percentage formatter: two
  decimal places below 100% in magnitude, one for larger moves. A 99.75% move
  therefore never becomes 99.8% in the summary.
- Every comparison names its span. Window figures say "N-session"; the summary
  explicitly distinguishes the day's move from any window return it includes.
- Split and dividend data are kept even when the price bars fall back from Yahoo to
  Nasdaq. The split ratio comes from that data; the bars decide whether it applies. If
  the split-day open jumps from the previous close by about the split factor, earlier
  prices and volumes are restated on the post-split basis. If not (the vendor already
  adjusted, or the split had not reached trading), the bars are left alone and the report
  describes only the observed continuity. A `no_gap` result does not establish
  vendor adjustment or the cause of the price move. When split adjustment is
  unconfirmed, the cross-split volume multiple is omitted and the methodology
  states that volume comparability is unconfirmed.
- A stock whose screener gain was only the split, and whose adjusted move falls below
  the next candidate's, is removed; the next candidate becomes the top candidate and the
  run record names the excluded stock. One that still leads keeps its place with the
  adjusted move. These screening diagnostics do not expand the email footer.
- A headline about a split that the split data does not show, or split data that could
  not be checked, is a stated limitation.
- Data-quality diagnostics (screener vs history gaps, single-vendor prices, flags) go to
  the run log and the run record (`email.notes`), not the email.

## Tried and removed: model rewrite

A model rewording the code-written report, shipped only when numbers, citations and
quotes survived deterministic checks and a meaning-only critic, was tried on 27 Oct 2026
(MiMo; 10 reports) and removed. 5 rewrites passed and kept every fact, but changed
little beyond wording such as "5-session" to "5-trading-day"; one softened "an extreme
spike" to "a sharp spike" without the critic noticing. Readability is improved in the
templates (`readable.py`) instead, where wording is fixed and tested.
