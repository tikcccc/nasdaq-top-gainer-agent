"""Reader-friendly wording is a deterministic function of the verified numbers."""

import pytest

from nasdaq_agent.reporting import readable


def analysis(returns, *, gspc=None, ixic=None, vol=0.03, drawdown=-0.05):
    bench = {k: v for k, v in (("GSPC", gspc), ("IXIC", ixic)) if v is not None}
    return {"daily_returns": returns, "benchmark": bench, "volatility_daily": vol, "max_drawdown": drawdown}


def lines(a, **kw):
    return readable.sentences(a, symbol="AIOK", trade_date="2026-10-23", **kw)


def test_aiok_like_window_reads_as_one_day_move():
    out = lines(analysis([-0.002, 0.24, -0.018, -0.12, 1.828], gspc=-0.0047))
    assert out["window_return"] == "Over the 5 sessions to Oct 23, AIOK rose 202.4%, with 2 up days and 3 down days."   # compounded by hand: 3.0243 / 1 - 1
    assert out["final_session_share"] == ("Almost all of the 5-session gain came on Oct 23, when the stock rose 182.8%; "
                                          "the 4 earlier sessions together added 6.94%.")
    assert out["vs_GSPC"] == "Its 5-session return was well ahead of the S&P 500, which fell 0.47% over the same dates."


@pytest.mark.parametrize("returns, expected", [
    ([0.01, 0.01, 0.01, 0.01, 0.10], "Most of the 5-session gain"),        # final share between 50% and 90%
    ([0.0, 0.0, 0.0, 0.0, 0.10], "All of the net gain came on Oct 23"),
    ([0.05, 0.05, 0.05, 0.05, -0.05], "cut into the earlier"),
    ([-0.05, -0.05, 0.0, 0.0, 0.20], "reversed the earlier"),
    ([0.02, 0.02, 0.02, 0.02, 0.0], "was flat; the earlier sessions account for the move"),
])
def test_final_session_wording_follows_the_numbers(returns, expected):
    assert expected in lines(analysis(returns))["final_session_share"]


@pytest.mark.parametrize("gap_bench, word", [
    (0.102, "in line with"), (0.05, "ahead of"), (-0.05, "well ahead of"), (0.15, "behind"), (0.30, "well behind"),
])
def test_benchmark_wording_bands(gap_bench, word):
    out = lines(analysis([0.02, 0.02, 0.02, 0.02, 0.02], gspc=gap_bench))   # stock window return is about +10.4%
    assert f"was {word} the S&P 500" in out["vs_GSPC"]


def test_both_benchmarks_share_one_sentence():
    one_day = lines(analysis([-0.002, 0.24, -0.018, -0.12, 1.828], gspc=-0.0047, ixic=-0.0071))["vs_both"]
    assert one_day == ("Over the same dates the S&P 500 fell 0.47% and the NASDAQ Composite fell 0.71%; "
                       "AIOK's 202.4% 5-session gain was much larger than either index's move.")
    broad = lines(analysis([0.02] * 5, gspc=0.08, ixic=0.09))["vs_both"]           # stock about +10.4%
    assert broad.endswith("so AIOK's 5-session return was ahead of both.")
    split = lines(analysis([0.02] * 5, gspc=0.0, ixic=0.30))["vs_both"]
    assert split == ("Its 5-session return was well ahead of the S&P 500, which was flat, "
                     "and well behind the NASDAQ Composite, which rose 30.00%, over the same dates.")
    assert "vs_both" not in lines(analysis([0.02] * 5, gspc=0.0))
    assert "S&P 500, which was flat" in lines(analysis([0.02] * 5, gspc=-0.00003))["vs_GSPC"]   # not "fell 0.00%"


@pytest.mark.parametrize("vol, label", [(0.08, "large"), (0.03, "moderate"), (0.01, "small")])
def test_volatility_labels(vol, label):
    assert f"swings were {label}" in lines(analysis([0.01] * 5, vol=vol))["volatility"]


def test_one_outsized_session_is_kept_out_of_typical_volatility():
    returns = [0.01, -0.02, 0.012, -0.01, 1.83]
    out = lines(analysis(returns, vol=0.819, drawdown=-0.02))
    assert out["volatility"] == ("Daily sample volatility was 1.56% over the 4 earlier returns "
                                 "and 81.90% over all 5; neither figure is annualized.")
    assert out["drawdown"] == "The deepest close-to-close pullback from a closing high in the window was 2.00%, reached before Oct 23."
    # A drawdown that does not match the return path gets no timing.
    assert lines(analysis(returns, drawdown=-0.05))["drawdown"] == "The deepest close-to-close pullback from a closing high in the window was 5.00%."


def test_all_days_one_direction_and_no_drawdown():
    out = lines(analysis([0.01] * 5, drawdown=0.0))
    assert "rising on all 5 days" in out["window_return"]
    assert out["drawdown"] == "The stock never closed below an earlier high in the window."


def test_volume_wording_follows_the_ratio():
    extreme = lines(analysis([0.01] * 5), volume={"shares": 42_564_200, "ratio": 876.5, "sessions": 5})["volume_spike"]
    heavy = lines(analysis([0.01] * 5), volume={"shares": 4_000_000, "ratio": 4.0, "sessions": 5})["volume_spike"]
    normal = lines(analysis([0.01] * 5), volume={"shares": 1_000_000, "ratio": 1.4, "sessions": 5})["volume_spike"]
    assert extreme == ("Oct 23 volume was 876 times the previous 5-session average of 48.6K shares a day"
                       " — an extreme spike in trading.")
    assert heavy.endswith("4.0 times the previous 5-session average of 1.0M shares a day — unusually heavy trading.")
    assert "unusually heavy" not in normal and "extreme" not in normal


@pytest.mark.parametrize("ratio, shown", [(876.5, "876"), (120.5, "120"), (12.3, "12"), (3.44, "3.4")])
def test_volume_ratio_is_rounded_for_reading(ratio, shown):
    assert readable.times(ratio) == shown


AIOK = [-0.002, 0.24, -0.018, -0.12, 1.828]
DIVIDEND = {"amount": 3.5, "ex_date": "2026-10-21", "prior_close": 4.25, "prior_date": "2026-10-20", "in_returns": True}


def brief(kinds, a=None, **extra):
    a = analysis(AIOK, gspc=-0.0047, ixic=-0.0071) if a is None else a
    return readable.headline(kinds, a, symbol="AIOK", trade_date="2026-10-23", session_move=1.828, **extra)


def test_headline_adds_what_the_first_finding_does_not_say():
    volume = {"ratio": 876.5}
    path = "AIOK jumped 182.8% on Oct 23 — almost all of its 202.4% gain over the last 5 sessions."
    # Finding 1 already explains the price path, so the brief brings in finding 2's fact instead.
    assert brief(["final_session_share", "volume_spike", "news", "vs_sp500"], volume=volume) == \
        "AIOK jumped 182.8% on Oct 23, with volume 876 times its recent average."
    assert brief(["window_return", "final_session_share", "vs_sp500"]) == \
        "AIOK jumped 182.8% on Oct 23; over the last 5 sessions it rose 202.4%, while the S&P 500 fell 0.47% and the NASDAQ Composite fell 0.71%."
    # Otherwise the price path comes first: never two near-equal numbers side by side.
    assert brief(["vs_sp500", "news"]) == path
    assert brief(["volume_spike", "drawdown"], volume=volume) == path
    assert brief(["final_session_share", "news"], dividend=DIVIDEND) == \
        "AIOK jumped 182.8% on Oct 23; a $3.50 dividend, 82.35% of the prior close, went ex on Oct 21."
    # Without the indexes' trade-date moves, both sides of the comparison use the window.
    assert brief(["final_session_share", "news"], volume={"ratio": 1.2}) == \
        "AIOK jumped 182.8% on Oct 23; over the last 5 sessions it rose 202.4%, while the S&P 500 fell 0.47% and the NASDAQ Composite fell 0.71%."
    assert brief(["final_session_share", "vs_nasdaq"], analysis(AIOK, ixic=0.00003)) == \
        "AIOK jumped 182.8% on Oct 23; over the last 5 sessions it rose 202.4%, while the NASDAQ Composite was flat."
    assert brief(["final_session_share", "news"], analysis(AIOK)) == "AIOK jumped 182.8% on Oct 23."
    # With them, the brief compares a day with a day.
    assert brief(["window_return", "vs_sp500"], index_day={"GSPC": -0.0052, "IXIC": 0.0004}) == \
        "AIOK jumped 182.8% on Oct 23, while the S&P 500 fell 0.52% and the NASDAQ Composite rose 0.04% that day."
    assert brief(["data_limitation", "vs_sp500"]).endswith("the data limitations below apply before relying on this result.")
    assert brief([], {}) == "AIOK jumped 182.8% on Oct 23."


def test_dividend_is_measured_against_the_close_before_its_ex_date():
    out = lines(analysis([0.01] * 5), dividend=DIVIDEND)
    assert out["dividend"] == ("A $3.50 per-share dividend went ex on Oct 21, equal to 82.35% of the Oct 20 close of $4.25; "
                               "these closes are not dividend-adjusted, so the Oct 21 return includes the payout coming off the price.")
    first_day = lines(analysis([0.01] * 5), dividend={**DIVIDEND, "prior_close": None, "prior_date": None, "in_returns": False})
    assert first_day["dividend"] == "A $3.50 per-share dividend went ex on Oct 21."
    assert "dividend" not in lines(analysis([0.01] * 5))


def test_headline_sourced_dividend_is_attributed_to_its_source():
    reported = {"amount": 3.5, "prior_date": "2026-10-22", "prior_close": 4.25, "ref": "n3", "publisher": "GlobeNewswire"}
    assert lines(analysis([0.01] * 5), dividend=reported)["dividend"] == (
        "The GlobeNewswire headline [n3] states a $3.50 per-share dividend, equal to 82.35% of AIOK's Oct 22 close of $4.25, "
        "the last close before it was published; the share price typically drops by about the dividend on the ex-dividend date.")
    assert brief(["final_session_share", "dividend"], dividend=reported) == \
        "AIOK jumped 182.8% on Oct 23; a $3.50 per-share dividend reported in a headline equals 82.35% of the prior close."


def test_stock_specific_needs_a_wide_move_and_quiet_indexes():
    assert readable.stock_specific(analysis(AIOK, gspc=-0.0047, ixic=-0.0071))
    assert not readable.stock_specific(analysis(AIOK, gspc=-0.0047))                  # one index missing
    assert readable.stock_specific(analysis([0.02] * 5, gspc=-0.0047, ixic=-0.0071))      # +10.4% vs indexes under a tenth of it
    assert not readable.stock_specific(analysis([0.02] * 5, gspc=-0.02, ixic=-0.015))     # well ahead, but indexes moved more
    assert not readable.stock_specific(analysis(AIOK, gspc=0.30, ixic=0.25))


def test_prior_shock_sentence_does_not_infer_a_cause_or_claim_no_news_exists():
    analysis = {"daily_returns": [-0.05, -0.065, -0.34, 0.068, 0.747]}
    bare = readable.sentences(analysis, symbol="NUWE", trade_date="2026-10-27",
                              shock={"date": "2026-10-23", "return": -0.34, "ref": None})
    assert bare['prior_shock'] == 'NUWE fell 34.00% on Oct 23; the cause has not been verified for this report.'
    covered = readable.sentences(analysis, symbol="NUWE", trade_date="2026-10-27",
                                 shock={"date": "2026-10-23", "return": -0.34, "ref": "n2"})
    assert 'a fetched headline is dated that day' in covered['prior_shock']
    assert 'the cause has not been verified' in covered['prior_shock']
    assert 'no news' not in bare['prior_shock']
