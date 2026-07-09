"""
scoring.py

Pure, deterministic scoring functions for the Stock Sentiment Analysis MVP.

Every function in this file:
  - Takes structured signal/score dicts as input.
  - Performs only arithmetic and rule-based comparisons.
  - Makes no external calls (no network, no AI, no LLM).
  - Is fully testable with hand-written sample dicts.

Functions:
  - score_news()
  - score_price()
  - score_technical()
  - aggregate_scores()
  - detect_contradiction()
"""

from __future__ import annotations

import logging
from typing import Optional, TypedDict

import config
from analyzers import AnalyzedArticle, PriceSignals, TechnicalSignals


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed Return Shapes
# ---------------------------------------------------------------------------

class NewsScore(TypedDict):
    score: Optional[float]
    article_count: int
    confidence: str  # "high" | "medium" | "low"


class PriceScore(TypedDict):
    score: Optional[float]
    contributing_signals: list[str]


class TechnicalScore(TypedDict):
    score: Optional[float]
    contributing_signals: list[str]


class AggregatedResult(TypedDict):
    final_score: Optional[float]
    sentiment_label: str
    weights_used: dict[str, float]


class ContradictionResult(TypedDict):
    has_contradiction: bool
    severity: str  # "none" | "low" | "medium" | "high"
    description: str


# ---------------------------------------------------------------------------
# Sentiment -> Numeric Mapping
# ---------------------------------------------------------------------------

_SENTIMENT_VALUE = {
    "positive": 1.0,
    "neutral": 0.0,
    "negative": -1.0,
}


# ---------------------------------------------------------------------------
# News Score
# ---------------------------------------------------------------------------

def score_news(analyzed_articles: list[AnalyzedArticle]) -> NewsScore:
    """
    Compute a 0-100 News Score from analyzed articles.

    Each article's sentiment (positive=+1, neutral=0, negative=-1) is
    weighted by its `effective_weight` (credibility x time decay, computed
    in analyzers.analyze_news). The weighted average is then normalized
    from the [-1, 1] range to [0, 100], where 50 is neutral.

    Args:
        analyzed_articles: Output of analyzers.analyze_news().

    Returns:
        A NewsScore dict. If `analyzed_articles` is empty, `score` is None
        and `confidence` is "low".

    Confidence is based on article_count:
        < 3 articles  -> "low"
        3-7 articles  -> "medium"
        8+ articles   -> "high"
    """
    article_count = len(analyzed_articles)

    if article_count == 0:
        logger.info("score_news called with 0 articles; returning score=None.")
        return NewsScore(score=None, article_count=0, confidence="low")

    total_weight = 0.0
    weighted_sum = 0.0

    for article in analyzed_articles:
        sentiment_value = _SENTIMENT_VALUE.get(article["sentiment"], 0.0)
        weight = article["effective_weight"]

        weighted_sum += sentiment_value * weight
        total_weight += weight

    if total_weight <= 0:
        # All effective weights were zero or negative (shouldn't normally
        # happen since credibility and decay are both positive, but guard
        # against division by zero defensively).
        logger.warning(
            "score_news: total_weight is %.6f (<= 0) across %d articles; "
            "returning neutral score of 50.0.",
            total_weight,
            article_count,
        )
        weighted_average = 0.0
    else:
        weighted_average = weighted_sum / total_weight

    # Map from [-1, 1] to [0, 100]. weighted_average=-1 -> 0, 0 -> 50, 1 -> 100.
    score = (weighted_average + 1.0) * 50.0
    score = max(0.0, min(100.0, score))

    confidence = _classify_news_confidence(article_count)

    result = NewsScore(score=score, article_count=article_count, confidence=confidence)
    logger.info("Computed news score: %s", result)
    return result


def _classify_news_confidence(article_count: int) -> str:
    """
    Classify confidence in the news score based on article count.
    """
    if article_count < 3:
        return "low"
    elif article_count <= 7:
        return "medium"
    else:
        return "high"


# ---------------------------------------------------------------------------
# Price Score
# ---------------------------------------------------------------------------

def score_price(price_signals: Optional[PriceSignals]) -> PriceScore:
    """
    Compute a 0-100 Price Score from price action signals using a
    point-based system starting from a 50-point baseline.

    Scoring rules (each may add or subtract from the baseline):
        trend_direction == "uptrend"          -> +15
        trend_direction == "downtrend"        -> -15
        volume_signal == "high"               -> +5
        volume_signal == "low"                -> -5
        near_52w_high == True                 -> +7
        near_52w_low == True                  -> -7
        day_change_pct > 2                    -> +5
        day_change_pct < -2                   -> -5

    The result is capped to [0, 100].

    Args:
        price_signals: Output of analyzers.compute_price_signals(), or None.

    Returns:
        A PriceScore dict. If price_signals is None, `score` is None and
        `contributing_signals` is [].
    """
    if price_signals is None:
        logger.info("score_price called with price_signals=None; returning score=None.")
        return PriceScore(score=None, contributing_signals=[])

    points = 50.0
    contributing_signals: list[str] = []

    trend_direction = price_signals["trend_direction"]
    if trend_direction == "uptrend":
        points += 15
        contributing_signals.append("uptrend (+15)")
    elif trend_direction == "downtrend":
        points -= 15
        contributing_signals.append("downtrend (-15)")

    volume_signal = price_signals["volume_signal"]
    if volume_signal == "high":
        points += 5
        contributing_signals.append("high volume (+5)")
    elif volume_signal == "low":
        points -= 5
        contributing_signals.append("low volume (-5)")

    if price_signals["near_52w_high"]:
        points += 7
        contributing_signals.append("near 52-week high (+7)")

    if price_signals["near_52w_low"]:
        points -= 7
        contributing_signals.append("near 52-week low (-7)")

    day_change_pct = price_signals["day_change_pct"]
    if day_change_pct > 2:
        points += 5
        contributing_signals.append(f"strong day gain {day_change_pct:.2f}% (+5)")
    elif day_change_pct < -2:
        points -= 5
        contributing_signals.append(f"strong day decline {day_change_pct:.2f}% (-5)")

    score = max(0.0, min(100.0, points))

    result = PriceScore(score=score, contributing_signals=contributing_signals)
    logger.info("Computed price score: %s", result)
    return result


# ---------------------------------------------------------------------------
# Technical Score
# ---------------------------------------------------------------------------

def score_technical(technical_signals: Optional[TechnicalSignals]) -> TechnicalScore:
    """
    Compute a 0-100 Technical Score from technical indicator signals using
    a point-based system starting from a 50-point baseline.

    Scoring rules (each may add or subtract from the baseline):
        price_vs_ema20 == "above"   -> +10
        price_vs_ema20 == "below"   -> -10
        price_vs_ema50 == "above"   -> +10
        price_vs_ema50 == "below"   -> -10
        macd_trend == "bullish"     -> +10
        macd_trend == "bearish"     -> -10
        rsi_zone == "oversold"      -> +5
        rsi_zone == "overbought"    -> -5

    The result is capped to [0, 100].

    Args:
        technical_signals: Output of analyzers.compute_technical_signals(),
            or None.

    Returns:
        A TechnicalScore dict. If technical_signals is None, `score` is
        None and `contributing_signals` is [].
    """
    if technical_signals is None:
        logger.info(
            "score_technical called with technical_signals=None; returning score=None."
        )
        return TechnicalScore(score=None, contributing_signals=[])

    points = 50.0
    contributing_signals: list[str] = []

    if technical_signals["price_vs_ema20"] == "above":
        points += 10
        contributing_signals.append("price above EMA20 (+10)")
    else:
        points -= 10
        contributing_signals.append("price below EMA20 (-10)")

    if technical_signals["price_vs_ema50"] == "above":
        points += 10
        contributing_signals.append("price above EMA50 (+10)")
    else:
        points -= 10
        contributing_signals.append("price below EMA50 (-10)")

    if technical_signals["macd_trend"] == "bullish":
        points += 10
        contributing_signals.append("MACD bullish (+10)")
    else:
        points -= 10
        contributing_signals.append("MACD bearish (-10)")

    rsi_zone = technical_signals["rsi_zone"]
    if rsi_zone == "oversold":
        points += 5
        contributing_signals.append("RSI oversold (+5)")
    elif rsi_zone == "overbought":
        points -= 5
        contributing_signals.append("RSI overbought (-5)")

    score = max(0.0, min(100.0, points))

    result = TechnicalScore(score=score, contributing_signals=contributing_signals)
    logger.info("Computed technical score: %s", result)
    return result


# ---------------------------------------------------------------------------
# Score Aggregation
# ---------------------------------------------------------------------------

def aggregate_scores(
    news_score: NewsScore,
    price_score: PriceScore,
    technical_score: TechnicalScore,
) -> AggregatedResult:
    """
    Combine News, Price, and Technical scores into a single final score
    using the weights from config.py (NEWS_WEIGHT, PRICE_WEIGHT,
    TECHNICAL_WEIGHT).

    If any component score is None, its configured weight is redistributed
    proportionally across the remaining available scores so the effective
    weights always sum to 1.0.

    The final 0-100 score is mapped to a sentiment label using
    config.SENTIMENT_THRESHOLDS:
        >= 70 -> "STRONGLY_BULLISH"
        >= 58 -> "BULLISH"
        >= 43 -> "NEUTRAL"
        >= 31 -> "BEARISH"
        <  31 -> "STRONGLY_BEARISH"

    Args:
        news_score: Output of score_news().
        price_score: Output of score_price().
        technical_score: Output of score_technical().

    Returns:
        An AggregatedResult dict. If all three component scores are None,
        `final_score` is None and `sentiment_label` is
        "INSUFFICIENT_DATA", with `weights_used` as {}.
    """
    available: dict[str, float] = {}
    base_weights: dict[str, float] = {
        "news": config.NEWS_WEIGHT,
        "price": config.PRICE_WEIGHT,
        "technical": config.TECHNICAL_WEIGHT,
    }

    if news_score["score"] is not None:
        available["news"] = news_score["score"]
    if price_score["score"] is not None:
        available["price"] = price_score["score"]
    if technical_score["score"] is not None:
        available["technical"] = technical_score["score"]

    if not available:
        logger.warning(
            "aggregate_scores: all component scores are None; "
            "returning INSUFFICIENT_DATA."
        )
        return AggregatedResult(
            final_score=None,
            sentiment_label="INSUFFICIENT_DATA",
            weights_used={},
        )

    # Redistribute weight proportionally across available components.
    total_available_weight = sum(base_weights[name] for name in available)

    if total_available_weight <= 0:
        # Defensive: should not happen given config weights sum to 1.0 and
        # are all positive, but guard against misconfiguration.
        logger.error(
            "aggregate_scores: total_available_weight is %.6f (<= 0); "
            "falling back to equal weighting across available components.",
            total_available_weight,
        )
        equal_weight = 1.0 / len(available)
        effective_weights = {name: equal_weight for name in available}
    else:
        effective_weights = {
            name: base_weights[name] / total_available_weight for name in available
        }

    final_score = sum(available[name] * effective_weights[name] for name in available)
    final_score = max(0.0, min(100.0, final_score))

    sentiment_label = _classify_sentiment(final_score)

    result = AggregatedResult(
        final_score=final_score,
        sentiment_label=sentiment_label,
        weights_used=effective_weights,
    )
    logger.info("Computed aggregated result: %s", result)
    return result


def _classify_sentiment(final_score: float) -> str:
    """
    Map a final 0-100 score to a sentiment label using
    config.SENTIMENT_THRESHOLDS.

    config.SENTIMENT_THRESHOLDS defines the *minimum* score for each label
    (except STRONGLY_BEARISH, which is the implicit fallback below the
    BEARISH threshold). Thresholds are checked from highest to lowest.
    """
    thresholds = config.SENTIMENT_THRESHOLDS

    if final_score >= thresholds["STRONGLY_BULLISH"]:
        return "STRONGLY_BULLISH"
    elif final_score >= thresholds["BULLISH"]:
        return "BULLISH"
    elif final_score >= thresholds["NEUTRAL"]:
        return "NEUTRAL"
    elif final_score >= thresholds["BEARISH"]:
        return "BEARISH"
    else:
        return "STRONGLY_BEARISH"


# ---------------------------------------------------------------------------
# Contradiction Detection
# ---------------------------------------------------------------------------

def detect_contradiction(news_score: NewsScore, price_score: PriceScore) -> ContradictionResult:
    """
    Detect a contradiction between the News Score and the Price Score.

    A contradiction is flagged when the absolute difference between the
    two scores exceeds config.CONTRADICTION_THRESHOLD_LOW. Severity is
    classified using config.CONTRADICTION_THRESHOLD_LOW and
    config.CONTRADICTION_THRESHOLD_HIGH:

        difference <= CONTRADICTION_THRESHOLD_LOW                    -> "none"
        CONTRADICTION_THRESHOLD_LOW < difference <= THRESHOLD_HIGH    -> "low"
            (Note: see implementation for the exact "low"/"medium" split)
        difference > CONTRADICTION_THRESHOLD_HIGH                     -> "high"

    Args:
        news_score: Output of score_news().
        price_score: Output of score_price().

    Returns:
        A ContradictionResult dict. If either score is None, returns
        has_contradiction=False, severity="none", with a description
        noting insufficient data.
    """
    if news_score["score"] is None or price_score["score"] is None:
        logger.info(
            "detect_contradiction: news_score or price_score is None; "
            "returning no contradiction."
        )
        return ContradictionResult(
            has_contradiction=False,
            severity="none",
            description="Insufficient data to compare news and price sentiment.",
        )

    news_value = news_score["score"]
    price_value = price_score["score"]
    difference = abs(news_value - price_value)

    low_threshold = config.CONTRADICTION_THRESHOLD_LOW
    high_threshold = config.CONTRADICTION_THRESHOLD_HIGH

    if difference <= low_threshold:
        severity = "none"
        has_contradiction = False
    elif difference <= high_threshold:
        severity = "medium"
        has_contradiction = True
    else:
        severity = "high"
        has_contradiction = True

    if has_contradiction:
        news_direction = "bullish" if news_value > price_value else "bearish"
        price_direction = "bearish" if news_value > price_value else "bullish"
        description = (
            f"News sentiment is {news_direction} ({news_value:.1f}) but price "
            f"action is {price_direction} ({price_value:.1f}) "
            f"(difference: {difference:.1f})."
        )
    else:
        description = (
            f"News score ({news_value:.1f}) and price score ({price_value:.1f}) "
            f"are in general agreement (difference: {difference:.1f})."
        )

    result = ContradictionResult(
        has_contradiction=has_contradiction,
        severity=severity,
        description=description,
    )
    logger.info("Computed contradiction result: %s", result)
    return result