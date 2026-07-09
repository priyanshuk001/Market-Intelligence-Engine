"""
orchestrator.py

The main pipeline for the Stock Sentiment Analysis MVP.

run_analysis(symbol) is the single entry point that ties together every
other module:

  data_fetchers  -> analyzers -> scoring -> llm_reasoning -> final report

This file is intentionally sequential (no async) for the MVP. Each step's
output feeds the next. Every step handles None / [] inputs gracefully, so
a missing data source degrades the report rather than crashing the
pipeline.

main.py and test_run.py both call run_analysis() as their only entry point
into the pipeline.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import analyzers
import data_fetchers
import llm_reasoning
import scoring
import config


logger = logging.getLogger(__name__)


def run_analysis(symbol: str) -> dict[str, Any]:
    """
    Run the complete sentiment analysis pipeline for a single stock symbol.

    Steps:
      1. Fetch raw news, price data, and market context.
      2. Analyze news sentiment (FinBERT), price action signals, and
         technical indicator signals.
      3. Compute News, Price, and Technical scores.
      4. Aggregate scores into a final verdict and detect contradictions.
      5. Select top articles by effective_weight for the LLM prompt.
      6. Build the LLM prompt, call Claude, and parse the response.
      7. Assemble and return the final report dict.

    This function does not raise under normal failure conditions (missing
    news, invalid symbol, LLM failure, etc.) -- each step degrades
    gracefully and the resulting report reflects what data was and wasn't
    available. It may raise only on truly unexpected programming errors.

    Args:
        symbol: Stock ticker symbol, e.g. "TATAMOTORS". Expected to be the
            raw NSE symbol (no exchange suffix).

    Returns:
        A dict matching the final report shape described in the project's
        architecture specification (symbol, timestamp, overall_sentiment,
        final_score, component_scores, weights_used, contradiction,
        market_context, top_articles, reasoning, positive_factors,
        negative_factors, risk_factors, summary).
    """
    symbol = symbol.strip().upper()
    logger.info("=" * 60)
    logger.info("Starting analysis for symbol=%s", symbol)
    logger.info("=" * 60)

    # -----------------------------------------------------------------
    # Step 1: Fetch raw data
    # -----------------------------------------------------------------
    logger.info("Step 1/7: Fetching raw data...")

    raw_articles = data_fetchers.fetch_news(symbol)
    price_data = data_fetchers.fetch_price_data(symbol)
    market_context = data_fetchers.fetch_market_context()

    logger.info(
        "Step 1/7 complete: articles=%d, price_data=%s, market_context=%s",
        len(raw_articles),
        "available" if price_data is not None else "unavailable",
        "available" if market_context is not None else "unavailable",
    )

    # -----------------------------------------------------------------
    # Step 2: Analyze
    # -----------------------------------------------------------------
    logger.info("Step 2/7: Running analysis (FinBERT sentiment, price action, technicals)...")

    analyzed_articles = analyzers.analyze_news(raw_articles)
    price_signals = analyzers.compute_price_signals(price_data)
    technical_signals = analyzers.compute_technical_signals(price_data)

    logger.info(
        "Step 2/7 complete: analyzed_articles=%d, price_signals=%s, technical_signals=%s",
        len(analyzed_articles),
        "available" if price_signals is not None else "unavailable",
        "available" if technical_signals is not None else "unavailable",
    )

    # -----------------------------------------------------------------
    # Step 3: Score
    # -----------------------------------------------------------------
    logger.info("Step 3/7: Computing component scores...")

    news_score = scoring.score_news(analyzed_articles)
    price_score = scoring.score_price(price_signals)
    technical_score = scoring.score_technical(technical_signals)

    logger.info(
        "Step 3/7 complete: news=%s, price=%s, technical=%s",
        news_score["score"],
        price_score["score"],
        technical_score["score"],
    )

    # -----------------------------------------------------------------
    # Step 4: Aggregate + contradiction detection
    # -----------------------------------------------------------------
    logger.info("Step 4/7: Aggregating scores and detecting contradictions...")

    aggregated = scoring.aggregate_scores(news_score, price_score, technical_score)
    contradiction = scoring.detect_contradiction(news_score, price_score)

    logger.info(
        "Step 4/7 complete: final_score=%s, sentiment=%s, has_contradiction=%s",
        aggregated["final_score"],
        aggregated["sentiment_label"],
        contradiction["has_contradiction"],
    )

    # -----------------------------------------------------------------
    # Step 5: Select top articles for the prompt
    # -----------------------------------------------------------------
    logger.info("Step 5/7: Selecting top articles for LLM prompt...")

    top_articles = _select_top_articles(analyzed_articles, config.TOP_ARTICLES_FOR_PROMPT)

    logger.info("Step 5/7 complete: selected %d/%d articles.", len(top_articles), len(analyzed_articles))

    # -----------------------------------------------------------------
    # Step 6: LLM reasoning
    # -----------------------------------------------------------------
    logger.info("Step 6/7: Building prompt and calling Claude...")

    prompt = llm_reasoning.build_prompt(
        symbol=symbol,
        aggregated=aggregated,
        news_score=news_score,
        price_score=price_score,
        technical_score=technical_score,
        contradiction=contradiction,
        top_articles=top_articles,
        price_signals=price_signals,
        technical_signals=technical_signals,
        market_context=market_context,
    )

    raw_response = llm_reasoning.call_gemini(prompt)
    llm_result = llm_reasoning.parse_response(raw_response)

    logger.info("Step 6/7 complete: llm_error=%s", llm_result["error"])

    # -----------------------------------------------------------------
    # Step 7: Assemble final report
    # -----------------------------------------------------------------
    logger.info("Step 7/7: Assembling final report...")

    report = _assemble_report(
        symbol=symbol,
        aggregated=aggregated,
        news_score=news_score,
        price_score=price_score,
        technical_score=technical_score,
        contradiction=contradiction,
        market_context=market_context,
        top_articles=top_articles,
        llm_result=llm_result,
    )

    logger.info("Analysis complete for symbol=%s. Overall sentiment: %s", symbol, report["overall_sentiment"])
    logger.info("=" * 60)

    return report


def _select_top_articles(
    analyzed_articles: list[analyzers.AnalyzedArticle],
    top_n: int,
) -> list[analyzers.AnalyzedArticle]:
    """
    Select the top `top_n` analyzed articles by effective_weight,
    descending.

    Args:
        analyzed_articles: Output of analyzers.analyze_news().
        top_n: Maximum number of articles to return.

    Returns:
        A list of at most `top_n` AnalyzedArticle dicts, sorted by
        effective_weight descending. Returns [] if `analyzed_articles`
        is empty.
    """
    if not analyzed_articles:
        return []

    sorted_articles = sorted(
        analyzed_articles, key=lambda article: article["effective_weight"], reverse=True
    )
    return sorted_articles[:top_n]


def _assemble_report(
    symbol: str,
    aggregated: scoring.AggregatedResult,
    news_score: scoring.NewsScore,
    price_score: scoring.PriceScore,
    technical_score: scoring.TechnicalScore,
    contradiction: scoring.ContradictionResult,
    market_context: data_fetchers.MarketContext | None,
    top_articles: list[analyzers.AnalyzedArticle],
    llm_result: llm_reasoning.LLMResult,
) -> dict[str, Any]:
    """
    Assemble the final report dict from all pipeline outputs.

    Pure assembly -- no computation, no inference. This is the exact shape
    returned by run_analysis() and serialized as JSON by main.py.

    Args:
        symbol: The (normalized, uppercase) stock symbol.
        aggregated: Output of scoring.aggregate_scores().
        news_score: Output of scoring.score_news().
        price_score: Output of scoring.score_price().
        technical_score: Output of scoring.score_technical().
        contradiction: Output of scoring.detect_contradiction().
        market_context: Output of data_fetchers.fetch_market_context(), or
            None.
        top_articles: Output of _select_top_articles().
        llm_result: Output of llm_reasoning.parse_response().

    Returns:
        The final report dict.
    """
    return {
        "symbol": symbol,
        "timestamp": datetime.now(timezone.utc),
        "overall_sentiment": aggregated["sentiment_label"],
        "final_score": aggregated["final_score"],
        "component_scores": {
            "news": {
                "score": news_score["score"],
                "article_count": news_score["article_count"],
                "confidence": news_score["confidence"],
            },
            "price": {
                "score": price_score["score"],
                "contributing_signals": price_score["contributing_signals"],
            },
            "technical": {
                "score": technical_score["score"],
                "contributing_signals": technical_score["contributing_signals"],
            },
        },
        "weights_used": aggregated["weights_used"],
        "contradiction": {
            "has_contradiction": contradiction["has_contradiction"],
            "severity": contradiction["severity"],
            "description": contradiction["description"],
        },
        "market_context": market_context,
        "top_articles": [
            {
                "headline": article["headline"],
                "sentiment": article["sentiment"],
                "sentiment_confidence": article["sentiment_confidence"],
                "effective_weight": article["effective_weight"],
                "published_at": article["published_at"],
            }
            for article in top_articles
        ],
        "reasoning": llm_result["reasoning"],
        "positive_factors": llm_result["positive_factors"],
        "negative_factors": llm_result["negative_factors"],
        "risk_factors": llm_result["risk_factors"],
        "summary": llm_result["summary"],
        "llm_error": llm_result["error"],
    }