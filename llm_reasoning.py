"""
llm_reasoning.py

The LLM explanation layer for the Stock Sentiment Analysis MVP.

This is the ONLY file that communicates with the Google Gemini API.
Its role is strictly to EXPLAIN pre-computed, deterministic scores in
plain language -- it never generates or overrides any numeric score.

Functions:
  - build_prompt()    -> assembles a structured prompt from pipeline outputs
  - call_gemini()     -> sends the prompt to Gemini, returns raw text
  - parse_response()  -> parses Gemini's JSON response, with safe fallback
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional, TypedDict

from google import genai
from google.genai import types

import config
from analyzers import AnalyzedArticle, PriceSignals, TechnicalSignals
from data_fetchers import MarketContext
from scoring import AggregatedResult, ContradictionResult, NewsScore, PriceScore, TechnicalScore


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed Return Shapes
# ---------------------------------------------------------------------------

class LLMResult(TypedDict):
    reasoning: str
    positive_factors: list[str]
    negative_factors: list[str]
    risk_factors: list[str]
    summary: str
    error: bool


# ---------------------------------------------------------------------------
# Gemini Client (module-level, created once)
# ---------------------------------------------------------------------------

_client = genai.Client(api_key=config.GEMINI_API_KEY)


# ---------------------------------------------------------------------------
# Prompt Building
# ---------------------------------------------------------------------------

def build_prompt(
    symbol: str,
    aggregated: AggregatedResult,
    news_score: NewsScore,
    price_score: PriceScore,
    technical_score: TechnicalScore,
    contradiction: ContradictionResult,
    top_articles: list[AnalyzedArticle],
    price_signals: Optional[PriceSignals],
    technical_signals: Optional[TechnicalSignals],
    market_context: Optional[MarketContext],
) -> str:
    """
    Assemble a structured text prompt for Gemini from all upstream pipeline
    outputs.

    The prompt explicitly instructs Gemini to:
      - Treat all provided scores and signals as ground truth.
      - NOT invent, recalculate, or override any numeric score.
      - Explain WHY the sentiment is what it is, using only the data given.
      - Respond in strict JSON with five specific fields.

    Returns:
        A single prompt string ready to send to call_gemini().
    """
    sections: list[str] = []

    sections.append(
        "You are a financial analyst assistant. You will be given "
        "pre-computed sentiment scores and market signals for an Indian "
        "stock. Your task is to EXPLAIN these results in plain language.\n\n"
        "CRITICAL RULES:\n"
        "- Do NOT generate, recalculate, or override any numeric score.\n"
        "- Do NOT invent facts, numbers, or events not present in the data below.\n"
        "- Use ONLY the data provided to construct your explanation.\n"
        "- If data for a component is marked as unavailable, acknowledge "
        "that in your reasoning rather than guessing."
    )

    sections.append(f"\n--- STOCK ---\nSymbol: {symbol}")

    sections.append(
        "\n--- FINAL VERDICT (already computed, do not change) ---\n"
        f"Overall Sentiment: {aggregated['sentiment_label']}\n"
        f"Final Score (0-100): {_format_score(aggregated['final_score'])}\n"
        f"Weights Used: {_format_weights(aggregated['weights_used'])}"
    )

    sections.append(
        "\n--- COMPONENT SCORES (already computed, do not change) ---\n"
        f"News Score: {_format_score(news_score['score'])} "
        f"(based on {news_score['article_count']} articles, "
        f"confidence: {news_score['confidence']})\n"
        f"Price Score: {_format_score(price_score['score'])}\n"
        f"  Contributing signals: {_format_signal_list(price_score['contributing_signals'])}\n"
        f"Technical Score: {_format_score(technical_score['score'])}\n"
        f"  Contributing signals: {_format_signal_list(technical_score['contributing_signals'])}"
    )

    sections.append(
        "\n--- CONTRADICTION ANALYSIS (already computed, do not change) ---\n"
        f"Has Contradiction: {contradiction['has_contradiction']}\n"
        f"Severity: {contradiction['severity']}\n"
        f"Description: {contradiction['description']}"
    )

    sections.append(_format_price_signals_section(price_signals))
    sections.append(_format_technical_signals_section(technical_signals))
    sections.append(_format_market_context_section(market_context))
    sections.append(_format_top_articles_section(top_articles))

    sections.append(
        "\n--- YOUR TASK ---\n"
        "Respond with ONLY a JSON object (no markdown code fences, no "
        "preamble, no extra text) with exactly these five fields:\n"
        "{\n"
        '  "reasoning": "A 3-5 sentence explanation of why the overall '
        'sentiment is what it is, referencing the scores and signals above.",\n'
        '  "positive_factors": ["short bullet point", "..."],\n'
        '  "negative_factors": ["short bullet point", "..."],\n'
        '  "risk_factors": ["short bullet point", "..."],\n'
        '  "summary": "A 1-2 sentence plain-English summary."\n'
        "}\n"
        "If there is a contradiction (has_contradiction is true), address "
        "it explicitly in your reasoning."
    )

    prompt = "\n".join(sections)
    logger.info("Built Gemini prompt for symbol=%s (length=%d chars)", symbol, len(prompt))
    return prompt


def _format_score(score: Optional[float]) -> str:
    if score is None:
        return "unavailable"
    return f"{score:.1f}"


def _format_weights(weights_used: dict[str, float]) -> str:
    if not weights_used:
        return "none (insufficient data)"
    parts = [f"{name}={weight:.2f}" for name, weight in weights_used.items()]
    return ", ".join(parts)


def _format_signal_list(signals: list[str]) -> str:
    if not signals:
        return "none"
    return "; ".join(signals)


def _format_price_signals_section(price_signals: Optional[PriceSignals]) -> str:
    if price_signals is None:
        return "\n--- PRICE ACTION SIGNALS ---\nUnavailable (no price data could be fetched)."
    return (
        "\n--- PRICE ACTION SIGNALS ---\n"
        f"Trend Direction: {price_signals['trend_direction']}\n"
        f"Volume Signal: {price_signals['volume_signal']}\n"
        f"Position in 52-Week Range (0=low, 1=high): "
        f"{price_signals['position_in_52w_range']:.2f}\n"
        f"Near 52-Week High: {price_signals['near_52w_high']}\n"
        f"Near 52-Week Low: {price_signals['near_52w_low']}\n"
        f"Day Change: {price_signals['day_change_pct']:.2f}%"
    )


def _format_technical_signals_section(technical_signals: Optional[TechnicalSignals]) -> str:
    if technical_signals is None:
        return (
            "\n--- TECHNICAL SIGNALS ---\n"
            "Unavailable (insufficient price history for indicators)."
        )
    return (
        "\n--- TECHNICAL SIGNALS ---\n"
        f"EMA20: {technical_signals['ema20']:.2f}\n"
        f"EMA50: {technical_signals['ema50']:.2f}\n"
        f"RSI(14): {technical_signals['rsi']:.2f} ({technical_signals['rsi_zone']})\n"
        f"MACD Line: {technical_signals['macd_line']:.4f}\n"
        f"MACD Signal Line: {technical_signals['macd_signal_line']:.4f}\n"
        f"MACD Trend: {technical_signals['macd_trend']}\n"
        f"Price vs EMA20: {technical_signals['price_vs_ema20']}\n"
        f"Price vs EMA50: {technical_signals['price_vs_ema50']}"
    )


def _format_market_context_section(market_context: Optional[MarketContext]) -> str:
    if market_context is None:
        return "\n--- BROADER MARKET CONTEXT ---\nUnavailable."
    return (
        "\n--- BROADER MARKET CONTEXT ---\n"
        f"NIFTY 50 Value: {market_context['nifty_value']:.2f}\n"
        f"NIFTY 50 Day Change: {market_context['nifty_change_pct']:.2f}%"
    )


def _format_top_articles_section(top_articles: list[AnalyzedArticle]) -> str:
    if not top_articles:
        return "\n--- TOP NEWS ARTICLES ---\nNo recent news articles found."
    lines = ["\n--- TOP NEWS ARTICLES (most influential, by weight) ---"]
    for i, article in enumerate(top_articles, start=1):
        lines.append(
            f"{i}. \"{article['headline']}\" "
            f"-- sentiment: {article['sentiment']} "
            f"(confidence: {article['sentiment_confidence']:.2f})"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gemini API Call
# ---------------------------------------------------------------------------

def call_gemini(prompt: str) -> str:
    """
    Send the prompt to Gemini and return the raw text response.

    Uses the new google-genai SDK (google.genai). No retry logic in the
    MVP -- on any failure logs the exception and returns "" so
    parse_response() can return the fallback template gracefully.

    Returns:
        The raw text content of Gemini's response, or "" on failure.
    """
    logger.info("Calling Gemini (model=%s)...", config.GEMINI_MODEL)

    try:
        response = _client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=1024,
            ),
        )
    except Exception:
        logger.exception("Gemini API call failed.")
        return ""

    if not response.candidates:
        logger.warning("Gemini returned no candidates (possible safety block).")
        return ""

    try:
        raw_text = response.text
    except Exception:
        logger.exception(
            "Failed to extract text from Gemini response "
            "(possible safety block or empty content)."
        )
        return ""

    if not raw_text or not raw_text.strip():
        logger.warning("Gemini returned an empty text response.")
        return ""

    logger.info("Received Gemini response (length=%d chars).", len(raw_text))
    return raw_text


# ---------------------------------------------------------------------------
# Response Parsing
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS = ("reasoning", "positive_factors", "negative_factors", "risk_factors", "summary")

_FALLBACK_RESULT: LLMResult = LLMResult(
    reasoning="Unable to generate detailed reasoning at this time.",
    positive_factors=[],
    negative_factors=[],
    risk_factors=[],
    summary="Analysis completed but explanation generation failed.",
    error=True,
)

_CODE_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_response(raw_response: str) -> LLMResult:
    """
    Parse Gemini's raw text response into an LLMResult.

    Handles markdown code fences (```json ... ```) which Gemini frequently
    adds despite instructions not to.

    Returns:
        An LLMResult dict. On any parsing or validation failure returns
        a copy of _FALLBACK_RESULT (error=True).
    """
    if not raw_response or not raw_response.strip():
        logger.warning("parse_response received empty raw_response; using fallback.")
        return dict(_FALLBACK_RESULT)

    cleaned = _CODE_FENCE_PATTERN.sub("", raw_response.strip()).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.exception(
            "Failed to parse Gemini response as JSON. Raw response: %r", raw_response
        )
        return dict(_FALLBACK_RESULT)

    if not isinstance(parsed, dict):
        logger.error("Parsed Gemini response is not a JSON object: %r", parsed)
        return dict(_FALLBACK_RESULT)

    missing_fields = [f for f in _REQUIRED_FIELDS if f not in parsed]
    if missing_fields:
        logger.error(
            "Gemini response missing required fields: %s. Got: %s",
            missing_fields,
            list(parsed.keys()),
        )
        return dict(_FALLBACK_RESULT)

    reasoning = parsed["reasoning"]
    summary = parsed["summary"]
    positive_factors = parsed["positive_factors"]
    negative_factors = parsed["negative_factors"]
    risk_factors = parsed["risk_factors"]

    if not isinstance(reasoning, str) or not reasoning.strip():
        logger.error("'reasoning' is not a non-empty string: %r", reasoning)
        return dict(_FALLBACK_RESULT)

    if not isinstance(summary, str) or not summary.strip():
        logger.error("'summary' is not a non-empty string: %r", summary)
        return dict(_FALLBACK_RESULT)

    for field_name, value in (
        ("positive_factors", positive_factors),
        ("negative_factors", negative_factors),
        ("risk_factors", risk_factors),
    ):
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            logger.error(
                "Field '%s' is not a list of strings: %r", field_name, value
            )
            return dict(_FALLBACK_RESULT)

    result = LLMResult(
        reasoning=reasoning.strip(),
        positive_factors=positive_factors,
        negative_factors=negative_factors,
        risk_factors=risk_factors,
        summary=summary.strip(),
        error=False,
    )
    logger.info("Successfully parsed Gemini response.")
    return result