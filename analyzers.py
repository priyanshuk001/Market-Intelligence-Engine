"""
analyzers.py

Transforms raw data (from data_fetchers.py) into structured signals.

Three analysis concerns:
  - analyze_news()              -> FinBERT sentiment + source credibility +
                                    exponential time decay, per article.
  - compute_price_signals()      -> Pure price-action signals derived from
                                    OHLCV data.
  - compute_technical_signals()  -> EMA20/EMA50/RSI14/MACD via pandas-ta,
                                    plus derived structural signals.

FinBERT model loading happens once at module import time (module-level
globals below), not per-call. If loading fails, this module raises
immediately on import — better to fail loudly at application startup than
silently mid-request.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Optional, TypedDict

import pandas as pd
import ta as ta_lib
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config
from data_fetchers import NewsArticle, PriceData


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed Return Shapes
# ---------------------------------------------------------------------------

class AnalyzedArticle(TypedDict):
    headline: str
    sentiment: str  # "positive" | "negative" | "neutral"
    sentiment_confidence: float
    credibility_score: float
    time_decay_weight: float
    effective_weight: float
    published_at: datetime


class PriceSignals(TypedDict):
    trend_direction: str  # "uptrend" | "downtrend" | "sideways"
    volume_signal: str  # "high" | "normal" | "low"
    position_in_52w_range: float
    near_52w_high: bool
    near_52w_low: bool
    day_change_pct: float


class TechnicalSignals(TypedDict):
    ema20: float
    ema50: float
    rsi: float
    macd_line: float
    macd_signal_line: float
    price_vs_ema20: str  # "above" | "below"
    price_vs_ema50: str  # "above" | "below"
    macd_trend: str  # "bullish" | "bearish"
    rsi_zone: str  # "oversold" | "neutral" | "overbought"


# ---------------------------------------------------------------------------
# FinBERT Model Loading (module-level, executed once at import time)
# ---------------------------------------------------------------------------

_FINBERT_MODEL_NAME = "ProsusAI/finbert"

# FinBERT's label order for ProsusAI/finbert is:
#   0 -> positive, 1 -> negative, 2 -> neutral
# This is fixed by the model's config and is documented in its model card.
_FINBERT_LABELS = ["positive", "negative", "neutral"]

logger.info("Loading FinBERT model '%s' (this happens once at startup)...", _FINBERT_MODEL_NAME)

try:
    _finbert_tokenizer = AutoTokenizer.from_pretrained(_FINBERT_MODEL_NAME)
    _finbert_model = AutoModelForSequenceClassification.from_pretrained(_FINBERT_MODEL_NAME)
    _finbert_model.eval()
except Exception:
    logger.exception("Failed to load FinBERT model '%s' at startup", _FINBERT_MODEL_NAME)
    raise

logger.info("FinBERT model loaded successfully.")


# ---------------------------------------------------------------------------
# News Analysis
# ---------------------------------------------------------------------------

def analyze_news(articles: list[NewsArticle]) -> list[AnalyzedArticle]:
    """
    Analyze each news article's sentiment using FinBERT, attach source
    credibility (from config.SOURCE_CREDIBILITY) and an exponential
    time-decay weight, then combine these into an effective_weight used
    for downstream scoring.

    Args:
        articles: Output of data_fetchers.fetch_news().

    Returns:
        A list of AnalyzedArticle dicts, one per input article. Returns []
        if `articles` is empty. This function does not raise — if FinBERT
        inference fails for an individual article, that article is skipped
        and logged, and processing continues with the remaining articles.
    """
    if not articles:
        logger.info("analyze_news called with 0 articles; returning [].")
        return []

    logger.info("Analyzing sentiment for %d articles using FinBERT.", len(articles))

    headlines = [article["headline"] for article in articles]

    try:
        sentiments = _run_finbert_batch(headlines)
    except Exception:
        logger.exception(
            "FinBERT batch inference failed for %d articles; returning [].",
            len(articles),
        )
        return []

    now = datetime.now(timezone.utc)
    analyzed: list[AnalyzedArticle] = []

    for article, (sentiment_label, sentiment_confidence) in zip(articles, sentiments):
        credibility_score = config.SOURCE_CREDIBILITY.get(
            article["source_domain"], config.SOURCE_CREDIBILITY_DEFAULT
        )

        time_decay_weight = _compute_time_decay(article["published_at"], now)
        effective_weight = credibility_score * time_decay_weight

        analyzed.append(
            AnalyzedArticle(
                headline=article["headline"],
                sentiment=sentiment_label,
                sentiment_confidence=sentiment_confidence,
                credibility_score=credibility_score,
                time_decay_weight=time_decay_weight,
                effective_weight=effective_weight,
                published_at=article["published_at"],
            )
        )

    logger.info("Successfully analyzed %d/%d articles.", len(analyzed), len(articles))
    return analyzed


def _run_finbert_batch(headlines: list[str]) -> list[tuple[str, float]]:
    """
    Run FinBERT inference on a batch of headlines.

    Args:
        headlines: List of headline strings.

    Returns:
        A list of (sentiment_label, confidence) tuples, one per headline,
        in the same order as the input. sentiment_label is one of
        "positive", "negative", "neutral". confidence is the softmax
        probability of the predicted label (0.0 - 1.0).

    Raises:
        Any exception from tokenization or model inference is propagated
        to the caller (analyze_news), which handles it.
    """
    inputs = _finbert_tokenizer(
        headlines,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    )

    with torch.no_grad():
        outputs = _finbert_model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

    results: list[tuple[str, float]] = []
    for row in probabilities:
        predicted_index = int(torch.argmax(row).item())
        confidence = float(row[predicted_index].item())
        sentiment_label = _FINBERT_LABELS[predicted_index]
        results.append((sentiment_label, confidence))

    return results


def _compute_time_decay(published_at: datetime, now: datetime) -> float:
    """
    Compute an exponential time-decay weight for a news article.

    decay_weight = e^(-TIME_DECAY_LAMBDA * hours_since_published)

    Args:
        published_at: When the article was published (timezone-aware UTC).
        now: The current time (timezone-aware UTC), passed explicitly so
             all articles in a batch are decayed relative to the same
             reference point.

    Returns:
        A float in (0.0, 1.0]. Articles published in the future (clock
        skew / feed inconsistencies) are clamped to a decay of 1.0
        (hours_since_published treated as 0).
    """
    hours_since_published = (now - published_at).total_seconds() / 3600.0

    if hours_since_published < 0:
        logger.debug(
            "Article published_at=%s is in the future relative to now=%s; "
            "clamping hours_since_published to 0.",
            published_at,
            now,
        )
        hours_since_published = 0.0

    return math.exp(-config.TIME_DECAY_LAMBDA * hours_since_published)


# ---------------------------------------------------------------------------
# Price Action Signals
# ---------------------------------------------------------------------------

def compute_price_signals(price_data: Optional[PriceData]) -> Optional[PriceSignals]:
    """
    Derive price-action signals from raw price data.

    Args:
        price_data: Output of data_fetchers.fetch_price_data(), or None.

    Returns:
        A PriceSignals dict, or None if price_data is None or does not
        contain enough OHLCV history to compute a trend (at least 20 bars
        are needed for the 20-day moving average comparison).
    """
    if price_data is None:
        logger.info("compute_price_signals called with price_data=None; returning None.")
        return None

    ohlcv = price_data["ohlcv"]

    if len(ohlcv) < 20:
        logger.warning(
            "Insufficient OHLCV history to compute price signals "
            "(have %d bars, need >= 20); returning None.",
            len(ohlcv),
        )
        return None

    closes = pd.Series([bar["close"] for bar in ohlcv])

    current_price = price_data["current_price"]
    moving_average_20 = float(closes.tail(20).mean())

    trend_direction = _classify_trend(current_price, moving_average_20)
    volume_signal = _classify_volume(price_data["volume_today"], price_data["avg_volume_20d"])

    fifty_two_week_high = price_data["fifty_two_week_high"]
    fifty_two_week_low = price_data["fifty_two_week_low"]

    position_in_52w_range = _position_in_range(
        current_price, fifty_two_week_low, fifty_two_week_high
    )

    near_52w_high = _is_near(current_price, fifty_two_week_high, tolerance_pct=3.0)
    near_52w_low = _is_near(current_price, fifty_two_week_low, tolerance_pct=3.0)

    signals = PriceSignals(
        trend_direction=trend_direction,
        volume_signal=volume_signal,
        position_in_52w_range=position_in_52w_range,
        near_52w_high=near_52w_high,
        near_52w_low=near_52w_low,
        day_change_pct=price_data["day_change_pct"],
    )

    logger.info("Computed price signals: %s", signals)
    return signals


def _classify_trend(current_price: float, moving_average_20: float) -> str:
    """
    Classify trend direction by comparing current price to the 20-day
    moving average. A 0.5% buffer around the moving average is treated
    as "sideways" to avoid noise from marginal differences.
    """
    if moving_average_20 == 0:
        return "sideways"

    pct_diff = ((current_price - moving_average_20) / moving_average_20) * 100.0

    if pct_diff > 0.5:
        return "uptrend"
    elif pct_diff < -0.5:
        return "downtrend"
    else:
        return "sideways"


def _classify_volume(volume_today: int, avg_volume_20d: float) -> str:
    """
    Classify today's volume relative to the 20-day average.
    "high" if >= 1.5x average, "low" if <= 0.5x average, else "normal".
    """
    if avg_volume_20d <= 0:
        return "normal"

    ratio = volume_today / avg_volume_20d

    if ratio >= 1.5:
        return "high"
    elif ratio <= 0.5:
        return "low"
    else:
        return "normal"


def _position_in_range(current_price: float, low: float, high: float) -> float:
    """
    Compute where current_price sits within [low, high], normalized to
    0.0 (at low) - 1.0 (at high). Returns 0.5 if low == high (degenerate
    range, avoids division by zero).
    """
    if high == low:
        return 0.5

    position = (current_price - low) / (high - low)
    return max(0.0, min(1.0, position))


def _is_near(current_price: float, reference_price: float, tolerance_pct: float) -> bool:
    """
    Returns True if current_price is within tolerance_pct percent of
    reference_price. Returns False if reference_price is 0.
    """
    if reference_price == 0:
        return False

    pct_diff = abs((current_price - reference_price) / reference_price) * 100.0
    return pct_diff <= tolerance_pct


# ---------------------------------------------------------------------------
# Technical Indicator Signals
# ---------------------------------------------------------------------------

def compute_technical_signals(price_data: Optional[PriceData]) -> Optional[TechnicalSignals]:
    """
    Compute technical indicators (EMA20, EMA50, RSI14, MACD) from OHLCV
    data using pandas-ta, then derive structural signals from them.

    Args:
        price_data: Output of data_fetchers.fetch_price_data(), or None.

    Returns:
        A TechnicalSignals dict, or None if price_data is None, or if
        fewer than config.MIN_BARS_FOR_TECHNICAL bars are available, or
        if indicator computation produces no valid (non-NaN) final values
        (e.g. insufficient warm-up period for EMA50/MACD).
    """
    if price_data is None:
        logger.info("compute_technical_signals called with price_data=None; returning None.")
        return None

    ohlcv = price_data["ohlcv"]

    if len(ohlcv) < config.MIN_BARS_FOR_TECHNICAL:
        logger.warning(
            "Insufficient OHLCV history to compute technical signals "
            "(have %d bars, need >= %d); returning None.",
            len(ohlcv),
            config.MIN_BARS_FOR_TECHNICAL,
        )
        return None

    df = pd.DataFrame(ohlcv)
    df = df.sort_values("date").reset_index(drop=True)

    close = df["close"]

    try:
        # ta library uses class-based API. Each indicator returns a Series.
        ema20_series = ta_lib.trend.EMAIndicator(close=close, window=20).ema_indicator()
        ema50_series = ta_lib.trend.EMAIndicator(close=close, window=50).ema_indicator()
        rsi_series   = ta_lib.momentum.RSIIndicator(close=close, window=14).rsi()

        macd_obj        = ta_lib.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        macd_line_series   = macd_obj.macd()
        macd_signal_series = macd_obj.macd_signal()
    except Exception:
        logger.exception("ta indicator computation raised an exception.")
        return None

    ema20_latest       = ema20_series.iloc[-1]
    ema50_latest       = ema50_series.iloc[-1]
    rsi_latest         = rsi_series.iloc[-1]
    macd_line_latest   = macd_line_series.iloc[-1]
    macd_signal_latest = macd_signal_series.iloc[-1]

    values_to_check = {
        "ema20":            ema20_latest,
        "ema50":            ema50_latest,
        "rsi":              rsi_latest,
        "macd_line":        macd_line_latest,
        "macd_signal_line": macd_signal_latest,
    }

    nan_fields = [name for name, value in values_to_check.items() if pd.isna(value)]
    if nan_fields:
        logger.warning(
            "Technical indicators contain NaN values (likely insufficient "
            "warm-up period) for fields: %s; returning None.",
            nan_fields,
        )
        return None

    ema20 = float(ema20_latest)
    ema50 = float(ema50_latest)
    rsi = float(rsi_latest)
    macd_line = float(macd_line_latest)
    macd_signal_line = float(macd_signal_latest)

    current_price = price_data["current_price"]

    price_vs_ema20 = "above" if current_price > ema20 else "below"
    price_vs_ema50 = "above" if current_price > ema50 else "below"
    macd_trend = "bullish" if macd_line > macd_signal_line else "bearish"
    rsi_zone = _classify_rsi_zone(rsi)

    signals = TechnicalSignals(
        ema20=ema20,
        ema50=ema50,
        rsi=rsi,
        macd_line=macd_line,
        macd_signal_line=macd_signal_line,
        price_vs_ema20=price_vs_ema20,
        price_vs_ema50=price_vs_ema50,
        macd_trend=macd_trend,
        rsi_zone=rsi_zone,
    )

    logger.info("Computed technical signals: %s", signals)
    return signals


def _classify_rsi_zone(rsi: float) -> str:
    """
    Classify RSI(14) into oversold (<30), overbought (>70), or neutral.
    """
    if rsi < 30:
        return "oversold"
    elif rsi > 70:
        return "overbought"
    else:
        return "neutral"