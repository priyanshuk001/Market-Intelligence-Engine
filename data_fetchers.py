"""
data_fetchers.py

All external data retrieval for the Stock Sentiment Analysis MVP.

Three independent functions, one per data source:
  - fetch_news()           -> Google News RSS
  - fetch_price_data()      -> yfinance (price + OHLCV history)
  - fetch_market_context()  -> yfinance (NIFTY 50 index)

Design principles:
  - Every function is fully typed.
  - Every external call is wrapped in error handling. No function raises
    on a failed external call — they return None / [] so the orchestrator
    can proceed with partial data (graceful degradation).
  - Every significant step (success, failure, fallback) is logged so
    issues can be diagnosed from logs alone without a debugger.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, TypedDict
from urllib.parse import quote, urlparse


import feedparser
import yfinance as yf

import config


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed Return Shapes
# ---------------------------------------------------------------------------

class NewsArticle(TypedDict):
    headline: str
    source_domain: str
    published_at: datetime
    url: str


class OHLCVBar(TypedDict):
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


class PriceData(TypedDict):
    current_price: float
    previous_close: float
    day_change_pct: float
    volume_today: int
    avg_volume_20d: float
    fifty_two_week_high: float
    fifty_two_week_low: float
    ohlcv: list[OHLCVBar]


class MarketContext(TypedDict):
    nifty_value: float
    nifty_change_pct: float


# ---------------------------------------------------------------------------
# News Fetching
# ---------------------------------------------------------------------------

def fetch_news(symbol: str) -> list[NewsArticle]:
    """
    Fetch recent news articles for the given stock symbol via Google News RSS.

    Uses config.SYMBOL_TO_COMPANY_NAME to build a higher-quality search query
    (e.g. "Tata Motors" instead of "TATAMOTORS"). Falls back to the raw symbol
    if no mapping exists.

    Filters to articles published within config.NEWS_LOOKBACK_HOURS and caps
    the result at config.MAX_ARTICLES.

    Args:
        symbol: Stock ticker symbol, e.g. "TATAMOTORS".

    Returns:
        A list of NewsArticle dicts, newest-relevant first (as returned by
        the feed). Returns [] if no articles are found, the feed fails to
        parse, or an unexpected error occurs. This function never raises.
    """
    query = config.SYMBOL_TO_COMPANY_NAME.get(symbol, symbol)

    rss_url = (
        f"https://news.google.com/rss/search?"
        f"q={quote(query)}&hl=en-IN&gl=IN&ceid=IN:en" # cite: 1
    )

    logger.info("Fetching news for symbol=%s using query='%s'", symbol, query)

    try:
        feed = feedparser.parse(rss_url)
    except Exception:
        logger.exception("feedparser.parse() raised an exception for symbol=%s", symbol)
        return []

    # feedparser sets bozo=1 (and bozo_exception) on malformed feeds rather
    # than raising. Treat that as a soft failure: log and continue, since
    # some feeds set bozo=1 even when entries are still usable.
    if getattr(feed, "bozo", 0):
        logger.warning(
            "Feed for symbol=%s reported bozo=1 (possible malformed XML): %s",
            symbol,
            getattr(feed, "bozo_exception", "unknown reason"),
        )

    entries = getattr(feed, "entries", None)
    if not entries:
        logger.warning("No entries returned from news feed for symbol=%s", symbol)
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(hours=config.NEWS_LOOKBACK_HOURS)
    articles: list[NewsArticle] = []
    skipped_no_date = 0
    skipped_too_old = 0

    for entry in entries:
        headline = getattr(entry, "title", None)
        link = getattr(entry, "link", None)

        if not headline or not link:
            continue

        published_at = _parse_published_date(entry)
        if published_at is None:
            # Skip rather than guess — time decay scoring depends on this
            # field being accurate.
            skipped_no_date += 1
            continue

        if published_at < cutoff:
            skipped_too_old += 1
            continue

        source_domain = _extract_domain(link)

        articles.append(
            NewsArticle(
                headline=headline.strip(),
                source_domain=source_domain,
                published_at=published_at,
                url=link,
            )
        )

        if len(articles) >= config.MAX_ARTICLES:
            break

    logger.info(
        "Fetched %d articles for symbol=%s (skipped %d with no date, %d too old)",
        len(articles),
        symbol,
        skipped_no_date,
        skipped_too_old,
    )

    return articles


def _parse_published_date(entry: feedparser.FeedParserDict) -> Optional[datetime]:
    """
    Extract a timezone-aware UTC datetime from a feedparser entry.

    feedparser provides 'published_parsed' as a time.struct_time in UTC
    when available.

    Returns:
        A UTC datetime, or None if no usable date is present on the entry.
    """
    parsed_time = getattr(entry, "published_parsed", None)
    if parsed_time is None:
        return None

    try:
        return datetime(*parsed_time[:6], tzinfo=timezone.utc)
    except (TypeError, ValueError) as exc:
        logger.debug("Failed to parse published_parsed=%r: %s", parsed_time, exc)
        return None


def _extract_domain(url: str) -> str:
    """
    Extract a clean domain from a URL for source credibility lookup.

    Strips a leading "www." so "www.livemint.com" matches the
    "livemint.com" key in config.SOURCE_CREDIBILITY.

    Note: Google News RSS links often point to news.google.com redirect
    URLs rather than the original publisher's domain. In that case this
    returns "news.google.com", which falls back to
    SOURCE_CREDIBILITY_DEFAULT in analyzers.py. Resolving the true
    publisher domain would require following the redirect (an extra HTTP
    call per article) and is deferred beyond the MVP.

    Returns:
        The lowercase domain, or "" if the URL cannot be parsed.
    """
    try:
        netloc = urlparse(url).netloc.lower() # cite: 1
    except Exception:
        logger.debug("Failed to parse domain from url=%r", url)
        return ""

    if netloc.startswith("www."):
        netloc = netloc[4:]

    return netloc


# ---------------------------------------------------------------------------
# Price Data Fetching
# ---------------------------------------------------------------------------

def fetch_price_data(symbol: str) -> Optional[PriceData]:
    """
    Fetch current price snapshot and historical OHLCV data for the given
    symbol using yfinance. Appends config.EXCHANGE_SUFFIX (".NS") for NSE.

    Args:
        symbol: Stock ticker symbol, e.g. "TATAMOTORS".

    Returns:
        A PriceData dict, or None if the ticker is invalid, no data is
        returned, or fewer than 2 valid rows of history are available
        (at least 2 rows are required to compute day_change_pct).
        This function never raises.
    """
    # Try NSE first (.NS), then BSE (.BO) as fallback.
    # Some symbols vary in availability between exchanges on yfinance.
    suffixes_to_try = [config.EXCHANGE_SUFFIX, ".BO"]
    history = None

    for suffix in suffixes_to_try:
        ticker_symbol = f"{symbol}{suffix}"
        logger.info("Fetching price data for symbol=%s (ticker=%s)", symbol, ticker_symbol)
        try:
            ticker = yf.Ticker(ticker_symbol)
            candidate = ticker.history(period=f"{config.PRICE_HISTORY_DAYS}d")
        except Exception:
            logger.warning("yfinance raised an exception for ticker=%s; trying next.", ticker_symbol)
            continue
        if candidate is not None and not candidate.empty:
            history = candidate
            logger.info("Price data found using ticker=%s", ticker_symbol)
            break
        logger.warning("yfinance returned empty history for ticker=%s; trying next.", ticker_symbol)

    if history is None or history.empty:
        logger.warning("No price data found for symbol=%s after trying suffixes: %s", symbol, suffixes_to_try)
        return None

    required_columns = {"Open", "High", "Low", "Close", "Volume"}
    missing_columns = required_columns - set(history.columns)
    if missing_columns:
        logger.error(
            "yfinance history for symbol=%s is missing expected columns: %s",
            symbol,
            missing_columns,
        )
        return None

    history = history.dropna(subset=list(required_columns))

    if history.empty or len(history) < 2:
        logger.warning(
            "Insufficient price history for symbol=%s after dropping NaNs (rows=%d, need >= 2)",
            symbol,
            len(history),
        )
        return None

    try:
        latest = history.iloc[-1]
        previous = history.iloc[-2]

        current_price = float(latest["Close"])
        previous_close = float(previous["Close"])
        volume_today = int(latest["Volume"])

        day_change_pct = _safe_pct_change(current_price, previous_close)

        # Average volume over the last 20 trading days, excluding today
        # (today's volume may be partial/incomplete intraday).
        volume_window = history["Volume"].iloc[:-1].tail(20)
        avg_volume_20d = (
            float(volume_window.mean()) if not volume_window.empty else float(volume_today)
        )

        fifty_two_week_high = float(history["High"].max())
        fifty_two_week_low = float(history["Low"].min())

        ohlcv: list[OHLCVBar] = []
        for date_index, row in history.iterrows():
            ohlcv.append(
                OHLCVBar(
                    date=date_index.to_pydatetime(),
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=int(row["Volume"]),
                )
            )
    except (KeyError, IndexError, ValueError, TypeError):
        logger.exception("Failed to process price history for symbol=%s", symbol)
        return None

    logger.info(
        "Successfully fetched price data for symbol=%s: price=%.2f, day_change=%.2f%%, bars=%d",
        symbol,
        current_price,
        day_change_pct,
        len(ohlcv),
    )

    return PriceData(
        current_price=current_price,
        previous_close=previous_close,
        day_change_pct=day_change_pct,
        volume_today=volume_today,
        avg_volume_20d=avg_volume_20d,
        fifty_two_week_high=fifty_two_week_high,
        fifty_two_week_low=fifty_two_week_low,
        ohlcv=ohlcv,
    )


def _safe_pct_change(current: float, previous: float) -> float:
    """
    Compute percentage change, guarding against division by zero.

    Returns:
        The percentage change, or 0.0 if previous is 0 (defensive — should
        not occur for real stock prices, but avoids a crash on bad data).
    """
    if previous == 0:
        logger.debug("_safe_pct_change called with previous=0; returning 0.0")
        return 0.0
    return ((current - previous) / previous) * 100.0


# ---------------------------------------------------------------------------
# Market Context Fetching
# ---------------------------------------------------------------------------

def fetch_market_context() -> Optional[MarketContext]:
    """
    Fetch NIFTY 50 index value and day change percentage to provide broad
    market context for the LLM prompt.

    Returns:
        A MarketContext dict, or None on failure. The rest of the pipeline
        must work without market context — it is used only as a minor
        annotation in the LLM prompt in this MVP, not as a scoring input.
        This function never raises.
    """
    logger.info("Fetching market context (NIFTY 50)")

    try:
        ticker = yf.Ticker("^NSEI")
        history = ticker.history(period="5d")
    except Exception:
        logger.exception("yfinance raised an exception fetching NIFTY 50 data")
        return None

    if history is None or history.empty:
        logger.warning("yfinance returned empty history for NIFTY 50 (^NSEI)")
        return None

    if "Close" not in history.columns:
        logger.error("yfinance history for ^NSEI is missing 'Close' column")
        return None

    history = history.dropna(subset=["Close"])

    if history.empty or len(history) < 2:
        logger.warning(
            "Insufficient NIFTY 50 history after dropping NaNs (rows=%d, need >= 2)",
            len(history),
        )
        return None

    try:
        latest_close = float(history["Close"].iloc[-1])
        previous_close = float(history["Close"].iloc[-2])
    except (KeyError, IndexError, ValueError, TypeError):
        logger.exception("Failed to extract NIFTY 50 close prices")
        return None

    nifty_change_pct = _safe_pct_change(latest_close, previous_close)

    logger.info(
        "Successfully fetched market context: NIFTY 50=%.2f, change=%.2f%%",
        latest_close,
        nifty_change_pct,
    )

    return MarketContext(
        nifty_value=latest_close,
        nifty_change_pct=nifty_change_pct,
    )