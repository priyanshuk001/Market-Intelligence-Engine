"""
config.py

Central configuration for the Stock Sentiment Analysis MVP.

This file contains ONLY constants and lookup tables. No logic, no functions
beyond environment loading. Every other module imports from here.
"""

import os
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment Variables
# ---------------------------------------------------------------------------

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is not set. "
        "Create a .env file with GEMINI_API_KEY=your_key_here"
    )

# ---------------------------------------------------------------------------
# LLM Configuration
# ---------------------------------------------------------------------------

# Model used for the final reasoning/explanation layer.
# Options: "gemini-1.5-flash" (faster, cheaper) or "gemini-1.5-pro" (stronger)
# GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_MODEL = "gemini-2.0-flash-lite"
# ---------------------------------------------------------------------------
# Symbol -> Company Name Mapping
# ---------------------------------------------------------------------------
# Used to build better news search queries. Searching "TATAMOTORS" on a
# news feed returns poor results; searching "Tata Motors" returns good ones.
#
# Extend this dict as more symbols are tested. Symbols not present here will
# fall back to using the raw ticker as the search query (lower quality
# results, but the pipeline still works).

SYMBOL_TO_COMPANY_NAME = {
    "TATAMOTORS": "Tata Motors",
    "RELIANCE": "Reliance Industries",
    "TCS": "Tata Consultancy Services",
    "INFY": "Infosys",
    "HDFCBANK": "HDFC Bank",
    "ICICIBANK": "ICICI Bank",
    "WIPRO": "Wipro",
    "SBIN": "State Bank of India",
    "ITC": "ITC Limited",
    "BHARTIARTL": "Bharti Airtel",
}

# ---------------------------------------------------------------------------
# Source Credibility Registry (Hardcoded for MVP)
# ---------------------------------------------------------------------------
# Maps a news source's domain to a credibility score between 0.0 and 1.0.
# Higher score = more weight given to articles from that source.
#
# Any domain not found in this dict falls back to SOURCE_CREDIBILITY_DEFAULT.

SOURCE_CREDIBILITY = {
    "economictimes.indiatimes.com": 0.85,
    "livemint.com": 0.85,
    "moneycontrol.com": 0.70,
    "business-standard.com": 0.80,
    "financialexpress.com": 0.70,
    "ndtv.com": 0.75,
    "ndtvprofit.com": 0.75,
    "reuters.com": 0.90,
    "bloomberg.com": 0.90,
    "cnbctv18.com": 0.75,
}

# Credibility score assigned to any source not found in the dict above.
SOURCE_CREDIBILITY_DEFAULT = 0.50

# ---------------------------------------------------------------------------
# Scoring Weights
# ---------------------------------------------------------------------------
# Must sum to 1.0. Used by aggregate_scores() in scoring.py to combine the
# three component scores into a final sentiment score.

NEWS_WEIGHT = 0.40
PRICE_WEIGHT = 0.35
TECHNICAL_WEIGHT = 0.25

# ---------------------------------------------------------------------------
# Time Decay
# ---------------------------------------------------------------------------
# Controls how quickly older news articles lose weight.
# decay_weight = e^(-TIME_DECAY_LAMBDA * hours_since_published)
#
# Higher lambda = faster decay (older news matters less, sooner).

TIME_DECAY_LAMBDA = 0.05

# ---------------------------------------------------------------------------
# News Fetching Limits
# ---------------------------------------------------------------------------

# Only consider articles published within this many hours of "now".
NEWS_LOOKBACK_HOURS = 72

# Maximum number of articles to fetch and analyze per query.
MAX_ARTICLES = 15

# Number of top articles (by effective_weight) to include in the LLM prompt.
TOP_ARTICLES_FOR_PROMPT = 5

# ---------------------------------------------------------------------------
# Price Data Limits
# ---------------------------------------------------------------------------

# How many days of daily OHLCV history to fetch.
# 100 days is sufficient for EMA50 and RSI14. EMA200 is out of scope for MVP.
PRICE_HISTORY_DAYS = 100

# Minimum number of OHLCV bars required to compute technical signals.
# If fewer bars are available, compute_technical_signals() returns None.
MIN_BARS_FOR_TECHNICAL = 50

# ---------------------------------------------------------------------------
# Contradiction Detection Thresholds
# ---------------------------------------------------------------------------
# Used by detect_contradiction() to classify the severity of a divergence
# between the News Score and the Price Score.

CONTRADICTION_THRESHOLD_LOW = 15   # difference > 15  -> at least LOW severity
CONTRADICTION_THRESHOLD_HIGH = 30  # difference > 30  -> HIGH severity

# ---------------------------------------------------------------------------
# Sentiment Label Thresholds
# ---------------------------------------------------------------------------
# Used by aggregate_scores() to map the final 0-100 score to a label.

SENTIMENT_THRESHOLDS = {
    "STRONGLY_BULLISH": 70,   # score >= 70
    "BULLISH": 58,            # score >= 58
    "NEUTRAL": 43,            # score >= 43
    "BEARISH": 31,            # score >= 31
    # below 31 -> STRONGLY_BEARISH
}

# ---------------------------------------------------------------------------
# Exchange Suffix
# ---------------------------------------------------------------------------
# Appended to symbols for yfinance lookups (NSE = National Stock Exchange).

EXCHANGE_SUFFIX = ".NS"