"""
main.py

FastAPI application for the Stock Sentiment Analysis MVP.

Exposes a single primary endpoint:

    POST /analyze
        Request:  {"symbol": "TATAMOTORS"}
        Response: the full report dict from orchestrator.run_analysis()

Also exposes:

    GET /health
        Returns {"status": "ok"} once the app (and FinBERT model, loaded
        via the analyzers import) has started successfully.

This file is intentionally thin. All pipeline logic lives in
orchestrator.run_analysis(); this file's only responsibilities are HTTP
request/response handling, input validation, and converting unexpected
exceptions into clean error responses.
"""

from __future__ import annotations

import logging
import re

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

import orchestrator


# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
# Configured here, at the application entry point, so that log messages
# from data_fetchers, analyzers, scoring, llm_reasoning, and orchestrator
# (all of which use `logging.getLogger(__name__)` and emit but do not
# configure) become visible.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# App Initialization
# ---------------------------------------------------------------------------
# Note: importing `orchestrator` (which imports `analyzers`) triggers
# FinBERT model loading at import time, i.e. when this module is first
# loaded by uvicorn -- before the app starts accepting requests. If model
# loading fails, the app will fail to start entirely (by design -- see
# analyzers.py).

app = FastAPI(
    title="Stock Sentiment Analysis Engine",
    description="On-demand sentiment analysis for Indian equities.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

_SYMBOL_PATTERN = re.compile(r"^[A-Za-z0-9&\-]{1,20}$")


class AnalyzeRequest(BaseModel):
    symbol: str

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, value: str) -> str:
        """
        Basic symbol format validation: 1-20 characters, alphanumeric plus
        '&' and '-' (covers symbols like "M&M" and "L&T"). Normalizes to
        uppercase with surrounding whitespace stripped.

        This is a format check only -- it does not verify the symbol
        actually exists. An unrecognized symbol will be handled gracefully
        downstream by orchestrator.run_analysis(), which returns a report
        with overall_sentiment="INSUFFICIENT_DATA" rather than an error.
        """
        cleaned = value.strip().upper()

        if not cleaned:
            raise ValueError("symbol must not be empty")

        if not _SYMBOL_PATTERN.match(cleaned):
            raise ValueError(
                "symbol must be 1-20 characters long and contain only "
                "letters, digits, '&', or '-'"
            )

        return cleaned


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    """
    Health check endpoint. A 200 response confirms the application
    (including FinBERT model loading at import time) started successfully.
    """
    return {"status": "ok"}


@app.post("/analyze")
def analyze(request: AnalyzeRequest) -> JSONResponse:
    """
    Run the full sentiment analysis pipeline for the given stock symbol
    and return the resulting report.

    Request body:
        {"symbol": "TATAMOTORS"}

    Returns:
        200: The report dict from orchestrator.run_analysis(), as JSON.
        422: Automatic FastAPI/Pydantic response if `symbol` fails
             validation (empty, too long, invalid characters).
        500: If run_analysis() raises an unexpected exception. The
             pipeline is designed to degrade gracefully rather than raise
             under normal failure conditions (missing data, LLM failure,
             unknown symbol), so a 500 here indicates a genuine bug rather
             than expected real-world data issues.
    """
    symbol = request.symbol

    logger.info("Received /analyze request for symbol=%s", symbol)

    try:
        report = orchestrator.run_analysis(symbol)
    except Exception:
        logger.exception("Unexpected error running analysis for symbol=%s", symbol)
        raise HTTPException(
            status_code=500,
            detail=(
                "An unexpected error occurred while analyzing this symbol. "
                "Please try again later."
            ),
        )

    # FastAPI's default JSON encoder (jsonable_encoder, used internally by
    # JSONResponse via Response when given a dict) handles datetime objects
    # by converting them to ISO 8601 strings, so report["timestamp"] and
    # the published_at fields in top_articles serialize correctly without
    # custom handling here.
    return JSONResponse(content=_jsonable(report))


# ---------------------------------------------------------------------------
# Serialization Helper
# ---------------------------------------------------------------------------

from fastapi.encoders import jsonable_encoder


def _jsonable(report: dict) -> dict:
    """
    Convert the report dict (which may contain datetime objects) into a
    JSON-serializable structure using FastAPI's jsonable_encoder.

    JSONResponse does not apply FastAPI's automatic encoding the way
    returning a dict directly from a route does, so this conversion is
    done explicitly to ensure datetime fields (report["timestamp"] and
    each article's "published_at") serialize to ISO 8601 strings rather
    than raising a TypeError.
    """
    return jsonable_encoder(report)