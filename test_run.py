"""
test_run.py

Developer convenience script for the Stock Sentiment Analysis MVP.

Runs orchestrator.run_analysis() directly, without needing the FastAPI
server running, and pretty-prints the resulting report to the console.
This is the fastest feedback loop during development -- use this instead
of starting uvicorn and making HTTP requests for every iteration.

Usage:
    python test_run.py                # defaults to TATAMOTORS
    python test_run.py RELIANCE
    python test_run.py tcs            # symbol is case-insensitive
"""

from __future__ import annotations

import json
import logging
import sys

import orchestrator


# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
# Configured here (same as main.py) so that the step-by-step INFO logs
# from orchestrator, data_fetchers, analyzers, scoring, and llm_reasoning
# are visible when running this script directly.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


DEFAULT_SYMBOL = "RELIANCE"  # used if no symbol is provided on the command line    


def main() -> None:
    """
    Parse the symbol from command-line arguments (or fall back to
    DEFAULT_SYMBOL), run the analysis pipeline, and pretty-print the
    resulting report as JSON.

    `default=str` is passed to json.dumps() because the report contains
    datetime objects (report["timestamp"] and each article's
    "published_at"), which are not JSON-serializable by default --
    `default=str` converts them to their string representation for
    display purposes.
    """
    if len(sys.argv) > 1:
        symbol = sys.argv[1]
    else:
        symbol = DEFAULT_SYMBOL
        print(f"No symbol provided. Using default: {DEFAULT_SYMBOL}\n")

    report = orchestrator.run_analysis(symbol)

    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()