"""Observability: structured logging, optional LangSmith tracing, and node timing.

Usage:
    from observability import logger, node_timer, setup_logging, setup_langsmith

    setup_logging()          # Call once at startup
    setup_langsmith()        # Enable LangSmith if LANGCHAIN_API_KEY is set

    with node_timer("agent"):
        response = llm.invoke(messages)
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager

# ---------------------------------------------------------------------------
# Module-level logger — all graph nodes and app code import this
# ---------------------------------------------------------------------------
logger = logging.getLogger("portfolio_chatbot")


def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging. Call once at application startup."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Avoid duplicate handlers on Streamlit reruns
    if logger.handlers:
        return

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.setLevel(log_level)
    logger.addHandler(handler)
    logger.info("Logging initialized at %s level", level)


def setup_langsmith() -> None:
    """Enable LangSmith tracing if LANGCHAIN_API_KEY is set in environment.

    LangSmith provides full trace visibility for every graph invocation:
    node inputs/outputs, LLM token usage, tool calls, latencies, and errors.
    Set LANGCHAIN_API_KEY and optionally LANGCHAIN_PROJECT in .env to enable.
    """
    if os.getenv("LANGCHAIN_API_KEY"):
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_PROJECT", "ai-portfolio-chatbot")
        logger.info(
            "LangSmith tracing enabled (project: %s)",
            os.environ.get("LANGCHAIN_PROJECT"),
        )
    else:
        logger.debug("LangSmith tracing disabled (no LANGCHAIN_API_KEY)")


@contextmanager
def node_timer(node_name: str):
    """Context manager that logs node execution start, duration, and errors.

    Usage:
        with node_timer("agent"):
            response = llm.invoke(messages)
    """
    start = time.perf_counter()
    logger.debug("Node '%s' started", node_name)
    try:
        yield
    except Exception:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.error("Node '%s' FAILED after %.1fms", node_name, elapsed_ms)
        raise
    else:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info("Node '%s' completed in %.1fms", node_name, elapsed_ms)
