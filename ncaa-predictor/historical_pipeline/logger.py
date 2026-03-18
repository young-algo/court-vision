"""Structured logging for the historical pipeline.

Outputs JSON-formatted log lines for machine parseability. Falls back to
plain print if structlog is not installed.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone


def _emit(level: str, msg: str, **kwargs: object) -> None:
    entry = {
        "level": level,
        "msg": msg,
        "ts": datetime.now(timezone.utc).isoformat(),
        **kwargs,
    }
    stream = sys.stderr if level == "error" else sys.stdout
    stream.write(json.dumps(entry, default=str) + "\n")
    stream.flush()


def info(msg: str, **kwargs: object) -> None:
    _emit("info", msg, **kwargs)


def warn(msg: str, **kwargs: object) -> None:
    _emit("warn", msg, **kwargs)


def error(msg: str, **kwargs: object) -> None:
    _emit("error", msg, **kwargs)


def debug(msg: str, **kwargs: object) -> None:
    _emit("debug", msg, **kwargs)
