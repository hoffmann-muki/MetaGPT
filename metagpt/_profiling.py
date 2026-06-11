#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Optional profiling helpers for Nsight Systems/NVTX traces.

The helpers in this module intentionally degrade to no-ops when ``nvtx`` is not
installed. This keeps profiling instrumentation safe to leave in normal runs.
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Any

try:
    import nvtx
except ImportError:  # pragma: no cover - depends on optional profiling extra
    nvtx = None


_FALSE_VALUES = {"0", "false", "off", "no"}
_MAX_MESSAGE_LEN = 180


def is_enabled() -> bool:
    """Return whether NVTX ranges should be emitted."""
    if nvtx is None:
        return False
    return os.getenv("METAGPT_NVTX", "1").lower() not in _FALSE_VALUES


def _clean(value: Any, limit: int = _MAX_MESSAGE_LEN) -> str:
    text = str(value).replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


def label(name: str, **fields: Any) -> str:
    """Build a compact NVTX message from a component name and small metadata."""
    metadata = " ".join(
        f"{key}={_clean(value)}" for key, value in fields.items() if value is not None
    )
    return f"{name} {metadata}".strip()


def span(message: str, *, domain: str = "metagpt", color: str | None = None):
    """Return an NVTX context manager, or a no-op context manager if disabled."""
    if not is_enabled():
        return nullcontext()
    kwargs = {"message": _clean(message), "domain": domain}
    if color:
        kwargs["color"] = color
    try:
        return nvtx.annotate(**kwargs)
    except Exception:  # pragma: no cover - defensive against unsupported colors/domains
        return nullcontext()


def mark(message: str, *, domain: str = "metagpt", color: str | None = None) -> None:
    """Emit a point marker when NVTX is enabled."""
    if not is_enabled():
        return
    kwargs = {"message": _clean(message), "domain": domain}
    if color:
        kwargs["color"] = color
    try:
        nvtx.mark(**kwargs)
    except Exception:  # pragma: no cover - defensive against unsupported colors/domains
        return
