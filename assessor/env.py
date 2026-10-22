"""Load a local .env file into os.environ without a dependency.

Rules: KEY=VALUE lines, optional quotes, '#' comments. Values already present in the
environment win, so shell-exported secrets always take precedence. Never log the values.
"""
from __future__ import annotations

import os
from pathlib import Path

_SECRET_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD")


def load_env(path: str | os.PathLike | None = None) -> None:
    p = Path(path) if path else Path(__file__).resolve().parent.parent / ".env"
    if not p.exists():
        return
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, val)


def redacted_env(keys: list[str]) -> dict[str, str]:
    """Values for display: secrets become masked, everything else stays visible."""
    out = {}
    for k in keys:
        v = os.environ.get(k)
        if v is None:
            out[k] = None
        elif any(h in k.upper() for h in _SECRET_HINTS):
            out[k] = v[:4] + "…" + f"(len {len(v)})"
        else:
            out[k] = v
    return out
