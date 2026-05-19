"""LLM-driven entity sector classification.

Runs once per scan. Inputs: entity name, domain, and a short policy excerpt.
Output: one of the canonical sectors (exact string match) or None when the
LLM call errors out — letting callers fall back to their pre-scan seed
(e.g. the web keyword classifier).
"""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Callable
from typing import Any

from engine.prompts.sector import SECTOR_SYSTEM_PROMPT, get_sector_user_prompt

logger = logging.getLogger(__name__)

# Mirror of apps/web/lib/constants.ts:SECTORS. Keep in sync.
SECTORS: list[str] = [
    "Banking & Finance",
    "Technology & SaaS",
    "Telecom",
    "Retail & Ecommerce",
    "Health & Wellness",
    "Real Estate",
    "Gambling & Gaming",
    "Government & Utilities",
    "Other",
]

# Cap the policy excerpt. Homepage-quality privacy signal is front-loaded;
# a few KB is plenty and keeps per-scan classifier cost predictable.
_POLICY_EXCERPT_CHAR_LIMIT = 2000


def classify_sector(
    name: str,
    domain: str,
    policy_text: str,
    llm_client: Callable[..., str],
) -> str | None:
    """Ask the LLM to place the entity into one of the canonical sectors.

    Returns:
        A sector string from SECTORS on success, or None when the LLM call
        errored. Invalid / unknown sector strings are coerced to "Other"
        rather than None so the caller can still persist *a* value.
    """
    excerpt = (policy_text or "")[:_POLICY_EXCERPT_CHAR_LIMIT]
    user_prompt = get_sector_user_prompt(name, domain, excerpt)

    try:
        raw = _call_with_system(
            llm_client,
            user_prompt=user_prompt,
            system=SECTOR_SYSTEM_PROMPT,
            max_tokens=64,
        )
    except Exception:
        logger.warning(
            "Sector classification LLM call failed — caller should use seed",
            exc_info=True,
        )
        return None

    return _parse_sector(raw)


def _parse_sector(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```"):
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Sector response was not valid JSON; defaulting to Other")
        return "Other"

    if not isinstance(parsed, dict):
        return "Other"

    sector = parsed.get("sector")
    if not isinstance(sector, str) or sector not in SECTORS:
        logger.warning(
            "Sector response missing/invalid ('%s'); defaulting to Other",
            sector,
        )
        return "Other"
    return sector


def _call_with_system(
    llm_client: Callable[..., str],
    *,
    user_prompt: str,
    system: str,
    max_tokens: int,
) -> str:
    """Prefer system+max_tokens kwargs when the callable supports them."""
    if _supports_system_kwarg(llm_client):
        return llm_client(user_prompt, system=system, max_tokens=max_tokens)
    return llm_client(f"{system}\n\n{user_prompt}")


def _supports_system_kwarg(llm_client: Callable[..., Any]) -> bool:
    try:
        params = inspect.signature(llm_client).parameters
    except (TypeError, ValueError):
        return False
    return "system" in params
