"""Claim extraction from privacy policy text.

The extractor is a pure function: policy text in, structured claims out.
The LLM client is injected as a callable — the engine never imports any
specific LLM library. The callable may optionally accept `system` and
`cached_prefix` kwargs; if present, we use prompt caching to avoid
resending the policy text and system prompt on every dimension call.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
from collections.abc import Callable
from typing import Any

from engine import __version__
from engine.models import Claim, ExtractionResult, DIMENSIONS
from engine.prompts.extraction import (
    SYSTEM_PROMPT,
    get_dimension_instruction,
    get_dimension_prompt,
    wrap_policy_text,
)

logger = logging.getLogger(__name__)

# Hard ceiling on policy text length. A genuine privacy policy — even an
# aggregated multi-page one — is far shorter; anything larger is almost
# certainly a runaway fetch or hostile input. Truncating (rather than
# rejecting) keeps a score available while bounding LLM cost and memory.
MAX_POLICY_TEXT_CHARS = 500_000


def _cap_policy_text(policy_text: str) -> str:
    """Truncate over-long policy text, logging a warning when it does."""
    if len(policy_text) <= MAX_POLICY_TEXT_CHARS:
        return policy_text
    logger.warning(
        "Policy text is %d chars; truncating to %d for extraction.",
        len(policy_text),
        MAX_POLICY_TEXT_CHARS,
    )
    return policy_text[:MAX_POLICY_TEXT_CHARS]


def extract_claims(
    policy_text: str,
    llm_client: Callable[..., str],
) -> ExtractionResult:
    """Extract claims from privacy policy text across all 10 dimensions.

    Args:
        policy_text: Raw privacy policy text to analyse.
        llm_client: A callable that takes a prompt string and returns the LLM
            response string. The caller is responsible for injecting the
            appropriate client (e.g. Anthropic Claude).

    Returns:
        ExtractionResult with all valid claims aggregated across dimensions.
    """
    policy_text = _cap_policy_text(policy_text)
    supports_cached = _supports_cached_call(llm_client)
    all_claims: list[Claim] = []

    for dimension in DIMENSIONS:
        try:
            claims = _extract_for_dimension(
                policy_text, dimension, llm_client, supports_cached
            )
            all_claims.extend(claims)
        except Exception:
            logger.warning(
                "Failed to extract claims for dimension '%s'", dimension
            )

    policy_text_hash = hashlib.sha256(policy_text.encode("utf-8")).hexdigest()

    return ExtractionResult(
        claims=all_claims,
        policy_text_hash=policy_text_hash,
        engine_version=__version__,
    )


def extract_claims_for_dimension(
    policy_text: str,
    dimension: str,
    llm_client: Callable[..., str],
) -> list[Claim]:
    """Extract claims for a single dimension.

    Args:
        policy_text: Raw privacy policy text to analyse.
        dimension: One of the 10 dimension keys.
        llm_client: A callable that takes a prompt string and returns the LLM
            response string.

    Returns:
        List of valid Claim objects for the given dimension.
        Public API — single-dimension extraction. Re-checks support on each call.
    """
    return _extract_for_dimension(
        _cap_policy_text(policy_text),
        dimension,
        llm_client,
        _supports_cached_call(llm_client),
    )


def _extract_for_dimension(
    policy_text: str,
    dimension: str,
    llm_client: Callable[..., str],
    supports_cached: bool,
) -> list[Claim]:
    if supports_cached:
        user_suffix = get_dimension_instruction(dimension)
        raw_response = llm_client(
            user_suffix,
            system=SYSTEM_PROMPT,
            cached_prefix=wrap_policy_text(policy_text),
        )
    else:
        prompt = _build_legacy_prompt(dimension, policy_text)
        raw_response = llm_client(prompt)

    return _parse_claims(raw_response, dimension)


def _supports_cached_call(llm_client: Callable[..., Any]) -> bool:
    """True when the callable accepts `system` and `cached_prefix` kwargs."""
    try:
        sig = inspect.signature(llm_client)
    except (TypeError, ValueError):
        return False
    params = sig.parameters
    return "system" in params and "cached_prefix" in params


def _build_legacy_prompt(dimension: str, policy_text: str) -> str:
    """Build a flat prompt for callables that don't support caching kwargs."""
    dimension_prompt = get_dimension_prompt(dimension, policy_text)
    return f"{SYSTEM_PROMPT}\n\n{dimension_prompt}"


def _parse_claims(raw_response: str, dimension: str) -> list[Claim]:
    """Parse LLM JSON response into validated Claim objects. Handles malformed responses gracefully — logs warnings and skips invalid claims rather than crashing."""
    claims: list[Claim] = []

    try:
        parsed = _extract_json_array(raw_response)
    except (json.JSONDecodeError, ValueError):
        logger.warning(
            "Could not parse LLM response as JSON for dimension '%s'",
            dimension,
        )
        return claims

    if not isinstance(parsed, list):
        logger.warning(
            "LLM response for dimension '%s' is not a JSON array", dimension
        )
        return claims

    for item in parsed:
        claim = _validate_claim_item(item, dimension)
        if claim is not None:
            claims.append(claim)

    return claims


def _extract_json_array(raw: str) -> Any:
    """Extract a JSON array from the response, handling markdown code blocks."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return json.loads(text)


def _validate_claim_item(item: Any, dimension: str) -> Claim | None:
    """Validate a single claim dict into a Claim model. Returns None if validation fails."""
    if not isinstance(item, dict):
        logger.warning("Skipping non-dict claim item in dimension '%s'", dimension)
        return None
    item["dimension"] = dimension
    try:
        return Claim.model_validate(item)
    except Exception as e:
        logger.warning(
            "Skipping invalid claim in dimension '%s': %s", dimension, e
        )
        return None
