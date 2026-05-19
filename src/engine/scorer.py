"""Dimension scoring and overall weighted score computation.

The scorer evaluates extracted claims across all 10 dimensions in a single
batched LLM call, then computes a deterministic overall weighted score and
letter grade. The LLM client is injected as a callable — the engine never
imports any specific LLM library.

A per-dimension `score_dimension` function is retained for use by callers
that explicitly want single-dimension scoring (e.g. tests or the
remediator), but the default `compute_scores` path makes exactly one call.
"""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Callable

from engine import __version__
from engine.models import (
    Claim,
    DimensionScore,
    ScoringResult,
    DIMENSIONS,
    DIMENSION_WEIGHTS,
    GRADE_THRESHOLDS,
)
from engine.prompts.scoring import (
    SCORING_SYSTEM_PROMPT,
    get_batched_scoring_prompt,
    get_scoring_prompt,
)

logger = logging.getLogger(__name__)


def compute_scores(
    claims: list[Claim],
    llm_client: Callable[..., str],
    excluded_dimensions: set[str] | None = None,
) -> ScoringResult:
    """Score extracted claims across all 10 dimensions in ONE LLM call.

    Args:
        claims: List of extracted Claim objects.
        llm_client: A callable that takes a prompt string and returns the
            LLM response. May optionally accept `system=` / `max_tokens=`.
        excluded_dimensions: Dimensions to treat as not applicable when
            computing the weighted overall (see compute_overall_score).
            Per-dimension scores are still produced for transparency.

    Returns:
        ScoringResult with dimension scores, overall score, and letter grade.
    """
    dimension_scores = score_all_dimensions(claims, llm_client)
    overall = compute_overall_score(dimension_scores, excluded_dimensions)
    grade = assign_grade(overall)
    return ScoringResult(
        dimension_scores=dimension_scores,
        overall_score=overall,
        letter_grade=grade,
        engine_version=__version__,
    )


def score_all_dimensions(
    claims: list[Claim],
    llm_client: Callable[..., str],
) -> list[DimensionScore]:
    """Score all 10 dimensions with a single batched LLM call.

    When there are no claims at all, short-circuits without hitting the LLM
    and returns zero-score rows for every dimension.
    """
    claims_by_dim = _group_claims_by_dimension(claims)

    if not claims:
        return [
            DimensionScore(
                dimension=dim, score=0,
                rationale="No relevant claims found in policy.",
            )
            for dim in DIMENSIONS
        ]

    claims_payload = {
        dim: [c.model_dump(mode="json") for c in claims_by_dim.get(dim, [])]
        for dim in DIMENSIONS
    }
    claims_json = json.dumps(claims_payload, indent=2)

    user_prompt = get_batched_scoring_prompt(claims_json)

    # Retry once on JSON parse failure or LLM exception. Empirically these
    # are transient — a second sampling of the same prompt usually returns
    # parseable JSON. Without this, a single bad batched response silently
    # publishes a Grade-F score for an entity that may be genuinely strong
    # (e.g. Origin Energy's first 11k-char PDF scan hit this on 2026-05-12).
    parsed: dict = {}
    for attempt in (1, 2):
        try:
            raw_response = _call_with_system(
                llm_client,
                user_prompt=user_prompt,
                system=SCORING_SYSTEM_PROMPT,
                max_tokens=4096,
            )
        except Exception:
            logger.warning(
                "Batched scoring LLM call failed (attempt %d/2)",
                attempt, exc_info=True,
            )
            if attempt == 2:
                break
            continue
        try:
            parsed = _parse_batched_response(raw_response)
            break  # success
        except (json.JSONDecodeError, ValueError):
            logger.warning(
                "Batched scoring response was not parseable JSON (attempt %d/2)",
                attempt, exc_info=True,
            )
            if attempt == 2:
                parsed = {}

    return [
        _to_dimension_score(dim, parsed.get(dim))
        for dim in DIMENSIONS
    ]


def score_dimension(
    dimension: str,
    claims: list[Claim],
    llm_client: Callable[..., str],
) -> DimensionScore:
    """Score a single dimension. Retained for single-dimension callers.

    Args:
        dimension: One of the 10 dimension keys.
        claims: Claims for this dimension (may be empty).
        llm_client: LLM callable.

    Returns:
        DimensionScore with score 0-10 and rationale.
    """
    if not claims:
        return DimensionScore(
            dimension=dimension,
            score=0,
            rationale="No relevant claims found in policy.",
        )

    claims_json = json.dumps(
        [c.model_dump(mode="json") for c in claims],
        indent=2,
    )
    prompt = get_scoring_prompt(dimension, claims_json)

    try:
        raw_response = _call_with_system(
            llm_client,
            user_prompt=prompt,
            system=SCORING_SYSTEM_PROMPT,
            max_tokens=512,
        )
        return _parse_single_scoring_response(raw_response, dimension)
    except Exception:
        logger.warning(
            "Failed to score dimension '%s', defaulting to 0", dimension,
            exc_info=True,
        )
        return DimensionScore(
            dimension=dimension,
            score=0,
            rationale="Scoring failed — unable to evaluate claims.",
        )


def compute_overall_score(
    dimension_scores: list[DimensionScore],
    excluded_dimensions: set[str] | None = None,
) -> float:
    """Compute the overall weighted score (0-100). Pure deterministic.

    When ``excluded_dimensions`` is provided, those dimensions are treated as
    not applicable (e.g. ``childrens_data`` for services whose T&Cs restrict
    use to adults; see ADR 001). They are dropped from the weighted sum and
    the remaining dimension weights are renormalised so the result stays on
    the 0-100 scale. If every dimension is excluded, returns 0.
    """
    excluded = excluded_dimensions or set()
    score_map = {ds.dimension: ds.score for ds in dimension_scores}
    weighted_sum = 0.0
    total_weight = 0.0
    for dimension, weight in DIMENSION_WEIGHTS.items():
        if dimension in excluded:
            continue
        dim_score = score_map.get(dimension, 0)
        weighted_sum += dim_score * weight * 10
        total_weight += weight
    if total_weight == 0:
        return 0.0
    return round(weighted_sum / total_weight, 2)


def assign_grade(overall_score: float) -> str:
    """Assign a letter grade based on the overall score.

    Deterministic threshold lookup: A/B/C/D/F.
    """
    for threshold, grade in GRADE_THRESHOLDS:
        if overall_score >= threshold:
            return grade
    return "F"


# --- internals ---

def _call_with_system(
    llm_client: Callable[..., str],
    *,
    user_prompt: str,
    system: str,
    max_tokens: int,
) -> str:
    """Call the LLM preferring system+max_tokens kwargs when supported.

    Falls back to a flat-string prompt for legacy callables.
    """
    if _supports_system_kwarg(llm_client):
        return llm_client(user_prompt, system=system, max_tokens=max_tokens)
    return llm_client(f"{system}\n\n{user_prompt}")


def _supports_system_kwarg(llm_client: Callable[..., object]) -> bool:
    try:
        params = inspect.signature(llm_client).parameters
    except (TypeError, ValueError):
        return False
    return "system" in params


def _group_claims_by_dimension(claims: list[Claim]) -> dict[str, list[Claim]]:
    groups: dict[str, list[Claim]] = {}
    for claim in claims:
        groups.setdefault(claim.dimension, []).append(claim)
    return groups


def _parse_batched_response(raw: str) -> dict[str, dict]:
    """Parse the batched JSON response into {dimension: {score, rationale}}.

    Raises on invalid JSON — caller is responsible for try/except.
    """
    text = raw.strip()
    if text.startswith("```"):
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Batched scoring response is not a JSON object")
    return parsed


def _to_dimension_score(dimension: str, raw: dict | None) -> DimensionScore:
    """Coerce a {"score","rationale"} dict (or None) into a DimensionScore."""
    if raw is None or not isinstance(raw, dict):
        return DimensionScore(
            dimension=dimension, score=0,
            rationale="Scoring failed — unable to evaluate claims.",
        )

    score = raw.get("score", 0)
    rationale = raw.get("rationale", "No rationale provided.")

    if not isinstance(score, int):
        try:
            score = round(float(score))
        except (TypeError, ValueError):
            score = 0
    score = max(0, min(10, score))

    if not isinstance(rationale, str) or not rationale.strip():
        rationale = "No rationale provided."

    return DimensionScore(dimension=dimension, score=score, rationale=rationale)


def _parse_single_scoring_response(raw_response: str, dimension: str) -> DimensionScore:
    """Parse a single-dimension response. Used by score_dimension."""
    text = raw_response.strip()
    if text.startswith("```"):
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return DimensionScore(
            dimension=dimension, score=0,
            rationale="Scoring response was not valid JSON.",
        )
    if not isinstance(parsed, dict):
        return DimensionScore(
            dimension=dimension, score=0,
            rationale="Scoring response was not a JSON object.",
        )
    return _to_dimension_score(dimension, parsed)
