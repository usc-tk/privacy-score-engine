"""Tests for dimension scoring and overall weighted score computation."""

import json

import pytest

from engine.models import (
    Claim,
    DimensionScore,
    ScoringResult,
    DIMENSIONS,
    DIMENSION_WEIGHTS,
)
from engine.scorer import (
    assign_grade,
    compute_overall_score,
    compute_scores,
    score_dimension,
)

def _make_claim(dimension: str, claim_type: str = "test") -> Claim:
    return Claim(
        dimension=dimension,
        claim_type=claim_type,
        claim_value={"test": True},
        confidence=0.9,
        app_reference="APP 1",
        source_text="Test source text.",
    )


def _make_scoring_mock(scores_by_dim: dict[str, dict]) -> callable:
    """Mock LLM that returns scores based on dimension detected in prompt.

    Accepts the new keyword-only args (system, cached_prefix, max_tokens) so
    both legacy flat-prompt and new system-cached call sites work.

    If the prompt looks like a batched request (contains all 10 dimension
    keys), returns a full batched JSON payload keyed by dimension. Otherwise
    falls back to single-dimension substring matching for legacy callers.
    """
    from engine.models import DIMENSION_DISPLAY_NAMES, DIMENSIONS

    _display_to_dim = {
        name.lower().replace("-", " ").replace("'", ""): dim
        for dim, name in DIMENSION_DISPLAY_NAMES.items()
    }

    def mock_client(prompt: str, *, system=None, cached_prefix=None,
                    max_tokens=None) -> str:
        # Batched path: unique header string appears only in the batched template
        if "Evaluate ALL ten privacy dimensions" in prompt:
            payload = {
                dim: scores_by_dim.get(
                    dim, {"score": 5, "rationale": "Default mock score."}
                )
                for dim in DIMENSIONS
            }
            return json.dumps(payload)

        # Single-dimension legacy path
        prompt_normalized = prompt.lower().replace("-", " ").replace("'", "")
        for display_name, dim in sorted(
            _display_to_dim.items(), key=lambda x: -len(x[0])
        ):
            if display_name in prompt_normalized:
                if dim in scores_by_dim:
                    return json.dumps(scores_by_dim[dim])
                break
        return json.dumps({"score": 5, "rationale": "Default mock score."})
    return mock_client


class TestComputeScores:
    def test_returns_scoring_result(self):
        claims = [_make_claim("data_collection"), _make_claim("data_security")]
        mock_llm = _make_scoring_mock({
            "data_collection": {"score": 7, "rationale": "Good."},
            "data_security": {"score": 8, "rationale": "Strong."},
        })
        result = compute_scores(claims, mock_llm)

        assert isinstance(result, ScoringResult)
        assert len(result.dimension_scores) == 10
        assert result.engine_version == "0.1.0"
        assert result.scored_at is not None

    def test_ten_dimension_scores_always(self):
        mock_llm = _make_scoring_mock({})
        result = compute_scores([], mock_llm)

        assert len(result.dimension_scores) == 10
        dims = {ds.dimension for ds in result.dimension_scores}
        assert dims == set(DIMENSIONS)

    def test_each_score_valid_range(self):
        claims = [_make_claim(d) for d in DIMENSIONS]
        mock_llm = _make_scoring_mock({
            d: {"score": i, "rationale": f"Score {i}."} for i, d in enumerate(DIMENSIONS)
        })
        result = compute_scores(claims, mock_llm)

        for ds in result.dimension_scores:
            assert 0 <= ds.score <= 10
            assert len(ds.rationale) > 0

    def test_overall_score_in_range(self):
        claims = [_make_claim(d) for d in DIMENSIONS]
        mock_llm = _make_scoring_mock({
            d: {"score": 7, "rationale": "Good."} for d in DIMENSIONS
        })
        result = compute_scores(claims, mock_llm)

        assert 0.0 <= result.overall_score <= 100.0


class TestScoreDimension:
    def test_no_claims_returns_zero(self):
        mock_llm = _make_scoring_mock({})
        score = score_dimension("data_collection", [], mock_llm)

        assert score.score == 0
        assert "No relevant claims" in score.rationale

    def test_with_claims_calls_llm(self):
        claims = [_make_claim("cross_border_flows")]
        mock_llm = _make_scoring_mock({
            "cross_border_flows": {"score": 8, "rationale": "Specific destinations listed."},
        })
        score = score_dimension("cross_border_flows", claims, mock_llm)

        assert score.score == 8
        assert score.dimension == "cross_border_flows"


class TestComputeOverallScore:
    def test_all_tens_gives_hundred(self):
        scores = [
            DimensionScore(dimension=d, score=10, rationale="Perfect.")
            for d in DIMENSIONS
        ]
        overall = compute_overall_score(scores)
        assert overall == 100.0

    def test_all_zeros_gives_zero(self):
        scores = [
            DimensionScore(dimension=d, score=0, rationale="Nothing.")
            for d in DIMENSIONS
        ]
        overall = compute_overall_score(scores)
        assert overall == 0.0

    def test_weighted_average_correct(self):
        scores = [
            DimensionScore(dimension=d, score=5, rationale="Average.")
            for d in DIMENSIONS
        ]
        overall = compute_overall_score(scores)
        assert overall == 50.0

    def test_deterministic(self):
        scores = [
            DimensionScore(dimension=d, score=7, rationale="Good.")
            for d in DIMENSIONS
        ]
        result1 = compute_overall_score(scores)
        result2 = compute_overall_score(scores)
        result3 = compute_overall_score(scores)
        assert result1 == result2 == result3

    # ADR 001: excluded dimensions are treated as not applicable and their
    # weight is redistributed across the remaining dimensions.

    def test_excluded_dimension_is_rescaled_not_zeroed(self):
        """Excluding a dim should not drag the overall toward 0.

        All fives with childrens_data excluded should still score 50 —
        the remaining 9 dimensions still average to 5/10.
        """
        scores = [
            DimensionScore(dimension=d, score=5, rationale="Average.")
            for d in DIMENSIONS
        ]
        overall = compute_overall_score(scores, excluded_dimensions={"childrens_data"})
        assert overall == 50.0

    def test_excluding_zero_dim_raises_overall(self):
        """Excluding a dim an entity scored 0 on should raise the overall.

        Sportsbet case: if childrens_data=0 is included, overall is pulled
        down by its 5% weight. Excluding it rescales — the remaining dims'
        average is unchanged but is now the entity's whole score.
        """
        scores = [
            DimensionScore(dimension=d, score=10, rationale="Perfect.")
            for d in DIMENSIONS if d != "childrens_data"
        ] + [DimensionScore(dimension="childrens_data", score=0, rationale="N/A.")]

        included = compute_overall_score(scores)
        excluded = compute_overall_score(scores, excluded_dimensions={"childrens_data"})
        assert included == 95.0  # 10 across 9 dims (weight 0.95), 0 on one (weight 0.05)
        assert excluded == 100.0  # just the 9 dims, rescaled

    def test_empty_excluded_matches_default(self):
        scores = [
            DimensionScore(dimension=d, score=7, rationale="Good.")
            for d in DIMENSIONS
        ]
        assert compute_overall_score(scores) == compute_overall_score(
            scores, excluded_dimensions=set()
        )

    def test_all_dimensions_excluded_returns_zero(self):
        """Pathological input — defensively returns 0 rather than divide-by-zero."""
        scores = [
            DimensionScore(dimension=d, score=10, rationale="Perfect.")
            for d in DIMENSIONS
        ]
        overall = compute_overall_score(scores, excluded_dimensions=set(DIMENSIONS))
        assert overall == 0.0


class TestAssignGrade:
    @pytest.mark.parametrize("score,expected_grade", [
        (100, "A"),
        (80, "A"),
        (79, "B"),
        (65, "B"),
        (64, "C"),
        (50, "C"),
        (49, "D"),
        (35, "D"),
        (34, "F"),
        (0, "F"),
    ])
    def test_grade_boundaries(self, score, expected_grade):
        assert assign_grade(score) == expected_grade

    def test_fractional_scores(self):
        assert assign_grade(79.9) == "B"
        assert assign_grade(80.0) == "A"
        assert assign_grade(64.5) == "C"


class TestMalformedScoringResponses:
    def test_invalid_json(self):
        def bad_llm(prompt: str) -> str:
            return "not json"

        score = score_dimension(
            "data_collection", [_make_claim("data_collection")], bad_llm
        )
        assert score.score == 0

    def test_non_object_json(self):
        def array_llm(prompt: str) -> str:
            return "[1, 2, 3]"

        score = score_dimension(
            "data_collection", [_make_claim("data_collection")], array_llm
        )
        assert score.score == 0

    def test_score_clamped_high(self):
        def high_llm(prompt: str) -> str:
            return json.dumps({"score": 99, "rationale": "Too high."})

        score = score_dimension(
            "data_collection", [_make_claim("data_collection")], high_llm
        )
        assert score.score == 10

    def test_negative_score_clamped_to_zero(self):
        def neg_llm(prompt: str) -> str:
            return json.dumps({"score": -5, "rationale": "Negative."})

        score = score_dimension(
            "data_collection", [_make_claim("data_collection")], neg_llm
        )
        assert score.score == 0

    def test_float_score_rounded(self):
        def float_llm(prompt: str) -> str:
            return json.dumps({"score": 7.8, "rationale": "Float score."})

        score = score_dimension(
            "data_collection", [_make_claim("data_collection")], float_llm
        )
        assert score.score == 8  # round(7.8) = 8, not int(7.8) = 7

    def test_llm_exception_handled(self):
        def exploding_llm(prompt: str) -> str:
            raise RuntimeError("LLM down")

        score = score_dimension(
            "data_collection", [_make_claim("data_collection")], exploding_llm
        )
        assert score.score == 0
        assert "failed" in score.rationale.lower()

    def test_markdown_fenced_response(self):
        def fenced_llm(prompt: str) -> str:
            return '```json\n{"score": 6, "rationale": "Decent."}\n```'

        score = score_dimension(
            "data_collection", [_make_claim("data_collection")], fenced_llm
        )
        assert score.score == 6


class TestReproducibility:
    def test_identical_claims_produce_identical_scores(self):
        claims = [_make_claim("data_collection"), _make_claim("data_security")]
        mock_llm = _make_scoring_mock({
            d: {"score": 7, "rationale": "Consistent."} for d in DIMENSIONS
        })

        result1 = compute_scores(claims, mock_llm)
        result2 = compute_scores(claims, mock_llm)
        result3 = compute_scores(claims, mock_llm)

        assert result1.overall_score == result2.overall_score == result3.overall_score
        assert result1.letter_grade == result2.letter_grade == result3.letter_grade

        for ds1, ds2, ds3 in zip(
            result1.dimension_scores,
            result2.dimension_scores,
            result3.dimension_scores,
        ):
            assert ds1.score == ds2.score == ds3.score


from engine.models import Claim, DIMENSIONS
from engine.scorer import compute_scores, score_all_dimensions


def _all_dim_claims():
    return [
        Claim(dimension=dim, claim_type="t", claim_value={"ok": True},
              confidence=0.9, app_reference="APP 1",
              source_text="src")
        for dim in DIMENSIONS
    ]


def _batched_mock_response():
    return json.dumps({
        dim: {"score": 7, "rationale": f"ok for {dim}"}
        for dim in DIMENSIONS
    })


class TestBatchedScoring:
    def test_single_llm_call_produces_all_scores(self):
        calls = []

        def client(prompt, *, system=None, cached_prefix=None, max_tokens=None):
            calls.append(prompt)
            return _batched_mock_response()

        result = score_all_dimensions(_all_dim_claims(), client)

        assert len(calls) == 1
        assert len(result) == len(DIMENSIONS)
        assert {r.dimension for r in result} == set(DIMENSIONS)
        assert all(r.score == 7 for r in result)

    def test_compute_scores_uses_one_call(self):
        calls = []

        def client(prompt, *, system=None, cached_prefix=None, max_tokens=None):
            calls.append(prompt)
            return _batched_mock_response()

        result = compute_scores(_all_dim_claims(), client)

        assert len(calls) == 1
        assert len(result.dimension_scores) == len(DIMENSIONS)

    def test_missing_dimension_defaults_to_zero(self):
        """If the LLM omits a dimension in the batched response, that
        dimension gets score=0 with a 'scoring failed' rationale rather
        than crashing."""
        partial = json.dumps({
            dim: {"score": 5, "rationale": "ok"}
            for dim in DIMENSIONS[:5]
        })

        def client(prompt, *, system=None, cached_prefix=None, max_tokens=None):
            return partial

        result = score_all_dimensions(_all_dim_claims(), client)
        by_dim = {r.dimension: r for r in result}

        for dim in DIMENSIONS[:5]:
            assert by_dim[dim].score == 5
            assert by_dim[dim].rationale == "ok"
        for dim in DIMENSIONS[5:]:
            assert by_dim[dim].score == 0
            assert "scoring failed" in by_dim[dim].rationale.lower()

    def test_malformed_response_defaults_all_to_zero(self):
        def client(prompt, *, system=None, cached_prefix=None, max_tokens=None):
            return "this is not json"

        result = score_all_dimensions(_all_dim_claims(), client)
        assert all(r.score == 0 for r in result)
        assert len(result) == len(DIMENSIONS)

    def test_empty_claims_short_circuits_without_llm_call(self):
        """When there are zero claims total, don't call the LLM at all —
        every dimension gets a zero with the standard empty rationale."""
        calls = []

        def client(prompt, *, system=None, cached_prefix=None, max_tokens=None):
            calls.append(prompt)
            return _batched_mock_response()

        result = score_all_dimensions([], client)

        assert calls == []
        assert all(r.score == 0 for r in result)
