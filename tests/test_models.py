"""Tests for Pydantic models."""

import pytest
from pydantic import ValidationError

from engine.models import (
    Claim,
    DimensionScore,
    ExtractionResult,
    ScoringResult,
    DIMENSIONS,
    DIMENSION_WEIGHTS,
    GRADE_THRESHOLDS,
)


class TestClaim:
    def test_valid_claim(self):
        claim = Claim(
            dimension="data_collection",
            claim_type="collects",
            claim_value={"data_type": "financial_transactions"},
            confidence=0.92,
            app_reference="APP 3",
            source_text="We collect your financial transaction data.",
        )
        assert claim.dimension == "data_collection"
        assert claim.confidence == 0.92

    def test_invalid_dimension(self):
        with pytest.raises(ValidationError):
            Claim(
                dimension="invalid_dimension",
                claim_type="collects",
                claim_value={"data_type": "test"},
                confidence=0.5,
                app_reference="APP 1",
                source_text="test",
            )

    def test_confidence_too_high(self):
        with pytest.raises(ValidationError):
            Claim(
                dimension="data_collection",
                claim_type="collects",
                claim_value={"data_type": "test"},
                confidence=1.5,
                app_reference="APP 3",
                source_text="test",
            )

    def test_confidence_too_low(self):
        with pytest.raises(ValidationError):
            Claim(
                dimension="data_collection",
                claim_type="collects",
                claim_value={"data_type": "test"},
                confidence=-0.1,
                app_reference="APP 3",
                source_text="test",
            )

    def test_empty_claim_value_rejected(self):
        with pytest.raises(ValidationError):
            Claim(
                dimension="data_collection",
                claim_type="collects",
                claim_value={},
                confidence=0.5,
                app_reference="APP 3",
                source_text="test",
            )

    def test_empty_claim_type_rejected(self):
        with pytest.raises(ValidationError):
            Claim(
                dimension="data_collection",
                claim_type="",
                claim_value={"data_type": "test"},
                confidence=0.5,
                app_reference="APP 3",
                source_text="test",
            )

    def test_boundary_confidence_values(self):
        claim_zero = Claim(
            dimension="data_security",
            claim_type="encryption_stated",
            claim_value={"method": "AES-256"},
            confidence=0.0,
            app_reference="APP 11",
            source_text="test",
        )
        assert claim_zero.confidence == 0.0

        claim_one = Claim(
            dimension="data_security",
            claim_type="encryption_stated",
            claim_value={"method": "AES-256"},
            confidence=1.0,
            app_reference="APP 11",
            source_text="test",
        )
        assert claim_one.confidence == 1.0


class TestExtractionResult:
    def test_valid_result(self):
        claim = Claim(
            dimension="cross_border_flows",
            claim_type="transfers_to",
            claim_value={"destination": "US"},
            confidence=0.88,
            app_reference="APP 8",
            source_text="Data may be transferred to the United States.",
        )
        result = ExtractionResult(
            claims=[claim],
            policy_text_hash="abc123",
            engine_version="0.1.0",
        )
        assert len(result.claims) == 1
        assert result.extracted_at is not None

    def test_empty_claims_list(self):
        result = ExtractionResult(
            claims=[],
            policy_text_hash="abc123",
            engine_version="0.1.0",
        )
        assert len(result.claims) == 0


class TestDimensionScore:
    def test_valid_dimension_score(self):
        ds = DimensionScore(
            dimension="data_collection",
            score=7,
            rationale="Good disclosure of collection practices.",
        )
        assert ds.score == 7

    def test_score_out_of_range(self):
        with pytest.raises(ValidationError):
            DimensionScore(dimension="data_collection", score=11, rationale="test")

    def test_score_negative(self):
        with pytest.raises(ValidationError):
            DimensionScore(dimension="data_collection", score=-1, rationale="test")

    def test_empty_rationale_rejected(self):
        with pytest.raises(ValidationError):
            DimensionScore(dimension="data_collection", score=5, rationale="")


class TestScoringResult:
    def test_valid_scoring_result(self):
        ds = DimensionScore(
            dimension="data_collection", score=7, rationale="Good."
        )
        result = ScoringResult(
            dimension_scores=[ds],
            overall_score=70.0,
            letter_grade="B",
            engine_version="0.1.0",
        )
        assert result.letter_grade == "B"
        assert result.scored_at is not None

    def test_invalid_grade(self):
        with pytest.raises(ValidationError):
            ScoringResult(
                dimension_scores=[],
                overall_score=50.0,
                letter_grade="X",
                engine_version="0.1.0",
            )


class TestDimensions:
    def test_ten_dimensions_defined(self):
        assert len(DIMENSIONS) == 10

    def test_all_dimensions_are_snake_case(self):
        for dim in DIMENSIONS:
            assert dim == dim.lower()
            assert " " not in dim

    def test_weights_sum_to_one(self):
        assert abs(sum(DIMENSION_WEIGHTS.values()) - 1.0) < 1e-9

    def test_weights_cover_all_dimensions(self):
        assert set(DIMENSION_WEIGHTS.keys()) == set(DIMENSIONS)

    def test_grade_thresholds_ordered(self):
        thresholds = [t[0] for t in GRADE_THRESHOLDS]
        assert thresholds == sorted(thresholds, reverse=True)
