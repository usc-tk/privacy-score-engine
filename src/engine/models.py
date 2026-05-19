"""Pydantic models for claims, scores, and extraction results."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Dimension = Literal[
    "transparency_clarity",
    "data_collection",
    "third_party_sharing",
    "purpose_limitation",
    "consumer_rights",
    "data_security",
    "automated_decision_making",
    "childrens_data",
    "cross_border_flows",
    "policy_maintenance",
]

DIMENSIONS: list[str] = list(Dimension.__args__)  # type: ignore[attr-defined]

DIMENSION_DISPLAY_NAMES: dict[str, str] = {
    "transparency_clarity": "Transparency & Clarity",
    "data_collection": "Data Collection Disclosure",
    "third_party_sharing": "Third-Party Sharing & Disclosure",
    "purpose_limitation": "Purpose Limitation & Use",
    "consumer_rights": "Consumer Rights & Control",
    "data_security": "Data Security",
    "automated_decision_making": "Automated Decision-Making",
    "childrens_data": "Children's Data",
    "cross_border_flows": "Cross-Border Data Flows",
    "policy_maintenance": "Policy Maintenance & Accountability",
}

DIMENSION_APP_REFERENCES: dict[str, str] = {
    "transparency_clarity": "APP 1",
    "data_collection": "APP 3, APP 6",
    "third_party_sharing": "APP 6, APP 8",
    "purpose_limitation": "APP 6",
    "consumer_rights": "APP 12, APP 13",
    "data_security": "APP 11",
    "automated_decision_making": "APP 1.4",
    "childrens_data": "APP 3.5",
    "cross_border_flows": "APP 8",
    "policy_maintenance": "APP 1",
}


class Claim(BaseModel):
    """A single extracted claim from a privacy policy."""

    dimension: Dimension
    claim_type: str = Field(min_length=1)
    claim_value: dict
    confidence: float = Field(ge=0.0, le=1.0)
    app_reference: str = Field(min_length=1)
    source_text: str = Field(min_length=1)

    @field_validator("claim_value")
    @classmethod
    def claim_value_must_be_non_empty(cls, v: dict) -> dict:
        if not v:
            raise ValueError("claim_value must be a non-empty dict")
        return v


class ExtractionResult(BaseModel):
    """Result of extracting claims from a privacy policy."""

    claims: list[Claim]
    policy_text_hash: str
    engine_version: str
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# --- Scoring models (Story 1.4) ---

DIMENSION_WEIGHTS: dict[str, float] = {
    "transparency_clarity": 0.15,
    "data_collection": 0.15,
    "third_party_sharing": 0.15,
    "purpose_limitation": 0.10,
    "consumer_rights": 0.10,
    "data_security": 0.10,
    "automated_decision_making": 0.10,
    "childrens_data": 0.05,
    "cross_border_flows": 0.05,
    "policy_maintenance": 0.05,
}

GRADE_THRESHOLDS: list[tuple[int, str]] = [
    (80, "A"),
    (65, "B"),
    (50, "C"),
    (35, "D"),
    (0, "F"),
]

LetterGrade = Literal["A", "B", "C", "D", "F"]


class DimensionScore(BaseModel):
    """Score for a single privacy dimension."""

    dimension: Dimension
    score: int = Field(ge=0, le=10)
    rationale: str = Field(min_length=1)


class ScoringResult(BaseModel):
    """Result of scoring extracted claims."""

    dimension_scores: list[DimensionScore]
    overall_score: float = Field(ge=0.0, le=100.0)
    letter_grade: LetterGrade
    engine_version: str
    scored_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# --- Synthesis models (Privacy Synthesis Engine) ---

FindingCategory = Literal["dimension_gap", "content_gap", "data_quality"]
Severity = Literal["critical", "high", "medium", "info"]


class Finding(BaseModel):
    """A single structured advisory finding produced by the synthesis engine."""

    category: FindingCategory
    severity: Severity
    dimension: str | None = None
    disclosure_id: str | None = None
    headline: str = Field(min_length=1)
    payload: dict = Field(default_factory=dict)
    entities_named: list[str] = Field(default_factory=list)


class SynthesisResult(BaseModel):
    """The full output of one synthesis run."""

    findings: list[Finding]
    entity_count: int = Field(ge=0)
    scored_count: int = Field(ge=0)
    engine_version: str = Field(min_length=1)
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    summary: dict = Field(default_factory=dict)
