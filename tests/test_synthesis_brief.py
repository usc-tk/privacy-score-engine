"""Tests for the synthesis markdown brief renderer."""

from datetime import datetime, timezone

from engine.models import Finding, SynthesisResult
from engine.synthesis_brief import render_brief


def _result():
    return SynthesisResult(
        findings=[
            Finding(category="dimension_gap", severity="critical",
                    dimension="childrens_data",
                    headline="Children's Data averages 1.6/10",
                    payload={"average": 1.6}, entities_named=["A", "B"]),
            Finding(category="content_gap", severity="high",
                    dimension="data_security", disclosure_id="encryption",
                    headline="10 of 44 policies disclose encryption",
                    payload={"coverage_pct": 22.7}),
            Finding(category="data_quality", severity="medium",
                    headline="2 scans used a partial policy page",
                    entities_named=["Bupa"]),
        ],
        entity_count=50, scored_count=44, engine_version="0.1.0",
        generated_at=datetime(2026, 5, 18, tzinfo=timezone.utc),
        summary={"grade_distribution": {"B": 18, "C": 19},
                 "overall_average": 61.2},
    )


def test_render_brief_has_sections_and_findings():
    md = render_brief(_result())
    assert "# Privacy Synthesis Brief" in md
    assert "2026-05-18" in md
    assert "Systemic Dimension Gaps" in md
    assert "Content Gaps" in md
    assert "Data Quality" in md
    assert "Children's Data averages 1.6/10" in md
    assert "44" in md  # scored count appears


def test_render_brief_is_deterministic():
    assert render_brief(_result()) == render_brief(_result())
