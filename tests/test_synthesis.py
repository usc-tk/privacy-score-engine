"""Tests for the privacy synthesis engine."""

import pytest

from engine.models import Finding, SynthesisResult


def test_finding_minimal_construction():
    f = Finding(
        category="dimension_gap",
        severity="critical",
        headline="childrens_data averages 1.6/10",
    )
    assert f.dimension is None
    assert f.disclosure_id is None
    assert f.payload == {}
    assert f.entities_named == []


def test_finding_rejects_unknown_category():
    with pytest.raises(ValueError):
        Finding(category="bogus", severity="high", headline="x")


def test_synthesis_result_construction():
    f = Finding(category="data_quality", severity="medium", headline="x")
    result = SynthesisResult(
        findings=[f],
        entity_count=50,
        scored_count=44,
        engine_version="0.1.0",
    )
    assert result.scored_count == 44
    assert result.generated_at is not None
    assert result.summary == {}


from engine.synthesis import normalise_claim_type, match_disclosures


def test_normalise_claim_type_tokenises():
    assert normalise_claim_type("privacy_contact_information") == {
        "privacy", "contact", "information"
    }
    assert normalise_claim_type("AI-ML Usage") == {"ai", "ml", "usage"}


def test_match_disclosures_matches_variant_claim_types():
    # Three textually different claim_types that all mean "privacy contact".
    claims = [
        {"dimension": "transparency_clarity",
         "claim_type": "contact_information", "claim_value": {},
         "suppressed": False},
        {"dimension": "transparency_clarity",
         "claim_type": "privacy_contact_information", "claim_value": {},
         "suppressed": False},
    ]
    covered = match_disclosures(claims)
    assert "privacy_contact" in covered


def test_match_disclosures_ignores_suppressed_claims():
    claims = [
        {"dimension": "data_security", "claim_type": "encryption_practices",
         "claim_value": {}, "suppressed": True},
    ]
    assert match_disclosures(claims) == set()


def test_match_disclosures_uses_claim_value_keys():
    claims = [
        {"dimension": "data_security", "claim_type": "security_measure",
         "claim_value": {"encryption": "AES-256"}, "suppressed": False},
    ]
    assert "encryption" in match_disclosures(claims)


def test_match_disclosures_respects_dimension_boundary():
    # "access" keyword exists in consumer_rights, not data_collection.
    claims = [
        {"dimension": "data_collection", "claim_type": "access_logs",
         "claim_value": {}, "suppressed": False},
    ]
    assert "access_right" not in match_disclosures(claims)


from engine.synthesis import build_entity_records


def _fixture_data():
    entities = [
        {"id": "e1", "name": "Acme Bank", "category": "Banking & Finance"},
        {"id": "e2", "name": "Bupa", "category": "Health & Wellness"},
    ]
    scans = [
        {"id": "s1", "entity_id": "e1", "scan_date": "2026-05-01T00:00:00Z",
         "status": "completed", "partial_policy": False},
        {"id": "s1b", "entity_id": "e1", "scan_date": "2026-05-10T00:00:00Z",
         "status": "completed", "partial_policy": False},
        {"id": "s2", "entity_id": "e2", "scan_date": "2026-05-02T00:00:00Z",
         "status": "unable_to_assess", "partial_policy": False},
    ]
    scores = [
        {"scan_id": "s1b", "dimension": "data_security", "score": 7,
         "rationale": "Strong encryption disclosed."},
        {"scan_id": "s1", "dimension": "data_security", "score": 2,
         "rationale": "stale older scan"},
    ]
    claims = [
        {"scan_id": "s1b", "dimension": "data_security",
         "claim_type": "encryption_practices", "claim_value": {},
         "suppressed": False},
    ]
    return entities, scans, scores, claims


def test_build_entity_records_picks_latest_scan():
    entities, scans, scores, claims = _fixture_data()
    records = build_entity_records(entities, scans, scores, claims)
    by_name = {r.name: r for r in records}
    acme = by_name["Acme Bank"]
    assert acme.scored is True
    assert acme.scores["data_security"] == 7  # from s1b, not stale s1
    assert "encryption" in acme.covered_disclosures


def test_build_entity_records_marks_unscored():
    entities, scans, scores, claims = _fixture_data()
    records = build_entity_records(entities, scans, scores, claims)
    bupa = {r.name: r for r in records}["Bupa"]
    assert bupa.scored is False
    assert bupa.status == "unable_to_assess"


from engine.synthesis import compute_dimension_gaps, EntityRecord


def _scored_record(name, scores, claim_counts=None):
    return EntityRecord(
        name=name, category="Test", status="completed",
        partial_policy=False, scores=scores,
        dimension_claim_counts=claim_counts or {},
    )


def test_compute_dimension_gaps_one_finding_per_dimension():
    records = [
        _scored_record("A", {d: 5 for d in
            __import__("engine.models", fromlist=["DIMENSIONS"]).DIMENSIONS}),
    ]
    findings = compute_dimension_gaps(records)
    assert len(findings) == 10
    assert all(f.category == "dimension_gap" for f in findings)


def test_compute_dimension_gaps_severity_and_zeros():
    from engine.models import DIMENSIONS
    base = {d: 8 for d in DIMENSIONS}
    rec_a = _scored_record("A", {**base, "childrens_data": 0})
    rec_b = _scored_record("B", {**base, "childrens_data": 0})
    findings = compute_dimension_gaps([rec_a, rec_b])
    kids = next(f for f in findings if f.dimension == "childrens_data")
    assert kids.severity == "critical"  # average 0.0 < 3
    assert kids.payload["zero_count"] == 2
    assert kids.payload["average"] == 0.0
    assert set(kids.entities_named) == {"A", "B"}


from engine.synthesis import compute_content_gaps


def test_compute_content_gaps_emits_below_threshold_only():
    # 4 entities; only 1 covers data_security.encryption -> 25% coverage.
    records = [
        _scored_record("A", {"data_security": 5}),
        _scored_record("B", {"data_security": 5}),
        _scored_record("C", {"data_security": 5}),
        _scored_record("D", {"data_security": 5}),
    ]
    records[0].covered_disclosures = {"encryption"}
    findings = compute_content_gaps(records)
    enc = [f for f in findings if f.disclosure_id == "encryption"]
    assert len(enc) == 1
    assert enc[0].category == "content_gap"
    assert enc[0].severity == "critical"  # 25% coverage
    assert enc[0].payload["coverage_pct"] == 25.0
    assert set(enc[0].entities_named) == {"B", "C", "D"}  # the missers


def test_compute_content_gaps_skips_well_covered_disclosures():
    # All 4 cover encryption -> 100% -> no finding.
    records = [_scored_record(n, {"data_security": 5}) for n in "ABCD"]
    for r in records:
        r.covered_disclosures = {"encryption"}
    findings = compute_content_gaps(records)
    assert not [f for f in findings if f.disclosure_id == "encryption"]


from engine.synthesis import compute_data_quality_findings


def test_data_quality_flags_partial_and_unassessed():
    records = [
        EntityRecord("Partial Co", "Test", "completed", True,
                     scores={"data_security": 4}),
        EntityRecord("Failed Co", "Test", "unable_to_assess", False),
        EntityRecord("Fine Co", "Test", "completed", False,
                     scores={"data_security": 7}),
    ]
    findings = compute_data_quality_findings(records)
    partial = next(f for f in findings if "partial" in f.headline.lower())
    assert partial.severity == "medium"
    assert "Partial Co" in partial.entities_named
    unassessed = next(f for f in findings
                      if "could not be assessed" in f.headline.lower())
    assert unassessed.severity == "high"
    assert "Failed Co" in unassessed.entities_named


def test_data_quality_flags_rescan_candidates():
    # An entity whose latest scan has many "no relevant claims" rationales.
    rationales = {f"d{i}": "No relevant claims found in policy." for i in range(4)}
    rec = EntityRecord("Thin Co", "Test", "completed", False,
                       scores={f"d{i}": 0 for i in range(4)},
                       rationales=rationales)
    findings = compute_data_quality_findings([rec])
    rescan = next(f for f in findings if "re-scan" in f.headline.lower())
    assert rescan.severity == "high"
    assert "Thin Co" in rescan.entities_named


from engine.synthesis import synthesise


def test_synthesise_end_to_end():
    entities, scans, scores, claims = _fixture_data()
    result = synthesise(entities, scans, scores, claims)
    assert result.entity_count == 2
    assert result.scored_count == 1  # Acme scored; Bupa unable_to_assess
    assert result.engine_version
    # findings span all three categories where applicable
    cats = {f.category for f in result.findings}
    assert "dimension_gap" in cats
    assert "data_quality" in cats
    assert "grade_distribution" in result.summary


def test_synthesise_handles_empty_input():
    result = synthesise([], [], [], [])
    assert result.entity_count == 0
    assert result.scored_count == 0
    assert result.findings == []
