"""Privacy Synthesis Engine — deterministic analysis of scan data.

Pure functions only: no database, no LLM, no network. Inputs are plain
dicts as loaded by the runner script (see `scripts/run-synthesis.py`);
outputs are the `Finding` / `SynthesisResult` models from `engine.models`.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from engine import __version__
from engine.models import (
    DIMENSIONS,
    DIMENSION_DISPLAY_NAMES,
    DIMENSION_WEIGHTS,
    Finding,
    GRADE_THRESHOLDS,
    SynthesisResult,
)
from engine.taxonomy import DISCLOSURES

_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")


def normalise_claim_type(text: str) -> set[str]:
    """Lowercase and split a free-text claim_type into a token set."""
    return {t for t in _TOKEN_SPLIT.split(text.lower()) if t}


def match_disclosures(claims: list[dict]) -> set[str]:
    """Return the set of disclosure_ids covered by a single entity's claims.

    A disclosure is covered if at least one non-suppressed claim in the
    same dimension has token overlap between its claim_type (plus the keys
    of its claim_value) and the disclosure's match_keywords.
    """
    covered: set[str] = set()
    for claim in claims:
        if claim.get("suppressed"):
            continue
        dimension = claim.get("dimension")
        tokens = normalise_claim_type(claim.get("claim_type", ""))
        value = claim.get("claim_value") or {}
        if isinstance(value, dict):
            for key in value:
                tokens |= normalise_claim_type(str(key))
        for spec in DISCLOSURES:
            if spec.dimension != dimension:
                continue
            if tokens.intersection(spec.match_keywords):
                covered.add(spec.disclosure_id)
    return covered


@dataclass
class EntityRecord:
    """One entity reduced to its latest scan, ready for analysis."""

    name: str
    category: str
    status: str
    partial_policy: bool
    scores: dict[str, int] = field(default_factory=dict)
    rationales: dict[str, str] = field(default_factory=dict)
    covered_disclosures: set[str] = field(default_factory=set)
    dimension_claim_counts: dict[str, int] = field(default_factory=dict)

    @property
    def scored(self) -> bool:
        """True when the latest scan completed and produced dimension scores."""
        return self.status == "completed" and bool(self.scores)


def build_entity_records(
    entities: list[dict],
    scans: list[dict],
    scores: list[dict],
    claims: list[dict],
) -> list[EntityRecord]:
    """Reduce raw scan-database rows to one EntityRecord per entity.

    For each entity, the latest scan (by scan_date) is selected; its scores
    and claims are attached.
    """
    latest_by_entity: dict[str, dict] = {}
    for scan in scans:
        eid = scan["entity_id"]
        current = latest_by_entity.get(eid)
        if current is None or scan["scan_date"] > current["scan_date"]:
            latest_by_entity[eid] = scan

    scores_by_scan: dict[str, list[dict]] = defaultdict(list)
    for row in scores:
        scores_by_scan[row["scan_id"]].append(row)
    claims_by_scan: dict[str, list[dict]] = defaultdict(list)
    for claim in claims:
        claims_by_scan[claim["scan_id"]].append(claim)

    records: list[EntityRecord] = []
    for entity in entities:
        scan = latest_by_entity.get(entity["id"])
        if scan is None:
            records.append(EntityRecord(
                name=entity["name"], category=entity.get("category", ""),
                status="no_scan", partial_policy=False))
            continue
        scan_scores = scores_by_scan.get(scan["id"], [])
        scan_claims = claims_by_scan.get(scan["id"], [])
        dim_counts: dict[str, int] = defaultdict(int)
        for claim in scan_claims:
            if not claim.get("suppressed"):
                dim_counts[claim["dimension"]] += 1
        records.append(EntityRecord(
            name=entity["name"],
            category=entity.get("category", ""),
            status=scan["status"],
            partial_policy=bool(scan.get("partial_policy")),
            scores={s["dimension"]: s["score"] for s in scan_scores},
            rationales={s["dimension"]: (s.get("rationale") or "")
                        for s in scan_scores},
            covered_disclosures=match_disclosures(scan_claims),
            dimension_claim_counts=dict(dim_counts),
        ))
    return records


def _dimension_severity(average: float) -> str:
    """Severity for a dimension gap, from the market average (0-10 scale)."""
    if average < 3:
        return "critical"
    if average < 5:
        return "high"
    if average < 7:
        return "medium"
    return "info"


def compute_dimension_gaps(records: list[EntityRecord]) -> list[Finding]:
    """One Finding per dimension summarising market-wide score performance.

    Only scored entities contribute to the statistics.
    """
    scored = [r for r in records if r.scored]
    findings: list[Finding] = []
    for dim in DIMENSIONS:
        values = [r.scores[dim] for r in scored if dim in r.scores]
        if not values:
            continue
        average = round(sum(values) / len(values), 2)
        zero_entities = sorted(
            r.name for r in scored if r.scores.get(dim) == 0
        )
        with_claims = sum(
            1 for r in scored if r.dimension_claim_counts.get(dim, 0) > 0
        )
        display = DIMENSION_DISPLAY_NAMES.get(dim, dim)
        findings.append(Finding(
            category="dimension_gap",
            severity=_dimension_severity(average),
            dimension=dim,
            headline=(
                f"{display} averages {average}/10 across "
                f"{len(values)} scored entities"
            ),
            payload={
                "average": average,
                "min": min(values),
                "max": max(values),
                "zero_count": len(zero_entities),
                "scored_count": len(values),
                "entities_with_claims": with_claims,
            },
            entities_named=zero_entities,
        ))
    return findings


def _content_severity(coverage_pct: float) -> str | None:
    """Severity for a content gap; None means the disclosure is well covered."""
    if coverage_pct <= 25:
        return "critical"
    if coverage_pct < 50:
        return "high"
    if coverage_pct < 75:
        return "medium"
    return None


def compute_content_gaps(records: list[EntityRecord]) -> list[Finding]:
    """One Finding per disclosure that fewer than 75% of entities cover."""
    scored = [r for r in records if r.scored]
    findings: list[Finding] = []
    if not scored:
        return findings
    total = len(scored)
    for spec in DISCLOSURES:
        missers = sorted(
            r.name for r in scored
            if spec.disclosure_id not in r.covered_disclosures
        )
        covered_count = total - len(missers)
        coverage_pct = round(covered_count / total * 100, 1)
        severity = _content_severity(coverage_pct)
        if severity is None:
            continue
        display = DIMENSION_DISPLAY_NAMES.get(spec.dimension, spec.dimension)
        findings.append(Finding(
            category="content_gap",
            severity=severity,
            dimension=spec.dimension,
            disclosure_id=spec.disclosure_id,
            headline=(
                f"{covered_count} of {total} policies disclose "
                f"\"{spec.label}\" ({display})"
            ),
            payload={
                "disclosure_label": spec.label,
                "coverage_pct": coverage_pct,
                "covered_count": covered_count,
                "total": total,
            },
            entities_named=missers,
        ))
    return findings


_NO_CLAIMS_MARKER = "no relevant claims"
_RESCAN_RATIONALE_THRESHOLD = 3


def compute_data_quality_findings(
    records: list[EntityRecord],
) -> list[Finding]:
    """Findings for data-quality issues that distort the synthesis.

    Three aggregate findings (emitted only when non-empty): partial-policy
    scans, entities that could not be assessed, and re-scan candidates
    (scored entities whose latest scan has many empty-claim rationales).
    """
    findings: list[Finding] = []

    partial = sorted(r.name for r in records if r.partial_policy)
    if partial:
        findings.append(Finding(
            category="data_quality", severity="medium",
            headline=(
                f"{len(partial)} scans used a partial policy page "
                f"— results may understate true coverage"
            ),
            payload={"count": len(partial)},
            entities_named=partial,
        ))

    unassessed = sorted(
        r.name for r in records
        if r.status in ("unable_to_assess", "no_public_policy", "failed",
                        "no_scan")
    )
    if unassessed:
        findings.append(Finding(
            category="data_quality", severity="high",
            headline=(
                f"{len(unassessed)} entities could not be assessed "
                f"— excluded from all statistics"
            ),
            payload={"count": len(unassessed)},
            entities_named=unassessed,
        ))

    rescan: list[str] = []
    for r in records:
        if not r.scored:
            continue
        empty = sum(
            1 for text in r.rationales.values()
            if _NO_CLAIMS_MARKER in text.lower()
        )
        if empty >= _RESCAN_RATIONALE_THRESHOLD:
            rescan.append(r.name)
    rescan.sort()
    if rescan:
        findings.append(Finding(
            category="data_quality", severity="high",
            headline=(
                f"{len(rescan)} entities are re-scan candidates "
                f"— many dimensions found no claims (likely thin extraction)"
            ),
            payload={
                "count": len(rescan),
                "rationale_threshold": _RESCAN_RATIONALE_THRESHOLD,
            },
            entities_named=rescan,
        ))

    return findings


def _overall_score(scores: dict[str, int]) -> float:
    """Weighted overall score (0-100) from per-dimension scores."""
    return round(
        sum(scores.get(d, 0) / 10 * w for d, w in DIMENSION_WEIGHTS.items())
        * 100,
        1,
    )


def _grade(overall: float) -> str:
    """Letter grade for an overall score, using engine GRADE_THRESHOLDS."""
    for minimum, letter in GRADE_THRESHOLDS:
        if overall >= minimum:
            return letter
    return "F"


def synthesise(
    entities: list[dict],
    scans: list[dict],
    scores: list[dict],
    claims: list[dict],
) -> SynthesisResult:
    """Run the full synthesis: build records, compute all findings, summarise."""
    records = build_entity_records(entities, scans, scores, claims)
    scored = [r for r in records if r.scored]

    findings: list[Finding] = []
    findings.extend(compute_dimension_gaps(records))
    findings.extend(compute_content_gaps(records))
    findings.extend(compute_data_quality_findings(records))

    grade_distribution: dict[str, int] = {}
    overalls: list[float] = []
    for r in scored:
        overall = _overall_score(r.scores)
        overalls.append(overall)
        letter = _grade(overall)
        grade_distribution[letter] = grade_distribution.get(letter, 0) + 1

    summary = {
        "grade_distribution": grade_distribution,
        "overall_average": round(sum(overalls) / len(overalls), 1)
        if overalls else 0.0,
    }

    return SynthesisResult(
        findings=findings,
        entity_count=len(records),
        scored_count=len(scored),
        engine_version=__version__,
        summary=summary,
    )
