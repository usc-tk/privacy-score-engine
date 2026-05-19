"""Render a SynthesisResult as a dated markdown brief for editorial use."""

from __future__ import annotations

from engine.models import SynthesisResult

_SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "info": 3}
_SECTIONS = [
    ("dimension_gap", "Systemic Dimension Gaps"),
    ("content_gap", "Content Gaps"),
    ("data_quality", "Data Quality"),
]


def _render_finding(finding) -> list[str]:
    lines = [f"- **[{finding.severity.upper()}]** {finding.headline}"]
    if finding.entities_named:
        shown = ", ".join(finding.entities_named[:12])
        more = len(finding.entities_named) - 12
        if more > 0:
            shown += f", +{more} more"
        lines.append(f"  - Entities: {shown}")
    return lines


def render_brief(result: SynthesisResult) -> str:
    """Return the full markdown brief for a synthesis run."""
    date = result.generated_at.date().isoformat()
    grades = result.summary.get("grade_distribution", {})
    grade_line = " · ".join(
        f"{g}: {grades[g]}" for g in ("A", "B", "C", "D", "F") if g in grades
    )
    lines = [
        f"# Privacy Synthesis Brief — {date}",
        "",
        f"Engine version `{result.engine_version}` · "
        f"{result.scored_count} of {result.entity_count} entities scored.",
        "",
        f"**Overall average:** {result.summary.get('overall_average', 0)} · "
        f"**Grades:** {grade_line or 'n/a'}",
        "",
    ]
    for category, title in _SECTIONS:
        section = sorted(
            (f for f in result.findings if f.category == category),
            key=lambda f: _SEVERITY_ORDER.get(f.severity, 9),
        )
        if not section:
            continue
        lines.append(f"## {title}")
        lines.append("")
        for finding in section:
            lines.extend(_render_finding(finding))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
