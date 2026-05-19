"""Curated taxonomy of expected privacy-policy disclosures.

Each `DisclosureSpec` describes one disclosure a well-formed Australian
privacy policy is expected to make, derived from the per-dimension
extraction prompts in `engine/prompts/extraction.py`. The synthesis engine
uses `match_keywords` to decide, via token overlap, whether a scanned
policy covers a disclosure — see `engine.synthesis.match_disclosures`.

This file is plain data. A future LLM-assisted matcher would replace the
matching logic in `synthesis.py`, not this taxonomy.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DisclosureSpec:
    """One expected disclosure within a scoring dimension."""

    dimension: str
    disclosure_id: str
    label: str
    match_keywords: tuple[str, ...]


DISCLOSURES: tuple[DisclosureSpec, ...] = (
    # --- transparency_clarity ---
    DisclosureSpec("transparency_clarity", "scope_statement",
        "Scope statement (who the policy applies to)",
        ("scope", "applies", "coverage", "applicability")),
    DisclosureSpec("transparency_clarity", "plain_language",
        "Plain-language commitment",
        ("plain", "readable", "readability", "clarity")),
    DisclosureSpec("transparency_clarity", "definitions_glossary",
        "Definitions / glossary",
        ("definitions", "glossary", "terms")),
    DisclosureSpec("transparency_clarity", "section_overview",
        "Section overview / table of contents",
        ("overview", "contents", "toc", "introductory", "summary")),
    DisclosureSpec("transparency_clarity", "privacy_contact",
        "Privacy contact information",
        ("contact", "inquiries", "enquiries")),
    DisclosureSpec("transparency_clarity", "last_updated_date",
        "Last-updated date / version",
        ("updated", "version", "revised", "effective", "date")),

    # --- data_collection ---
    DisclosureSpec("data_collection", "information_types",
        "Types of personal information collected",
        ("types", "categories", "kinds")),
    DisclosureSpec("data_collection", "collection_methods",
        "Collection methods",
        ("methods", "method", "directly", "sources", "source")),
    DisclosureSpec("data_collection", "sensitive_information",
        "Sensitive information disclosure",
        ("sensitive", "biometric", "health")),
    DisclosureSpec("data_collection", "consent_basis",
        "Legal basis / consent for collection",
        ("consent", "basis", "lawful", "legal")),
    DisclosureSpec("data_collection", "automated_collection",
        "Automated collection (cookies / tracking)",
        ("cookies", "tracking", "analytics", "pixels", "fingerprinting")),

    # --- third_party_sharing ---
    DisclosureSpec("third_party_sharing", "third_party_categories",
        "Categories of third parties",
        ("categories", "recipients", "providers", "parties")),
    DisclosureSpec("third_party_sharing", "named_third_parties",
        "Specific named third parties",
        ("named", "specific", "list")),
    DisclosureSpec("third_party_sharing", "sharing_purpose",
        "Purpose of each sharing arrangement",
        ("purpose", "reason")),
    DisclosureSpec("third_party_sharing", "third_party_obligations",
        "Third parties contractually bound",
        ("contractual", "contract", "bound", "obligations", "agreements")),
    DisclosureSpec("third_party_sharing", "law_enforcement_disclosure",
        "Government / law-enforcement disclosure",
        ("enforcement", "government", "law", "court")),

    # --- purpose_limitation ---
    DisclosureSpec("purpose_limitation", "stated_purposes",
        "Stated purposes for use",
        ("purpose", "purposes", "use", "uses")),
    DisclosureSpec("purpose_limitation", "secondary_use",
        "Secondary-use disclosure",
        ("secondary", "additional", "beyond")),
    DisclosureSpec("purpose_limitation", "marketing_use",
        "Marketing use of data",
        ("marketing", "promotional", "advertising")),
    DisclosureSpec("purpose_limitation", "purpose_limitation_commitment",
        "Explicit purpose-limitation commitment",
        ("limitation", "limited", "necessary", "only")),

    # --- consumer_rights ---
    DisclosureSpec("consumer_rights", "access_right",
        "Right to access personal information",
        ("access",)),
    DisclosureSpec("consumer_rights", "correction_right",
        "Right to correct personal information",
        ("correct", "correction", "update", "rectify")),
    DisclosureSpec("consumer_rights", "deletion_right",
        "Right to deletion",
        ("delete", "deletion", "erasure", "removal")),
    DisclosureSpec("consumer_rights", "marketing_optout",
        "Right to opt out of marketing",
        ("optout", "unsubscribe", "marketing")),
    DisclosureSpec("consumer_rights", "rights_mechanism",
        "Mechanism to exercise rights",
        ("mechanism", "form", "process", "exercise")),
    DisclosureSpec("consumer_rights", "complaint_oaic",
        "Right to lodge a complaint (OAIC)",
        ("complaint", "oaic", "commissioner", "lodge")),
    DisclosureSpec("consumer_rights", "response_timeframe",
        "Response timeframes for requests",
        ("timeframe", "timeframes", "days", "response")),

    # --- data_security ---
    DisclosureSpec("data_security", "encryption",
        "Encryption practices",
        ("encryption", "encrypted", "ssl", "tls")),
    DisclosureSpec("data_security", "access_controls",
        "Access controls / authentication",
        ("controls", "authentication", "passwords", "passwd")),
    DisclosureSpec("data_security", "breach_notification",
        "Data-breach notification commitment",
        ("breach", "notification", "incident")),
    DisclosureSpec("data_security", "security_standards",
        "Security certifications / standards",
        ("certification", "certifications", "standards", "iso")),
    DisclosureSpec("data_security", "staff_training",
        "Employee security training",
        ("training", "employee", "staff")),
    DisclosureSpec("data_security", "security_audits",
        "Regular security audits or assessments",
        ("audit", "audits", "assessment", "assessments")),

    # --- automated_decision_making ---
    DisclosureSpec("automated_decision_making", "adm_disclosure",
        "Automated decision-making disclosed",
        ("automated", "decision", "decisions", "profiling")),
    DisclosureSpec("automated_decision_making", "ai_ml_use",
        "AI / machine-learning use disclosed",
        ("ai", "ml", "machine", "learning", "artificial")),
    DisclosureSpec("automated_decision_making", "human_review",
        "Right to human review of automated decisions",
        ("human", "review", "manual")),
    DisclosureSpec("automated_decision_making", "adm_optout",
        "Right to opt out of automated decisions",
        ("optout", "opt")),
    DisclosureSpec("automated_decision_making", "adm_logic_transparency",
        "Transparency about decision logic",
        ("logic", "transparency", "explanation", "explain")),

    # --- childrens_data ---
    DisclosureSpec("childrens_data", "age_verification",
        "Age verification mechanism",
        ("verification", "verify")),
    DisclosureSpec("childrens_data", "parental_consent",
        "Parental / guardian consent",
        ("parental", "guardian", "parent")),
    DisclosureSpec("childrens_data", "child_protections",
        "Special protections for children's data",
        ("protection", "protections", "children", "minors")),
    DisclosureSpec("childrens_data", "age_threshold",
        "Age threshold definition",
        ("threshold", "age")),
    DisclosureSpec("childrens_data", "child_collection_restrictions",
        "Restrictions on collecting children's data",
        ("restriction", "restrictions", "restrict", "prohibited")),

    # --- cross_border_flows ---
    DisclosureSpec("cross_border_flows", "transfer_locations",
        "Countries / regions of transfer",
        ("countries", "country", "regions", "overseas", "international")),
    DisclosureSpec("cross_border_flows", "transfer_safeguards",
        "Safeguards for cross-border transfers",
        ("safeguards", "adequacy", "clauses", "scc", "bcr")),
    DisclosureSpec("cross_border_flows", "overseas_recipient_obligations",
        "Overseas recipients bound by equivalent protections",
        ("recipients", "bound", "equivalent")),
    DisclosureSpec("cross_border_flows", "cloud_providers",
        "Cloud providers / data-centre locations",
        ("cloud", "hosting", "datacentre", "datacenter", "servers")),
    DisclosureSpec("cross_border_flows", "transfer_notification",
        "User notification of cross-border transfers",
        ("notification", "notify", "inform")),

    # --- policy_maintenance ---
    DisclosureSpec("policy_maintenance", "review_frequency",
        "Policy review frequency",
        ("review", "frequency", "periodically", "regularly")),
    DisclosureSpec("policy_maintenance", "version_history",
        "Version history / changelog",
        ("version", "history", "changelog", "changes")),
    DisclosureSpec("policy_maintenance", "change_notification",
        "Notification of policy changes",
        ("notification", "notify", "inform")),
    DisclosureSpec("policy_maintenance", "privacy_officer",
        "Named privacy officer / data-protection contact",
        ("officer", "dpo", "contact")),
    DisclosureSpec("policy_maintenance", "governance_framework",
        "Governance / accountability framework",
        ("governance", "accountability", "framework", "monitoring")),
    DisclosureSpec("policy_maintenance", "privacy_impact_assessment",
        "Privacy impact assessment commitment",
        ("impact", "assessment", "pia")),
)


def disclosures_for(dimension: str) -> list[DisclosureSpec]:
    """Return all disclosure specs belonging to a dimension."""
    return [d for d in DISCLOSURES if d.dimension == dimension]
