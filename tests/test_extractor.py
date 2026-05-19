"""Tests for claim extraction pipeline."""

import hashlib
import json
from pathlib import Path

import pytest

from engine.extractor import (
    MAX_POLICY_TEXT_CHARS,
    extract_claims,
    extract_claims_for_dimension,
)
from engine.models import Claim, ExtractionResult, DIMENSIONS
from engine.prompts.extraction import (
    POLICY_TEXT_CLOSE,
    POLICY_TEXT_OPEN,
    SYSTEM_PROMPT,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "sample_policies"


def _load_fixture(name: str) -> str:
    return (FIXTURES_DIR / name).read_text()


def _load_json_fixture(name: str) -> list[dict]:
    return json.loads((FIXTURES_DIR / name).read_text())


def _make_mock_llm(claims: list[dict]) -> callable:
    """Create a mock LLM client that returns pre-built claims JSON."""
    def mock_client(prompt: str) -> str:
        return json.dumps(claims)
    return mock_client


def _make_dimension_aware_mock(fixture_claims: list[dict]) -> callable:
    """Create a mock LLM that returns claims matching the requested dimension.

    Matches by checking if any word from the dimension key appears in the prompt,
    handling hyphens, apostrophes, and underscores robustly.
    """
    def _normalize(text: str) -> str:
        return text.lower().replace("-", " ").replace("_", " ").replace("'", "")

    def mock_client(prompt: str) -> str:
        normalized_prompt = _normalize(prompt)
        for dim in DIMENSIONS:
            normalized_dim = _normalize(dim)
            if normalized_dim in normalized_prompt:
                matching = [c for c in fixture_claims if c["dimension"] == dim]
                if matching:
                    return json.dumps(matching)
        return "[]"
    return mock_client


class TestExtractClaims:
    def test_returns_extraction_result(self):
        claims_data = [
            {
                "claim_type": "collects",
                "claim_value": {"data_type": "email"},
                "confidence": 0.9,
                "app_reference": "APP 3",
                "source_text": "We collect your email.",
            }
        ]
        mock_llm = _make_mock_llm(claims_data)
        result = extract_claims("Some policy text", mock_llm)

        assert isinstance(result, ExtractionResult)
        assert len(result.claims) > 0
        assert result.engine_version == "0.1.0"
        assert result.extracted_at is not None

    def test_policy_text_hash_consistent(self):
        mock_llm = _make_mock_llm([])
        policy = "Consistent policy text for hashing"

        result1 = extract_claims(policy, mock_llm)
        result2 = extract_claims(policy, mock_llm)

        assert result1.policy_text_hash == result2.policy_text_hash
        expected_hash = hashlib.sha256(policy.encode("utf-8")).hexdigest()
        assert result1.policy_text_hash == expected_hash

    def test_claims_across_multiple_dimensions(self):
        commbank_claims = _load_json_fixture("expected_claims_commbank.json")
        mock_llm = _make_dimension_aware_mock(commbank_claims)
        policy_text = _load_fixture("commbank_privacy_excerpt.txt")

        result = extract_claims(policy_text, mock_llm)

        dimensions_found = {c.dimension for c in result.claims}
        assert len(dimensions_found) >= 3, (
            f"Expected claims across 3+ dimensions, got: {dimensions_found}"
        )

    def test_all_claims_have_required_fields(self):
        commbank_claims = _load_json_fixture("expected_claims_commbank.json")
        mock_llm = _make_dimension_aware_mock(commbank_claims)
        policy_text = _load_fixture("commbank_privacy_excerpt.txt")

        result = extract_claims(policy_text, mock_llm)

        for claim in result.claims:
            assert claim.dimension in DIMENSIONS
            assert len(claim.claim_type) > 0
            assert len(claim.claim_value) > 0
            assert 0.0 <= claim.confidence <= 1.0
            assert len(claim.app_reference) > 0
            assert len(claim.source_text) > 0

    def test_confidence_within_range(self):
        commbank_claims = _load_json_fixture("expected_claims_commbank.json")
        mock_llm = _make_dimension_aware_mock(commbank_claims)
        policy_text = _load_fixture("commbank_privacy_excerpt.txt")

        result = extract_claims(policy_text, mock_llm)

        for claim in result.claims:
            assert 0.0 <= claim.confidence <= 1.0


class TestExtractClaimsForDimension:
    def test_single_dimension_extraction(self):
        claims_data = [
            {
                "claim_type": "transfers_to",
                "claim_value": {"destination": "US"},
                "confidence": 0.88,
                "app_reference": "APP 8",
                "source_text": "Data transferred to the US.",
            }
        ]
        mock_llm = _make_mock_llm(claims_data)

        claims = extract_claims_for_dimension(
            "Some policy text", "cross_border_flows", mock_llm
        )

        assert len(claims) == 1
        assert claims[0].dimension == "cross_border_flows"
        assert claims[0].claim_type == "transfers_to"


class TestCriticalDimensions:
    """Test extraction patterns for critical dimensions per AC 10."""

    def test_cross_border_claims(self):
        telstra_claims = _load_json_fixture("expected_claims_telstra.json")
        cross_border = [c for c in telstra_claims if c["dimension"] == "cross_border_flows"]
        mock_llm = _make_mock_llm(cross_border)

        claims = extract_claims_for_dimension(
            _load_fixture("telstra_privacy_excerpt.txt"),
            "cross_border_flows",
            mock_llm,
        )

        assert len(claims) >= 1
        assert any(c.claim_type == "transfers_to" for c in claims)

    def test_automated_decision_making_claims(self):
        telstra_claims = _load_json_fixture("expected_claims_telstra.json")
        adm = [c for c in telstra_claims if c["dimension"] == "automated_decision_making"]
        mock_llm = _make_mock_llm(adm)

        claims = extract_claims_for_dimension(
            _load_fixture("telstra_privacy_excerpt.txt"),
            "automated_decision_making",
            mock_llm,
        )

        assert len(claims) >= 1
        assert any(c.claim_type == "adm_disclosed" for c in claims)

    def test_third_party_sharing_claims(self):
        telstra_claims = _load_json_fixture("expected_claims_telstra.json")
        sharing = [c for c in telstra_claims if c["dimension"] == "third_party_sharing"]
        mock_llm = _make_mock_llm(sharing)

        claims = extract_claims_for_dimension(
            _load_fixture("telstra_privacy_excerpt.txt"),
            "third_party_sharing",
            mock_llm,
        )

        assert len(claims) >= 1
        assert any(c.claim_type == "shares_with" for c in claims)

    def test_sensitive_data_collection(self):
        commbank_claims = _load_json_fixture("expected_claims_commbank.json")
        sensitive = [c for c in commbank_claims if c["claim_type"] == "collects_sensitive"]
        mock_llm = _make_mock_llm(sensitive)

        claims = extract_claims_for_dimension(
            _load_fixture("commbank_privacy_excerpt.txt"),
            "data_collection",
            mock_llm,
        )

        assert len(claims) >= 1
        assert any(c.claim_type == "collects_sensitive" for c in claims)


class TestMalformedResponses:
    def test_invalid_json_returns_empty(self):
        def bad_llm(prompt: str) -> str:
            return "this is not json at all"

        claims = extract_claims_for_dimension(
            "Some text", "data_collection", bad_llm
        )
        assert claims == []

    def test_non_array_json_returns_empty(self):
        def obj_llm(prompt: str) -> str:
            return '{"not": "an array"}'

        claims = extract_claims_for_dimension(
            "Some text", "data_collection", obj_llm
        )
        assert claims == []

    def test_partial_valid_claims_kept(self):
        def mixed_llm(prompt: str) -> str:
            return json.dumps([
                {
                    "claim_type": "collects",
                    "claim_value": {"data_type": "email"},
                    "confidence": 0.9,
                    "app_reference": "APP 3",
                    "source_text": "We collect email.",
                },
                {
                    "claim_type": "",
                    "claim_value": {},
                    "confidence": 999,
                    "app_reference": "",
                    "source_text": "",
                },
            ])

        claims = extract_claims_for_dimension(
            "Some text", "data_collection", mixed_llm
        )
        assert len(claims) == 1
        assert claims[0].claim_type == "collects"

    def test_markdown_code_fence_handled(self):
        def fenced_llm(prompt: str) -> str:
            return '```json\n[{"claim_type": "collects", "claim_value": {"type": "test"}, "confidence": 0.8, "app_reference": "APP 3", "source_text": "test"}]\n```'

        claims = extract_claims_for_dimension(
            "Some text", "data_collection", fenced_llm
        )
        assert len(claims) == 1

    def test_llm_exception_handled_gracefully(self):
        def exploding_llm(prompt: str) -> str:
            raise RuntimeError("LLM service unavailable")

        result = extract_claims("Some text", exploding_llm)
        assert isinstance(result, ExtractionResult)
        assert len(result.claims) == 0


class TestCachedCallStructure:
    def test_cached_path_sends_system_and_prefix_separately(self):
        """When the client supports system/cached_prefix kwargs, the
        extractor must pass SYSTEM_PROMPT as system and the policy text
        as cached_prefix, so only the dimension-specific instruction
        varies across the 10 calls."""
        calls = []

        def client(prompt, *, system=None, cached_prefix=None, max_tokens=None):
            calls.append({
                "prompt": prompt,
                "system": system,
                "cached_prefix": cached_prefix,
            })
            return "[]"

        policy_text = "POLICY BODY TEXT"
        extract_claims_for_dimension(policy_text, "data_collection", client)

        assert len(calls) == 1
        assert calls[0]["system"] == SYSTEM_PROMPT
        assert "POLICY BODY TEXT" in calls[0]["cached_prefix"]
        # Dimension-specific suffix does NOT repeat the policy text
        assert "POLICY BODY TEXT" not in calls[0]["prompt"]
        # Dimension-specific suffix mentions the dimension topic
        assert "data collection" in calls[0]["prompt"].lower() \
            or "Data Collection" in calls[0]["prompt"]

    def test_legacy_callable_still_works(self):
        """Clients without system/cached_prefix support get a flat-string prompt."""
        calls = []

        def legacy_client(prompt):   # No keyword args
            calls.append(prompt)
            return "[]"

        extract_claims_for_dimension("POLICY", "data_collection", legacy_client)

        assert len(calls) == 1
        # Flat prompt contains system text, policy, and dimension instruction
        assert "privacy policy analyst" in calls[0].lower()  # SYSTEM_PROMPT marker
        assert "POLICY" in calls[0]
        assert "Data Collection" in calls[0]


class TestOpenSourceBoundary:
    def test_no_supabase_references(self):
        engine_src = Path(__file__).parent.parent / "src" / "engine"
        for py_file in engine_src.rglob("*.py"):
            content = py_file.read_text()
            assert "supabase" not in content.lower(), (
                f"Found 'supabase' reference in {py_file}"
            )

    def test_no_os_environ_usage(self):
        engine_src = Path(__file__).parent.parent / "src" / "engine"
        for py_file in engine_src.rglob("*.py"):
            content = py_file.read_text()
            assert "import os" not in content, (
                f"Found 'import os' in {py_file}"
            )
            assert "environ" not in content, (
                f"Found 'environ' reference in {py_file}"
            )

    def test_no_llm_library_imports(self):
        engine_src = Path(__file__).parent.parent / "src" / "engine"
        banned = ["import anthropic", "from anthropic", "import openai", "from openai"]
        for py_file in engine_src.rglob("*.py"):
            content = py_file.read_text()
            for term in banned:
                assert term not in content, (
                    f"Found '{term}' in {py_file}"
                )


class TestInjectionAndSizeGuardrails:
    """Guardrails: untrusted-input delimiters and the policy-size cap."""

    def test_oversized_policy_text_is_truncated(self, caplog):
        oversized = "x" * (MAX_POLICY_TEXT_CHARS + 5000)
        mock_llm = _make_mock_llm([])

        with caplog.at_level("WARNING"):
            result = extract_claims(oversized, mock_llm)

        assert isinstance(result, ExtractionResult)
        # The hash reflects the truncated text that was actually analysed.
        truncated = oversized[:MAX_POLICY_TEXT_CHARS]
        expected_hash = hashlib.sha256(truncated.encode("utf-8")).hexdigest()
        assert result.policy_text_hash == expected_hash
        assert any(
            "truncat" in r.getMessage().lower() for r in caplog.records
        )

    def test_normal_policy_text_is_not_truncated(self):
        mock_llm = _make_mock_llm([])
        policy = "A perfectly normal short privacy policy."

        result = extract_claims(policy, mock_llm)

        expected_hash = hashlib.sha256(policy.encode("utf-8")).hexdigest()
        assert result.policy_text_hash == expected_hash

    def test_cached_prefix_wraps_policy_in_untrusted_delimiters(self):
        captured = []

        def client(prompt, *, system=None, cached_prefix=None, max_tokens=None):
            captured.append(cached_prefix)
            return "[]"

        extract_claims_for_dimension("SCANNED POLICY BODY", "data_security", client)

        assert len(captured) == 1
        prefix = captured[0]
        assert POLICY_TEXT_OPEN in prefix
        assert POLICY_TEXT_CLOSE in prefix
        assert "SCANNED POLICY BODY" in prefix

    def test_legacy_prompt_wraps_policy_in_untrusted_delimiters(self):
        captured = []

        def legacy_client(prompt):
            captured.append(prompt)
            return "[]"

        extract_claims_for_dimension(
            "SCANNED POLICY BODY", "data_security", legacy_client
        )

        assert len(captured) == 1
        assert POLICY_TEXT_OPEN in captured[0]
        assert POLICY_TEXT_CLOSE in captured[0]
        assert "SCANNED POLICY BODY" in captured[0]

    def test_system_prompt_is_hardened_against_injection(self):
        lowered = SYSTEM_PROMPT.lower()
        assert POLICY_TEXT_OPEN in SYSTEM_PROMPT
        assert "untrusted" in lowered
        assert "instruction" in lowered
