"""Tests for LLM-driven sector classification."""

import json

from engine.sector_classifier import SECTORS, classify_sector


def _mock_llm(response: str):
    def client(prompt: str, *, system=None, cached_prefix=None,
               max_tokens=None) -> str:
        return response
    return client


def _raising_llm(exc: Exception):
    def client(prompt: str, *, system=None, cached_prefix=None,
               max_tokens=None) -> str:
        raise exc
    return client


class TestClassifySector:
    def test_returns_valid_sector(self):
        llm = _mock_llm(json.dumps({"sector": "Banking & Finance"}))
        result = classify_sector(
            name="YouI",
            domain="youi.com.au",
            policy_text="We are an insurance provider...",
            llm_client=llm,
        )
        assert result == "Banking & Finance"

    def test_strips_markdown_code_fences(self):
        llm = _mock_llm(
            '```json\n{"sector": "Telecom"}\n```'
        )
        assert classify_sector(
            name="Telstra", domain="telstra.com.au",
            policy_text="ISP policy", llm_client=llm,
        ) == "Telecom"

    def test_invalid_sector_falls_back_to_other(self):
        llm = _mock_llm(json.dumps({"sector": "Aerospace"}))
        assert classify_sector(
            name="SpaceCo", domain="spaceco.com",
            policy_text="...", llm_client=llm,
        ) == "Other"

    def test_missing_sector_field_falls_back_to_other(self):
        llm = _mock_llm(json.dumps({"foo": "bar"}))
        assert classify_sector(
            name="X", domain="x.com", policy_text="...", llm_client=llm,
        ) == "Other"

    def test_malformed_json_falls_back_to_other(self):
        llm = _mock_llm("not json at all")
        assert classify_sector(
            name="X", domain="x.com", policy_text="...", llm_client=llm,
        ) == "Other"

    def test_llm_exception_returns_none(self):
        """None lets the caller keep its pre-scan seed category."""
        llm = _raising_llm(RuntimeError("api down"))
        assert classify_sector(
            name="X", domain="x.com", policy_text="...", llm_client=llm,
        ) is None

    def test_truncates_long_policy_text(self):
        """Classifier should cap policy text to keep tokens predictable."""
        captured_prompts: list[str] = []

        def client(prompt: str, *, system=None, cached_prefix=None,
                   max_tokens=None) -> str:
            captured_prompts.append(prompt)
            return json.dumps({"sector": "Other"})

        huge = "x" * 50_000
        classify_sector(
            name="N", domain="n.com", policy_text=huge, llm_client=client,
        )
        assert captured_prompts
        assert len(captured_prompts[0]) < 5_000

    def test_passes_system_prompt_to_cache_aware_client(self):
        captured: dict = {}

        def client(prompt: str, *, system=None, cached_prefix=None,
                   max_tokens=None) -> str:
            captured["system"] = system
            captured["max_tokens"] = max_tokens
            return json.dumps({"sector": "Retail & Ecommerce"})

        classify_sector(
            name="Bunnings", domain="bunnings.com.au",
            policy_text="We sell hardware.", llm_client=client,
        )
        assert captured["system"] is not None
        assert "sector" in captured["system"].lower()
        assert isinstance(captured["max_tokens"], int)

    def test_works_with_legacy_flat_prompt_client(self):
        """Clients without system/max_tokens kwargs still work."""
        def legacy_client(prompt: str) -> str:
            return json.dumps({"sector": "Gambling & Gaming"})

        assert classify_sector(
            name="Sportsbet", domain="sportsbet.com.au",
            policy_text="Wagering policy", llm_client=legacy_client,
        ) == "Gambling & Gaming"

    def test_sectors_match_canonical_list(self):
        """Guard against drift from the web's SECTORS constant."""
        assert SECTORS == [
            "Banking & Finance",
            "Technology & SaaS",
            "Telecom",
            "Retail & Ecommerce",
            "Health & Wellness",
            "Real Estate",
            "Gambling & Gaming",
            "Government & Utilities",
            "Other",
        ]
