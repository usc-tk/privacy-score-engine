"""Tests for the disclosure taxonomy."""

from engine.models import DIMENSIONS
from engine.taxonomy import DISCLOSURES, DisclosureSpec, disclosures_for


def test_every_disclosure_has_a_valid_dimension():
    for d in DISCLOSURES:
        assert d.dimension in DIMENSIONS, d.dimension


def test_disclosure_ids_unique_within_dimension():
    for dim in DIMENSIONS:
        ids = [d.disclosure_id for d in disclosures_for(dim)]
        assert len(ids) == len(set(ids)), f"duplicate disclosure_id in {dim}"


def test_every_disclosure_has_keywords_and_label():
    for d in DISCLOSURES:
        assert d.label.strip(), d.disclosure_id
        assert d.match_keywords, d.disclosure_id
        assert all(k == k.lower() for k in d.match_keywords), d.disclosure_id


def test_every_dimension_has_disclosures():
    for dim in DIMENSIONS:
        assert disclosures_for(dim), f"{dim} has no disclosures"
