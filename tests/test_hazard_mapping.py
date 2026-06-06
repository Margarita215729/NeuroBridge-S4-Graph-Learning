"""Tests for Phase 5 hazard_mapping.py — HRP five-hazard context mapping.

These tests use small in-memory mock data and do not require project files.
"""

import numpy as np
import pandas as pd
import pytest

from neurobridge_graph.hazard_mapping import (
    HAZARD_CANONICAL,
    HAZARD_DISPLAY_NAMES,
    get_default_hazard_domain_mapping,
    normalize_domain_name,
    compute_hazard_relevance_scores,
    compute_hazard_coverage,
    export_hazard_domain_mapping,
    interpret_hazard_score,
)


def _mock_node_features() -> pd.DataFrame:
    """Two subjects, partial domain coverage (no sleep/cognitive/etc.)."""
    rows = []
    domains = {
        "Cardiovascular regulation": (0.5, 0.2),
        "Metabolic regulation": (1.2, 0.6),
        "Inflammation / immune-adjacent": (0.4, 1.6),
        "Hematologic / oxygen-carrying": (2.0, 0.7),
    }
    for domain, (a1, a2) in domains.items():
        rows.append({"subject_id": "S1", "domain": domain, "activation": a1})
        rows.append({"subject_id": "S2", "domain": domain, "activation": a2})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Mapping + normalization
# ---------------------------------------------------------------------------

def test_default_mapping_has_five_hazards():
    mapping = get_default_hazard_domain_mapping()
    assert set(mapping["hazard"].unique()) == set(HAZARD_CANONICAL)
    assert len(HAZARD_CANONICAL) == 5
    assert set(HAZARD_DISPLAY_NAMES) == set(HAZARD_CANONICAL)


def test_default_mapping_weights_in_range():
    mapping = get_default_hazard_domain_mapping()
    assert mapping["weight"].between(0.0, 1.0).all()
    assert {"hazard", "hazard_display", "domain", "weight", "interpretation"}.issubset(
        mapping.columns
    )


def test_normalize_domain_aliases():
    assert (
        normalize_domain_name("Inflammation / immune-adjacent")
        == "inflammation / immune-adjacent status"
    )
    assert (
        normalize_domain_name("Hematologic / oxygen-carrying")
        == "hematologic / oxygen-carrying capacity"
    )
    # Idempotent on canonical names.
    assert normalize_domain_name("cardiovascular regulation") == "cardiovascular regulation"
    assert normalize_domain_name("  CARDIOVASCULAR   regulation ") == "cardiovascular regulation"


# ---------------------------------------------------------------------------
# Relevance scoring
# ---------------------------------------------------------------------------

def test_relevance_scores_on_mock():
    nf = _mock_node_features()
    scores = compute_hazard_relevance_scores(nf)
    # One row per (subject, hazard).
    assert len(scores) == 2 * len(HAZARD_CANONICAL)
    assert {
        "subject_id", "hazard", "hazard_relevance_score",
        "available_domain_count", "expected_domain_count",
        "coverage_fraction", "coverage_note", "top_contributing_domain",
        "interpretation",
    }.issubset(scores.columns)


def test_relevance_score_formula_space_radiation():
    nf = _mock_node_features()
    scores = compute_hazard_relevance_scores(nf)
    row = scores[(scores["subject_id"] == "S1") & (scores["hazard"] == "space_radiation")].iloc[0]
    # Available domains for S1 space_radiation: inflammation(0.8,act0.4),
    # hematologic(0.7,act2.0), cardiovascular(0.5,act0.5), recovery-related(0.6 -> missing),
    # cognitive load(0.4 -> missing). recovery-related markers absent in mock.
    # num = 0.4*0.8 + 2.0*0.7 + 0.5*0.5 = 0.32 + 1.4 + 0.25 = 1.97
    # den = 0.8 + 0.7 + 0.5 = 2.0 ; score = 0.985
    assert row["available_domain_count"] == 3
    assert row["expected_domain_count"] == 5
    assert np.isclose(row["hazard_relevance_score"], 0.985, atol=1e-3)
    assert 0 < row["coverage_fraction"] < 1


def test_missing_domains_do_not_crash_and_return_nan():
    # A hazard whose mapped domains are entirely absent -> NaN score.
    nf = pd.DataFrame([
        {"subject_id": "S1", "domain": "Cardiovascular regulation", "activation": 0.5},
    ])
    scores = compute_hazard_relevance_scores(nf)
    iso = scores[scores["hazard"] == "isolation_and_confinement"].iloc[0]
    # isolation maps to sleep/autonomic/emotional/cognitive/inflammation/recovery-capacity,
    # none present except possibly none -> NaN.
    assert np.isnan(iso["hazard_relevance_score"])
    assert iso["coverage_fraction"] == 0
    assert iso["coverage_note"] == "No mapped domains available in current proxy dataset."


def test_coverage_report_columns_and_fraction():
    nf = _mock_node_features()
    cov = compute_hazard_coverage(nf)
    assert len(cov) == len(HAZARD_CANONICAL)
    assert {
        "hazard", "expected_domain_count", "available_domain_count",
        "coverage_fraction", "available_domains", "missing_domains",
        "coverage_note",
    }.issubset(cov.columns)
    assert cov["coverage_fraction"].between(0.0, 1.0).all()
    # Gravity should have decent coverage (cardio, metabolic, hematologic present).
    grav = cov[cov["hazard"] == "gravity_fields"].iloc[0]
    assert grav["available_domain_count"] >= 3


def test_interpret_hazard_score_no_coverage():
    txt = interpret_hazard_score("isolation_and_confinement", float("nan"), 0.0)
    assert "cannot be estimated" in txt
    assert "Isolation and Confinement" in txt


def test_interpret_hazard_score_bands():
    high = interpret_hazard_score("gravity_fields", 1.8, 0.83)
    assert "high" in high
    assert "diagnosis" in high.lower()  # guardrail phrasing present
    low = interpret_hazard_score("gravity_fields", 0.3, 0.83)
    assert "low" in low


def test_interpret_hazard_score_low_coverage_caveat():
    txt = interpret_hazard_score("distance_from_earth", 0.9, 0.2)
    assert "coverage-limited" in txt


def test_export_hazard_domain_mapping(tmp_path):
    out = export_hazard_domain_mapping(tmp_path / "hazard_domain_mapping.csv")
    assert out.exists()
    df = pd.read_csv(out)
    assert set(df["hazard"].unique()) == set(HAZARD_CANONICAL)


def test_relevance_requires_columns():
    bad = pd.DataFrame([{"subject_id": "S1", "activation": 0.5}])  # no 'domain'
    with pytest.raises(ValueError):
        compute_hazard_relevance_scores(bad)
