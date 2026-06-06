"""Tests for Phase 10 domain_mapping.py (mock data only)."""

from __future__ import annotations

from neurobridge_graph.domain_mapping import (
    CANONICAL_DOMAINS,
    UNMAPPED_DOMAIN,
    get_default_variable_domain_mapping,
    normalize_variable_name,
    map_variable_to_domain,
    map_variables_dataframe,
    build_domain_coverage_report,
)


def test_default_mapping_exists_and_well_formed():
    m = get_default_variable_domain_mapping()
    assert not m.empty
    expected_cols = {"variable_pattern", "canonical_variable", "domain",
                     "data_stream", "expected_unit", "direction_hint",
                     "interpretation_note"}
    assert expected_cols.issubset(m.columns)
    # all mapped domains are canonical
    assert set(m["domain"]).issubset(set(CANONICAL_DOMAINS))


def test_normalize_variable_name():
    assert normalize_variable_name("HRV (RMSSD)") == "hrv_rmssd"
    assert normalize_variable_name("Sleep Duration") == "sleep_duration"


def test_map_common_variables():
    assert map_variable_to_domain("heart_rate")["domain"] == "cardiovascular regulation"
    assert map_variable_to_domain("glucose")["domain"] == "metabolic regulation"
    assert map_variable_to_domain("crp")["domain"] == "inflammation / immune-adjacent status"
    assert map_variable_to_domain("sleep_duration")["domain"] == "sleep / circadian regulation"
    # substring / variant matching
    assert map_variable_to_domain("hrv_rmssd")["domain"] == "autonomic regulation"
    assert map_variable_to_domain("Resting HR")["domain"] == "cardiovascular regulation"


def test_unmapped_variable_handled_safely():
    res = map_variable_to_domain("totally_unknown_metric")
    assert res["mapping_status"] == "unmapped"
    assert res["domain"] == UNMAPPED_DOMAIN
    assert res["data_stream"] == "unspecified"


def test_map_variables_dataframe_dedup():
    rep = map_variables_dataframe(["heart_rate", "Heart Rate", "glucose", "steps"])
    # heart_rate variants collapse to one normalized row
    assert rep["normalized_variable"].nunique() == 3
    statuses = dict(zip(rep["normalized_variable"], rep["mapping_status"]))
    assert statuses["steps"] == "unmapped"


def test_build_domain_coverage_report():
    rep = map_variables_dataframe(["heart_rate", "glucose", "steps"])
    cov = build_domain_coverage_report(rep)
    by_domain = dict(zip(cov["domain"], cov["coverage_status"]))
    assert by_domain["cardiovascular regulation"] == "covered"
    assert by_domain["metabolic regulation"] == "covered"
    assert by_domain["sleep / circadian regulation"] == "absent"
    # unmapped row present because 'steps' is unmapped
    assert UNMAPPED_DOMAIN in set(cov["domain"])
