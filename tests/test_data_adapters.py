"""Tests for Phase 10 data_adapters.py and adapter_reporting.py (mock data only)."""

from __future__ import annotations

import pandas as pd

from neurobridge_graph.data_adapters import (
    SCHEMA_TEMPLATE_DATA_TYPE,
    STANDARDIZED_COLUMNS,
    DOMAIN_SCORE_LONG_COLUMNS,
    create_data_templates,
    standardize_wide_longitudinal_table,
    standardize_long_longitudinal_table,
    combine_standardized_streams,
    compute_variable_baseline_deltas,
    build_domain_scores_from_variables,
    pivot_domain_scores_wide,
    run_adapter_pipeline,
)
from neurobridge_graph.adapter_reporting import generate_adapter_report


def _wide() -> pd.DataFrame:
    return pd.DataFrame([
        {"subject_id": "S1", "timepoint": "T0", "mission_phase": "baseline",
         "time_index": 0, "heart_rate": 60, "glucose": 90},
        {"subject_id": "S1", "timepoint": "T1", "mission_phase": "inflight",
         "time_index": 1, "heart_rate": 72, "glucose": 99},
    ])


def _long() -> pd.DataFrame:
    return pd.DataFrame([
        {"subject_id": "S1", "timepoint": "T0", "mission_phase": "baseline",
         "time_index": 0, "variable_name": "crp", "value": 1.0, "unit": "mg/L"},
        {"subject_id": "S1", "timepoint": "T1", "mission_phase": "inflight",
         "time_index": 1, "variable_name": "crp", "value": 2.0, "unit": "mg/L"},
    ])


# --------------------------------------------------------------------------
# Templates
# --------------------------------------------------------------------------

def test_create_templates_marks_schema_only(tmp_path):
    paths = create_data_templates(tmp_path / "templates")
    assert len(paths) == 7
    for p in paths:
        df = pd.read_csv(p)
        assert "data_type" in df.columns
        assert (df["data_type"] == SCHEMA_TEMPLATE_DATA_TYPE).all()


# --------------------------------------------------------------------------
# Standardization
# --------------------------------------------------------------------------

def test_standardize_wide():
    std, report = standardize_wide_longitudinal_table(_wide(), "wide_input")
    assert list(std.columns) == STANDARDIZED_COLUMNS
    assert set(std["variable_name"]) == {"heart_rate", "glucose"}
    assert len(std) == 4
    assert report.iloc[0]["detected_format"] == "wide_longitudinal"


def test_standardize_wide_missing_subject_safe():
    df = _wide().drop(columns=["subject_id"])
    std, report = standardize_wide_longitudinal_table(df, "bad")
    assert std.empty
    assert list(std.columns) == STANDARDIZED_COLUMNS


def test_standardize_long():
    std, report = standardize_long_longitudinal_table(_long(), "long_input")
    assert list(std.columns) == STANDARDIZED_COLUMNS
    assert set(std["variable_name"]) == {"crp"}
    assert (std["unit"] == "mg/L").all()


def test_standardize_long_derives_time_index_when_missing():
    df = _long().drop(columns=["time_index"])
    std, report = standardize_long_longitudinal_table(df, "long_input")
    assert "derived" in report.iloc[0]["assumptions"]
    assert sorted(std["time_index"].tolist()) == [0.0, 1.0]


# --------------------------------------------------------------------------
# Combine
# --------------------------------------------------------------------------

def test_combine_dedup():
    a, _ = standardize_wide_longitudinal_table(_wide(), "a")
    b, _ = standardize_long_longitudinal_table(_long(), "b")
    combined = combine_standardized_streams([a, b])
    assert set(combined["variable_name"]) == {"heart_rate", "glucose", "crp"}
    # duplicate subject/timepoint/variable collapses
    dup = combine_standardized_streams([a, a])
    assert len(dup) == len(a)


def test_combine_empty():
    assert combine_standardized_streams([]).empty


# --------------------------------------------------------------------------
# Baseline deltas
# --------------------------------------------------------------------------

def test_baseline_deltas_uses_baseline_phase():
    std, _ = standardize_wide_longitudinal_table(_wide(), "a")
    deltas = compute_variable_baseline_deltas(std)
    hr_t1 = deltas[(deltas["variable_name"] == "heart_rate")
                   & (deltas["timepoint"] == "T1")].iloc[0]
    assert hr_t1["baseline_value"] == 60
    assert hr_t1["delta_from_baseline"] == 12
    assert abs(hr_t1["percent_change_from_baseline"] - 20.0) < 1e-6


def test_baseline_deltas_assumes_earliest_when_no_baseline():
    df = _wide().copy()
    df["mission_phase"] = ["pre", "inflight"]
    std, _ = standardize_wide_longitudinal_table(df, "a")
    deltas = compute_variable_baseline_deltas(std)
    assert (deltas["baseline_assumption"]
            == "earliest_time_index_used_as_baseline").all()


# --------------------------------------------------------------------------
# Domain scores + pivot
# --------------------------------------------------------------------------

def test_build_domain_scores_and_pivot():
    std, _ = standardize_wide_longitudinal_table(_wide(), "a")
    deltas = compute_variable_baseline_deltas(std)
    scores, mapping_report = build_domain_scores_from_variables(deltas)
    assert list(scores.columns) == DOMAIN_SCORE_LONG_COLUMNS
    # heart_rate -> cardiovascular, glucose -> metabolic
    domains = set(scores["domain"])
    assert "cardiovascular regulation" in domains
    assert "metabolic regulation" in domains
    # baseline timepoint has zero mean_abs_delta
    base = scores[(scores["timepoint"] == "T0")
                  & (scores["domain"] == "cardiovascular regulation")].iloc[0]
    assert base["domain_score"] == 0.0
    assert base["domain_score_method"] == "mean_abs_delta"

    wide = pivot_domain_scores_wide(scores)
    assert {"subject_id", "timepoint", "mission_phase", "time_index"}.issubset(wide.columns)
    assert "cardiovascular regulation" in wide.columns


def test_build_domain_scores_empty_safe():
    scores, mapping_report = build_domain_scores_from_variables(pd.DataFrame())
    assert scores.empty
    assert list(scores.columns) == DOMAIN_SCORE_LONG_COLUMNS


# --------------------------------------------------------------------------
# Full pipeline + report
# --------------------------------------------------------------------------

def test_run_pipeline_with_templates(tmp_path):
    out_tables = tmp_path / "results" / "tables"
    templates = tmp_path / "templates"
    outputs = run_adapter_pipeline([], output_dir=out_tables, templates_dir=templates)
    # all expected output files exist
    for name in (
        "adapter_input_readiness_report.csv",
        "standardized_longitudinal_variables.csv",
        "variable_baseline_deltas.csv",
        "variable_domain_mapping_report.csv",
        "domain_coverage_report.csv",
        "adapter_domain_scores_long.csv",
        "adapter_domain_scores_wide.csv",
        "adapter_generated_longitudinal_domain_scores.csv",
        "phase10_data_adapter_report.txt",
    ):
        assert name in outputs
        assert outputs[name].exists()
    # generated graph-ready table is Phase-6 compatible
    gen = pd.read_csv(outputs["adapter_generated_longitudinal_domain_scores.csv"])
    assert {"subject_id", "timepoint", "mission_phase", "time_index", "data_type"}.issubset(gen.columns)
    assert (gen["data_type"] == SCHEMA_TEMPLATE_DATA_TYPE).all()


def test_run_pipeline_with_provided_input(tmp_path):
    inp = tmp_path / "my_input.csv"
    _wide().to_csv(inp, index=False)
    out_tables = tmp_path / "results" / "tables"
    outputs = run_adapter_pipeline([inp], output_dir=out_tables,
                                   templates_dir=tmp_path / "templates")
    gen = pd.read_csv(outputs["adapter_generated_longitudinal_domain_scores.csv"])
    assert "cardiovascular regulation" in gen.columns
    assert (gen["data_type"] == "adapter_generated_from_provided_input").all()


def test_adapter_report_includes_guardrail():
    std, _ = standardize_wide_longitudinal_table(_wide(), "a")
    deltas = compute_variable_baseline_deltas(std)
    scores, mapping_report = build_domain_scores_from_variables(deltas)
    from neurobridge_graph.data_validation import build_input_readiness_report
    from neurobridge_graph.domain_mapping import build_domain_coverage_report
    readiness = build_input_readiness_report({"a": _wide()})
    coverage = build_domain_coverage_report(mapping_report)
    text = generate_adapter_report(readiness, mapping_report, coverage, scores,
                                   data_provenance_note="test note")
    lowered = text.lower()
    assert "does not diagnose, score risk, infer exposure, or recommend treatment" in lowered
    # forbidden standalone phrasing must not appear
    for bad in ("clinical diagnosis", "disease detection", "treatment recommendation",
                "predicts astronaut health", "automated health assessment"):
        assert bad not in lowered
