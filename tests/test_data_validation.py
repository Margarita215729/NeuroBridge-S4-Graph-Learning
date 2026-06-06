"""Tests for Phase 10 data_validation.py (mock data only)."""

from __future__ import annotations

import pandas as pd

from neurobridge_graph.data_validation import (
    normalize_column_name,
    detect_standard_columns,
    detect_table_format,
    measurement_columns,
    validate_required_columns,
    validate_longitudinal_structure,
    summarize_missingness,
    build_input_readiness_report,
)


def _wide() -> pd.DataFrame:
    return pd.DataFrame([
        {"subject_id": "S1", "timepoint": "T0", "mission_phase": "baseline",
         "time_index": 0, "heart_rate": 60, "crp": 1.0},
        {"subject_id": "S1", "timepoint": "T1", "mission_phase": "inflight",
         "time_index": 1, "heart_rate": 70, "crp": 1.5},
    ])


def _long() -> pd.DataFrame:
    return pd.DataFrame([
        {"subject_id": "S1", "timepoint": "T0", "mission_phase": "baseline",
         "time_index": 0, "variable_name": "heart_rate", "value": 60, "unit": "bpm"},
        {"subject_id": "S1", "timepoint": "T1", "mission_phase": "inflight",
         "time_index": 1, "variable_name": "heart_rate", "value": 70, "unit": "bpm"},
    ])


def test_normalize_column_name():
    assert normalize_column_name("Heart Rate (bpm)") == "heart_rate_bpm"
    assert normalize_column_name("  Mission-Phase ") == "mission_phase"
    assert normalize_column_name("subject_id") == "subject_id"


def test_detect_standard_columns_variants():
    df = pd.DataFrame(columns=["crew_id", "visit", "phase", "mission_day", "hr"])
    std = detect_standard_columns(df)
    assert std["subject_id"] == "crew_id"
    assert std["timepoint"] == "visit"
    assert std["mission_phase"] == "phase"
    assert std["time_index"] == "mission_day"


def test_detect_table_format_wide_long_unknown():
    assert detect_table_format(_wide()) == "wide_longitudinal"
    assert detect_table_format(_long()) == "long_longitudinal"
    assert detect_table_format(pd.DataFrame({"a": [1], "b": [2]})) == "unknown"


def test_measurement_columns_excludes_index():
    cols = measurement_columns(_wide())
    assert set(cols) == {"heart_rate", "crp"}


def test_validate_required_columns():
    rep = validate_required_columns(_wide(), ["subject_id", "timepoint", "glucose"])
    statuses = dict(zip(rep["required_column"], rep["status"]))
    assert statuses["subject_id"] == "present"
    assert statuses["timepoint"] == "present"
    assert statuses["glucose"] == "missing"


def test_validate_longitudinal_structure_ok():
    rep = validate_longitudinal_structure(_wide())
    checks = dict(zip(rep["check"], rep["status"]))
    assert checks["subject_column_present"] == "ok"
    assert checks["timepoint_column_present"] == "ok"
    assert checks["numeric_measurements"] == "ok"


def test_validate_longitudinal_structure_flags_nonnumeric():
    df = _wide()
    df["heart_rate"] = df["heart_rate"].astype(object)
    df.loc[0, "heart_rate"] = "bad"
    rep = validate_longitudinal_structure(df)
    checks = dict(zip(rep["check"], rep["status"]))
    assert checks["numeric_measurements"] == "warning"


def test_summarize_missingness():
    df = _wide()
    df.loc[0, "crp"] = None
    miss = summarize_missingness(df)
    crp_row = miss[miss["column"] == "crp"].iloc[0]
    assert crp_row["n_missing"] == 1


def test_build_input_readiness_report():
    rep = build_input_readiness_report({"wide": _wide(), "long": _long(),
                                        "empty": pd.DataFrame()})
    by_name = dict(zip(rep["table_name"], rep["detected_format"]))
    assert by_name["wide"] == "wide_longitudinal"
    assert by_name["long"] == "long_longitudinal"
    assert by_name["empty"] == "unknown"
    wide_row = rep[rep["table_name"] == "wide"].iloc[0]
    assert wide_row["subject_count"] == 1
    assert wide_row["timepoint_count"] == 2
