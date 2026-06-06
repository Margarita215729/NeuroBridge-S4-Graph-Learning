"""Tests for Phase 11 resilience interpretation orchestration (synthetic data)."""

import pandas as pd

from neurobridge_graph.resilience_interpretation import (
    build_phase11_input_readiness_report,
    core_inputs_available,
    get_subject_timepoint_pairs,
    interpret_subject_timepoint_resilience,
    build_resilience_state_table,
    build_mission_relevance_translation,
)


def _node_attr():
    return pd.DataFrame([
        {"subject_id": "S1", "timepoint": "T1", "mission_phase": "inflight",
         "domain": "metabolic regulation", "contribution_share": 0.5, "direction": "increase"},
        {"subject_id": "S1", "timepoint": "T1", "mission_phase": "inflight",
         "domain": "cardiovascular regulation", "contribution_share": 0.3, "direction": "increase"},
    ])


def _subgraph_attr():
    return pd.DataFrame([
        {"subject_id": "S1", "timepoint": "T1", "mission_phase": "inflight",
         "subgraph_name": "cardiometabolic", "n_available_domains": 3,
         "total_contribution_share": 0.7},
    ])


def _envelope_summary():
    return pd.DataFrame([{
        "subject_id": "S1", "timepoint": "T1", "mission_phase": "inflight",
        "n_outside_node_envelope": 1, "n_outside_graph_envelope": 0,
        "n_outside_hazard_envelope": 0, "overall_envelope_flag": "near"}])


def _tables():
    return {
        "trajectory_node_attribution": _node_attr(),
        "trajectory_subgraph_attribution": _subgraph_attr(),
        "phase8_reference_calibrated_summary": _envelope_summary(),
    }


def test_core_inputs_available():
    assert core_inputs_available(_tables()) is True
    assert core_inputs_available({}) is False


def test_readiness_report_marks_missing_required():
    rdy = build_phase11_input_readiness_report({})
    req_missing = rdy[(rdy["required_or_optional"] == "required")
                      & (rdy["status"] == "missing")]
    assert not req_missing.empty


def test_get_subject_timepoint_pairs():
    pairs = get_subject_timepoint_pairs(_tables())
    assert list(pairs.columns) == ["subject_id", "timepoint", "mission_phase"]
    assert len(pairs) == 1


def test_interpret_subject_timepoint_returns_full_object():
    interp = interpret_subject_timepoint_resilience("S1", "T1", _tables())
    for key in ("resilience_state", "resilience_state_label", "confidence_level",
                "dominant_adaptation_mode", "evidence_chain", "hazard_context_alignment",
                "recovery_persistence_interpretation", "data_gap_interpretation",
                "mission_relevance_context", "guardrail"):
        assert key in interp
    assert isinstance(interp["evidence_chain"], list)


def test_build_resilience_state_table_one_row_per_pair():
    st = build_resilience_state_table(_tables())
    assert len(st) == 1
    assert st.iloc[0]["subject_id"] == "S1"
    assert "resilience_state_label" in st.columns


def test_build_state_table_empty_inputs():
    st = build_resilience_state_table({})
    assert st.empty
    assert "resilience_state_label" in st.columns


def test_mission_relevance_translation_no_risk_language():
    st = build_resilience_state_table(_tables())
    mr = build_mission_relevance_translation(st, _tables())
    assert not mr.empty
    blob = " ".join(
        mr[["mission_relevance_context", "expert_review_context"]]
        .astype(str).values.ravel()).lower()
    for bad in ("risk score", "diagnosis", "treatment", "mission readiness",
                "readiness score", "exposure measurement", "fit or unfit"):
        assert bad not in blob


def test_missing_optional_inputs_do_not_crash():
    # Only the required tables, no optional attribution/recovery/coverage.
    interp = interpret_subject_timepoint_resilience("S1", "T1", _tables())
    assert interp["resilience_state"] in {
        "stable_compensated", "localized_adaptive_shift", "distributed_adaptive_load",
        "systemic_strain_pattern", "persistent_displacement", "recovery_lag_pattern",
        "multi_domain_instability", "coverage_limited_interpretation"}
