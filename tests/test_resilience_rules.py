"""Tests for the Phase 11 transparent resilience rule engine.

These tests do not depend on real project data; all inputs are synthetic.
"""

import pandas as pd
import pytest

from neurobridge_graph.resilience_rules import (
    RESILIENCE_STATES,
    DOMINANT_ADAPTATION_MODES,
    classify_resilience_state,
    classify_dominant_adaptation_mode,
    evaluate_coverage_limitations,
    derive_resilience_evidence,
    build_evidence_chain,
)

_ADEQUATE_COVERAGE = {"coverage_limited": False, "coverage_fraction": 0.9,
                      "coverage_note": "ok", "missing_domains": []}


def _base_evidence(**overrides):
    ev = {
        "subject_id": "S1", "timepoint": "T1", "mission_phase": "inflight",
        "top_domain_contributor": "metabolic regulation", "top_domain_share": 0.1,
        "n_node_contributors": 0, "n_increasing": 0, "n_decreasing": 0,
        "direction_mixed": False,
        "top_graph_metric_contributor": "density", "top_graph_share": 0.1,
        "top_subgraph_contributor": "cardiometabolic", "top_subgraph_share": 0.1,
        "n_subgraphs_involved": 0,
        "top_hazard_context": "n/a", "top_hazard_share": 0.0,
        "n_outside_node": 0, "n_outside_graph": 0, "n_outside_hazard": 0,
        "total_outside": 0, "overall_envelope_flag": "within_expected_envelope",
        "recovery_categories": [], "min_recovery_fraction": float("nan"),
        "persistent": False, "recovery_lag": False,
        "coverage": dict(_ADEQUATE_COVERAGE),
    }
    ev.update(overrides)
    return ev


# --------------------------------------------------------------------------
# State classification
# --------------------------------------------------------------------------

def test_stable_compensated():
    res = classify_resilience_state(_base_evidence())
    assert res["resilience_state"] == "stable_compensated"
    assert res["resilience_state_label"] == RESILIENCE_STATES["stable_compensated"]
    assert res["confidence_level"] in ("high", "moderate", "low")


def test_localized_adaptive_shift():
    ev = _base_evidence(total_outside=1, n_outside_node=1, n_node_contributors=1,
                        top_domain_share=0.4)
    res = classify_resilience_state(ev)
    assert res["resilience_state"] == "localized_adaptive_shift"


def test_distributed_adaptive_load():
    ev = _base_evidence(n_node_contributors=4, n_subgraphs_involved=3,
                        total_outside=2, n_outside_node=2)
    res = classify_resilience_state(ev)
    assert res["resilience_state"] == "distributed_adaptive_load"


def test_systemic_strain_pattern():
    ev = _base_evidence(total_outside=5, n_outside_node=3, n_outside_graph=2,
                        n_subgraphs_involved=3)
    res = classify_resilience_state(ev)
    assert res["resilience_state"] == "systemic_strain_pattern"


def test_persistent_displacement_in_recovery():
    ev = _base_evidence(mission_phase="recovery", persistent=True, total_outside=2,
                        n_outside_node=2, recovery_categories=["persistent_shift"])
    res = classify_resilience_state(ev)
    assert res["resilience_state"] == "persistent_displacement"


def test_recovery_lag_pattern():
    ev = _base_evidence(mission_phase="pre_mission", recovery_lag=True,
                        recovery_categories=["partial_recovery"],
                        min_recovery_fraction=0.5, total_outside=0)
    res = classify_resilience_state(ev)
    assert res["resilience_state"] == "recovery_lag_pattern"


def test_multi_domain_instability():
    ev = _base_evidence(direction_mixed=True, n_increasing=3, n_decreasing=2,
                        n_subgraphs_involved=2, total_outside=1, n_outside_node=1)
    res = classify_resilience_state(ev)
    assert res["resilience_state"] == "multi_domain_instability"


def test_coverage_limited_interpretation():
    ev = _base_evidence(coverage={"coverage_limited": True, "coverage_fraction": 0.2,
                                  "coverage_note": "limited", "missing_domains": ["x"]})
    res = classify_resilience_state(ev)
    assert res["resilience_state"] == "coverage_limited_interpretation"
    assert res["confidence_level"] == "coverage_limited"


def test_no_risk_or_diagnosis_language_in_state():
    # Scan rule triggers + the affirmative meaning only; the fixed guardrail
    # sentence legitimately negates these terms ("It is not diagnosis ...").
    for ev in (_base_evidence(), _base_evidence(total_outside=6, n_outside_node=4)):
        res = classify_resilience_state(ev)
        meaning = res["interpretation"].split("This is a baseline-relative")[0]
        text = (meaning + " " + " ".join(res["rule_triggers"])).lower()
        for bad in ("risk score", "diagnosis", "treatment", "danger",
                    "mission readiness", "fit or unfit"):
            assert bad not in text


# --------------------------------------------------------------------------
# Coverage evaluation
# --------------------------------------------------------------------------

def test_evaluate_coverage_none_does_not_force_limited():
    cov = evaluate_coverage_limitations(None)
    assert cov["coverage_limited"] is False


def test_evaluate_coverage_limited_when_sparse():
    df = pd.DataFrame({
        "domain": ["a", "b", "c", "d", "unmapped"],
        "coverage_status": ["covered", "absent", "absent", "absent", "absent"],
    })
    cov = evaluate_coverage_limitations(df)
    assert cov["coverage_limited"] is True
    assert cov["n_covered_domains"] == 1


# --------------------------------------------------------------------------
# Evidence derivation + chain
# --------------------------------------------------------------------------

def _node_attr():
    return pd.DataFrame([
        {"subject_id": "S1", "timepoint": "T1", "mission_phase": "inflight",
         "domain": "metabolic regulation", "contribution_share": 0.5,
         "direction": "increase"},
        {"subject_id": "S1", "timepoint": "T1", "mission_phase": "inflight",
         "domain": "cardiovascular regulation", "contribution_share": 0.3,
         "direction": "decrease"},
    ])


def _subgraph_attr():
    return pd.DataFrame([
        {"subject_id": "S1", "timepoint": "T1", "mission_phase": "inflight",
         "subgraph_name": "cardiometabolic", "n_available_domains": 3,
         "total_contribution_share": 0.6},
        {"subject_id": "S1", "timepoint": "T1", "mission_phase": "inflight",
         "subgraph_name": "immune_metabolic", "n_available_domains": 2,
         "total_contribution_share": 0.4},
    ])


def test_derive_evidence_collects_top_contributors():
    ev = derive_resilience_evidence(
        "S1", "T1",
        node_attr=_node_attr(), graph_attr=None, subgraph_attr=_subgraph_attr(),
        hazard_attr=None, recovery_attr=None,
        envelope_summary=pd.DataFrame([{
            "subject_id": "S1", "timepoint": "T1", "mission_phase": "inflight",
            "n_outside_node_envelope": 2, "n_outside_graph_envelope": 1,
            "n_outside_hazard_envelope": 0, "overall_envelope_flag": "outside"}]),
        node_envelope=None, graph_envelope=None, hazard_envelope=None,
        coverage_report=None)
    assert ev["top_domain_contributor"] == "metabolic regulation"
    assert ev["n_subgraphs_involved"] == 2
    assert ev["total_outside"] == 3


def test_build_evidence_chain_is_ordered_list():
    ev = _base_evidence(total_outside=2, n_outside_node=2, top_hazard_context="radiation")
    chain = build_evidence_chain(ev)
    assert isinstance(chain, list) and len(chain) >= 2
    assert all(isinstance(x, str) for x in chain)


# --------------------------------------------------------------------------
# Dominant adaptation mode
# --------------------------------------------------------------------------

def test_dominant_mode_single_subgraph():
    sg = pd.DataFrame([{
        "subject_id": "S1", "timepoint": "T1", "subgraph_name": "cardiometabolic",
        "n_available_domains": 3, "total_contribution_share": 0.8}])
    res = classify_dominant_adaptation_mode(sg, None, None, "S1", "T1")
    assert res["dominant_adaptation_mode"] == "cardiometabolic_recovery_dominant"


def test_dominant_mode_distributed():
    res = classify_dominant_adaptation_mode(_subgraph_attr(), None, None, "S1", "T1")
    assert res["dominant_adaptation_mode"] == "multi_subgraph_distributed"


def test_dominant_mode_coverage_limited():
    df = pd.DataFrame({"domain": ["a", "b", "c"],
                       "coverage_status": ["covered", "absent", "absent"]})
    res = classify_dominant_adaptation_mode(None, None, df, "S1", "T1")
    assert res["dominant_adaptation_mode"] == "coverage_limited"
