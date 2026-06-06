"""Tests for Phase 11 resilience reporting (synthetic data)."""

import pandas as pd

from neurobridge_graph.resilience_reporting import (
    generate_resilience_card,
    generate_phase11_report,
)
from neurobridge_graph.resilience_interpretation import (
    build_phase11_input_readiness_report,
)

_GUARDRAIL_FRAGMENT = ("not diagnosis, treatment guidance, health risk scoring, "
                       "exposure measurement, or an operational medical decision")


def _interpretation():
    return {
        "subject_id": "S1", "timepoint": "T1", "mission_phase": "inflight",
        "resilience_state": "localized_adaptive_shift",
        "resilience_state_label": "Localized adaptive shift",
        "confidence_level": "moderate",
        "dominant_adaptation_mode_label": "Cardiometabolic-recovery dominant",
        "primary_displacement_pattern": "localized to the cardiometabolic subgraph",
        "evidence_chain": ["Top domain contributor: metabolic regulation (0.50).",
                           "Envelope: 1 outside-envelope feature.",
                           "Coverage: 11/12 domains covered."],
        "hazard_context_alignment": "Leading hazard-context alignment is radiation.",
        "recovery_persistence_interpretation": "recovery categories observed: partial_recovery.",
        "data_gap_interpretation": "11/12 biological domains covered.",
        "mission_relevance_context": "Expert review may inspect the dominant domain.",
    }


def test_generate_card_includes_required_fields_and_guardrail():
    card = generate_resilience_card(_interpretation())
    assert "Adaptive Resilience Interpretation Card" in card
    assert "Subject: S1" in card
    assert "Adaptive resilience state: Localized adaptive shift" in card
    assert "Evidence chain:" in card
    assert _GUARDRAIL_FRAGMENT in card


def test_card_has_no_forbidden_language():
    # The fixed guardrail line negates these terms; scan everything before it.
    card = generate_resilience_card(_interpretation()).lower()
    body = card.split("guardrail:")[0]
    for bad in ("risk score", "diagnosis", "treatment guidance", "mission readiness",
                "fitness classification", "disease detection"):
        assert bad not in body


def test_generate_phase11_report_runs_with_data():
    st = pd.DataFrame([{
        "subject_id": "S1", "timepoint": "T1", "mission_phase": "inflight",
        "resilience_state": "localized_adaptive_shift",
        "resilience_state_label": "Localized adaptive shift",
        "confidence_level": "moderate",
        "dominant_adaptation_mode": "Cardiometabolic-recovery dominant",
        "evidence_chain_short": "metabolic regulation",
        "data_gap_summary": "ok",
    }])
    mr = pd.DataFrame([{
        "subject_id": "S1", "timepoint": "T1", "mission_phase": "inflight",
        "resilience_state_label": "Localized adaptive shift",
        "mission_relevance_context": "review", "expert_review_context": "review",
        "data_streams_that_would_strengthen_interpretation": "none", "guardrail": "g"}])
    rdy = build_phase11_input_readiness_report({})
    report = generate_phase11_report(st, mr, rdy)
    assert "Phase 11" in report
    assert "Adaptive resilience states" in report
    assert "Subject/timepoints interpreted: 1" in report
    assert _GUARDRAIL_FRAGMENT in report


def test_generate_phase11_report_handles_empty():
    rdy = build_phase11_input_readiness_report({})
    report = generate_phase11_report(pd.DataFrame(), pd.DataFrame(), rdy)
    assert "Subject/timepoints interpreted: 0" in report
    assert _GUARDRAIL_FRAGMENT in report
