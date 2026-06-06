"""Phase 11 — Plain-language reporting for operational resilience interpretation.

Generates per subject/timepoint Adaptive Resilience Interpretation Cards and a
full Phase 11 report.

Operational resilience interpretation is a research-review layer. It describes
baseline-relative adaptation patterns for expert interpretation. It is not
diagnosis, treatment guidance, health risk scoring, exposure measurement, or an
operational medical decision.
"""

from __future__ import annotations

import pandas as pd

from neurobridge_graph.resilience_rules import (
    GUARDRAIL,
    RESILIENCE_STATES,
    DOMINANT_ADAPTATION_MODES,
)

_GUARDRAIL_CARD = (
    "Operational resilience interpretation is a research-review layer. It is "
    "not diagnosis, treatment guidance, health risk scoring, exposure "
    "measurement, or an operational medical decision."
)

_STATE_DEFINITIONS: dict[str, str] = {
    "stable_compensated":
        "Low baseline-relative deltas; few/no outside-envelope features; recovery near baseline.",
    "localized_adaptive_shift":
        "One dominant domain/subgraph contributor with limited spread across domains.",
    "distributed_adaptive_load":
        "Multiple active domains and subgraphs with broad graph-level displacement.",
    "systemic_strain_pattern":
        "High graph displacement across multiple features/subgraphs with recovery not established.",
    "persistent_displacement":
        "Graph remains displaced from baseline into the post-shift (postflight/recovery) phase.",
    "recovery_lag_pattern":
        "Partial or delayed return toward baseline in selected features.",
    "multi_domain_instability":
        "Mixed-direction deltas and alternating subgraph dominance across domains.",
    "coverage_limited_interpretation":
        "Interpretation limited by insufficient data coverage.",
}


def generate_resilience_card(interpretation: dict) -> str:
    """Generate one Adaptive Resilience Interpretation Card."""
    chain = interpretation.get("evidence_chain", []) or []
    chain_lines = "\n".join(f"{i + 1}. {b}" for i, b in enumerate(chain)) or "1. No evidence available."

    return "\n".join([
        "Adaptive Resilience Interpretation Card",
        "",
        f"Subject: {interpretation.get('subject_id', 'n/a')}",
        f"Timepoint: {interpretation.get('timepoint', 'n/a')}",
        f"Mission phase: {interpretation.get('mission_phase', 'unknown')}",
        "",
        f"Adaptive resilience state: {interpretation.get('resilience_state_label', 'n/a')} "
        f"(confidence: {interpretation.get('confidence_level', 'n/a')})",
        f"Dominant adaptation mode: {interpretation.get('dominant_adaptation_mode_label', 'n/a')}",
        f"Primary graph displacement pattern: {interpretation.get('primary_displacement_pattern', 'n/a')}",
        "",
        "Evidence chain:",
        chain_lines,
        "",
        f"HRP hazard-context alignment: {interpretation.get('hazard_context_alignment', 'n/a')}",
        "",
        f"Recovery / persistence interpretation: "
        f"{interpretation.get('recovery_persistence_interpretation', 'n/a')}",
        "",
        f"Data coverage and data gaps: {interpretation.get('data_gap_interpretation', 'n/a')}",
        "",
        f"Mission-relevant expert review context: "
        f"{interpretation.get('mission_relevance_context', 'n/a')}",
        "",
        f"Guardrail: {_GUARDRAIL_CARD}",
        "",
    ])


def generate_phase11_report(
    resilience_df: pd.DataFrame,
    mission_relevance_df: pd.DataFrame,
    readiness_report: pd.DataFrame,
    data_provenance_note: str | None = None,
) -> str:
    """Generate the full Phase 11 operational resilience interpretation report."""
    lines: list[str] = []
    lines.append("NeuroBridge-S4 Graph Learning")
    lines.append("Phase 11 — Operational Resilience Interpretation Report")
    lines.append("=" * 70)
    lines.append("")

    # Overview.
    lines.append("Overview")
    lines.append("-" * 70)
    lines.append(
        "This report translates within-subject longitudinal graph trajectories, "
        "attribution, reference-calibrated envelope status, recovery/persistence "
        "information, HRP hazard-context alignment, and adapter-layer data "
        "coverage into adaptive resilience state interpretations for expert "
        "review.")
    lines.append("")
    if data_provenance_note:
        lines.append(f"Data provenance: {data_provenance_note}")
        lines.append("")

    # Input readiness.
    lines.append("Input readiness")
    lines.append("-" * 70)
    if readiness_report is not None and not readiness_report.empty:
        for _, r in readiness_report.iterrows():
            lines.append(
                f"- {r['table_name']} [{r['required_or_optional']}]: {r['status']} "
                f"({int(r['rows'])} rows)")
    else:
        lines.append("- readiness report unavailable")
    lines.append("")

    # State definitions.
    lines.append("Adaptive resilience states")
    lines.append("-" * 70)
    for key, label in RESILIENCE_STATES.items():
        lines.append(f"- {label}: {_STATE_DEFINITIONS.get(key, '')}")
    lines.append("")

    n_interpreted = 0 if resilience_df is None or resilience_df.empty else len(resilience_df)
    lines.append(f"Subject/timepoints interpreted: {n_interpreted}")
    lines.append("")

    if n_interpreted:
        # State distribution.
        lines.append("Distribution of resilience states")
        lines.append("-" * 70)
        dist = resilience_df["resilience_state_label"].value_counts()
        for label, count in dist.items():
            lines.append(f"- {label}: {count}")
        lines.append("")

        # Dominant adaptation modes.
        lines.append("Dominant adaptation modes")
        lines.append("-" * 70)
        modes = resilience_df["dominant_adaptation_mode"].value_counts()
        for label, count in modes.items():
            lines.append(f"- {label}: {count}")
        lines.append("")

        # Example evidence chains.
        lines.append("Example evidence chains")
        lines.append("-" * 70)
        for _, r in resilience_df.head(3).iterrows():
            lines.append(
                f"- {r['subject_id']} @ {r['timepoint']} "
                f"({r['resilience_state_label']}): {r.get('evidence_chain_short', '')}")
        lines.append("")

        # Confidence summary.
        lines.append("Confidence levels")
        lines.append("-" * 70)
        for label, count in resilience_df["confidence_level"].value_counts().items():
            lines.append(f"- {label}: {count}")
        lines.append("")

    # Mission relevance.
    lines.append("Mission-relevant expert review context")
    lines.append("-" * 70)
    if mission_relevance_df is not None and not mission_relevance_df.empty:
        lines.append(
            "Mission-relevance entries describe expert review context only. They "
            "are not operational instructions, risk classifications, treatment "
            "recommendations, or readiness decisions.")
        lines.append(f"- entries generated: {len(mission_relevance_df)}")
    else:
        lines.append("- no mission-relevance entries generated")
    lines.append("")

    # Data gaps.
    lines.append("Data gaps")
    lines.append("-" * 70)
    if n_interpreted and "data_gap_summary" in resilience_df.columns:
        coverage_limited = (
            resilience_df["resilience_state"] == "coverage_limited_interpretation").sum()
        lines.append(
            f"- coverage-limited interpretations: {coverage_limited} of {n_interpreted}")
    lines.append(
        "- interpretation strength depends on available data streams; sparse "
        "coverage should not be overinterpreted.")
    lines.append("")

    # Limitations.
    lines.append("Limitations")
    lines.append("-" * 70)
    for item in (
        "Resilience states are rule-based research interpretations, not validated "
        "operational states.",
        "They are not clinical labels, mission readiness categories, or health "
        "risk levels.",
        "They are sensitive to available data streams and timepoint density.",
        "Coverage-limited outputs should not be overinterpreted.",
    ):
        lines.append(f"- {item}")
    lines.append("")

    # Guardrail.
    lines.append("Guardrail")
    lines.append("-" * 70)
    lines.append(GUARDRAIL)
    lines.append("")

    # Next phase.
    lines.append("Next phase")
    lines.append("-" * 70)
    lines.append(
        "Phase 12 — Self-supervised within-subject temporal graph learning: learn "
        "temporal graph representations to complement the rule-based "
        "interpretation layer.")
    lines.append("")

    return "\n".join(lines)
