"""Phase 11 — Operational resilience interpretation orchestration.

Loads Phase 6-10 outputs, builds per subject/timepoint adaptive resilience
interpretations, a resilience state table, and a mission-relevance review
context table.

Operational resilience interpretation is a research-review layer. It describes
baseline-relative adaptation patterns for expert interpretation. It is not
diagnosis, treatment guidance, health risk scoring, exposure measurement, or an
operational medical decision.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from neurobridge_graph.resilience_rules import (
    GUARDRAIL,
    RESILIENCE_STATES,
    classify_dominant_adaptation_mode,
    classify_resilience_state,
    derive_resilience_evidence,
    build_evidence_chain,
    evaluate_coverage_limitations,
)

# Tables Phase 11 consumes. ``required`` marks the minimum set that must be
# present (from Phases 6-8) for meaningful interpretation.
_INPUT_SPEC: list[tuple[str, str]] = [
    # filename (without .csv), required|optional
    ("longitudinal_node_deltas", "optional"),
    ("longitudinal_graph_deltas", "optional"),
    ("longitudinal_hazard_deltas", "optional"),
    ("recovery_metrics", "optional"),
    ("trajectory_node_attribution", "required"),
    ("trajectory_graph_metric_attribution", "optional"),
    ("trajectory_subgraph_attribution", "required"),
    ("trajectory_hazard_attribution", "optional"),
    ("recovery_attribution", "optional"),
    ("phase7_attribution_summary", "optional"),
    ("reference_calibrated_node_delta_scores", "optional"),
    ("reference_calibrated_graph_delta_scores", "optional"),
    ("reference_calibrated_hazard_delta_scores", "optional"),
    ("phase8_reference_calibrated_summary", "required"),
    ("domain_coverage_report", "optional"),
    ("variable_domain_mapping_report", "optional"),
    ("adapter_input_readiness_report", "optional"),
    ("adapter_generated_longitudinal_domain_scores", "optional"),
]

_CORE_MISSING_MESSAGE = (
    "Phase 11 requires Phase 6-8 outputs. Please run the longitudinal "
    "trajectory, attribution, and reference-envelope notebooks first."
)


def load_phase11_inputs(results_dir: str | Path = "results/tables") -> dict[str, pd.DataFrame]:
    """Load available Phase 6-10 tables. Missing optional tables do not fail."""
    results_dir = Path(results_dir)
    tables: dict[str, pd.DataFrame] = {}
    for name, _req in _INPUT_SPEC:
        path = results_dir / f"{name}.csv"
        if path.exists():
            try:
                tables[name] = pd.read_csv(path)
            except Exception:  # noqa: BLE001 - corrupt file treated as absent
                tables[name] = pd.DataFrame()
    return tables


def build_phase11_input_readiness_report(
    tables: dict[str, pd.DataFrame],
    results_dir: str | Path = "results/tables",
) -> pd.DataFrame:
    """Report presence/shape of each Phase 11 input table."""
    rows: list[dict] = []
    for name, req in _INPUT_SPEC:
        df = tables.get(name)
        present = isinstance(df, pd.DataFrame) and not df.empty
        if present:
            status = "available"
            note = "loaded"
        elif name in tables:
            status = "empty"
            note = "file present but empty"
        else:
            status = "missing"
            note = ("required core input missing" if req == "required"
                    else "optional input not present")
        rows.append({
            "table_name": name,
            "required_or_optional": req,
            "status": status,
            "rows": int(len(df)) if present else 0,
            "columns": int(df.shape[1]) if present else 0,
            "notes": note,
        })
    return pd.DataFrame(rows, columns=[
        "table_name", "required_or_optional", "status", "rows", "columns", "notes"])


def core_inputs_available(tables: dict[str, pd.DataFrame]) -> bool:
    """True when the required Phase 6-8 core tables are present and non-empty."""
    for name, req in _INPUT_SPEC:
        if req != "required":
            continue
        df = tables.get(name)
        if not (isinstance(df, pd.DataFrame) and not df.empty):
            return False
    return True


def get_subject_timepoint_pairs(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return subject_id/timepoint/mission_phase rows available for interpretation."""
    frames = []
    for name in ("phase8_reference_calibrated_summary", "trajectory_node_attribution",
                 "phase7_attribution_summary", "longitudinal_node_deltas"):
        df = tables.get(name)
        if isinstance(df, pd.DataFrame) and not df.empty and {
                "subject_id", "timepoint"}.issubset(df.columns):
            cols = ["subject_id", "timepoint"]
            if "mission_phase" in df.columns:
                cols.append("mission_phase")
            frames.append(df[cols].copy())
    if not frames:
        return pd.DataFrame(columns=["subject_id", "timepoint", "mission_phase"])
    combined = pd.concat(frames, ignore_index=True)
    if "mission_phase" not in combined.columns:
        combined["mission_phase"] = "unknown"
    combined = (combined
                .dropna(subset=["subject_id", "timepoint"])
                .drop_duplicates(subset=["subject_id", "timepoint"])
                .sort_values(["subject_id", "timepoint"])
                .reset_index(drop=True))
    return combined[["subject_id", "timepoint", "mission_phase"]]


def _hazard_alignment_text(evidence: dict) -> str:
    hz = evidence.get("top_hazard_context", "n/a")
    if hz in ("n/a", None):
        return ("No HRP hazard-context alignment available for this selection; "
                "hazard-context tables were not provided.")
    return (
        f"Leading hazard-context alignment is {str(hz).replace('_', ' ')} "
        f"(alignment share {evidence.get('top_hazard_share', 0.0):.2f}). This is "
        "hazard-context alignment for expert review only, not exposure "
        "measurement or causal attribution.")


def _recovery_persistence_text(evidence: dict) -> str:
    cats = evidence.get("recovery_categories", [])
    if not cats:
        return ("Recovery/persistence not assessed (recovery attribution not "
                "available for this subject).")
    parts = [f"recovery categories observed: {', '.join(cats)}"]
    frac = evidence.get("min_recovery_fraction")
    if isinstance(frac, float) and frac == frac:  # not NaN
        parts.append(f"minimum recovery fraction {frac:.2f}")
    if evidence.get("persistent"):
        parts.append("at least one feature remains shifted from baseline (persistent displacement)")
    if evidence.get("recovery_lag"):
        parts.append("partial/delayed return toward baseline (recovery lag) in selected features")
    return "; ".join(parts).capitalize() + "."


def _data_gap_text(evidence: dict) -> str:
    cov = evidence.get("coverage", {})
    base = cov.get("coverage_note", "Data coverage not assessed.")
    missing = cov.get("missing_domains", [])
    if missing:
        base += " Domains not covered by the adapter layer: " + ", ".join(missing) + "."
    base += (" Interpretation strength depends on available data streams; "
             "sparse coverage should not be overinterpreted.")
    return base


def _mission_relevance_text(state: str, evidence: dict) -> str:
    cov = evidence.get("coverage", {})
    streams = cov.get("missing_domains", [])
    context = {
        "stable_compensated":
            "Trajectory tracks personal baseline; useful as a within-subject "
            "reference point for subsequent expert review.",
        "localized_adaptive_shift":
            "A localized adaptive shift is visible; expert review may focus on "
            "the dominant domain/subgraph and its supporting data streams.",
        "distributed_adaptive_load":
            "A broader cross-system adaptive load is visible; expert review may "
            "consider multiple biological subgraphs together.",
        "systemic_strain_pattern":
            "A stronger multi-system strain-like pattern is visible and is "
            "flagged for expert review; this is not a risk score or alert.",
        "persistent_displacement":
            "The graph remains displaced from baseline into the post-shift phase; "
            "expert review may consider follow-up data collection.",
        "recovery_lag_pattern":
            "Return toward baseline appears delayed; expert review may consider "
            "additional recovery-phase timepoints.",
        "multi_domain_instability":
            "Cross-domain behaviour appears unstable under current coverage; "
            "expert review may consider data quality and additional timepoints.",
        "coverage_limited_interpretation":
            "Interpretation is limited by data coverage; expert review would "
            "benefit from additional data streams before drawing conclusions.",
    }.get(state, "Expert review context.")
    if streams:
        context += " Adding these data streams would strengthen interpretation: " + ", ".join(streams) + "."
    return context


def interpret_subject_timepoint_resilience(
    subject_id: str,
    timepoint: str,
    tables: dict[str, pd.DataFrame],
    thresholds: dict | None = None,
) -> dict:
    """Return a complete interpretation object for one subject/timepoint."""
    evidence = derive_resilience_evidence(
        subject_id=subject_id,
        timepoint=timepoint,
        node_attr=tables.get("trajectory_node_attribution"),
        graph_attr=tables.get("trajectory_graph_metric_attribution"),
        subgraph_attr=tables.get("trajectory_subgraph_attribution"),
        hazard_attr=tables.get("trajectory_hazard_attribution"),
        recovery_attr=tables.get("recovery_attribution"),
        envelope_summary=tables.get("phase8_reference_calibrated_summary"),
        node_envelope=tables.get("reference_calibrated_node_delta_scores"),
        graph_envelope=tables.get("reference_calibrated_graph_delta_scores"),
        hazard_envelope=tables.get("reference_calibrated_hazard_delta_scores"),
        coverage_report=tables.get("domain_coverage_report"),
        thresholds=thresholds,
    )

    state = classify_resilience_state(evidence, thresholds)
    mode = classify_dominant_adaptation_mode(
        subgraph_attr=tables.get("trajectory_subgraph_attribution"),
        hazard_attr=tables.get("trajectory_hazard_attribution"),
        coverage_report=tables.get("domain_coverage_report"),
        subject_id=subject_id,
        timepoint=timepoint,
        thresholds=thresholds,
    )
    evidence_chain = build_evidence_chain(evidence)

    primary_displacement = _primary_displacement(evidence)

    return {
        "subject_id":   str(subject_id),
        "timepoint":    str(timepoint),
        "mission_phase": evidence.get("mission_phase", "unknown"),
        "resilience_state":       state["resilience_state"],
        "resilience_state_label": state["resilience_state_label"],
        "confidence_level":       state["confidence_level"],
        "rule_triggers":          state["rule_triggers"],
        "dominant_adaptation_mode":       mode["dominant_adaptation_mode"],
        "dominant_adaptation_mode_label": mode["dominant_adaptation_mode_label"],
        "adaptation_modes":               mode["modes"],
        "mode_basis":                     mode["mode_basis"],
        "primary_displacement_pattern":   primary_displacement,
        "evidence_chain":                 evidence_chain,
        "hazard_context_alignment":       _hazard_alignment_text(evidence),
        "recovery_persistence_interpretation": _recovery_persistence_text(evidence),
        "data_gap_interpretation":        _data_gap_text(evidence),
        "mission_relevance_context":      _mission_relevance_text(state["resilience_state"], evidence),
        "interpretation":                 state["interpretation"],
        "evidence":                       evidence,
        "guardrail":                      GUARDRAIL,
    }


def _primary_displacement(evidence: dict) -> str:
    n_sg = evidence.get("n_subgraphs_involved", 0)
    n_nodes = evidence.get("n_node_contributors", 0)
    if n_sg >= 2 and n_nodes >= 3:
        return "distributed across multiple biological subgraphs"
    if evidence.get("top_subgraph_contributor", "n/a") not in ("n/a", None):
        return f"localized to the {evidence['top_subgraph_contributor']} subgraph"
    if evidence.get("top_domain_contributor", "n/a") not in ("n/a", None):
        return f"localized to the {evidence['top_domain_contributor']} domain"
    return "no clear graph displacement under current data coverage"


def build_resilience_state_table(
    tables: dict[str, pd.DataFrame],
    thresholds: dict | None = None,
) -> pd.DataFrame:
    """Build one row per subject/timepoint with resilience interpretation."""
    columns = [
        "subject_id", "timepoint", "mission_phase",
        "resilience_state", "resilience_state_label", "confidence_level",
        "dominant_adaptation_mode", "top_domain_contributor",
        "top_graph_metric_contributor", "top_subgraph_contributor",
        "top_hazard_context_alignment", "outside_envelope_summary",
        "recovery_persistence_summary", "data_gap_summary",
        "evidence_chain_short", "interpretation",
    ]
    pairs = get_subject_timepoint_pairs(tables)
    if pairs.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict] = []
    for _, p in pairs.iterrows():
        interp = interpret_subject_timepoint_resilience(
            p["subject_id"], p["timepoint"], tables, thresholds)
        ev = interp["evidence"]
        chain_short = " | ".join(interp["evidence_chain"][:3])
        rows.append({
            "subject_id":   interp["subject_id"],
            "timepoint":    interp["timepoint"],
            "mission_phase": interp["mission_phase"],
            "resilience_state":       interp["resilience_state"],
            "resilience_state_label": interp["resilience_state_label"],
            "confidence_level":       interp["confidence_level"],
            "dominant_adaptation_mode": interp["dominant_adaptation_mode_label"],
            "top_domain_contributor":   ev.get("top_domain_contributor", "n/a"),
            "top_graph_metric_contributor": ev.get("top_graph_metric_contributor", "n/a"),
            "top_subgraph_contributor": ev.get("top_subgraph_contributor", "n/a"),
            "top_hazard_context_alignment": ev.get("top_hazard_context", "n/a"),
            "outside_envelope_summary": (
                f"{ev.get('total_outside', 0)} outside-envelope features "
                f"(flag: {ev.get('overall_envelope_flag', 'n/a')})"),
            "recovery_persistence_summary": interp["recovery_persistence_interpretation"],
            "data_gap_summary":         ev.get("coverage", {}).get("coverage_note", "not assessed"),
            "evidence_chain_short":     chain_short,
            "interpretation":           interp["interpretation"],
        })
    return pd.DataFrame(rows, columns=columns)


def build_mission_relevance_translation(
    resilience_df: pd.DataFrame,
    tables: dict[str, pd.DataFrame] | None = None,
    thresholds: dict | None = None,
) -> pd.DataFrame:
    """Translate resilience states into mission-relevant expert review context.

    This produces expert review context only — not treatment, risk, readiness,
    or operational recommendations.
    """
    columns = [
        "subject_id", "timepoint", "mission_phase", "resilience_state_label",
        "mission_relevance_context", "expert_review_context",
        "data_streams_that_would_strengthen_interpretation", "guardrail",
    ]
    if resilience_df is None or resilience_df.empty:
        return pd.DataFrame(columns=columns)

    coverage = (tables or {}).get("domain_coverage_report")
    rows: list[dict] = []
    for _, r in resilience_df.iterrows():
        state = str(r.get("resilience_state", ""))
        cov = evaluate_coverage_limitations(
            coverage, r.get("subject_id"), r.get("timepoint"), thresholds)
        missing = cov.get("missing_domains", [])
        streams = (", ".join(missing) if missing
                   else "current coverage adequate; additional recovery-phase timepoints would help")
        review = _expert_review_context(state)
        rows.append({
            "subject_id":   r.get("subject_id"),
            "timepoint":    r.get("timepoint"),
            "mission_phase": r.get("mission_phase", "unknown"),
            "resilience_state_label": r.get("resilience_state_label", ""),
            "mission_relevance_context": _mission_relevance_text(state, {"coverage": cov}),
            "expert_review_context": review,
            "data_streams_that_would_strengthen_interpretation": streams,
            "guardrail": GUARDRAIL,
        })
    return pd.DataFrame(rows, columns=columns)


def _expert_review_context(state: str) -> str:
    return {
        "stable_compensated":
            "Suitable as a within-subject reference; no specific feature flagged for review.",
        "localized_adaptive_shift":
            "Expert review may inspect the dominant domain/subgraph and its source variables.",
        "distributed_adaptive_load":
            "Expert review may consider multiple subgraphs and shared drivers together.",
        "systemic_strain_pattern":
            "Flagged for expert review of the multi-system pattern; interpretation only, not an alert.",
        "persistent_displacement":
            "Expert review may consider whether follow-up timepoints confirm persistence.",
        "recovery_lag_pattern":
            "Expert review may consider additional recovery-phase data before conclusions.",
        "multi_domain_instability":
            "Expert review may assess data quality and timepoint density before interpreting.",
        "coverage_limited_interpretation":
            "Expert review should treat this as provisional pending more complete data coverage.",
    }.get(state, "Expert review context.")
