"""Phase 8 — Plain-language reporting for the reference-calibrated envelope.

Generates careful, reviewer-facing interpretations. Outside-envelope means a
baseline-relative graph change is larger than expected under the current
calibration data — **not** disease, danger, or health risk.
"""

from __future__ import annotations

import pandas as pd

from neurobridge_graph.reference_envelope import (
    CORE_ENVELOPE_STATEMENT,
    INSUFFICIENT,
    NEAR,
    OUTSIDE,
    SCHEMA_DEMO_DATA_TYPE,
    WITHIN,
)

_GUARDRAIL = (
    "Outside-envelope does not mean disease, danger, or health risk. It means "
    "the baseline-relative graph change is larger than expected under the "
    "current calibration envelope and may deserve expert review."
)

_LAYER_FEATURE = {"node": "domain", "graph": "metric", "hazard": "hazard"}
_LAYER_DELTA = {
    "node": "delta_activation",
    "graph": "delta_value",
    "hazard": "delta_hazard_relevance",
}


def generate_envelope_interpretation(row: pd.Series, layer: str) -> str:
    """Generate a careful interpretation for one envelope-scored row."""
    feature_col = _LAYER_FEATURE.get(layer, "feature")
    delta_col = _LAYER_DELTA.get(layer, "delta_value")
    feature = row.get(feature_col, "feature")
    position = row.get("envelope_position", INSUFFICIENT)
    delta = row.get(delta_col)
    rz = row.get("robust_z")
    layer_word = {"node": "domain", "graph": "graph metric", "hazard": "hazard-context"}.get(
        layer, "feature")

    delta_txt = f"{float(delta):+.4f}" if pd.notna(delta) else "n/a"
    rz_txt = f"{float(rz):+.2f}" if pd.notna(rz) else "n/a"

    if position == OUTSIDE:
        return (
            f"The {layer_word} delta for {feature} ({delta_txt}, robust z {rz_txt}) "
            "is outside the expected variability envelope. This identifies a "
            f"baseline-relative change that may deserve expert review. {_GUARDRAIL}"
        )
    if position == NEAR:
        return (
            f"The {layer_word} delta for {feature} ({delta_txt}, robust z {rz_txt}) "
            f"is near the expected variability envelope boundary. {_GUARDRAIL}"
        )
    if position == INSUFFICIENT:
        return (
            f"Reference calibration data are insufficient to score the {layer_word} "
            f"delta for {feature}; envelope position is undetermined."
        )
    return (
        f"The {layer_word} delta for {feature} ({delta_txt}) remains within the "
        "expected variability envelope under the current calibration data."
    )


def _provenance(
    envelope_df: pd.DataFrame | None,
    data_provenance_note: str | None,
) -> tuple[str, bool]:
    """Return a provenance sentence and whether example/demo data were used."""
    is_demo = False
    if envelope_df is not None and not envelope_df.empty and "data_type" in envelope_df.columns:
        is_demo = bool((envelope_df["data_type"].astype(str) == SCHEMA_DEMO_DATA_TYPE).any())
    if data_provenance_note:
        return data_provenance_note, is_demo
    if is_demo:
        return (
            "Calibration source: EXAMPLE schema-demonstration envelope only. "
            "This is not scientific evidence and must be replaced with real "
            "analog/reference variability data.", True,
        )
    return "Calibration source: provided reference/analog variability data.", False


def _top_outside_table(scores: pd.DataFrame, feature_col: str, lines: list[str]) -> None:
    if scores is None or scores.empty:
        lines.append("  (no scores available)")
        return
    outside = scores[scores["envelope_position"] == OUTSIDE].copy()
    if outside.empty:
        lines.append("  None outside the expected variability envelope.")
        return
    outside["_rank"] = outside["envelope_exceedance"].abs().fillna(0)
    outside = outside.sort_values("_rank", ascending=False).head(8)
    for _, r in outside.iterrows():
        rz = r.get("robust_z")
        rz_txt = f"{float(rz):+.2f}" if pd.notna(rz) else "n/a"
        lines.append(
            f"  - {r['subject_id']} @ {r['timepoint']}: {r[feature_col]} "
            f"(robust z {rz_txt}, exceedance {float(r['_rank']):.3f})"
        )


def generate_phase8_report(
    phase8_summary: pd.DataFrame,
    node_scores: pd.DataFrame,
    graph_scores: pd.DataFrame,
    hazard_scores: pd.DataFrame | None = None,
    envelope_df: pd.DataFrame | None = None,
    data_provenance_note: str | None = None,
) -> str:
    """Generate the plain-language Phase 8 report."""
    provenance_sentence, is_demo = _provenance(envelope_df, data_provenance_note)
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("NeuroBridge-S4 Graph Learning")
    lines.append("Phase 8 — Reference-Calibrated Trajectory Envelope")
    lines.append("=" * 78)
    lines.append("")

    lines.append("OVERVIEW")
    lines.append("-" * 78)
    lines.append(
        "Phase 8 keeps within-subject self-baseline tracking as the primary "
        "method and adds a reference-calibrated variability envelope. It "
        "estimates whether each subject's own baseline-relative graph change is "
        "small, moderate, or unusually large relative to expected variability."
    )
    lines.append("")
    lines.append("WHAT THE ENVELOPE IS")
    lines.append("-" * 78)
    lines.append("  " + CORE_ENVELOPE_STATEMENT)
    lines.append("")
    lines.append("WHAT THE ENVELOPE IS NOT")
    lines.append("-" * 78)
    lines.append("  - It is not a diagnosis or abnormality label.")
    lines.append("  - It is not a health risk score.")
    lines.append("  - It is not treatment guidance.")
    lines.append("  - It is not an exposure measurement.")
    lines.append("  - It does not define a healthy-vs-unhealthy endpoint.")
    lines.append("")

    lines.append("INPUT / CALIBRATION DATA STATUS")
    lines.append("-" * 78)
    lines.append("  " + provenance_sentence)
    if is_demo:
        lines.append(
            "  NOTE: Example envelope used only to demonstrate the calibration "
            "workflow; results are illustrative, not scientific findings."
        )
    n_features = 0 if envelope_df is None else len(envelope_df)
    lines.append(f"  Envelope features calibrated: {n_features}")
    layers = {
        "Node / domain delta scoring":   node_scores is not None and not node_scores.empty,
        "Graph metric delta scoring":    graph_scores is not None and not graph_scores.empty,
        "Hazard-context delta scoring":  hazard_scores is not None and not hazard_scores.empty,
    }
    for layer, present in layers.items():
        lines.append(f"  - {layer}: {'available' if present else 'unavailable'}")
    lines.append("")

    n_st = 0 if phase8_summary is None or phase8_summary.empty else (
        phase8_summary[["subject_id", "timepoint"]].drop_duplicates().shape[0])
    lines.append("SCOPE")
    lines.append("-" * 78)
    lines.append(f"  Subject-timepoints scored: {n_st}")
    if phase8_summary is not None and not phase8_summary.empty:
        flag_counts = phase8_summary["overall_envelope_flag"].value_counts()
        for flag, n in flag_counts.items():
            lines.append(f"  - {flag}: {n} subject-timepoint(s)")
    lines.append("")

    lines.append("DOMAIN DELTAS OUTSIDE EXPECTED ENVELOPE")
    lines.append("-" * 78)
    _top_outside_table(node_scores, "domain", lines)
    lines.append("")

    lines.append("GRAPH METRIC DELTAS OUTSIDE EXPECTED ENVELOPE")
    lines.append("-" * 78)
    _top_outside_table(graph_scores, "metric", lines)
    lines.append("")

    lines.append("HAZARD-CONTEXT DELTAS OUTSIDE EXPECTED ENVELOPE (alignment, not exposure)")
    lines.append("-" * 78)
    if hazard_scores is not None and not hazard_scores.empty:
        _top_outside_table(hazard_scores, "hazard", lines)
    else:
        lines.append("  Hazard-context envelope scoring unavailable.")
    lines.append("")

    lines.append("PER SUBJECT-TIMEPOINT SUMMARY")
    lines.append("-" * 78)
    if phase8_summary is not None and not phase8_summary.empty:
        for _, r in phase8_summary.iterrows():
            lines.append(
                f"  [{r['subject_id']} @ {r['timepoint']}] flag={r['overall_envelope_flag']} "
                f"(node:{r['n_outside_node_envelope']} graph:{r['n_outside_graph_envelope']} "
                f"hazard:{r['n_outside_hazard_envelope']})"
            )
    else:
        lines.append("  No summary available.")
    lines.append("")

    lines.append("LIMITATIONS")
    lines.append("-" * 78)
    lines.append("  - Envelope quality depends entirely on reference/analog data quality.")
    lines.append("  - Example envelopes are schema demonstration only and are not evidence.")
    lines.append("  - Envelope exceedance is descriptive, not a clinical threshold.")
    lines.append("  - The envelope does not validate clinical or operational thresholds.")
    lines.append("  - Domain coverage limitations affect which deltas can be scored.")
    lines.append("")

    lines.append("HOW THIS SUPPORTS HRP REVIEW")
    lines.append("-" * 78)
    lines.append(
        "  Astronauts may show operationally meaningful within-subject shifts that "
        "still sit inside population-normal ranges. A reference-calibrated envelope "
        "helps reviewers avoid overreacting to expected variability while flagging "
        "unusually large self-baseline shifts for closer expert review — useful when "
        "crews are too small for group inference."
    )
    lines.append("")
    lines.append("  " + _GUARDRAIL)
    lines.append("")

    lines.append("NEXT PHASE RECOMMENDATION")
    lines.append("-" * 78)
    lines.append(
        "  Phase 9 — Interactive longitudinal dashboard: let reviewers explore "
        "trajectories, attribution, and envelope exceedance together."
    )
    lines.append("")
    lines.append("=" * 78)
    return "\n".join(lines)
