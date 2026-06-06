"""Phase 7 — Plain-language trajectory explanation generator.

Turns the transparent attribution tables produced by
:mod:`neurobridge_graph.trajectory_attribution` into reviewer-facing, plain
language. Tone is professional and scientifically careful: no overclaiming, no
diagnosis, no treatment guidance, no causal or exposure claims. Explanations
are written to be understandable by non-programmer reviewers.
"""

from __future__ import annotations

import pandas as pd

from neurobridge_graph.hazard_mapping import HAZARD_DISPLAY_NAMES

_GUARDRAIL_LINE = (
    "Guardrail: This is a monitoring-relevant pattern for expert review. It is "
    "not diagnosis, not treatment guidance, not causal proof, and not exposure "
    "measurement. Hazard-context alignment is not hazard exposure."
)


def _fmt_pct(x: float) -> str:
    return f"{float(x) * 100:.0f}%"


def _top_node_phrase(node_rows: pd.DataFrame, top_k: int = 2) -> tuple[str, float]:
    """Return a phrase naming the top-k domain contributors and their summed share."""
    ranked = node_rows.sort_values("contribution_share", ascending=False).head(top_k)
    ranked = ranked[ranked["contribution_share"] > 0]
    if ranked.empty:
        return "no domain (graph matches personal baseline)", 0.0
    domains = list(ranked["domain"])
    summed = float(ranked["contribution_share"].sum())
    if len(domains) == 1:
        return domains[0], summed
    return " and ".join(domains), summed


def generate_subject_timepoint_explanation(
    subject_id: str,
    timepoint: str,
    node_attr: pd.DataFrame,
    graph_attr: pd.DataFrame,
    subgraph_attr: pd.DataFrame | None = None,
    hazard_attr: pd.DataFrame | None = None,
    recovery_attr: pd.DataFrame | None = None,
) -> str:
    """Generate a reviewer-facing explanation for one subject/timepoint."""
    n_rows = node_attr[
        (node_attr["subject_id"] == subject_id) & (node_attr["timepoint"] == timepoint)
    ]
    if n_rows.empty:
        return (
            f"For subject {subject_id} at timepoint {timepoint}, no node-level "
            f"attribution is available. {_GUARDRAIL_LINE}"
        )

    phase = str(n_rows.iloc[0]["mission_phase"])
    domains_phrase, summed_share = _top_node_phrase(n_rows, top_k=2)

    if summed_share <= 0:
        return (
            f"For subject {subject_id} at the {phase} timepoint {timepoint}, the "
            "biological adaptation graph matches the personal baseline, so there "
            f"is no baseline-relative shift to attribute. {_GUARDRAIL_LINE}"
        )

    parts: list[str] = []
    parts.append(
        f"For subject {subject_id} at the {phase} timepoint {timepoint}, the "
        f"baseline-relative graph shift is driven primarily by {domains_phrase}. "
        f"Together these domains account for {_fmt_pct(summed_share)} of the "
        "observed node-level activation change."
    )

    # Subgraph layer.
    if subgraph_attr is not None and not subgraph_attr.empty:
        s_rows = subgraph_attr[
            (subgraph_attr["subject_id"] == subject_id)
            & (subgraph_attr["timepoint"] == timepoint)
            & (subgraph_attr["n_available_domains"] > 0)
        ]
        if not s_rows.empty:
            top_s = s_rows.sort_values("total_contribution_share", ascending=False).iloc[0]
            parts.append(
                "At the subgraph level, the largest shift maps to the "
                f"{top_s['subgraph_name']} pattern "
                f"({_fmt_pct(top_s['total_contribution_share'])} of node change)."
            )

    # Graph-metric layer.
    if graph_attr is not None and not graph_attr.empty:
        g_rows = graph_attr[
            (graph_attr["subject_id"] == subject_id) & (graph_attr["timepoint"] == timepoint)
        ]
        g_rows = g_rows[g_rows["contribution_share"] > 0]
        if not g_rows.empty:
            top_g = g_rows.sort_values("contribution_share", ascending=False).iloc[0]
            parts.append(
                f"The graph-metric change is led by {top_g['metric']} "
                f"({_fmt_pct(top_g['contribution_share'])})."
            )

    # Hazard-context layer.
    if hazard_attr is not None and not hazard_attr.empty:
        h_rows = hazard_attr[
            (hazard_attr["subject_id"] == subject_id) & (hazard_attr["timepoint"] == timepoint)
        ]
        h_rows = h_rows[h_rows["contribution_share"] > 0]
        if not h_rows.empty:
            top_h = h_rows.sort_values("contribution_share", ascending=False).iloc[0]
            display = HAZARD_DISPLAY_NAMES.get(
                str(top_h["hazard"]), str(top_h["hazard"]).replace("_", " ")
            )
            parts.append(
                f"The hazard-context layer aligns most strongly with {display} "
                "relevance, but this is not evidence of actual hazard exposure or "
                "causality."
            )

    # Recovery layer.
    if recovery_attr is not None and not recovery_attr.empty:
        r_rows = recovery_attr[recovery_attr["subject_id"] == subject_id]
        if not r_rows.empty:
            counts = r_rows["recovery_category"].value_counts()
            persistent = int(counts.get("persistent_shift", 0))
            returned = int(counts.get("returned_near_baseline", 0))
            if persistent > 0:
                parts.append(
                    f"On recovery, {persistent} metric(s) remained shifted from "
                    f"baseline while {returned} returned near baseline."
                )
            elif returned > 0:
                parts.append(
                    f"On recovery, {returned} metric(s) returned near personal baseline."
                )

    parts.append(
        "This attribution identifies a monitoring-relevant pattern for expert "
        "review."
    )
    parts.append(_GUARDRAIL_LINE)
    return " ".join(parts)


def generate_phase7_report(
    attribution_summary: pd.DataFrame,
    node_attr: pd.DataFrame,
    graph_attr: pd.DataFrame,
    subgraph_attr: pd.DataFrame | None = None,
    hazard_attr: pd.DataFrame | None = None,
    recovery_attr: pd.DataFrame | None = None,
    data_provenance_note: str | None = None,
) -> str:
    """Generate the full Phase 7 report as plain text."""
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("NeuroBridge-S4 Graph Learning")
    lines.append("Phase 7 — Explainable Within-Subject Trajectory Attribution")
    lines.append("=" * 78)
    lines.append("")

    # Overview.
    lines.append("OVERVIEW")
    lines.append("-" * 78)
    lines.append(
        "Phase 6 measured how each subject's biological adaptation graph changed "
        "from personal baseline over time. Phase 7 explains which biological "
        "domains, subgraphs, graph metrics, HRP hazard contexts, and recovery "
        "components contributed most to those baseline-relative changes. "
        "Attribution is transparent arithmetic (absolute delta / total absolute "
        "delta), not a black-box model."
    )
    lines.append("")

    # Provenance / input status.
    lines.append("INPUT DATA STATUS")
    lines.append("-" * 78)
    if data_provenance_note:
        lines.append(data_provenance_note)
    layers = {
        "Node-level attribution":     not node_attr.empty,
        "Graph-metric attribution":   graph_attr is not None and not graph_attr.empty,
        "Subgraph attribution":       subgraph_attr is not None and not subgraph_attr.empty,
        "Hazard-context attribution": hazard_attr is not None and not hazard_attr.empty,
        "Recovery attribution":       recovery_attr is not None and not recovery_attr.empty,
    }
    for layer, present in layers.items():
        lines.append(f"  - {layer}: {'available' if present else 'unavailable'}")
    lines.append("")

    # Counts.
    n_subjects = node_attr["subject_id"].nunique() if not node_attr.empty else 0
    n_tp = (
        node_attr[["subject_id", "timepoint"]].drop_duplicates().shape[0]
        if not node_attr.empty else 0
    )
    lines.append("SCOPE")
    lines.append("-" * 78)
    lines.append(f"  Subjects analyzed:           {n_subjects}")
    lines.append(f"  Subject-timepoints analyzed: {n_tp}")
    lines.append("")

    # Strongest domain contributors.
    lines.append("STRONGEST BIOLOGICAL DOMAIN CONTRIBUTORS (across subject-timepoints)")
    lines.append("-" * 78)
    moving = node_attr[node_attr["contribution_share"] > 0] if not node_attr.empty else node_attr
    if not moving.empty:
        dom_rank = (
            moving.groupby("domain")["contribution_share"].mean()
            .sort_values(ascending=False).head(5)
        )
        for dom, share in dom_rank.items():
            lines.append(f"  - {dom}: mean contribution share {_fmt_pct(share)}")
    else:
        lines.append("  No baseline-relative node change observed.")
    lines.append("")

    # Strongest subgraph contributors.
    lines.append("STRONGEST SUBGRAPH CONTRIBUTORS")
    lines.append("-" * 78)
    if subgraph_attr is not None and not subgraph_attr.empty:
        s_moving = subgraph_attr[subgraph_attr["n_available_domains"] > 0]
        if not s_moving.empty:
            s_rank = (
                s_moving.groupby("subgraph_name")["total_contribution_share"].mean()
                .sort_values(ascending=False).head(5)
            )
            for name, share in s_rank.items():
                lines.append(f"  - {name}: mean contribution share {_fmt_pct(share)}")
        else:
            lines.append("  No subgraph domains available in this dataset.")
    else:
        lines.append("  Subgraph attribution unavailable.")
    lines.append("")

    # Strongest hazard-context contributors.
    lines.append("STRONGEST HAZARD-CONTEXT CONTRIBUTORS (alignment, not exposure)")
    lines.append("-" * 78)
    if hazard_attr is not None and not hazard_attr.empty:
        h_moving = hazard_attr[hazard_attr["contribution_share"] > 0]
        if not h_moving.empty:
            h_rank = (
                h_moving.groupby("hazard")["contribution_share"].mean()
                .sort_values(ascending=False).head(5)
            )
            for hz, share in h_rank.items():
                display = HAZARD_DISPLAY_NAMES.get(str(hz), str(hz).replace("_", " "))
                lines.append(f"  - {display}: mean contribution share {_fmt_pct(share)}")
        else:
            lines.append("  No hazard-context shift observed.")
    else:
        lines.append("  Hazard-context attribution unavailable.")
    lines.append("")

    # Recovery patterns.
    lines.append("RECOVERY PATTERNS")
    lines.append("-" * 78)
    if recovery_attr is not None and not recovery_attr.empty:
        cat_counts = recovery_attr["recovery_category"].value_counts()
        for cat, n in cat_counts.items():
            lines.append(f"  - {cat}: {n} metric(s)")
    else:
        lines.append("  Recovery attribution unavailable.")
    lines.append("")

    # Per subject-timepoint interpretations.
    lines.append("PER SUBJECT-TIMEPOINT INTERPRETATION")
    lines.append("-" * 78)
    if not attribution_summary.empty:
        for _, row in attribution_summary.iterrows():
            lines.append(f"  [{row['subject_id']} @ {row['timepoint']}] {row['interpretation']}")
    else:
        lines.append("  No attribution summary available.")
    lines.append("")

    # Limitations.
    lines.append("LIMITATIONS")
    lines.append("-" * 78)
    lines.append("  - Attribution is descriptive arithmetic, not causal inference.")
    lines.append("  - If example data are used, they are schema demonstration only and")
    lines.append("    are not scientific evidence.")
    lines.append("  - Hazard-context attribution is hazard-context alignment, not exposure")
    lines.append("    attribution.")
    lines.append("  - Recovery metrics depend on timepoint availability and baseline quality.")
    lines.append("  - Sparse domains limit subgraph attribution coverage.")
    lines.append("")

    # Why attribution supports expert review.
    lines.append("WHY ATTRIBUTION SUPPORTS EXPERT REVIEW")
    lines.append("-" * 78)
    lines.append(
        "  For small-N astronaut monitoring it is not enough to know that a graph "
        "changed. Reviewers need to know what drove the change and whether it "
        "appears localized, distributed, persistent, or recovery-associated. "
        "Transparent attribution provides that evidence trail."
    )
    lines.append("")

    # Next phase.
    lines.append("NEXT PHASE RECOMMENDATION")
    lines.append("-" * 78)
    lines.append(
        "  Phase 8 — Reference-calibrated trajectory envelope: place each "
        "within-subject trajectory against a reference band to flag deviations "
        "that warrant closer expert review."
    )
    lines.append("")
    lines.append("=" * 78)
    return "\n".join(lines)
