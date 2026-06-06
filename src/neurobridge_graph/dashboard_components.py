"""Phase 9 — Streamlit rendering components for the review dashboard.

These helpers wrap Streamlit primitives so ``app.py`` stays readable. They are
display-only: all data shaping happens in :mod:`dashboard_data`. Streamlit is
imported at module load, so this module is not imported by the data tests.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from neurobridge_graph.dashboard_text import (
    HEADER_GUARDRAIL_SHORT,
    explain_attribution,
    explain_hazard_context,
    explain_recovery,
    explain_reference_envelope,
    explain_self_baseline,
)

_ENVELOPE_FLAG_STYLE = {
    "within_expected_envelope":  ("Within expected envelope", "🟢"),
    "near_envelope_boundary":    ("Near envelope boundary", "🟡"),
    "outside_expected_envelope": ("Outside expected envelope", "🔴"),
    "insufficient_reference":    ("Insufficient reference", "⚪"),
    "n/a":                       ("Not available", "⚪"),
}


def render_guardrail_box() -> None:
    """Render the always-visible guardrail banner."""
    st.warning(HEADER_GUARDRAIL_SHORT, icon="⚠️")


def render_readiness_panel(readiness_df: pd.DataFrame) -> None:
    """Render the input-table readiness report."""
    if readiness_df is None or readiness_df.empty:
        st.info("No readiness information available.")
        return
    missing_required = readiness_df[
        (readiness_df["required_or_optional"] == "required")
        & (readiness_df["status"] != "loaded")
    ]
    if missing_required.empty:
        st.success("All required Phase 6 core tables are present.")
    else:
        st.error("Some required tables are missing; core panels may be empty.")
    st.dataframe(readiness_df, width="stretch", hide_index=True)


def _metric_row(items: list[tuple[str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.metric(label, value)


def render_subject_overview(
    context: dict,
    envelope_data: dict,
    attribution_data: dict,
) -> None:
    """Render the Overview tab metric cards and summary text."""
    phase = context.get("mission_phase", "unknown")
    _metric_row([
        ("Subject", str(context.get("subject_id", "n/a"))),
        ("Timepoint", str(context.get("timepoint", "n/a"))),
        ("Mission phase", str(phase)),
    ])

    summary = attribution_data.get("summary", {}) or {}
    top_domain = summary.get("top_domain_contributor", "n/a")
    top_subgraph = summary.get("top_subgraph_contributor", "n/a")
    top_hazard = summary.get("top_hazard_context_contributor", "n/a")
    if isinstance(top_hazard, str):
        top_hazard = top_hazard.replace("_", " ")

    _metric_row([
        ("Top domain contributor", str(top_domain)),
        ("Top subgraph contributor", str(top_subgraph)),
        ("Top hazard-context alignment", str(top_hazard)),
    ])

    flag = str(envelope_data.get("overall_flag", "n/a"))
    label, icon = _ENVELOPE_FLAG_STYLE.get(flag, (flag, "⚪"))
    n_out = (envelope_data.get("n_outside_node", 0)
             + envelope_data.get("n_outside_graph", 0)
             + envelope_data.get("n_outside_hazard", 0))
    recovery_summary = summary.get("recovery_summary", "n/a")
    _metric_row([
        ("Reference envelope", f"{icon} {label}"),
        ("Deltas outside envelope", str(n_out)),
        ("Recovery summary", str(recovery_summary)),
    ])

    explanation = attribution_data.get("explanation", "")
    if explanation:
        st.markdown(f"**Interpretation.** {explanation}")
    else:
        st.info("No attribution interpretation available for this selection.")


def render_domain_delta_chart(domain_delta_df: pd.DataFrame) -> None:
    """Line chart of domain activation deltas over time (per domain)."""
    st.caption(explain_self_baseline())
    if domain_delta_df is None or domain_delta_df.empty:
        st.info("No domain delta data available for this subject.")
        return
    if {"timepoint", "domain", "delta_activation"}.issubset(domain_delta_df.columns):
        pivot = domain_delta_df.pivot_table(
            index="timepoint", columns="domain", values="delta_activation", aggfunc="mean")
        pivot = _order_timepoints(pivot)
        st.line_chart(pivot)
    st.markdown("**Domain deltas relative to personal baseline**")
    st.dataframe(domain_delta_df, width="stretch", hide_index=True)


def render_graph_metric_chart(graph_metric_df: pd.DataFrame) -> None:
    """Line charts of selected graph metrics over time."""
    if graph_metric_df is None or graph_metric_df.empty:
        st.info("No graph metric data available for this subject.")
        return
    if {"timepoint", "metric", "delta_value"}.issubset(graph_metric_df.columns):
        pivot = graph_metric_df.pivot_table(
            index="timepoint", columns="metric", values="delta_value", aggfunc="mean")
        pivot = _order_timepoints(pivot)
        st.markdown("**Graph metric deltas relative to personal baseline**")
        st.line_chart(pivot)
    st.dataframe(graph_metric_df, width="stretch", hide_index=True)


def render_hazard_context_chart(hazard_df: pd.DataFrame) -> None:
    """Line chart of hazard-context relevance deltas over time."""
    st.caption(explain_hazard_context())
    if hazard_df is None or hazard_df.empty:
        st.info(
            "Hazard-context trajectory data are unavailable because the required "
            "domain-delta and hazard-mapping inputs were not found. Run Phase 6 "
            "hazard delta generation or provide `longitudinal_hazard_deltas.csv`."
        )
        return
    if {"timepoint", "hazard", "delta_hazard_relevance"}.issubset(hazard_df.columns):
        pivot = hazard_df.pivot_table(
            index="timepoint", columns="hazard", values="delta_hazard_relevance", aggfunc="mean")
        pivot = _order_timepoints(pivot)
        st.line_chart(pivot)
    if "coverage_fraction" in hazard_df.columns:
        st.caption(
            "Coverage fraction indicates how much proxy data support each hazard "
            "context; low coverage means limited proxy data, not absence of relevance."
        )
    st.dataframe(hazard_df, width="stretch", hide_index=True)


def render_attribution_panel(attribution_data: dict) -> None:
    """Render the Attribution tab tables and explanation card."""
    st.caption(explain_attribution())
    if not attribution_data.get("available"):
        st.info(attribution_data.get("note", "No attribution data available."))
        return

    explanation = attribution_data.get("explanation", "")
    if explanation:
        st.markdown(f"**Explanation card.** {explanation}")

    _table_block("Top biological domain contributors", attribution_data.get("top_domains"))
    _table_block("Top graph metric contributors", attribution_data.get("top_graph_metrics"))
    _table_block("Top subgraph contributors", attribution_data.get("top_subgraphs"))
    _table_block("Top hazard-context contributors (alignment, not exposure)",
                 attribution_data.get("top_hazards"))


def render_envelope_panel(envelope_data: dict) -> None:
    """Render the Reference envelope tab."""
    st.caption(explain_reference_envelope())
    if not envelope_data.get("available"):
        st.info(envelope_data.get("note", "No reference-envelope data available."))
        return

    flag = str(envelope_data.get("overall_flag", "n/a"))
    label, icon = _ENVELOPE_FLAG_STYLE.get(flag, (flag, "⚪"))
    _metric_row([
        ("Overall envelope status", f"{icon} {label}"),
        ("Outside (node)", str(envelope_data.get("n_outside_node", 0))),
        ("Outside (graph)", str(envelope_data.get("n_outside_graph", 0))),
        ("Outside (hazard)", str(envelope_data.get("n_outside_hazard", 0))),
    ])
    st.info(
        "Outside-envelope is not diagnosis and not risk. It means the "
        "baseline-relative change is larger than expected under the current "
        "calibration data and may deserve expert review."
    )
    _table_block("Node / domain envelope scores", envelope_data.get("node_scores"))
    _table_block("Graph metric envelope scores", envelope_data.get("graph_scores"))
    _table_block("Hazard-context envelope scores", envelope_data.get("hazard_scores"))


def render_recovery_panel(recovery_df: pd.DataFrame) -> None:
    """Render the Recovery tab."""
    st.caption(explain_recovery())
    if recovery_df is None or recovery_df.empty:
        st.info("No recovery data available for this subject.")
        return
    if "recovery_category" in recovery_df.columns:
        counts = recovery_df["recovery_category"].value_counts()
        st.bar_chart(counts)
    st.dataframe(recovery_df, width="stretch", hide_index=True)


_RESILIENCE_GUARDRAIL = (
    "Operational resilience interpretation is a research-review layer. It is not "
    "diagnosis, treatment guidance, health risk scoring, exposure measurement, or "
    "an operational medical decision."
)


def render_resilience_panel(resilience_data: dict) -> None:
    """Render the Phase 11 Operational resilience tab."""
    st.caption(
        "Adaptive resilience interpretation derived from within-subject graph "
        "trajectories, attribution, reference-calibrated envelope status, "
        "recovery/persistence, HRP hazard-context alignment, and data coverage. "
        "Research-review interpretation only."
    )
    if not resilience_data or not resilience_data.get("available"):
        st.info(resilience_data.get("note", _RESILIENCE_GUARDRAIL)
                if resilience_data else _RESILIENCE_GUARDRAIL)
        return

    r = resilience_data.get("state_row", {})
    _metric_row([
        ("Resilience state", str(r.get("resilience_state_label", "n/a"))),
        ("Dominant adaptation mode", str(r.get("dominant_adaptation_mode", "n/a"))),
        ("Confidence", str(r.get("confidence_level", "n/a"))),
    ])

    st.markdown(f"**Interpretation.** {r.get('interpretation', 'n/a')}")

    chain = resilience_data.get("evidence_chain", [])
    if chain:
        st.markdown("**Evidence chain**")
        for i, bullet in enumerate(chain, start=1):
            st.markdown(f"{i}. {bullet}")

    st.markdown(
        f"**HRP hazard-context alignment.** "
        f"{r.get('top_hazard_context_alignment', 'n/a')}")
    st.markdown(
        f"**Recovery / persistence.** "
        f"{r.get('recovery_persistence_summary', 'n/a')}")
    st.markdown(
        f"**Data coverage and data gaps.** {r.get('data_gap_summary', 'n/a')}")

    mr = resilience_data.get("mission_relevance", {})
    if mr:
        st.markdown(
            f"**Mission-relevant expert review context.** "
            f"{mr.get('mission_relevance_context', 'n/a')}")
        if mr.get("data_streams_that_would_strengthen_interpretation"):
            st.caption(
                "Data streams that would strengthen interpretation: "
                f"{mr['data_streams_that_would_strengthen_interpretation']}")

    st.warning(_RESILIENCE_GUARDRAIL, icon="⚠️")


def render_data_table_section(
    title: str,
    df: pd.DataFrame,
    explanation: str | None = None,
) -> None:
    """Render a titled raw-table section with an optional explanation."""
    st.subheader(title)
    if explanation:
        st.caption(explanation)
    if df is None or df.empty:
        st.info("Table not available.")
        return
    st.dataframe(df, width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _table_block(title: str, df: pd.DataFrame | None) -> None:
    st.markdown(f"**{title}**")
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        st.caption("Not available for this selection.")
        return
    st.dataframe(df, width="stretch", hide_index=True)


def _timepoint_sort_key(tp: str):
    s = str(tp)
    if s and s[0] in ("T", "t") and len(s) > 1 and s[1].isdigit():
        digits = ""
        for ch in s[1:]:
            if ch.isdigit():
                digits += ch
            else:
                break
        if digits:
            return (0, int(digits), s)
    return (1, 0, s)


def _order_timepoints(pivot: pd.DataFrame) -> pd.DataFrame:
    """Order a timepoint-indexed pivot table by embedded timepoint index."""
    try:
        ordered = sorted(pivot.index, key=_timepoint_sort_key)
        return pivot.reindex(ordered)
    except Exception:  # noqa: BLE001
        return pivot
