"""NeuroBridge-S4 — Phase 9 longitudinal review dashboard (Streamlit app).

Run with::

    pip install -r requirements.txt
    streamlit run app.py

This is a local research-review prototype. It is not a clinical monitoring
system, not diagnosis, not treatment guidance, not exposure measurement, and
not health risk scoring.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the src package importable when launched from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st

from neurobridge_graph.dashboard_data import (
    build_dashboard_readiness_report,
    get_attribution_panel_data,
    get_available_subjects,
    get_available_timepoints,
    get_domain_delta_panel_data,
    get_envelope_panel_data,
    get_graph_metric_panel_data,
    get_hazard_context_panel_data,
    get_recovery_panel_data,
    get_subject_timepoint_context,
    has_required_tables,
    load_dashboard_tables,
    MISSING_REQUIRED_MESSAGE,
)
from neurobridge_graph.dashboard_text import (
    DASHBOARD_SUBTITLE,
    DASHBOARD_TITLE,
    get_dashboard_intro_text,
    get_guardrail_text,
)
from neurobridge_graph import dashboard_components as dc

st.set_page_config(
    page_title="NeuroBridge-S4 Longitudinal Review Dashboard",
    layout="wide",
)

RESULTS_TABLES = _REPO_ROOT / "results" / "tables"


@st.cache_data(show_spinner=False)
def _load(results_dir: str) -> dict:
    return load_dashboard_tables(results_dir)


def main() -> None:
    st.title(DASHBOARD_TITLE)
    st.caption(DASHBOARD_SUBTITLE)
    dc.render_guardrail_box()

    tables = _load(str(RESULTS_TABLES))
    readiness = build_dashboard_readiness_report(tables, RESULTS_TABLES)

    if not has_required_tables(tables):
        st.error(MISSING_REQUIRED_MESSAGE)
        st.markdown(
            "Run the Phase 6\u20138 notebooks first:\n\n"
            "- `notebooks/04_Within_Subject_Longitudinal_Graph_Trajectories.ipynb`\n"
            "- `notebooks/05_Explainable_Trajectory_Attribution.ipynb`\n"
            "- `notebooks/06_Reference_Calibrated_Trajectory_Envelope.ipynb`"
        )
        with st.expander("Input readiness report", expanded=True):
            dc.render_readiness_panel(readiness)
        return

    subjects = get_available_subjects(tables)
    if not subjects:
        st.error(MISSING_REQUIRED_MESSAGE)
        dc.render_readiness_panel(readiness)
        return

    # ---- Sidebar -------------------------------------------------------
    st.sidebar.header("Review controls")
    subject_id = st.sidebar.selectbox("Subject", subjects)
    timepoints = get_available_timepoints(tables, subject_id)
    if not timepoints:
        st.warning("No timepoints found for the selected subject.")
        return
    timepoint = st.sidebar.selectbox("Timepoint", timepoints)

    st.sidebar.divider()
    show_raw = st.sidebar.checkbox("Show raw tables", value=False)
    show_notes = st.sidebar.checkbox("Show method notes", value=False)
    show_readiness = st.sidebar.checkbox("Show input readiness report", value=False)

    if show_notes:
        st.sidebar.divider()
        st.sidebar.caption(get_dashboard_intro_text())

    # ---- Data for current selection ------------------------------------
    context = get_subject_timepoint_context(tables, subject_id, timepoint)
    attribution_data = get_attribution_panel_data(tables, subject_id, timepoint)
    envelope_data = get_envelope_panel_data(tables, subject_id, timepoint)
    domain_panel = get_domain_delta_panel_data(tables, subject_id)
    graph_panel = get_graph_metric_panel_data(tables, subject_id)
    hazard_panel = get_hazard_context_panel_data(tables, subject_id)
    recovery_panel = get_recovery_panel_data(tables, subject_id)

    tabs = st.tabs([
        "Overview", "Domain trajectory", "Graph metrics", "HRP hazard context",
        "Attribution", "Reference envelope", "Recovery", "Data & limitations",
    ])

    with tabs[0]:
        st.subheader(f"Overview — {subject_id} @ {timepoint}")
        dc.render_subject_overview(context, envelope_data, attribution_data)

    with tabs[1]:
        st.subheader("Domain trajectory (baseline-relative)")
        dc.render_domain_delta_chart(domain_panel)

    with tabs[2]:
        st.subheader("Graph metric trajectory (baseline-relative)")
        dc.render_graph_metric_chart(graph_panel)

    with tabs[3]:
        st.subheader("HRP hazard-context shifts")
        dc.render_hazard_context_chart(hazard_panel)

    with tabs[4]:
        st.subheader(f"Attribution — {subject_id} @ {timepoint}")
        dc.render_attribution_panel(attribution_data)

    with tabs[5]:
        st.subheader(f"Reference-calibrated envelope — {subject_id} @ {timepoint}")
        dc.render_envelope_panel(envelope_data)

    with tabs[6]:
        st.subheader("Recovery behavior")
        dc.render_recovery_panel(recovery_panel)

    with tabs[7]:
        st.subheader("Data & limitations")
        st.markdown(f"**Guardrail.** {get_guardrail_text()}")
        data_type = str(context.get("data_type", "unknown"))
        if "schema_demonstration" in data_type:
            st.warning(
                "Data provenance: schema demonstration only — not scientific "
                "evidence. Replace with real longitudinal/analog data before "
                "drawing conclusions."
            )
        st.markdown(
            "**Limitations.**\n"
            "- The dashboard depends on Phase 6\u20138 output tables; missing "
            "tables reduce available panels.\n"
            "- This is a local prototype, not real-time monitoring.\n"
            "- It is not validated for operational decisions and provides no "
            "clinical interpretation.\n"
            "- Example data, if used, are not scientific evidence."
        )
        dc.render_readiness_panel(readiness)

    if show_readiness:
        st.divider()
        st.subheader("Input readiness report")
        dc.render_readiness_panel(readiness)

    if show_raw:
        st.divider()
        st.subheader("Raw tables for current selection")
        dc.render_data_table_section(
            "Domain deltas (subject)", domain_panel,
            "Per-domain activation change from personal baseline across timepoints.")
        dc.render_data_table_section(
            "Graph metric deltas (subject)", graph_panel,
            "Graph-level metric change from personal baseline across timepoints.")
        dc.render_data_table_section(
            "Hazard-context deltas (subject)", hazard_panel,
            "Hazard-context relevance change (alignment, not exposure).")


if __name__ == "__main__":
    main()
