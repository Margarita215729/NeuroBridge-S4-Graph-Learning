"""Phase 9 — Data access layer for the longitudinal review dashboard.

This module loads the Phase 6–8 output tables and reshapes them for the
reviewer-facing dashboard. It deliberately has **no Streamlit dependency** so it
can be unit-tested without launching any UI.

All functions tolerate missing tables and missing columns: they return empty
DataFrames or safe dictionaries (with a ``note``) rather than raising, so the
dashboard can degrade gracefully when only part of the pipeline has been run.

The dashboard is a local research-review prototype. It is not a clinical
monitoring system, not diagnosis, not treatment guidance, not exposure
measurement, and not health risk scoring.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Logical table name -> filename. Required tables block the dashboard if absent.
REQUIRED_TABLES: dict[str, str] = {
    "node_deltas":  "longitudinal_node_deltas.csv",
    "graph_deltas": "longitudinal_graph_deltas.csv",
}

OPTIONAL_TABLES: dict[str, str] = {
    # Phase 6
    "hazard_deltas":        "longitudinal_hazard_deltas.csv",
    "trajectory_summary":   "longitudinal_trajectory_summary.csv",
    "recovery_metrics":     "recovery_metrics.csv",
    # Phase 7
    "node_attribution":     "trajectory_node_attribution.csv",
    "graph_attribution":    "trajectory_graph_metric_attribution.csv",
    "subgraph_attribution": "trajectory_subgraph_attribution.csv",
    "hazard_attribution":   "trajectory_hazard_attribution.csv",
    "recovery_attribution": "recovery_attribution.csv",
    "attribution_summary":  "phase7_attribution_summary.csv",
    # Phase 8
    "node_envelope_scores":   "reference_calibrated_node_delta_scores.csv",
    "graph_envelope_scores":  "reference_calibrated_graph_delta_scores.csv",
    "hazard_envelope_scores": "reference_calibrated_hazard_delta_scores.csv",
    "envelope_summary":       "phase8_reference_calibrated_summary.csv",
    "reference_envelope":     "reference_trajectory_envelope.csv",
}

# Tables that carry subject_id + timepoint (used for subject/timepoint discovery).
_SUBJECT_TABLES = [
    "node_deltas", "graph_deltas", "node_attribution", "attribution_summary",
    "node_envelope_scores", "envelope_summary",
]

MISSING_REQUIRED_MESSAGE = (
    "This dashboard requires Phase 6\u20138 outputs. "
    "Please run the corresponding notebooks first."
)


def _empty_note(note: str) -> dict:
    return {"available": False, "note": note}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_dashboard_tables(
    results_dir: "str | Path" = "results/tables",
) -> dict[str, pd.DataFrame]:
    """Load available dashboard input tables.

    Returns a dictionary mapping logical table name to a DataFrame. Tables not
    present on disk are simply omitted; missing optional tables never raise.
    """
    results_dir = Path(results_dir)
    tables: dict[str, pd.DataFrame] = {}
    for key, fname in {**REQUIRED_TABLES, **OPTIONAL_TABLES}.items():
        fpath = results_dir / fname
        if fpath.exists():
            try:
                tables[key] = pd.read_csv(fpath)
            except Exception:  # noqa: BLE001 - corrupt/empty file should not crash UI
                continue

    # Fallback: derive the longitudinal hazard-context delta table when the
    # explicit file is missing but domain deltas + hazard mapping are available.
    if "hazard_deltas" not in tables or tables["hazard_deltas"].empty:
        try:
            from neurobridge_graph.trajectory_features import (
                ensure_longitudinal_hazard_deltas,
            )

            derived = ensure_longitudinal_hazard_deltas(results_dir)
            if derived is not None and not derived.empty:
                tables["hazard_deltas"] = derived
        except Exception:  # noqa: BLE001 - fallback must never break loading
            pass

    return tables


def has_required_tables(tables: dict[str, pd.DataFrame]) -> bool:
    """True when all required (Phase 6 core) tables are present and non-empty."""
    return all(
        key in tables and not tables[key].empty for key in REQUIRED_TABLES
    )


def build_dashboard_readiness_report(
    tables: dict[str, pd.DataFrame],
    results_dir: "str | Path" = "results/tables",
) -> pd.DataFrame:
    """Return a readiness DataFrame describing each input table's status."""
    rows: list[dict] = []
    for key, fname in {**REQUIRED_TABLES, **OPTIONAL_TABLES}.items():
        req = "required" if key in REQUIRED_TABLES else "optional"
        if key in tables:
            df = tables[key]
            rows.append({
                "table_name":          fname,
                "required_or_optional": req,
                "status":              "loaded" if not df.empty else "empty",
                "rows":                len(df),
                "columns":             df.shape[1],
                "notes":               "ok" if not df.empty else "file present but empty",
            })
        else:
            rows.append({
                "table_name":          fname,
                "required_or_optional": req,
                "status":              "missing",
                "rows":                0,
                "columns":             0,
                "notes":               ("blocks dashboard" if req == "required"
                                        else "panel will be limited"),
            })
    return pd.DataFrame(rows, columns=[
        "table_name", "required_or_optional", "status", "rows", "columns", "notes",
    ])


# ---------------------------------------------------------------------------
# Subject / timepoint discovery
# ---------------------------------------------------------------------------

def get_available_subjects(tables: dict[str, pd.DataFrame]) -> list[str]:
    """Return sorted unique subject IDs across available tables."""
    subjects: set[str] = set()
    for key in _SUBJECT_TABLES:
        df = tables.get(key)
        if df is not None and not df.empty and "subject_id" in df.columns:
            subjects.update(df["subject_id"].dropna().astype(str).unique())
    return sorted(subjects)


def _timepoint_sort_key(tp: str):
    """Order timepoints by an embedded index when present (e.g. T2_inflight)."""
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


def get_available_timepoints(
    tables: dict[str, pd.DataFrame],
    subject_id: str,
) -> list[str]:
    """Return sorted timepoints for the selected subject across available tables."""
    timepoints: set[str] = set()
    for key in _SUBJECT_TABLES:
        df = tables.get(key)
        if (df is not None and not df.empty
                and {"subject_id", "timepoint"}.issubset(df.columns)):
            sub = df[df["subject_id"].astype(str) == str(subject_id)]
            timepoints.update(sub["timepoint"].dropna().astype(str).unique())
    return sorted(timepoints, key=_timepoint_sort_key)


def filter_subject_timepoint(
    df: pd.DataFrame,
    subject_id: str,
    timepoint: str,
) -> pd.DataFrame:
    """Filter a DataFrame to one subject/timepoint, tolerating missing columns."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df
    if "subject_id" in out.columns:
        out = out[out["subject_id"].astype(str) == str(subject_id)]
    if "timepoint" in out.columns:
        out = out[out["timepoint"].astype(str) == str(timepoint)]
    return out.reset_index(drop=True)


def _filter_subject(df: pd.DataFrame, subject_id: str) -> pd.DataFrame:
    if df is None or df.empty or "subject_id" not in df.columns:
        return pd.DataFrame() if df is None else df.iloc[0:0]
    return df[df["subject_id"].astype(str) == str(subject_id)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------

def get_subject_timepoint_context(
    tables: dict[str, pd.DataFrame],
    subject_id: str,
    timepoint: str,
) -> dict:
    """Return mission phase, baseline info, and available context for selection."""
    context: dict = {
        "subject_id": subject_id,
        "timepoint": timepoint,
        "mission_phase": "unknown",
        "time_index": None,
        "data_type": "unknown",
        "available": False,
        "note": "",
    }
    nd = filter_subject_timepoint(tables.get("node_deltas", pd.DataFrame()),
                                  subject_id, timepoint)
    if nd.empty:
        context["note"] = "No node delta data for this subject/timepoint."
        return context

    context["available"] = True
    if "mission_phase" in nd.columns:
        context["mission_phase"] = str(nd.iloc[0]["mission_phase"])
    if "time_index" in nd.columns:
        context["time_index"] = nd.iloc[0]["time_index"]
    if "data_type" in nd.columns:
        context["data_type"] = str(nd.iloc[0]["data_type"])

    # Trajectory summary adds high-level descriptors when available.
    ts = filter_subject_timepoint(tables.get("trajectory_summary", pd.DataFrame()),
                                  subject_id, timepoint)
    if not ts.empty:
        row = ts.iloc[0]
        for col in ("dominant_domain", "dominant_domain_direction",
                    "n_domains_increased", "n_domains_decreased",
                    "max_absolute_node_delta"):
            if col in ts.columns:
                context[col] = row[col]
    return context


# ---------------------------------------------------------------------------
# Trajectory panels (per subject, across timepoints)
# ---------------------------------------------------------------------------

def get_domain_delta_panel_data(
    tables: dict[str, pd.DataFrame],
    subject_id: str,
) -> pd.DataFrame:
    """Return domain delta trajectory for the subject across timepoints."""
    df = _filter_subject(tables.get("node_deltas", pd.DataFrame()), subject_id)
    if df.empty:
        return pd.DataFrame()
    keep = [c for c in ("subject_id", "timepoint", "mission_phase", "time_index",
                        "domain", "baseline_activation", "current_activation",
                        "delta_activation", "absolute_delta_activation", "direction")
            if c in df.columns]
    out = df[keep].copy()
    if {"time_index", "timepoint"}.issubset(out.columns):
        out = out.sort_values(["time_index", "domain"])
    return out.reset_index(drop=True)


def get_graph_metric_panel_data(
    tables: dict[str, pd.DataFrame],
    subject_id: str,
) -> pd.DataFrame:
    """Return graph metric trajectory for the subject across timepoints."""
    df = _filter_subject(tables.get("graph_deltas", pd.DataFrame()), subject_id)
    if df.empty:
        return pd.DataFrame()
    keep = [c for c in ("subject_id", "timepoint", "mission_phase", "time_index",
                        "metric", "baseline_value", "current_value", "delta_value",
                        "absolute_delta_value")
            if c in df.columns]
    out = df[keep].copy()
    if "time_index" in out.columns:
        out = out.sort_values(["time_index", "metric"])
    return out.reset_index(drop=True)


def get_hazard_context_panel_data(
    tables: dict[str, pd.DataFrame],
    subject_id: str,
) -> pd.DataFrame:
    """Return hazard-context delta trajectory for the subject across timepoints."""
    df = _filter_subject(tables.get("hazard_deltas", pd.DataFrame()), subject_id)
    if df.empty:
        return pd.DataFrame()
    keep = [c for c in ("subject_id", "timepoint", "mission_phase", "hazard",
                        "baseline_hazard_relevance", "current_hazard_relevance",
                        "delta_hazard_relevance", "coverage_fraction",
                        "top_contributing_domain")
            if c in df.columns]
    return df[keep].reset_index(drop=True)


def get_recovery_panel_data(
    tables: dict[str, pd.DataFrame],
    subject_id: str,
) -> pd.DataFrame:
    """Return recovery metrics joined with recovery attribution for the subject."""
    metrics = _filter_subject(tables.get("recovery_metrics", pd.DataFrame()), subject_id)
    attr = _filter_subject(tables.get("recovery_attribution", pd.DataFrame()), subject_id)

    if metrics.empty and attr.empty:
        return pd.DataFrame()
    if metrics.empty:
        return attr.reset_index(drop=True)
    if attr.empty:
        return metrics.reset_index(drop=True)

    # Merge category from attribution onto metric rows.
    cat_cols = [c for c in ("metric", "recovery_category") if c in attr.columns]
    if "metric" in metrics.columns and "metric" in cat_cols:
        merged = metrics.merge(
            attr[cat_cols].drop_duplicates("metric"), on="metric", how="left",
        )
        return merged.reset_index(drop=True)
    return metrics.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Attribution / envelope panels (per subject + timepoint)
# ---------------------------------------------------------------------------

def get_attribution_panel_data(
    tables: dict[str, pd.DataFrame],
    subject_id: str,
    timepoint: str,
) -> dict:
    """Return top contributors for the selected subject/timepoint.

    Always returns a dict; missing optional tables yield ``available: False``
    with a ``note`` and empty sub-tables rather than raising.
    """
    out: dict = {
        "available": False,
        "note": "",
        "summary": {},
        "top_domains": pd.DataFrame(),
        "top_graph_metrics": pd.DataFrame(),
        "top_subgraphs": pd.DataFrame(),
        "top_hazards": pd.DataFrame(),
        "explanation": "",
    }

    summary = filter_subject_timepoint(
        tables.get("attribution_summary", pd.DataFrame()), subject_id, timepoint)
    if not summary.empty:
        out["available"] = True
        out["summary"] = summary.iloc[0].to_dict()
        if "interpretation" in summary.columns:
            out["explanation"] = str(summary.iloc[0]["interpretation"])

    nd = filter_subject_timepoint(
        tables.get("node_attribution", pd.DataFrame()), subject_id, timepoint)
    if not nd.empty:
        out["available"] = True
        cols = [c for c in ("domain", "delta_activation", "contribution_share",
                            "direction", "attribution_rank") if c in nd.columns]
        nd = nd[nd.get("contribution_share", 1) > 0] if "contribution_share" in nd.columns else nd
        out["top_domains"] = (nd.sort_values("contribution_share", ascending=False)[cols].head(8)
                              if "contribution_share" in nd.columns else nd[cols].head(8))

    gm = filter_subject_timepoint(
        tables.get("graph_attribution", pd.DataFrame()), subject_id, timepoint)
    if not gm.empty:
        out["available"] = True
        cols = [c for c in ("metric", "delta_value", "contribution_share",
                            "direction", "attribution_rank") if c in gm.columns]
        gm = gm[gm.get("contribution_share", 1) > 0] if "contribution_share" in gm.columns else gm
        out["top_graph_metrics"] = (gm.sort_values("contribution_share", ascending=False)[cols].head(8)
                                    if "contribution_share" in gm.columns else gm[cols].head(8))

    sg = filter_subject_timepoint(
        tables.get("subgraph_attribution", pd.DataFrame()), subject_id, timepoint)
    if not sg.empty:
        out["available"] = True
        if "n_available_domains" in sg.columns:
            sg = sg[sg["n_available_domains"] > 0]
        cols = [c for c in ("subgraph_name", "total_contribution_share",
                            "dominant_domain", "n_available_domains") if c in sg.columns]
        out["top_subgraphs"] = (sg.sort_values("total_contribution_share", ascending=False)[cols].head(8)
                                if "total_contribution_share" in sg.columns else sg[cols].head(8))

    hz = filter_subject_timepoint(
        tables.get("hazard_attribution", pd.DataFrame()), subject_id, timepoint)
    if not hz.empty:
        out["available"] = True
        cols = [c for c in ("hazard", "delta_hazard_relevance", "contribution_share",
                            "attribution_rank") if c in hz.columns]
        hz = hz[hz.get("contribution_share", 1) > 0] if "contribution_share" in hz.columns else hz
        out["top_hazards"] = (hz.sort_values("contribution_share", ascending=False)[cols].head(8)
                              if "contribution_share" in hz.columns else hz[cols].head(8))

    if not out["available"]:
        out["note"] = "No Phase 7 attribution tables available for this selection."
    return out


def get_envelope_panel_data(
    tables: dict[str, pd.DataFrame],
    subject_id: str,
    timepoint: str,
) -> dict:
    """Return reference-calibrated envelope status for the selection.

    Always returns a dict; missing Phase 8 tables yield ``available: False``.
    """
    out: dict = {
        "available": False,
        "note": "",
        "overall_flag": "n/a",
        "summary": {},
        "node_scores": pd.DataFrame(),
        "graph_scores": pd.DataFrame(),
        "hazard_scores": pd.DataFrame(),
        "n_outside_node": 0,
        "n_outside_graph": 0,
        "n_outside_hazard": 0,
    }

    summary = filter_subject_timepoint(
        tables.get("envelope_summary", pd.DataFrame()), subject_id, timepoint)
    if not summary.empty:
        out["available"] = True
        row = summary.iloc[0].to_dict()
        out["summary"] = row
        out["overall_flag"] = str(row.get("overall_envelope_flag", "n/a"))
        out["n_outside_node"] = int(row.get("n_outside_node_envelope", 0) or 0)
        out["n_outside_graph"] = int(row.get("n_outside_graph_envelope", 0) or 0)
        out["n_outside_hazard"] = int(row.get("n_outside_hazard_envelope", 0) or 0)

    for tkey, okey in (("node_envelope_scores", "node_scores"),
                       ("graph_envelope_scores", "graph_scores"),
                       ("hazard_envelope_scores", "hazard_scores")):
        sc = filter_subject_timepoint(tables.get(tkey, pd.DataFrame()),
                                      subject_id, timepoint)
        if not sc.empty:
            out["available"] = True
            out[okey] = sc

    if not out["available"]:
        out["note"] = "No Phase 8 reference-envelope tables available for this selection."
    return out


# ---------------------------------------------------------------------------
# Phase 11 — operational resilience interpretation
# ---------------------------------------------------------------------------

PHASE11_MISSING_MESSAGE = (
    "Operational resilience interpretation is unavailable. "
    "Run the Phase 11 notebook first."
)


def load_resilience_tables(
    results_dir: "str | Path" = "results/tables",
) -> dict[str, pd.DataFrame]:
    """Load Phase 11 resilience outputs if present. Never raises."""
    results_dir = Path(results_dir)
    out: dict[str, pd.DataFrame] = {}
    for key, fname in (("resilience_state", "resilience_state_table.csv"),
                       ("mission_relevance", "mission_relevance_translation.csv"),
                       ("evidence_chains", "resilience_evidence_chains.csv")):
        fpath = results_dir / fname
        if fpath.exists():
            try:
                out[key] = pd.read_csv(fpath)
            except Exception:  # noqa: BLE001 - corrupt file should not crash UI
                continue
    return out


def get_resilience_panel_data(
    resilience_tables: dict[str, pd.DataFrame],
    subject_id: str,
    timepoint: str,
) -> dict:
    """Return the Phase 11 resilience interpretation for one subject/timepoint."""
    state = resilience_tables.get("resilience_state", pd.DataFrame())
    if state is None or state.empty:
        return _empty_note(PHASE11_MISSING_MESSAGE)

    row = filter_subject_timepoint(state, subject_id, timepoint)
    if row.empty:
        return _empty_note(
            "No operational resilience interpretation for this subject/timepoint.")

    r = row.iloc[0].to_dict()
    out: dict = {"available": True, "state_row": r}

    # Evidence chain bullets (ordered) for this subject/timepoint.
    ev = resilience_tables.get("evidence_chains", pd.DataFrame())
    ev = filter_subject_timepoint(ev, subject_id, timepoint)
    if not ev.empty and "evidence" in ev.columns:
        if "evidence_order" in ev.columns:
            ev = ev.sort_values("evidence_order")
        out["evidence_chain"] = ev["evidence"].astype(str).tolist()
    else:
        short = str(r.get("evidence_chain_short", ""))
        out["evidence_chain"] = [s.strip() for s in short.split("|") if s.strip()]

    # Mission-relevance review context.
    mr = resilience_tables.get("mission_relevance", pd.DataFrame())
    mr = filter_subject_timepoint(mr, subject_id, timepoint)
    out["mission_relevance"] = mr.iloc[0].to_dict() if not mr.empty else {}
    return out
