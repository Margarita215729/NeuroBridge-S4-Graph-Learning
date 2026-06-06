"""Phase 6 — Within-subject longitudinal graph trajectories.

The primary signal in astronaut monitoring is **within-subject change** from an
individual's own baseline, not one-time comparison to a generic healthy cohort.

Conceptual model::

    participant + timepoint → biological adaptation graph
    participant trajectory → sequence of biological adaptation graphs
    trajectory signal → delta from personal baseline

    G_subject_baseline → G_subject_mission → G_subject_postflight → G_subject_recovery
    DeltaGraph(t) = Graph(t) - Graph(baseline)

Reference cohorts remain useful for scale calibration, noise estimation, rarity
context, and feature geometry — but they are a **secondary calibration layer**,
not the main comparison target.

> The primary signal is within-subject change from the individual's own baseline.
> Population reference data are used only to calibrate scale, estimate noise,
> contextualize rarity, and stabilize feature geometry.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

import networkx as nx
import numpy as np
import pandas as pd

from neurobridge_graph.graph_builder import build_subject_graph
from neurobridge_graph.graph_features import (
    extract_graph_level_features,
    extract_node_level_features,
)
from neurobridge_graph.hazard_mapping import compute_hazard_relevance_scores

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_LONGITUDINAL_COLS: tuple[str, ...] = (
    "subject_id", "timepoint", "mission_phase", "time_index",
)

RECOMMENDED_MISSION_PHASES: tuple[str, ...] = (
    "baseline", "pre_mission", "inflight", "postflight", "recovery",
)

# Metadata columns that are never treated as biological domains.
_METADATA_COLS: frozenset[str] = frozenset({
    "subject_id", "timepoint", "mission_phase", "time_index",
    "data_type", "baseline_timepoint", "baci_score", "baci_category",
})

# Graph-level metrics tracked for baseline-relative deltas.
GRAPH_DELTA_METRICS: tuple[str, ...] = (
    "mean_node_activation",
    "max_node_activation",
    "total_node_activation",
    "n_active_domains",
    "active_domain_fraction",
    "graph_density",
    "coactivation_edge_count",
)

SCHEMA_DEMO_DATA_TYPE = "schema_demonstration_not_scientific_evidence"

PRIMARY_SIGNAL_STATEMENT = (
    "The primary signal is within-subject change from the individual's own "
    "baseline. Population reference data are used only to calibrate scale, "
    "estimate noise, contextualize rarity, and stabilize feature geometry."
)

SELF_BASELINE_STATEMENT = (
    "Healthy population ranges are insufficient for astronaut monitoring "
    "because operationally meaningful changes may occur within "
    "population-normal ranges, while stable individual baselines may differ "
    "from generic norms."
)


# ---------------------------------------------------------------------------
# Column detection and validation
# ---------------------------------------------------------------------------

def detect_longitudinal_columns(df: pd.DataFrame) -> dict[str, Any]:
    """Detect subject_id, timepoint, mission_phase, time_index, and domain columns.

    Returns
    -------
    dict
        Keys: ``required_cols``, ``domain_cols``, ``optional_cols``,
        ``missing_required``, ``has_data_type``.
    """
    cols = list(df.columns)
    required = [c for c in REQUIRED_LONGITUDINAL_COLS if c in cols]
    missing_required = [c for c in REQUIRED_LONGITUDINAL_COLS if c not in cols]
    domain_cols = [
        c for c in cols
        if c not in _METADATA_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]
    optional = [c for c in cols if c in _METADATA_COLS and c not in required]
    return {
        "required_cols":      required,
        "domain_cols":        domain_cols,
        "optional_cols":      optional,
        "missing_required":   missing_required,
        "has_data_type":      "data_type" in cols,
        "n_subjects":         int(df["subject_id"].nunique()) if "subject_id" in cols else 0,
        "n_timepoints":         int(df["timepoint"].nunique()) if "timepoint" in cols else 0,
    }


def validate_longitudinal_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a validation report for a longitudinal input table.

    Returns
    -------
    pandas.DataFrame
        One row per check with columns ``check``, ``status``, ``detail``.
    """
    detected = detect_longitudinal_columns(df)
    rows: list[dict] = []

    for col in REQUIRED_LONGITUDINAL_COLS:
        present = col in df.columns
        rows.append({
            "check":  f"required_column:{col}",
            "status": "ok" if present else "missing",
            "detail": f"column {'present' if present else 'absent'}",
        })

    rows.append({
        "check":  "domain_columns",
        "status": "ok" if detected["domain_cols"] else "warning",
        "detail": f"{len(detected['domain_cols'])} numeric domain columns detected",
    })

    if "subject_id" in df.columns:
        rows.append({
            "check":  "n_subjects",
            "status": "ok",
            "detail": str(df["subject_id"].nunique()),
        })
    if "timepoint" in df.columns:
        rows.append({
            "check":  "n_timepoints",
            "status": "ok",
            "detail": str(df["timepoint"].nunique()),
        })
    if "mission_phase" in df.columns:
        phases = sorted(df["mission_phase"].astype(str).unique())
        rows.append({
            "check":  "mission_phases",
            "status": "ok",
            "detail": ", ".join(phases),
        })
        has_baseline = any(
            str(p).lower() == "baseline" for p in df["mission_phase"].unique()
        )
        rows.append({
            "check":  "baseline_phase_present",
            "status": "ok" if has_baseline else "warning",
            "detail": "baseline phase found" if has_baseline else "no baseline phase; earliest time_index will be used",
        })

    if "data_type" in df.columns:
        dtypes = df["data_type"].astype(str).unique().tolist()
        rows.append({
            "check":  "data_type",
            "status": "ok",
            "detail": ", ".join(dtypes),
        })
    else:
        rows.append({
            "check":  "data_type",
            "status": "info",
            "detail": "no data_type column (assumed real or unspecified data)",
        })

    # Missing values in required columns.
    for col in REQUIRED_LONGITUDINAL_COLS:
        if col in df.columns:
            n_miss = int(df[col].isna().sum())
            rows.append({
                "check":  f"missing_values:{col}",
                "status": "ok" if n_miss == 0 else "warning",
                "detail": f"{n_miss} missing",
            })

    return pd.DataFrame(rows, columns=["check", "status", "detail"])


# ---------------------------------------------------------------------------
# Example longitudinal table (schema demonstration only)
# ---------------------------------------------------------------------------

def create_example_longitudinal_table(output_path: "str | Path | None" = None) -> pd.DataFrame:
    """Create a small example longitudinal table for pipeline testing only.

    This data is **schema demonstration / pipeline test data**. It must not be
    presented as scientific evidence or actual astronaut data.

    Parameters
    ----------
    output_path:
        Optional path to write the CSV (e.g.
        ``data/examples/example_longitudinal_domain_scores.csv``).

    Returns
    -------
    pandas.DataFrame
        Longitudinal table with ``data_type`` column set to
        ``schema_demonstration_not_scientific_evidence``.
    """
    rows: list[dict] = []

    # Two demonstration subjects, five timepoints each.
    demo_specs = [
        ("Demo_Crew_01", [
            ("T0_baseline",  "baseline",    0, 0.85, 0.90, 0.80, 0.75, 0.88, 0.82),
            ("T1_pre",       "pre_mission", 1, 0.88, 0.92, 0.82, 0.78, 0.90, 0.84),
            ("T2_inflight",  "inflight",    2, 1.10, 1.05, 1.20, 1.15, 1.08, 0.95),
            ("T3_post",      "postflight",  3, 1.25, 1.18, 1.35, 1.30, 1.20, 1.05),
            ("T4_recovery",  "recovery",    4, 0.95, 0.98, 0.92, 0.88, 0.95, 0.86),
        ]),
        ("Demo_Crew_02", [
            ("T0_baseline",  "baseline",    0, 0.70, 0.75, 0.68, 0.72, 0.78, 0.74),
            ("T1_pre",       "pre_mission", 1, 0.72, 0.78, 0.70, 0.74, 0.80, 0.76),
            ("T2_inflight",  "inflight",    2, 0.95, 1.10, 0.90, 1.05, 1.00, 0.88),
            ("T3_post",      "postflight",  3, 1.05, 1.20, 0.95, 1.15, 1.10, 0.92),
            ("T4_recovery",  "recovery",    4, 0.78, 0.85, 0.74, 0.80, 0.82, 0.77),
        ]),
    ]
    domain_names = [
        "Cardiovascular regulation",
        "Metabolic regulation",
        "Body composition / physical status",
        "Inflammation / immune-adjacent",
        "Hematologic / oxygen-carrying",
        "Recovery-related markers",
    ]

    for subject_id, timepoints in demo_specs:
        for tp, phase, tidx, *domain_vals in timepoints:
            row: dict[str, Any] = {
                "subject_id":    subject_id,
                "timepoint":     tp,
                "mission_phase": phase,
                "time_index":    tidx,
                "data_type":     SCHEMA_DEMO_DATA_TYPE,
            }
            for dname, val in zip(domain_names, domain_vals):
                row[dname] = val
            rows.append(row)

    df = pd.DataFrame(rows)
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
    return df


# ---------------------------------------------------------------------------
# Baseline identification
# ---------------------------------------------------------------------------

def identify_baseline_timepoint(
    subject_df: pd.DataFrame,
    phase_col: str = "mission_phase",
    time_col: str = "time_index",
) -> str:
    """Identify the baseline timepoint for one subject.

    Prefers ``mission_phase == 'baseline'``. Otherwise uses the earliest
    ``time_index``.
    """
    if subject_df.empty:
        raise ValueError("subject_df is empty")
    if "timepoint" not in subject_df.columns:
        raise ValueError("subject_df must contain a 'timepoint' column")

    if phase_col in subject_df.columns:
        baseline_rows = subject_df[
            subject_df[phase_col].astype(str).str.lower() == "baseline"
        ]
        if not baseline_rows.empty:
            if time_col in baseline_rows.columns:
                return str(baseline_rows.sort_values(time_col).iloc[0]["timepoint"])
            return str(baseline_rows.iloc[0]["timepoint"])

    if time_col in subject_df.columns:
        return str(subject_df.sort_values(time_col).iloc[0]["timepoint"])
    return str(subject_df.iloc[0]["timepoint"])


# ---------------------------------------------------------------------------
# Graph construction per timepoint
# ---------------------------------------------------------------------------

def _domain_row_from_longitudinal_row(
    row: pd.Series,
    domain_cols: list[str],
) -> pd.Series:
    """Extract domain scores as a Series indexed by domain name."""
    return pd.Series({d: row[d] for d in domain_cols if d in row.index})


def sanitize_graph_for_graphml(G: nx.Graph) -> nx.Graph:
    """Return a copy of *G* with GraphML-safe attribute types (no None)."""
    G2 = G.copy()
    for key, val in list(G2.graph.items()):
        if val is None:
            G2.graph[key] = "n/a"
        elif not isinstance(val, (str, int, float, bool)):
            G2.graph[key] = str(val)
    for _, attrs in G2.nodes(data=True):
        for key, val in list(attrs.items()):
            if val is None:
                attrs[key] = "n/a"
            elif not isinstance(val, (str, int, float, bool)):
                attrs[key] = str(val)
    for _, _, attrs in G2.edges(data=True):
        for key, val in list(attrs.items()):
            if val is None:
                attrs[key] = "n/a"
            elif not isinstance(val, (str, int, float, bool)):
                attrs[key] = str(val)
    return G2


def build_timepoint_graphs(
    longitudinal_df: pd.DataFrame,
    graph_builder_func: Callable[..., nx.Graph] | None = None,
    baci_df: pd.DataFrame | None = None,
) -> dict[str, dict[str, nx.Graph]]:
    """Build one biological adaptation graph per subject_id / timepoint.

    Returns
    -------
    dict[str, dict[str, nx.Graph]]
        ``graphs[subject_id][timepoint] = nx.Graph``

    Each graph carries longitudinal metadata in ``G.graph``:
    ``subject_id``, ``timepoint``, ``mission_phase``, ``time_index``,
    ``data_type``, ``baseline_timepoint``.
    """
    builder = graph_builder_func or build_subject_graph
    detected = detect_longitudinal_columns(longitudinal_df)
    if detected["missing_required"]:
        raise ValueError(
            f"Longitudinal table missing required columns: {detected['missing_required']}"
        )
    domain_cols = detected["domain_cols"]
    if not domain_cols:
        raise ValueError("No numeric domain columns detected in longitudinal table.")

    graphs: dict[str, dict[str, nx.Graph]] = {}

    for subject_id, sub_df in longitudinal_df.groupby("subject_id"):
        sid = str(subject_id)
        sub_df = sub_df.sort_values("time_index") if "time_index" in sub_df.columns else sub_df
        baseline_tp = identify_baseline_timepoint(sub_df)
        graphs[sid] = {}

        for _, row in sub_df.iterrows():
            tp = str(row["timepoint"])
            domain_row = _domain_row_from_longitudinal_row(row, domain_cols)
            graph_id = f"{sid}__{tp}"
            G = builder(graph_id, domain_row, baci_df=baci_df)

            G.graph["subject_id"] = sid
            G.graph["timepoint"] = tp
            G.graph["mission_phase"] = str(row.get("mission_phase", "unknown"))
            G.graph["time_index"] = int(row.get("time_index", 0))
            G.graph["baseline_timepoint"] = baseline_tp
            if "data_type" in row.index and pd.notna(row["data_type"]):
                G.graph["data_type"] = str(row["data_type"])
            G.graph["graph_type"] = "longitudinal_biological_adaptation_graph"
            graphs[sid][tp] = G

    return graphs


# ---------------------------------------------------------------------------
# Delta computation
# ---------------------------------------------------------------------------

def compute_node_activation_delta(
    baseline_graph: nx.Graph,
    current_graph: nx.Graph,
) -> pd.DataFrame:
    """Compute node activation changes from baseline to current timepoint.

    Returns
    -------
    pandas.DataFrame
        Columns: ``domain``, ``baseline_activation``, ``current_activation``,
        ``delta_activation``, ``absolute_delta_activation``, ``direction``,
        ``activation_level_current``.
    """
    rows: list[dict] = []
    all_nodes = set(baseline_graph.nodes()) | set(current_graph.nodes())

    for node in sorted(all_nodes):
        b_act = float(baseline_graph.nodes[node].get("activation", 0.0)) if node in baseline_graph else 0.0
        c_act = float(current_graph.nodes[node].get("activation", 0.0)) if node in current_graph else 0.0
        delta = round(c_act - b_act, 5)
        direction = "increase" if delta > 0.01 else ("decrease" if delta < -0.01 else "stable")
        domain = (
            current_graph.nodes[node].get("domain", node)
            if node in current_graph
            else baseline_graph.nodes[node].get("domain", node)
        )
        level = (
            current_graph.nodes[node].get("activation_level", "n/a")
            if node in current_graph else "n/a"
        )
        rows.append({
            "domain":                    domain,
            "baseline_activation":       round(b_act, 5),
            "current_activation":        round(c_act, 5),
            "delta_activation":          delta,
            "absolute_delta_activation": round(abs(delta), 5),
            "direction":                 direction,
            "activation_level_current":  level,
        })
    return pd.DataFrame(rows)


def compute_graph_metric_delta(
    baseline_features: dict,
    current_features: dict,
) -> dict[str, dict[str, float]]:
    """Compute graph-level feature deltas between baseline and current.

    Returns
    -------
    dict
        ``{metric: {baseline_value, current_value, delta_value,
        absolute_delta_value}}`` for each metric in ``GRAPH_DELTA_METRICS``.
    """
    result: dict[str, dict[str, float]] = {}
    for metric in GRAPH_DELTA_METRICS:
        b_val = baseline_features.get(metric, 0.0)
        c_val = current_features.get(metric, 0.0)
        try:
            b_f = float(b_val)
            c_f = float(c_val)
        except (TypeError, ValueError):
            b_f = 0.0
            c_f = 0.0
        delta = round(c_f - b_f, 5)
        result[metric] = {
            "baseline_value":        round(b_f, 5),
            "current_value":         round(c_f, 5),
            "delta_value":           delta,
            "absolute_delta_value":  round(abs(delta), 5),
        }
    return result


def compute_subject_trajectory_table(
    graphs_by_subject: dict[str, dict[str, nx.Graph]],
    graph_feature_func: Callable[[nx.Graph], dict] | None = None,
) -> pd.DataFrame:
    """Compute timepoint-by-timepoint graph trajectory features per subject.

    Returns
    -------
    pandas.DataFrame
        One row per (subject, timepoint) with graph-level features and
        longitudinal metadata.
    """
    feat_fn = graph_feature_func or extract_graph_level_features
    rows: list[dict] = []

    for subject_id, tp_graphs in graphs_by_subject.items():
        for timepoint, G in sorted(
            tp_graphs.items(),
            key=lambda x: x[1].graph.get("time_index", 0),
        ):
            feats = feat_fn(G)
            row = {
                "subject_id":    subject_id,
                "timepoint":     timepoint,
                "mission_phase": G.graph.get("mission_phase", "unknown"),
                "time_index":    G.graph.get("time_index", 0),
                "baseline_timepoint": G.graph.get("baseline_timepoint", "unknown"),
                "data_type":     G.graph.get("data_type", "unspecified"),
            }
            for k, v in feats.items():
                if k != "subject_id":
                    row[k] = v
            rows.append(row)

    return pd.DataFrame(rows)


def compute_longitudinal_delta_tables(
    graphs_by_subject: dict[str, dict[str, nx.Graph]],
    graph_feature_func: Callable[[nx.Graph], dict] | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute node, graph, and summary delta tables for all subjects.

    Returns
    -------
    dict
        Keys: ``node_delta_table``, ``graph_delta_table``,
        ``trajectory_summary``.
    """
    feat_fn = graph_feature_func or extract_graph_level_features
    node_rows: list[dict] = []
    graph_rows: list[dict] = []

    for subject_id, tp_graphs in graphs_by_subject.items():
        if not tp_graphs:
            continue
        # Identify baseline graph.
        baseline_tp = next(
            iter(tp_graphs.values())
        ).graph.get("baseline_timepoint")
        if baseline_tp not in tp_graphs:
            baseline_tp = identify_baseline_timepoint(
                pd.DataFrame([
                    {"timepoint": tp, "mission_phase": G.graph.get("mission_phase"),
                     "time_index": G.graph.get("time_index", 0)}
                    for tp, G in tp_graphs.items()
                ])
            )
        baseline_graph = tp_graphs.get(baseline_tp)
        if baseline_graph is None:
            continue
        baseline_feats = feat_fn(baseline_graph)

        for timepoint, current_graph in tp_graphs.items():
            mission_phase = current_graph.graph.get("mission_phase", "unknown")
            time_index = current_graph.graph.get("time_index", 0)
            data_type = current_graph.graph.get("data_type", "unspecified")

            # Node deltas.
            node_delta = compute_node_activation_delta(baseline_graph, current_graph)
            for _, nd in node_delta.iterrows():
                node_rows.append({
                    "subject_id":    subject_id,
                    "timepoint":     timepoint,
                    "mission_phase": mission_phase,
                    "time_index":    time_index,
                    "data_type":     data_type,
                    **nd.to_dict(),
                })

            # Graph metric deltas.
            current_feats = feat_fn(current_graph)
            metric_deltas = compute_graph_metric_delta(baseline_feats, current_feats)
            for metric, vals in metric_deltas.items():
                graph_rows.append({
                    "subject_id":    subject_id,
                    "timepoint":     timepoint,
                    "mission_phase": mission_phase,
                    "time_index":    time_index,
                    "metric":        metric,
                    "data_type":     data_type,
                    **vals,
                })

    node_delta_table = pd.DataFrame(node_rows) if node_rows else pd.DataFrame()
    graph_delta_table = pd.DataFrame(graph_rows) if graph_rows else pd.DataFrame()

    # Trajectory summary: one row per subject/timepoint with key aggregates.
    summary_rows: list[dict] = []
    if not node_delta_table.empty:
        for (sid, tp), grp in node_delta_table.groupby(["subject_id", "timepoint"]):
            top_domain = grp.loc[grp["absolute_delta_activation"].idxmax(), "domain"]
            max_delta = float(grp["absolute_delta_activation"].max())
            summary_rows.append({
                "subject_id":           sid,
                "timepoint":            tp,
                "mission_phase":        grp.iloc[0]["mission_phase"],
                "top_domain_delta":     top_domain,
                "max_absolute_node_delta": round(max_delta, 5),
                "n_domains_increased":  int((grp["direction"] == "increase").sum()),
                "n_domains_decreased":  int((grp["direction"] == "decrease").sum()),
            })
    trajectory_summary = pd.DataFrame(summary_rows)

    return {
        "node_delta_table":    node_delta_table,
        "graph_delta_table":   graph_delta_table,
        "trajectory_summary":  trajectory_summary,
    }


def compute_hazard_scores_per_timepoint(
    graphs_by_subject: dict[str, dict[str, nx.Graph]],
) -> pd.DataFrame:
    """Compute hazard relevance scores for each subject-timepoint separately.

    Returns a long-form table with ``subject_id``, ``timepoint``, ``mission_phase``,
    and all hazard relevance score columns — suitable for
    :func:`trajectory_features.compute_hazard_context_delta`.
    """
    frames: list[pd.DataFrame] = []
    for subject_id, tp_graphs in graphs_by_subject.items():
        for timepoint, G in tp_graphs.items():
            node_feats = extract_node_level_features(G)
            scores = compute_hazard_relevance_scores(node_feats)
            scores["timepoint"] = timepoint
            scores["mission_phase"] = G.graph.get("mission_phase", "unknown")
            frames.append(scores)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
