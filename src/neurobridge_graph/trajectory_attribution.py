"""Phase 7 — Explainable within-subject trajectory attribution.

Phase 6 answered *how* each subject's biological adaptation graph changed from
personal baseline over time. Phase 7 answers *which* biological domains,
subgraphs, graph metrics, HRP hazard contexts, and recovery components explain
that change.

Attribution here is **transparent arithmetic**, not a black-box model::

    contribution_share = absolute_delta / sum(absolute_delta) per subject-timepoint

This lets reviewers see which components drive the baseline-relative graph
shift. Outputs are monitoring-relevant research interpretations and candidates
for expert review — **not** diagnosis, treatment guidance, causal proof, or
exposure measurement. HRP hazard-context attribution is hazard-context
alignment, not exposure attribution.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from neurobridge_graph.hazard_mapping import (
    HAZARD_DISPLAY_NAMES,
    normalize_domain_name,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPSILON = 1e-9

# Default biological subgraph templates (canonical long-form domain names).
DEFAULT_SUBGRAPH_TEMPLATES: dict[str, list[str]] = {
    "cardiometabolic": [
        "cardiovascular regulation",
        "metabolic regulation",
        "body composition / physical status",
    ],
    "immune_metabolic": [
        "inflammation / immune-adjacent status",
        "metabolic regulation",
        "recovery-related markers",
    ],
    "hematologic_cardiovascular": [
        "hematologic / oxygen-carrying capacity",
        "cardiovascular regulation",
        "recovery-related markers",
    ],
    "sleep_autonomic_recovery": [
        "sleep / circadian regulation",
        "autonomic regulation",
        "recovery capacity",
    ],
    "cognitive_emotional_recovery": [
        "cognitive load",
        "emotional regulation",
        "recovery capacity",
        "recovery-related markers",
    ],
}

_PHASE6_FILES: dict[str, str] = {
    "node_deltas":         "longitudinal_node_deltas.csv",
    "graph_deltas":        "longitudinal_graph_deltas.csv",
    "hazard_deltas":       "longitudinal_hazard_deltas.csv",
    "recovery_metrics":    "recovery_metrics.csv",
    "trajectory_summary":  "longitudinal_trajectory_summary.csv",
}
_REQUIRED_PHASE6 = ("node_deltas", "graph_deltas")

_GUARDRAIL = (
    "Monitoring-relevant pattern for expert review. Not diagnosis, not "
    "treatment guidance, not causal proof, and not exposure measurement."
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_phase6_delta_tables(results_dir: "str | Path") -> dict[str, pd.DataFrame]:
    """Load Phase 6 delta tables if available.

    Parameters
    ----------
    results_dir:
        Path to the ``results`` directory containing a ``tables`` subfolder.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Keys: ``node_deltas``, ``graph_deltas``, ``hazard_deltas``,
        ``recovery_metrics``, ``trajectory_summary``. Only tables present on
        disk are included.

    Raises
    ------
    FileNotFoundError
        If a core required table (node or graph deltas) is missing.
    """
    tables_dir = Path(results_dir) / "tables"
    loaded: dict[str, pd.DataFrame] = {}
    for key, fname in _PHASE6_FILES.items():
        fpath = tables_dir / fname
        if fpath.exists():
            loaded[key] = pd.read_csv(fpath)

    missing_required = [k for k in _REQUIRED_PHASE6 if k not in loaded]
    if missing_required:
        raise FileNotFoundError(
            "Phase 7 requires Phase 6 longitudinal delta outputs. "
            "Please run the Phase 6 notebook first. "
            f"Missing: {[_PHASE6_FILES[k] for k in missing_required]}"
        )
    return loaded


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _maybe_phase(df: pd.DataFrame, group: pd.DataFrame) -> str:
    if "mission_phase" in df.columns and not group.empty:
        return str(group.iloc[0]["mission_phase"])
    return "unknown"


def _contribution_shares(abs_deltas: pd.Series) -> pd.Series:
    """Normalize absolute deltas into shares summing to 1 (0 when total is 0)."""
    total = float(abs_deltas.sum())
    if total <= _EPSILON:
        return pd.Series(0.0, index=abs_deltas.index)
    return abs_deltas / total


# ---------------------------------------------------------------------------
# Node-level attribution
# ---------------------------------------------------------------------------

def compute_node_attribution(
    node_delta_df: pd.DataFrame,
    subject_col: str = "subject_id",
    timepoint_col: str = "timepoint",
    domain_col: str = "domain",
    delta_col: str = "delta_activation",
) -> pd.DataFrame:
    """Compute node-level contribution shares per subject/timepoint.

    For each subject-timepoint, ``contribution_share`` is the absolute domain
    delta divided by the sum of absolute domain deltas. Shares sum to 1 when
    any nonzero delta exists, and to 0 when the graph is unchanged (baseline).
    """
    required = {subject_col, timepoint_col, domain_col, delta_col}
    missing = required - set(node_delta_df.columns)
    if missing:
        raise ValueError(f"node_delta_df missing columns: {missing}")

    has_phase = "mission_phase" in node_delta_df.columns
    out_rows: list[dict] = []

    for (sid, tp), grp in node_delta_df.groupby([subject_col, timepoint_col]):
        grp = grp.copy()
        abs_delta = grp[delta_col].abs()
        if "absolute_delta_activation" in grp.columns:
            abs_delta = grp["absolute_delta_activation"]
        shares = _contribution_shares(abs_delta)

        grp = grp.assign(abs_contrib=abs_delta.values, share_contrib=shares.values)
        grp = grp.sort_values("abs_contrib", ascending=False).reset_index(drop=True)

        for rank, (_, row_d) in enumerate(grp.iterrows(), start=1):
            delta_val = float(row_d.get(delta_col, 0.0))
            share = float(row_d["share_contrib"])
            direction = row_d.get("direction")
            if direction is None or (isinstance(direction, float) and np.isnan(direction)):
                direction = "increase" if delta_val > 0 else ("decrease" if delta_val < 0 else "stable")
            domain = row_d[domain_col]
            interp = (
                f"{domain} accounts for {share * 100:.0f}% of the baseline-relative "
                f"node activation change ({direction}). {_GUARDRAIL}"
            )
            out_rows.append({
                "subject_id":               sid,
                "timepoint":                tp,
                "mission_phase":            row_d.get("mission_phase", "unknown") if has_phase else "unknown",
                "domain":                   domain,
                "baseline_activation":      row_d.get("baseline_activation", np.nan),
                "current_activation":       row_d.get("current_activation", np.nan),
                "delta_activation":         round(delta_val, 5),
                "absolute_delta_activation": round(float(row_d["abs_contrib"]), 5),
                "contribution_share":       round(share, 5),
                "direction":                direction,
                "attribution_rank":         rank,
                "interpretation":           interp,
            })

    return pd.DataFrame(out_rows, columns=[
        "subject_id", "timepoint", "mission_phase", "domain",
        "baseline_activation", "current_activation", "delta_activation",
        "absolute_delta_activation", "contribution_share", "direction",
        "attribution_rank", "interpretation",
    ])


# ---------------------------------------------------------------------------
# Graph-metric attribution
# ---------------------------------------------------------------------------

def compute_graph_metric_attribution(
    graph_delta_df: pd.DataFrame,
    subject_col: str = "subject_id",
    timepoint_col: str = "timepoint",
    metric_col: str = "metric",
    delta_col: str = "delta_value",
) -> pd.DataFrame:
    """Compute contribution shares for graph-level metric changes.

    Handles a zero total delta gracefully (all shares 0).
    """
    required = {subject_col, timepoint_col, metric_col, delta_col}
    missing = required - set(graph_delta_df.columns)
    if missing:
        raise ValueError(f"graph_delta_df missing columns: {missing}")

    has_phase = "mission_phase" in graph_delta_df.columns
    out_rows: list[dict] = []

    for (sid, tp), grp in graph_delta_df.groupby([subject_col, timepoint_col]):
        grp = grp.copy()
        if "absolute_delta_value" in grp.columns:
            abs_delta = grp["absolute_delta_value"]
        else:
            abs_delta = grp[delta_col].abs()
        shares = _contribution_shares(abs_delta)
        grp = grp.assign(abs_contrib=abs_delta.values, share_contrib=shares.values)
        grp = grp.sort_values("abs_contrib", ascending=False).reset_index(drop=True)

        for rank, (_, row_d) in enumerate(grp.iterrows(), start=1):
            delta_val = float(row_d.get(delta_col, 0.0))
            share = float(row_d["share_contrib"])
            direction = "increase" if delta_val > 0 else ("decrease" if delta_val < 0 else "stable")
            metric = row_d[metric_col]
            interp = (
                f"{metric} accounts for {share * 100:.0f}% of the graph-metric "
                f"change ({direction}). {_GUARDRAIL}"
            )
            out_rows.append({
                "subject_id":           sid,
                "timepoint":            tp,
                "mission_phase":        row_d.get("mission_phase", "unknown") if has_phase else "unknown",
                "metric":               metric,
                "baseline_value":       row_d.get("baseline_value", np.nan),
                "current_value":        row_d.get("current_value", np.nan),
                "delta_value":          round(delta_val, 5),
                "absolute_delta_value": round(float(row_d["abs_contrib"]), 5),
                "contribution_share":   round(share, 5),
                "direction":            direction,
                "attribution_rank":     rank,
                "interpretation":       interp,
            })

    return pd.DataFrame(out_rows, columns=[
        "subject_id", "timepoint", "mission_phase", "metric",
        "baseline_value", "current_value", "delta_value",
        "absolute_delta_value", "contribution_share", "direction",
        "attribution_rank", "interpretation",
    ])


# ---------------------------------------------------------------------------
# Subgraph attribution
# ---------------------------------------------------------------------------

def compute_subgraph_attribution_from_node_deltas(
    node_attribution_df: pd.DataFrame,
    subgraph_templates: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Aggregate node attribution into biological subgraph attribution.

    Domain names are reconciled to canonical form (so short node-data spellings
    match the long template spellings). Templates with no available domains for
    a subject-timepoint are reported with ``n_available_domains == 0``.
    """
    templates = subgraph_templates or DEFAULT_SUBGRAPH_TEMPLATES
    if node_attribution_df.empty:
        return pd.DataFrame(columns=[
            "subject_id", "timepoint", "mission_phase", "subgraph_name",
            "available_domains", "n_available_domains", "total_contribution_share",
            "mean_delta_activation", "max_absolute_delta_activation",
            "dominant_domain", "interpretation",
        ])

    # Normalized template domain sets.
    norm_templates = {
        name: {normalize_domain_name(d) for d in domains}
        for name, domains in templates.items()
    }

    out_rows: list[dict] = []
    for (sid, tp), grp in node_attribution_df.groupby(["subject_id", "timepoint"]):
        phase = _maybe_phase(node_attribution_df, grp)
        grp = grp.copy()
        grp["_norm_domain"] = grp["domain"].map(normalize_domain_name)

        for name, norm_domains in norm_templates.items():
            matched = grp[grp["_norm_domain"].isin(norm_domains)]
            n_available = int(matched["_norm_domain"].nunique())
            if matched.empty:
                out_rows.append({
                    "subject_id":                sid,
                    "timepoint":                 tp,
                    "mission_phase":             phase,
                    "subgraph_name":             name,
                    "available_domains":         "none",
                    "n_available_domains":       0,
                    "total_contribution_share":  0.0,
                    "mean_delta_activation":     np.nan,
                    "max_absolute_delta_activation": np.nan,
                    "dominant_domain":           "n/a",
                    "interpretation":            (
                        f"No domains for the '{name}' subgraph are available in "
                        "this dataset; subgraph attribution unavailable."
                    ),
                })
                continue

            total_share = float(matched["contribution_share"].sum())
            mean_delta = float(matched["delta_activation"].mean())
            max_abs = float(matched["absolute_delta_activation"].max())
            dom_row = matched.loc[matched["absolute_delta_activation"].idxmax()]
            dominant = dom_row["domain"]
            out_rows.append({
                "subject_id":                sid,
                "timepoint":                 tp,
                "mission_phase":             phase,
                "subgraph_name":             name,
                "available_domains":         "; ".join(sorted(matched["domain"].unique())),
                "n_available_domains":       n_available,
                "total_contribution_share":  round(total_share, 5),
                "mean_delta_activation":     round(mean_delta, 5),
                "max_absolute_delta_activation": round(max_abs, 5),
                "dominant_domain":           dominant,
                "interpretation":            (
                    f"The '{name}' subgraph accounts for {total_share * 100:.0f}% of the "
                    f"baseline-relative node change; '{dominant}' is dominant. {_GUARDRAIL}"
                ),
            })

    return pd.DataFrame(out_rows, columns=[
        "subject_id", "timepoint", "mission_phase", "subgraph_name",
        "available_domains", "n_available_domains", "total_contribution_share",
        "mean_delta_activation", "max_absolute_delta_activation",
        "dominant_domain", "interpretation",
    ])


# ---------------------------------------------------------------------------
# Hazard-context attribution
# ---------------------------------------------------------------------------

def compute_hazard_context_attribution(
    hazard_delta_df: pd.DataFrame,
    subject_col: str = "subject_id",
    timepoint_col: str = "timepoint",
    hazard_col: str = "hazard",
    delta_col: str = "delta_hazard_relevance",
) -> pd.DataFrame:
    """Compute hazard-context contribution shares from hazard relevance deltas.

    Hazard attribution is **hazard-context alignment**, not exposure
    attribution: it shows which HRP hazard contexts the graph shift maps onto.
    """
    if hazard_delta_df is None or hazard_delta_df.empty:
        return pd.DataFrame(columns=[
            "subject_id", "timepoint", "mission_phase", "hazard",
            "baseline_hazard_relevance", "current_hazard_relevance",
            "delta_hazard_relevance", "absolute_delta_hazard_relevance",
            "contribution_share", "coverage_fraction", "attribution_rank",
            "interpretation",
        ])
    required = {subject_col, timepoint_col, hazard_col, delta_col}
    missing = required - set(hazard_delta_df.columns)
    if missing:
        raise ValueError(f"hazard_delta_df missing columns: {missing}")

    has_phase = "mission_phase" in hazard_delta_df.columns
    has_cov = "coverage_fraction" in hazard_delta_df.columns
    out_rows: list[dict] = []

    for (sid, tp), grp in hazard_delta_df.groupby([subject_col, timepoint_col]):
        grp = grp.copy()
        abs_delta = pd.to_numeric(grp[delta_col], errors="coerce").abs().fillna(0.0)
        shares = _contribution_shares(abs_delta)
        grp = grp.assign(abs_contrib=abs_delta.values, share_contrib=shares.values)
        grp = grp.sort_values("abs_contrib", ascending=False).reset_index(drop=True)

        for rank, (_, row_d) in enumerate(grp.iterrows(), start=1):
            hazard = str(row_d[hazard_col])
            display = HAZARD_DISPLAY_NAMES.get(hazard, hazard.replace("_", " "))
            delta_val = row_d.get(delta_col)
            delta_f = float(delta_val) if pd.notna(delta_val) else float("nan")
            share = float(row_d["share_contrib"])
            interp = (
                f"{display} hazard-context alignment accounts for {share * 100:.0f}% "
                "of the hazard-context shift. This is hazard-context alignment, not "
                "exposure attribution or causal proof."
            )
            out_rows.append({
                "subject_id":                    sid,
                "timepoint":                     tp,
                "mission_phase":                 row_d.get("mission_phase", "unknown") if has_phase else "unknown",
                "hazard":                        hazard,
                "baseline_hazard_relevance":     row_d.get("baseline_hazard_relevance", np.nan),
                "current_hazard_relevance":      row_d.get("current_hazard_relevance", np.nan),
                "delta_hazard_relevance":        round(delta_f, 5) if pd.notna(delta_f) else np.nan,
                "absolute_delta_hazard_relevance": round(float(row_d["abs_contrib"]), 5),
                "contribution_share":            round(share, 5),
                "coverage_fraction":             row_d.get("coverage_fraction", np.nan) if has_cov else np.nan,
                "attribution_rank":              rank,
                "interpretation":                interp,
            })

    return pd.DataFrame(out_rows, columns=[
        "subject_id", "timepoint", "mission_phase", "hazard",
        "baseline_hazard_relevance", "current_hazard_relevance",
        "delta_hazard_relevance", "absolute_delta_hazard_relevance",
        "contribution_share", "coverage_fraction", "attribution_rank",
        "interpretation",
    ])


# ---------------------------------------------------------------------------
# Recovery attribution
# ---------------------------------------------------------------------------

def _recovery_category(
    recovery_fraction: float | None,
    peak_delta: float,
    final_delta: float,
) -> str:
    if recovery_fraction is None or (isinstance(recovery_fraction, float) and np.isnan(recovery_fraction)):
        if abs(peak_delta) < _EPSILON:
            return "insufficient_data"
        return "insufficient_data"
    # Overshoot/reversal: final lands on the opposite side of baseline from peak.
    if peak_delta != 0 and final_delta != 0 and np.sign(final_delta) != np.sign(peak_delta):
        return "overshoot_or_reversal"
    if recovery_fraction >= 0.75:
        return "returned_near_baseline"
    if recovery_fraction >= 0.4:
        return "partial_recovery"
    return "persistent_shift"


def compute_recovery_attribution(
    recovery_metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    """Interpret which metrics returned toward baseline, persisted, or overshot.

    Categories: ``returned_near_baseline``, ``partial_recovery``,
    ``persistent_shift``, ``overshoot_or_reversal``, ``insufficient_data``.
    """
    if recovery_metrics_df is None or recovery_metrics_df.empty:
        return pd.DataFrame(columns=[
            "subject_id", "metric", "baseline_value", "peak_value", "final_value",
            "peak_delta_from_baseline", "final_delta_from_baseline",
            "recovery_fraction", "recovery_category", "interpretation",
        ])

    out_rows: list[dict] = []
    for _, row in recovery_metrics_df.iterrows():
        rec_frac = row.get("recovery_fraction")
        rec_frac_f = float(rec_frac) if pd.notna(rec_frac) else None
        peak_delta = float(row.get("peak_delta_from_baseline", 0.0) or 0.0)
        final_delta = float(row.get("final_delta_from_baseline", 0.0) or 0.0)
        category = _recovery_category(rec_frac_f, peak_delta, final_delta)

        metric = row.get("metric", "metric")
        cat_text = {
            "returned_near_baseline": "returned near personal baseline",
            "partial_recovery":       "partially recovered toward baseline",
            "persistent_shift":       "remained shifted from baseline (persistent)",
            "overshoot_or_reversal":  "overshot or reversed relative to baseline",
            "insufficient_data":      "has insufficient data to assess recovery",
        }[category]
        interp = (
            f"{metric} {cat_text}"
            + (f" (recovery fraction {rec_frac_f:.2f})." if rec_frac_f is not None else ".")
            + " Recovery persistence is a monitoring-relevant pattern, not diagnosis."
        )
        out_rows.append({
            "subject_id":                row.get("subject_id"),
            "metric":                    metric,
            "baseline_value":            row.get("baseline_value", np.nan),
            "peak_value":                row.get("peak_value", np.nan),
            "final_value":               row.get("final_value", np.nan),
            "peak_delta_from_baseline":  round(peak_delta, 5),
            "final_delta_from_baseline": round(final_delta, 5),
            "recovery_fraction":         round(rec_frac_f, 5) if rec_frac_f is not None else np.nan,
            "recovery_category":         category,
            "interpretation":            interp,
        })

    return pd.DataFrame(out_rows, columns=[
        "subject_id", "metric", "baseline_value", "peak_value", "final_value",
        "peak_delta_from_baseline", "final_delta_from_baseline",
        "recovery_fraction", "recovery_category", "interpretation",
    ])


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def build_phase7_attribution_summary(
    node_attr: pd.DataFrame,
    graph_attr: pd.DataFrame,
    subgraph_attr: pd.DataFrame | None = None,
    hazard_attr: pd.DataFrame | None = None,
    recovery_attr: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build one summary row per subject/timepoint of top contributors."""
    if node_attr.empty:
        return pd.DataFrame(columns=[
            "subject_id", "timepoint", "mission_phase",
            "top_domain_contributor", "top_domain_contribution_share",
            "top_graph_metric_contributor", "top_subgraph_contributor",
            "top_hazard_context_contributor", "recovery_summary", "interpretation",
        ])

    # Recovery summary is per-subject (not per-timepoint).
    recovery_by_subject: dict[str, str] = {}
    if recovery_attr is not None and not recovery_attr.empty:
        for sid, grp in recovery_attr.groupby("subject_id"):
            counts = grp["recovery_category"].value_counts()
            recovery_by_subject[sid] = "; ".join(
                f"{cat}:{n}" for cat, n in counts.items()
            )

    keys = node_attr[["subject_id", "timepoint"]].drop_duplicates()
    out_rows: list[dict] = []

    for _, key in keys.iterrows():
        sid, tp = key["subject_id"], key["timepoint"]
        n_grp = node_attr[(node_attr["subject_id"] == sid) & (node_attr["timepoint"] == tp)]
        phase = n_grp.iloc[0]["mission_phase"] if not n_grp.empty else "unknown"

        top_domain = "n/a"
        top_domain_share = 0.0
        if not n_grp.empty:
            top = n_grp.sort_values("contribution_share", ascending=False).iloc[0]
            top_domain = top["domain"]
            top_domain_share = round(float(top["contribution_share"]), 5)

        top_metric = "n/a"
        if not graph_attr.empty:
            g_grp = graph_attr[(graph_attr["subject_id"] == sid) & (graph_attr["timepoint"] == tp)]
            if not g_grp.empty:
                top_metric = g_grp.sort_values("contribution_share", ascending=False).iloc[0]["metric"]

        top_subgraph = "n/a"
        if subgraph_attr is not None and not subgraph_attr.empty:
            s_grp = subgraph_attr[(subgraph_attr["subject_id"] == sid) & (subgraph_attr["timepoint"] == tp)]
            s_grp = s_grp[s_grp["n_available_domains"] > 0]
            if not s_grp.empty:
                top_subgraph = s_grp.sort_values("total_contribution_share", ascending=False).iloc[0]["subgraph_name"]

        top_hazard = "n/a"
        if hazard_attr is not None and not hazard_attr.empty:
            h_grp = hazard_attr[(hazard_attr["subject_id"] == sid) & (hazard_attr["timepoint"] == tp)]
            h_grp = h_grp.dropna(subset=["delta_hazard_relevance"])
            if not h_grp.empty and h_grp["contribution_share"].sum() > 0:
                top_hazard = h_grp.sort_values("contribution_share", ascending=False).iloc[0]["hazard"]

        recovery_summary = recovery_by_subject.get(sid, "n/a")

        if top_domain_share <= 0:
            interp = (
                f"At {tp} ({phase}), the graph matches personal baseline; no "
                "baseline-relative shift to attribute."
            )
        else:
            interp = (
                f"At {tp} ({phase}), the baseline-relative graph shift is driven "
                f"primarily by {top_domain} ({top_domain_share * 100:.0f}% of node change)"
            )
            if top_subgraph != "n/a":
                interp += f"; the dominant subgraph is {top_subgraph}"
            if top_hazard != "n/a":
                interp += f"; hazard-context alignment is strongest for {top_hazard.replace('_', ' ')}"
            interp += f". {_GUARDRAIL}"

        out_rows.append({
            "subject_id":                     sid,
            "timepoint":                      tp,
            "mission_phase":                  phase,
            "top_domain_contributor":         top_domain,
            "top_domain_contribution_share":  top_domain_share,
            "top_graph_metric_contributor":   top_metric,
            "top_subgraph_contributor":       top_subgraph,
            "top_hazard_context_contributor": top_hazard,
            "recovery_summary":               recovery_summary,
            "interpretation":                 interp,
        })

    summary = pd.DataFrame(out_rows, columns=[
        "subject_id", "timepoint", "mission_phase",
        "top_domain_contributor", "top_domain_contribution_share",
        "top_graph_metric_contributor", "top_subgraph_contributor",
        "top_hazard_context_contributor", "recovery_summary", "interpretation",
    ])
    return summary.sort_values(["subject_id", "timepoint"]).reset_index(drop=True)
