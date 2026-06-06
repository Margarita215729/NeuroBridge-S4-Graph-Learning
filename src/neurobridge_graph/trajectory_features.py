"""Phase 6 — Trajectory feature extraction for longitudinal graph analysis.

Computes recovery metrics, hazard-context deltas from personal baseline, and
identifies dominant within-subject trajectory shifts.

All outputs describe within-subject graph changes. They are not diagnosis,
treatment guidance, exposure measurement, or health outcome prediction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from neurobridge_graph.hazard_mapping import interpret_hazard_score

_EPSILON = 1e-6


def compute_recovery_slope(
    trajectory_df: pd.DataFrame,
    metric_col: str,
    phase_col: str = "mission_phase",
    time_col: str = "time_index",
) -> float | None:
    """Estimate whether a metric moves back toward baseline during recovery.

    Fits a simple linear slope over recovery-phase timepoints. A negative slope
    after a peak suggests movement back toward baseline; a positive slope
    suggests continued deviation.

    Returns ``None`` when recovery phase has fewer than 2 points or the metric
    column is absent.
    """
    if metric_col not in trajectory_df.columns:
        return None
    if phase_col not in trajectory_df.columns or time_col not in trajectory_df.columns:
        return None

    recovery = trajectory_df[
        trajectory_df[phase_col].astype(str).str.lower() == "recovery"
    ].sort_values(time_col)
    if len(recovery) < 2:
        return None

    x = recovery[time_col].astype(float).values
    y = pd.to_numeric(recovery[metric_col], errors="coerce").values
    if np.any(np.isnan(y)):
        return None

    # Simple least-squares slope.
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom < _EPSILON:
        return None
    slope = float(((x - x_mean) * (y - y_mean)).sum() / denom)
    return round(slope, 5)


def compute_time_to_baseline_like_state(
    trajectory_df: pd.DataFrame,
    metric_col: str,
    tolerance: float = 0.25,
    time_col: str = "time_index",
    phase_col: str = "mission_phase",
) -> float | None:
    """Estimate first timepoint where metric returns near personal baseline.

    Compares each post-baseline timepoint's value to the baseline-phase value.
    Returns the ``time_index`` of the first timepoint where
    ``|value - baseline| <= tolerance``, or ``None`` if never reached.
    """
    if metric_col not in trajectory_df.columns:
        return None
    if time_col not in trajectory_df.columns:
        return None

    df = trajectory_df.sort_values(time_col)
    baseline_rows = df[df[phase_col].astype(str).str.lower() == "baseline"]
    if baseline_rows.empty:
        baseline_val = pd.to_numeric(df.iloc[0][metric_col], errors="coerce")
    else:
        baseline_val = pd.to_numeric(baseline_rows.iloc[0][metric_col], errors="coerce")

    if pd.isna(baseline_val):
        return None

    post_baseline = df[df[phase_col].astype(str).str.lower() != "baseline"]
    for _, row in post_baseline.iterrows():
        val = pd.to_numeric(row[metric_col], errors="coerce")
        if pd.isna(val):
            continue
        if abs(val - baseline_val) <= tolerance:
            return float(row[time_col])

    return None


def compute_recovery_fraction(
    baseline_value: float,
    peak_value: float,
    final_value: float,
) -> float | None:
    """Compute recovery fraction: how much of peak deviation has been recovered.

    ``recovery_fraction = 1 - |final - baseline| / max(|peak - baseline|, epsilon)``

    Clipped to ``[0, 1]``. Returns ``None`` when peak equals baseline (no deviation).
    """
    peak_delta = abs(peak_value - baseline_value)
    if peak_delta < _EPSILON:
        return None
    final_delta = abs(final_value - baseline_value)
    frac = 1.0 - final_delta / peak_delta
    return round(float(np.clip(frac, 0.0, 1.0)), 5)


def compute_hazard_context_delta(
    hazard_scores_longitudinal: pd.DataFrame,
    baseline_phase: str = "baseline",
) -> pd.DataFrame:
    """Compute hazard relevance score changes from personal baseline.

    Parameters
    ----------
    hazard_scores_longitudinal:
        Long-form table with ``subject_id``, ``timepoint``, ``mission_phase``,
        ``hazard``, ``hazard_relevance_score``, ``coverage_fraction``.

    Returns
    -------
    pandas.DataFrame
        Columns: ``subject_id``, ``timepoint``, ``mission_phase``, ``hazard``,
        ``baseline_hazard_relevance``, ``current_hazard_relevance``,
        ``delta_hazard_relevance``, ``coverage_fraction``, ``interpretation``.
    """
    required = {"subject_id", "timepoint", "mission_phase", "hazard",
                "hazard_relevance_score"}
    missing = required - set(hazard_scores_longitudinal.columns)
    if missing:
        raise ValueError(f"hazard_scores_longitudinal missing columns: {missing}")

    rows: list[dict] = []
    for subject_id, sub in hazard_scores_longitudinal.groupby("subject_id"):
        baseline_rows = sub[sub["mission_phase"].astype(str).str.lower() == baseline_phase]
        if baseline_rows.empty:
            baseline_rows = sub.sort_values("timepoint").iloc[:1]

        baseline_lookup: dict[str, float] = {}
        for _, br in baseline_rows.iterrows():
            hz = str(br["hazard"])
            val = br["hazard_relevance_score"]
            baseline_lookup[hz] = float(val) if pd.notna(val) else float("nan")

        for _, row in sub.iterrows():
            hz = str(row["hazard"])
            b_val = baseline_lookup.get(hz, float("nan"))
            c_val = row["hazard_relevance_score"]
            c_f = float(c_val) if pd.notna(c_val) else float("nan")
            if pd.isna(b_val) or pd.isna(c_f):
                delta = float("nan")
            else:
                delta = round(c_f - b_val, 5)
            cov = float(row.get("coverage_fraction", 0.0))
            interp = (
                f"Hazard-context shift for {hz.replace('_', ' ')}: "
                f"delta {delta:+.2f} from personal baseline. "
                "This is a monitoring-relevant pattern shift, not exposure "
                "measurement or causal proof."
                if pd.notna(delta) else
                f"No computable hazard-context delta for {hz} "
                "(insufficient domain coverage)."
            )
            rows.append({
                "subject_id":                  subject_id,
                "timepoint":                   row["timepoint"],
                "mission_phase":               row["mission_phase"],
                "hazard":                      hz,
                "baseline_hazard_relevance":   round(b_val, 5) if pd.notna(b_val) else float("nan"),
                "current_hazard_relevance":    round(c_f, 5) if pd.notna(c_f) else float("nan"),
                "delta_hazard_relevance":      delta,
                "coverage_fraction":           round(cov, 5),
                "interpretation":              interp,
            })

    return pd.DataFrame(rows)


def compute_recovery_metrics_table(
    trajectory_df: pd.DataFrame,
    metrics: list[str] | None = None,
    phase_col: str = "mission_phase",
    time_col: str = "time_index",
    subject_col: str = "subject_id",
    tolerance: float = 0.25,
) -> pd.DataFrame:
    """Compute per-subject recovery metrics for selected trajectory features.

    Returns
    -------
    pandas.DataFrame
        Columns: ``subject_id``, ``metric``, ``baseline_value``, ``peak_value``,
        ``final_value``, ``peak_delta_from_baseline``, ``final_delta_from_baseline``,
        ``recovery_fraction``, ``time_to_baseline_like_state``, ``interpretation``.
    """
    if metrics is None:
        metrics = [
            "mean_node_activation", "max_node_activation",
            "total_node_activation", "n_active_domains",
        ]
    metrics = [m for m in metrics if m in trajectory_df.columns]
    rows: list[dict] = []

    for subject_id, sub in trajectory_df.groupby(subject_col):
        sub = sub.sort_values(time_col)
        baseline_rows = sub[sub[phase_col].astype(str).str.lower() == "baseline"]
        if baseline_rows.empty:
            baseline_rows = sub.iloc[:1]

        for metric in metrics:
            vals = pd.to_numeric(sub[metric], errors="coerce")
            if vals.isna().all():
                continue
            b_val = float(pd.to_numeric(baseline_rows.iloc[0][metric], errors="coerce"))
            peak_val = float(vals.max())
            final_val = float(vals.iloc[-1])
            peak_delta = round(peak_val - b_val, 5)
            final_delta = round(final_val - b_val, 5)
            rec_frac = compute_recovery_fraction(b_val, peak_val, final_val)
            t2b = compute_time_to_baseline_like_state(sub, metric, tolerance=tolerance)
            slope = compute_recovery_slope(sub, metric)

            if rec_frac is not None and rec_frac >= 0.75:
                interp = (
                    f"{metric}: strong recovery toward personal baseline "
                    f"(recovery fraction {rec_frac:.2f})."
                )
            elif rec_frac is not None and rec_frac >= 0.4:
                interp = (
                    f"{metric}: partial recovery toward personal baseline "
                    f"(recovery fraction {rec_frac:.2f})."
                )
            elif peak_delta > 0.1:
                interp = (
                    f"{metric}: mission-phase shift detected "
                    f"(peak delta {peak_delta:+.2f} from baseline); "
                    "monitoring-relevant pattern, not diagnosis."
                )
            else:
                interp = f"{metric}: stable relative to personal baseline."

            rows.append({
                "subject_id":                  subject_id,
                "metric":                      metric,
                "baseline_value":              round(b_val, 5),
                "peak_value":                  round(peak_val, 5),
                "final_value":                 round(final_val, 5),
                "peak_delta_from_baseline":    peak_delta,
                "final_delta_from_baseline":   final_delta,
                "recovery_fraction":           rec_frac,
                "recovery_slope":              slope,
                "time_to_baseline_like_state": t2b,
                "interpretation":              interp,
            })

    return pd.DataFrame(rows)


def identify_dominant_trajectory_shift(
    node_delta_df: pd.DataFrame,
    graph_delta_df: pd.DataFrame,
    hazard_delta_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Identify top domain, graph feature, and hazard-context shifts per subject/timepoint.

    Returns
    -------
    pandas.DataFrame
        One row per (subject_id, timepoint) with dominant shift summaries.
    """
    rows: list[dict] = []

    if node_delta_df.empty:
        return pd.DataFrame()

    groups = node_delta_df.groupby(["subject_id", "timepoint"])
    for (sid, tp), node_grp in groups:
        top_node = node_grp.loc[node_grp["absolute_delta_activation"].idxmax()]
        row: dict[str, Any] = {
            "subject_id":              sid,
            "timepoint":               tp,
            "mission_phase":           node_grp.iloc[0].get("mission_phase", "unknown"),
            "dominant_domain":         top_node["domain"],
            "dominant_domain_delta":   top_node["delta_activation"],
            "dominant_domain_direction": top_node["direction"],
        }

        if not graph_delta_df.empty:
            g_grp = graph_delta_df[
                (graph_delta_df["subject_id"] == sid) &
                (graph_delta_df["timepoint"] == tp)
            ]
            if not g_grp.empty:
                top_g = g_grp.loc[g_grp["absolute_delta_value"].idxmax()]
                row["dominant_graph_metric"] = top_g["metric"]
                row["dominant_graph_delta"] = top_g["delta_value"]

        if hazard_delta_df is not None and not hazard_delta_df.empty:
            h_grp = hazard_delta_df[
                (hazard_delta_df["subject_id"] == sid) &
                (hazard_delta_df["timepoint"] == tp)
            ]
            if not h_grp.empty:
                valid = h_grp.dropna(subset=["delta_hazard_relevance"])
                if not valid.empty:
                    top_h = valid.loc[valid["delta_hazard_relevance"].abs().idxmax()]
                    row["dominant_hazard"] = top_h["hazard"]
                    row["dominant_hazard_delta"] = top_h["delta_hazard_relevance"]

        rows.append(row)

    return pd.DataFrame(rows)
