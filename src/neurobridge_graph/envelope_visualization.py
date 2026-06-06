"""Phase 8 — Matplotlib visualizations for the reference-calibrated envelope.

Figures are intentionally plain (no decorative styling) and carry guardrail
captions. They show whether within-subject baseline-relative deltas fall inside,
near, or outside an expected-variability envelope. Envelope exceedance is **not**
diagnosis, **not** a risk score, and **not** exposure measurement.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_GUARDRAIL = (
    "Envelope exceedance = larger-than-expected baseline-relative change. "
    "Not diagnosis, not risk, not exposure measurement."
)

_POSITION_COLOR = {
    "within_expected_envelope": "#2c7a2c",
    "near_envelope_boundary":   "#d99000",
    "outside_expected_envelope": "#b03030",
    "insufficient_reference":   "#888888",
}


def _save(fig: plt.Figure, output_path: "str | Path", caption_y: float = -0.04) -> Path:
    fig.text(0.5, caption_y, _GUARDRAIL, ha="center", va="top", fontsize=8,
             style="italic", color="#555555", wrap=True)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def _empty(output_path: "str | Path", message: str) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11)
    ax.axis("off")
    return _save(fig, output_path)


def _delta_col(scores: pd.DataFrame) -> str:
    for c in ("delta_activation", "delta_value", "delta_hazard_relevance"):
        if c in scores.columns:
            return c
    raise ValueError("No recognized delta column in scores table.")


def _feature_col(scores: pd.DataFrame) -> str:
    for c in ("domain", "metric", "hazard"):
        if c in scores.columns:
            return c
    raise ValueError("No recognized feature column in scores table.")


def _plot_delta_envelope(
    scores: pd.DataFrame,
    output_path: "str | Path",
    title: str,
    empty_msg: str,
) -> Path:
    if scores is None or scores.empty:
        return _empty(output_path, empty_msg)

    dcol = _delta_col(scores)
    fcol = _feature_col(scores)
    work = scores.copy()
    work["_st"] = work["subject_id"].astype(str) + " · " + work["timepoint"].astype(str)
    work["_label"] = work[fcol].astype(str) + "\n" + work["_st"]

    # Focus on the most informative rows: any outside/near, else largest |z|.
    informative = work[work["envelope_position"].isin(
        ["outside_expected_envelope", "near_envelope_boundary"])]
    if informative.empty:
        work["_absz"] = work["robust_z"].abs().fillna(0)
        informative = work.sort_values("_absz", ascending=False).head(12)
    else:
        informative = informative.sort_values("envelope_exceedance", ascending=False).head(16)

    informative = informative.reset_index(drop=True)
    n = len(informative)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * n + 1.5)))

    y = np.arange(n)
    # Envelope band as horizontal error bars from lower to upper bound.
    for i, row in informative.iterrows():
        lb, ub = row["lower_bound"], row["upper_bound"]
        if pd.notna(lb) and pd.notna(ub):
            ax.plot([lb, ub], [i, i], color="#9bbde0", linewidth=8, solid_capstyle="butt",
                    alpha=0.6, zorder=1)
        color = _POSITION_COLOR.get(row["envelope_position"], "#333333")
        ax.scatter(row[dcol], i, color=color, s=55, zorder=3,
                   edgecolor="black", linewidth=0.5)

    ax.axvline(0.0, color="#444444", linestyle=":", linewidth=1, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(informative["_label"], fontsize=7)
    ax.set_xlabel("Baseline-relative delta (point) vs expected envelope (band)")
    ax.set_title(title)
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                   markeredgecolor="black", markersize=8, label=lbl.replace("_", " "))
        for lbl, c in _POSITION_COLOR.items()
    ]
    ax.legend(handles=handles, fontsize=7, loc="best", framealpha=0.9)
    return _save(fig, output_path)


def plot_domain_delta_envelope(node_scores: pd.DataFrame, output_path: "str | Path") -> Path:
    """Plot subject/timepoint domain deltas against envelope bounds."""
    return _plot_delta_envelope(
        node_scores, output_path,
        title="Domain delta vs reference-calibrated variability envelope",
        empty_msg="No node delta envelope scores available.",
    )


def plot_graph_metric_envelope(graph_scores: pd.DataFrame, output_path: "str | Path") -> Path:
    """Plot graph metric deltas against envelope bounds."""
    return _plot_delta_envelope(
        graph_scores, output_path,
        title="Graph metric delta vs reference-calibrated variability envelope",
        empty_msg="No graph delta envelope scores available.",
    )


def plot_hazard_delta_envelope(hazard_scores: pd.DataFrame, output_path: "str | Path") -> Path:
    """Plot hazard-context deltas against envelope bounds."""
    return _plot_delta_envelope(
        hazard_scores, output_path,
        title="Hazard-context delta vs envelope (alignment, not exposure)",
        empty_msg="No hazard delta envelope scores available.",
    )


def plot_envelope_exceedance_heatmap(
    phase8_summary: pd.DataFrame,
    output_path: "str | Path",
) -> Path:
    """Heatmap of outside-envelope counts by subject/timepoint."""
    if phase8_summary is None or phase8_summary.empty:
        return _empty(output_path, "No Phase 8 summary available.")

    work = phase8_summary.copy()
    work["_st"] = work["subject_id"].astype(str) + " · " + work["timepoint"].astype(str)
    layers = [
        ("n_outside_node_envelope", "Node / domain"),
        ("n_outside_graph_envelope", "Graph metric"),
        ("n_outside_hazard_envelope", "Hazard-context"),
    ]
    layers = [(c, lbl) for c, lbl in layers if c in work.columns]
    matrix = np.array([work[c].fillna(0).to_numpy(dtype=float) for c, _ in layers])

    fig, ax = plt.subplots(figsize=(max(9, 1.25 * work.shape[0]), 4))
    im = ax.imshow(matrix, aspect="auto", cmap="OrRd")
    ax.set_xticks(range(work.shape[0]))
    ax.set_xticklabels(work["_st"], fontsize=8, rotation=30, ha="right")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([lbl for _, lbl in layers], fontsize=9)
    ax.set_title("Outside-envelope delta counts by subject-timepoint")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{int(matrix[i, j])}", ha="center", va="center",
                    color="black", fontsize=9)
    fig.colorbar(im, ax=ax, label="# deltas outside envelope")
    fig.subplots_adjust(bottom=0.28)
    return _save(fig, output_path, caption_y=-0.14)


def plot_reference_envelope_overview(
    envelope_df: pd.DataFrame,
    output_path: "str | Path",
) -> Path:
    """Show envelope width by feature (which features are naturally more variable)."""
    if envelope_df is None or envelope_df.empty:
        return _empty(output_path, "No reference envelope available.")

    work = envelope_df.copy()
    work["envelope_width"] = work["upper_bound"] - work["lower_bound"]
    work = work.dropna(subset=["envelope_width"])
    if work.empty:
        return _empty(output_path, "Reference envelope has no usable bounds.")
    work = work.sort_values("envelope_width", ascending=True)

    colors_by_type = {"node": "#3b6ea5", "graph": "#6a51a3", "hazard": "#2c7a2c"}
    bar_colors = [
        colors_by_type.get(str(t), "#777777") for t in work.get("feature_type", ["?"] * len(work))
    ]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.42 * len(work) + 1.5)))
    ax.barh(range(len(work)), work["envelope_width"], color=bar_colors)
    ax.set_yticks(range(len(work)))
    ax.set_yticklabels(work["feature"], fontsize=8)
    ax.set_xlabel("Expected variability envelope width (upper bound − lower bound)")
    ax.set_title("Reference envelope width by feature")
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    if "feature_type" in work.columns and work["feature_type"].notna().any():
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=c, label=t)
            for t, c in colors_by_type.items()
        ]
        ax.legend(handles=handles, fontsize=8, title="feature type", loc="lower right")
    return _save(fig, output_path)
