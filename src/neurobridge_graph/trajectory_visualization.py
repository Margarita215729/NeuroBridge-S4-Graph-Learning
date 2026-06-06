"""Phase 6 — Matplotlib visualizations for longitudinal graph trajectories.

All figures use non-diagnostic language. Captions emphasize that outputs
describe within-subject graph changes, not diagnosis or health outcome
prediction.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_GUARDRAIL = (
    "Within-subject graph change only. Not diagnosis, treatment guidance, "
    "or health outcome prediction."
)


def _save_fig(fig: plt.Figure, output_path: "str | Path | None") -> None:
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=130, bbox_inches="tight")


def plot_domain_delta_trajectory(
    node_delta_df: pd.DataFrame,
    domains: list[str] | None = None,
    output_path: "str | Path | None" = None,
    show: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot domain activation delta over time for selected domains.

    Parameters
    ----------
    node_delta_df:
        Output of ``compute_longitudinal_delta_tables`` node table.
    domains:
        Domain names to plot. Defaults to top 3 by mean absolute delta.
    """
    if node_delta_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No node delta data.", ha="center", va="center")
        _save_fig(fig, output_path)
        return fig, ax

    if domains is None:
        mean_abs = (
            node_delta_df.groupby("domain")["absolute_delta_activation"]
            .mean().sort_values(ascending=False)
        )
        domains = list(mean_abs.head(3).index)

    fig, ax = plt.subplots(figsize=(10, 5))
    subjects = sorted(node_delta_df["subject_id"].unique())

    for domain in domains:
        sort_col = "time_index" if "time_index" in node_delta_df.columns else "timepoint"
        sub = node_delta_df[node_delta_df["domain"] == domain].sort_values(sort_col)
        for sid in subjects:
            s = sub[sub["subject_id"] == sid]
            if s.empty:
                continue
            x = s[sort_col].values if sort_col in s.columns else range(len(s))
            ax.plot(x, s["delta_activation"].values,
                    marker="o", label=f"{sid} — {domain}", linewidth=1.5)

    phase_labels = []
    if not node_delta_df.empty:
        sort_col = "time_index" if "time_index" in node_delta_df.columns else "timepoint"
        first_sub = node_delta_df[node_delta_df["subject_id"] == subjects[0]].drop_duplicates("timepoint")
        if "mission_phase" in first_sub.columns:
            phase_labels = list(first_sub.sort_values(sort_col)["mission_phase"].values)
    if phase_labels:
        ax.set_xticks(range(len(phase_labels)))
        ax.set_xticklabels(phase_labels, rotation=20, ha="right", fontsize=8)

    ax.axhline(0, color="#888", lw=0.8, linestyle="--")
    ax.set_ylabel("Activation delta from personal baseline")
    ax.set_title("Domain activation delta trajectory (self-baseline)")
    ax.legend(fontsize=7, loc="best")
    fig.text(0.5, -0.02, _GUARDRAIL, ha="center", fontsize=8, color="#555", style="italic")
    plt.tight_layout()
    _save_fig(fig, output_path)
    if show:
        plt.show()
    return fig, ax


def plot_graph_metric_trajectory(
    trajectory_df: pd.DataFrame,
    metric: str = "mean_node_activation",
    output_path: "str | Path | None" = None,
    show: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a graph-level metric over mission phases per subject."""
    fig, ax = plt.subplots(figsize=(9, 5))
    if metric not in trajectory_df.columns:
        ax.text(0.5, 0.5, f"Metric '{metric}' not found.", ha="center", va="center")
        _save_fig(fig, output_path)
        return fig, ax

    time_col = "time_index" if "time_index" in trajectory_df.columns else "timepoint"
    for sid, sub in trajectory_df.groupby("subject_id"):
        sub = sub.sort_values(time_col)
        ax.plot(sub[time_col], sub[metric], marker="o", label=str(sid), linewidth=1.5)

    ax.set_xlabel("Time index")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(f"Graph metric trajectory: {metric.replace('_', ' ')}")
    ax.legend(fontsize=8)
    fig.text(0.5, -0.02, _GUARDRAIL, ha="center", fontsize=8, color="#555", style="italic")
    plt.tight_layout()
    _save_fig(fig, output_path)
    if show:
        plt.show()
    return fig, ax


def plot_hazard_context_trajectory(
    hazard_delta_df: pd.DataFrame,
    hazards: list[str] | None = None,
    output_path: "str | Path | None" = None,
    show: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot hazard-context relevance delta from personal baseline over time."""
    fig, ax = plt.subplots(figsize=(10, 5))
    if hazard_delta_df.empty:
        ax.text(0.5, 0.5, "No hazard-context delta data.", ha="center", va="center")
        _save_fig(fig, output_path)
        return fig, ax

    if hazards is None:
        hazards = list(hazard_delta_df["hazard"].unique()[:3])

    time_col = "timepoint"
    for hazard in hazards:
        sub = hazard_delta_df[hazard_delta_df["hazard"] == hazard]
        for sid, s in sub.groupby("subject_id"):
            s = s.sort_values(time_col)
            valid = s.dropna(subset=["delta_hazard_relevance"])
            if valid.empty:
                continue
            ax.plot(range(len(valid)), valid["delta_hazard_relevance"].values,
                    marker="s", label=f"{sid} — {hazard.replace('_', ' ')}", linewidth=1.5)

    ax.axhline(0, color="#888", lw=0.8, linestyle="--")
    ax.set_ylabel("Hazard relevance delta from personal baseline")
    ax.set_title("Hazard-context shift trajectory (HRP interpretation layer)")
    ax.legend(fontsize=7, loc="best")
    fig.text(0.5, -0.04,
             "Hazard-context delta is not exposure measurement. " + _GUARDRAIL,
             ha="center", fontsize=8, color="#555", style="italic")
    plt.tight_layout()
    _save_fig(fig, output_path)
    if show:
        plt.show()
    return fig, ax


def plot_trajectory_heatmap(
    node_delta_df: pd.DataFrame,
    output_path: "str | Path | None" = None,
    show: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Heatmap of domain activation deltas by subject and timepoint."""
    fig, ax = plt.subplots(figsize=(10, 6))
    if node_delta_df.empty:
        ax.text(0.5, 0.5, "No data.", ha="center", va="center")
        _save_fig(fig, output_path)
        return fig, ax

    node_delta_df = node_delta_df.copy()
    node_delta_df["row_label"] = (
        node_delta_df["subject_id"].astype(str) + " / " +
        node_delta_df["timepoint"].astype(str)
    )
    pivot = node_delta_df.pivot_table(
        index="row_label", columns="domain",
        values="delta_activation", aggfunc="mean",
    )
    data = pivot.values.astype(float)
    im = ax.imshow(data, cmap="RdBu_r", aspect="auto",
                   vmin=-np.nanmax(np.abs(data)) if np.any(~np.isnan(data)) else -1,
                   vmax=np.nanmax(np.abs(data)) if np.any(~np.isnan(data)) else 1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_title("Domain activation delta heatmap (self-baseline)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="delta activation")
    fig.text(0.5, -0.02, _GUARDRAIL, ha="center", fontsize=8, color="#555", style="italic")
    plt.tight_layout()
    _save_fig(fig, output_path)
    if show:
        plt.show()
    return fig, ax


def plot_recovery_summary(
    recovery_df: pd.DataFrame,
    output_path: "str | Path | None" = None,
    show: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Bar chart of recovery fraction per subject and metric."""
    fig, ax = plt.subplots(figsize=(9, 5))
    if recovery_df.empty or "recovery_fraction" not in recovery_df.columns:
        ax.text(0.5, 0.5, "No recovery data.", ha="center", va="center")
        _save_fig(fig, output_path)
        return fig, ax

    valid = recovery_df.dropna(subset=["recovery_fraction"])
    if valid.empty:
        ax.text(0.5, 0.5, "No computable recovery fractions.", ha="center", va="center")
        _save_fig(fig, output_path)
        return fig, ax

    valid = valid.copy()
    valid["label"] = valid["subject_id"].astype(str) + "\n" + valid["metric"].astype(str)
    colors = ["#2ecc71" if r >= 0.75 else ("#f39c12" if r >= 0.4 else "#e74c3c")
              for r in valid["recovery_fraction"]]
    ax.barh(range(len(valid)), valid["recovery_fraction"].values, color=colors)
    ax.set_yticks(range(len(valid)))
    ax.set_yticklabels(valid["label"], fontsize=8)
    ax.set_xlabel("Recovery fraction toward personal baseline")
    ax.set_xlim(0, 1)
    ax.set_title("Recovery summary (self-baseline longitudinal tracking)")
    fig.text(0.5, -0.02, _GUARDRAIL, ha="center", fontsize=8, color="#555", style="italic")
    plt.tight_layout()
    _save_fig(fig, output_path)
    if show:
        plt.show()
    return fig, ax
