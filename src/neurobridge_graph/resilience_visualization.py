"""Phase 11 — Matplotlib figures for operational resilience interpretation.

All figures degrade gracefully when the resilience table is empty (an
explanatory placeholder figure is written instead of crashing).

Operational resilience interpretation is a research-review layer for expert
interpretation, not diagnosis, treatment guidance, health risk scoring,
exposure measurement, or an operational medical decision.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from neurobridge_graph.resilience_rules import RESILIENCE_STATES

_CAPTION = ("Research-review interpretation layer — not diagnosis, risk scoring, "
            "exposure measurement, or a mission-readiness decision.")


def _placeholder(output_path: str | Path, message: str) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.text(0.5, 0.55, message, ha="center", va="center", wrap=True, fontsize=11)
    ax.text(0.5, 0.18, _CAPTION, ha="center", va="center", fontsize=8, color="dimgray")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _empty(resilience_df: pd.DataFrame | None) -> bool:
    return resilience_df is None or resilience_df.empty


def plot_resilience_state_summary(resilience_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Bar chart of resilience state counts."""
    if _empty(resilience_df) or "resilience_state_label" not in resilience_df.columns:
        return _placeholder(output_path, "No resilience interpretations available.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    order = [RESILIENCE_STATES[k] for k in RESILIENCE_STATES]
    counts = resilience_df["resilience_state_label"].value_counts()
    counts = counts.reindex(order, fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(counts.index, counts.values, color="#4C72B0")
    ax.set_xlabel("Number of subject/timepoints")
    ax.set_title("Adaptive resilience state distribution")
    ax.invert_yaxis()
    for i, v in enumerate(counts.values):
        if v:
            ax.text(v + 0.05, i, str(int(v)), va="center", fontsize=9)
    fig.text(0.5, -0.02, _CAPTION, ha="center", fontsize=8, color="dimgray")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_resilience_state_timeline(resilience_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Matrix of resilience state by subject (rows) and timepoint (columns)."""
    needed = {"subject_id", "timepoint", "resilience_state"}
    if _empty(resilience_df) or not needed.issubset(resilience_df.columns):
        return _placeholder(output_path, "No resilience interpretations available.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    state_keys = list(RESILIENCE_STATES.keys())
    state_to_idx = {s: i for i, s in enumerate(state_keys)}

    subjects = sorted(resilience_df["subject_id"].astype(str).unique())
    timepoints = sorted(resilience_df["timepoint"].astype(str).unique())
    mat = np.full((len(subjects), len(timepoints)), np.nan)
    for _, r in resilience_df.iterrows():
        si = subjects.index(str(r["subject_id"]))
        ti = timepoints.index(str(r["timepoint"]))
        mat[si, ti] = state_to_idx.get(str(r["resilience_state"]), np.nan)

    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(timepoints)), max(3, 0.7 * len(subjects) + 2)))
    cmap = plt.get_cmap("viridis", len(state_keys))
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=len(state_keys) - 1)
    ax.set_xticks(range(len(timepoints)))
    ax.set_xticklabels(timepoints, rotation=45, ha="right")
    ax.set_yticks(range(len(subjects)))
    ax.set_yticklabels(subjects)
    ax.set_title("Resilience state by subject and timepoint")

    cbar = fig.colorbar(im, ax=ax, ticks=range(len(state_keys)))
    cbar.ax.set_yticklabels([RESILIENCE_STATES[k] for k in state_keys], fontsize=7)
    fig.text(0.5, -0.04, _CAPTION, ha="center", fontsize=8, color="dimgray")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_adaptation_mode_heatmap(resilience_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Heatmap of dominant adaptation mode counts by mission phase."""
    needed = {"dominant_adaptation_mode", "mission_phase"}
    if _empty(resilience_df) or not needed.issubset(resilience_df.columns):
        return _placeholder(output_path, "No adaptation-mode interpretations available.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pivot = (resilience_df
             .groupby(["dominant_adaptation_mode", "mission_phase"])
             .size().unstack(fill_value=0))

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * pivot.shape[1] + 3),
                                    max(3, 0.6 * pivot.shape[0] + 2)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="magma")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Dominant adaptation mode by mission phase")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = int(pivot.values[i, j])
            if v:
                ax.text(j, i, str(v), ha="center", va="center",
                        color="white", fontsize=9)
    fig.colorbar(im, ax=ax, label="count")
    fig.text(0.5, -0.04, _CAPTION, ha="center", fontsize=8, color="dimgray")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_evidence_chain_summary(resilience_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Summary of top evidence drivers (top domain / subgraph contributors)."""
    if _empty(resilience_df):
        return _placeholder(output_path, "No evidence-chain data available.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    plotted = False
    for ax, col, title in (
        (axes[0], "top_domain_contributor", "Top domain contributors"),
        (axes[1], "top_subgraph_contributor", "Top subgraph contributors"),
    ):
        if col in resilience_df.columns:
            counts = (resilience_df[col].astype(str)
                      .replace("n/a", np.nan).dropna().value_counts().head(10))
            if not counts.empty:
                ax.barh(counts.index, counts.values, color="#55A868")
                ax.invert_yaxis()
                plotted = True
        ax.set_title(title)
        ax.set_xlabel("count")

    if not plotted:
        plt.close(fig)
        return _placeholder(output_path, "No evidence-chain drivers available.")

    fig.suptitle("Evidence-chain driver summary")
    fig.text(0.5, -0.02, _CAPTION, ha="center", fontsize=8, color="dimgray")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
