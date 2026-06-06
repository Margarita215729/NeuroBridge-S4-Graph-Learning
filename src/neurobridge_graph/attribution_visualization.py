"""Phase 7 — Matplotlib visualizations for trajectory attribution.

Figures are intentionally plain (no decorative styling) and carry guardrail
captions. They visualize *which* components contribute to the baseline-relative
graph shift; they are not diagnostic, causal, or exposure visualizations.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neurobridge_graph.hazard_mapping import HAZARD_DISPLAY_NAMES

_GUARDRAIL = (
    "Transparent attribution of within-subject graph change. Not diagnosis, "
    "causal proof, or exposure measurement."
)

_RECOVERY_ORDER = [
    "returned_near_baseline",
    "partial_recovery",
    "persistent_shift",
    "overshoot_or_reversal",
    "insufficient_data",
]


def _save(fig: plt.Figure, output_path: "str | Path") -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def _caption(fig: plt.Figure, y: float = -0.04) -> None:
    fig.text(0.5, y, _GUARDRAIL, ha="center", va="top", fontsize=8,
             style="italic", color="#555555", wrap=True)


def _empty(output_path: "str | Path", message: str) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11)
    ax.axis("off")
    return _save(fig, output_path)


def _label(row: pd.Series) -> str:
    return f"{row['subject_id']}\n{row['timepoint']}"


def plot_node_attribution_bar(
    node_attr: pd.DataFrame,
    output_path: "str | Path",
    top_n: int = 8,
) -> Path:
    """Bar chart of the top node (domain) contributors by contribution share."""
    moving = node_attr[node_attr["contribution_share"] > 0] if not node_attr.empty else node_attr
    if moving.empty:
        return _empty(output_path, "No baseline-relative node change to attribute.")

    agg = (
        moving.groupby("domain")["contribution_share"].mean()
        .sort_values(ascending=False).head(top_n)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(range(len(agg)), agg.values[::-1], color="#3b6ea5")
    ax.set_yticks(range(len(agg)))
    ax.set_yticklabels(list(agg.index)[::-1], fontsize=9)
    ax.set_xlabel("Mean contribution share (absolute delta / total)")
    ax.set_title(f"Top {len(agg)} biological domain contributors to graph shift")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    _caption(fig)
    return _save(fig, output_path)


def _pivot_heatmap(
    df: pd.DataFrame,
    index_col: str,
    value_col: str,
    output_path: "str | Path",
    title: str,
    xlabel: str,
    empty_msg: str,
    row_display: dict | None = None,
) -> Path:
    if df is None or df.empty:
        return _empty(output_path, empty_msg)
    work = df.copy()
    work["_st"] = work["subject_id"].astype(str) + "\n" + work["timepoint"].astype(str)
    pivot = work.pivot_table(
        index=index_col, columns="_st", values=value_col, aggfunc="mean"
    ).fillna(0.0)
    if pivot.empty:
        return _empty(output_path, empty_msg)

    if row_display:
        pivot.index = [row_display.get(str(i), str(i)) for i in pivot.index]

    fig, ax = plt.subplots(figsize=(max(9, 1.35 * pivot.shape[1]), max(4, 0.6 * pivot.shape[0] + 1)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(
        [str(c).replace("\n", " · ") for c in pivot.columns],
        fontsize=8, rotation=30, ha="right",
    )
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="white" if val < pivot.values.max() * 0.6 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="Contribution share")
    fig.subplots_adjust(bottom=0.32)
    _caption(fig, y=-0.16)
    return _save(fig, output_path)


def plot_subgraph_attribution_heatmap(
    subgraph_attr: pd.DataFrame,
    output_path: "str | Path",
) -> Path:
    """Heatmap of subgraph contribution share by subject/timepoint."""
    if subgraph_attr is not None and not subgraph_attr.empty:
        subgraph_attr = subgraph_attr[subgraph_attr["n_available_domains"] > 0]
    return _pivot_heatmap(
        subgraph_attr,
        index_col="subgraph_name",
        value_col="total_contribution_share",
        output_path=output_path,
        title="Subgraph contribution share by subject-timepoint",
        xlabel="Subject / timepoint",
        empty_msg="No subgraph attribution available (sparse domains).",
    )


def plot_hazard_context_attribution_heatmap(
    hazard_attr: pd.DataFrame,
    output_path: "str | Path",
) -> Path:
    """Heatmap of hazard-context contribution share by subject/timepoint."""
    return _pivot_heatmap(
        hazard_attr,
        index_col="hazard",
        value_col="contribution_share",
        output_path=output_path,
        title="Hazard-context alignment share by subject-timepoint (not exposure)",
        xlabel="Subject / timepoint",
        empty_msg="Hazard-context attribution unavailable.",
        row_display=HAZARD_DISPLAY_NAMES,
    )


def plot_recovery_attribution_summary(
    recovery_attr: pd.DataFrame,
    output_path: "str | Path",
) -> Path:
    """Categorical bar chart summarizing recovery categories."""
    if recovery_attr is None or recovery_attr.empty:
        return _empty(output_path, "Recovery attribution unavailable.")

    counts = recovery_attr["recovery_category"].value_counts()
    ordered = [c for c in _RECOVERY_ORDER if c in counts.index]
    values = [int(counts[c]) for c in ordered]

    colors = {
        "returned_near_baseline": "#2c7a2c",
        "partial_recovery":       "#9acd32",
        "persistent_shift":       "#d99000",
        "overshoot_or_reversal":  "#b03030",
        "insufficient_data":      "#888888",
    }
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(ordered)), values,
           color=[colors[c] for c in ordered])
    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels([c.replace("_", "\n") for c in ordered], fontsize=9)
    ax.set_ylabel("Number of metrics")
    ax.set_title("Recovery attribution: how metrics behaved relative to baseline")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    for i, v in enumerate(values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    _caption(fig)
    return _save(fig, output_path)


def plot_subject_explanation_panel(
    subject_summary: pd.DataFrame,
    output_path: "str | Path",
) -> Path:
    """Compact table-style panel of top contributors per subject/timepoint."""
    if subject_summary is None or subject_summary.empty:
        return _empty(output_path, "No attribution summary available.")

    cols = [
        ("subject_id", "Subject"),
        ("timepoint", "Timepoint"),
        ("mission_phase", "Phase"),
        ("top_domain_contributor", "Top domain"),
        ("top_subgraph_contributor", "Top subgraph"),
        ("top_hazard_context_contributor", "Top hazard-context"),
        ("top_graph_metric_contributor", "Top metric"),
    ]
    present = [(c, h) for c, h in cols if c in subject_summary.columns]
    headers = [h for _, h in present]

    def _cell(val: object) -> str:
        s = str(val)
        return s if len(s) <= 26 else s[:24] + "…"

    table_data = [
        [_cell(row[c]) for c, _ in present]
        for _, row in subject_summary.iterrows()
    ]

    fig_h = max(2.5, 0.5 * len(table_data) + 1.5)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    ax.axis("off")
    ax.set_title("Phase 7 — Top trajectory-shift contributors by subject/timepoint",
                 fontsize=12, pad=14)
    tbl = ax.table(cellText=table_data, colLabels=headers, loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)
    for j in range(len(headers)):
        tbl[0, j].set_facecolor("#3b6ea5")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    _caption(fig)
    return _save(fig, output_path)
