"""Static matplotlib visualizations for biological adaptation graphs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend safe for notebooks + CI
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx


# Activation-level colour map
_LEVEL_COLORS = {
    "low":      "#AED6F1",   # light blue
    "mild":     "#A9DFBF",   # light green
    "moderate": "#F9E79F",   # yellow
    "high":     "#F1948A",   # red-pink
}
_DEFAULT_COLOR = "#D5D8DC"

_LEVEL_ORDER = ["low", "mild", "moderate", "high"]


def draw_subject_graph(
    G: nx.Graph,
    output_path: str | Path | None = None,
    title: str | None = None,
    show: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Draw a subject-level biological adaptation graph.

    Node size is proportional to domain activation.
    Edge width is proportional to edge weight.
    A guardrail note is embedded in the figure.

    Parameters
    ----------
    G:
        NetworkX graph built by ``build_subject_graph``.
    output_path:
        If provided, save the figure as a PNG at this path.
    title:
        Figure title. Defaults to ``Subject <ID>: Biological Adaptation Graph``.
    show:
        If True, call ``plt.show()``.

    Returns
    -------
    (fig, ax)
    """
    subject_id = G.graph.get("subject_id", "unknown")
    if title is None:
        title = f"Subject {subject_id}: Biological Adaptation Graph"

    fig, ax = plt.subplots(figsize=(10, 7))

    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No nodes in graph.", ha="center", va="center")
        ax.set_title(title)
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=120, bbox_inches="tight")
        if show:
            plt.show()
        return fig, ax

    # Layout
    pos = nx.spring_layout(G, seed=42, k=2.5)

    # Node styling
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        attrs = G.nodes[node]
        level = attrs.get("activation_level", "low")
        activation = float(attrs.get("activation", 0.5))
        node_colors.append(_LEVEL_COLORS.get(level, _DEFAULT_COLOR))
        # Size: base 800 + 600 per unit of activation, capped at 4000
        node_sizes.append(min(800 + 600 * activation, 4000))

    # Edge styling
    edge_widths = []
    edge_colors = []
    for u, v, attrs in G.edges(data=True):
        w = float(attrs.get("weight", 1.0))
        etype = attrs.get("edge_type", "")
        edge_widths.append(max(0.8, min(w * 2.5, 6.0)))
        if "coactivation" in etype:
            edge_colors.append("#E74C3C")  # red for co-activation
        else:
            edge_colors.append("#7F8C8D")  # grey for conceptual

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=edge_widths,
        edge_color=edge_colors,
        alpha=0.6,
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.92,
    )

    # Labels — shortened to fit
    labels = {
        n: "\n".join(n.split(" / ")[0].split()[:3])
        for n in G.nodes()
    }
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=8)

    ax.set_title(title, fontsize=13, pad=14)
    ax.axis("off")

    # Legend: activation levels
    patches = [
        mpatches.Patch(color=_LEVEL_COLORS[lvl], label=f"Activation: {lvl}")
        for lvl in _LEVEL_ORDER
        if lvl in {G.nodes[n].get("activation_level") for n in G.nodes()}
    ]
    if patches:
        ax.legend(handles=patches, loc="upper left", fontsize=8, framealpha=0.8)

    # Guardrail note
    fig.text(
        0.5, 0.01,
        (
            "Node size reflects domain activation. "
            "Edges are conceptual or co-activation links. Not diagnostic."
        ),
        ha="center", fontsize=8, color="#555555", style="italic",
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=120, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


def draw_graph_summary_bar(
    summary_df,
    output_path: str | Path | None = None,
    show: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Bar chart comparing subjects by max domain activation and active domain count.

    Parameters
    ----------
    summary_df:
        DataFrame returned by ``export_graph_summary``.
    output_path:
        Optional PNG save path.
    show:
        Whether to call plt.show().
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    subjects = summary_df["subject_id"].astype(str).tolist()
    x = range(len(subjects))

    # Max activation
    ax = axes[0]
    vals = summary_df["max_domain_activation"].astype(float)
    bars = ax.bar(x, vals, color="#AED6F1", edgecolor="grey")
    ax.set_xticks(list(x))
    ax.set_xticklabels(subjects, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Max domain activation (mean |z|)")
    ax.set_title(
        "Which pseudo-crew graphs show the\nstrongest domain activation?",
        fontsize=10,
    )
    ax.axhline(1.0, color="#E74C3C", linestyle="--", linewidth=1, label="moderate threshold")
    ax.legend(fontsize=8)

    # Active domains
    ax2 = axes[1]
    vals2 = summary_df["n_active_domains"].astype(int)
    ax2.bar(x, vals2, color="#A9DFBF", edgecolor="grey")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(subjects, rotation=20, ha="right", fontsize=9)
    ax2.set_ylabel("Number of active domains (activation ≥ 1.0)")
    ax2.set_title("Active biological domains\nper pseudo-crew member", fontsize=10)

    fig.suptitle(
        "Phase 3 — Biological Adaptation Graph Summary",
        fontsize=12, y=1.02,
    )

    fig.text(
        0.5, -0.04,
        "Research interpretation only. Not diagnosis or treatment guidance.",
        ha="center", fontsize=8, color="#555555", style="italic",
    )

    plt.tight_layout()

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=120, bbox_inches="tight")

    if show:
        plt.show()

    return fig, axes
