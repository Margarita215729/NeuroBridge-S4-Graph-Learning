"""Phase 4 — Interpretable Graph Feature Extraction.

Converts each biological adaptation graph (NetworkX) produced in Phase 3
into tabular feature sets suitable for comparison, similarity mapping,
and downstream graph-learning experiments.

Three levels:
  - graph_level  : one row per participant
  - node_level   : one row per (participant, domain)
  - edge_level   : one row per (participant, edge)

All features are interpretable without machine-learning knowledge.
No diagnostic claims are made or implied.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

# A domain is "active" when its activation crosses this threshold.
ACTIVE_THRESHOLD: float = 1.0

_GUARDRAIL = (
    "Research interpretation only. "
    "Not diagnosis or treatment guidance."
)


# ---------------------------------------------------------------------------
# Graph-level features
# ---------------------------------------------------------------------------

def extract_graph_level_features(G: nx.Graph) -> dict[str, Any]:
    """Return a flat dict of graph-level features for one participant.

    Parameters
    ----------
    G:
        NetworkX graph produced by ``build_subject_graph``.

    Returns
    -------
    dict with keys described in the module docstring.
    """
    g_attrs = G.graph

    # Collect numeric activations from node attrs
    activations = []
    top_domain = None
    top_activation = 0.0
    for node, attrs in G.nodes(data=True):
        act = float(attrs.get("activation", 0.0))
        activations.append((node, act))
        if act > top_activation:
            top_activation = act
            top_domain = attrs.get("domain", node)

    act_values = [a for _, a in activations]
    n_nodes = len(act_values)
    n_active = sum(1 for a in act_values if a >= ACTIVE_THRESHOLD)
    active_fraction = n_active / n_nodes if n_nodes > 0 else 0.0

    # Edge stats
    weights = [float(d.get("weight", 1.0)) for _, _, d in G.edges(data=True)]
    n_edges = len(weights)
    conceptual_count = sum(
        1 for _, _, d in G.edges(data=True)
        if d.get("edge_type", "") == "conceptual_biological_relationship"
    )
    coact_count = sum(
        1 for _, _, d in G.edges(data=True)
        if d.get("edge_type", "") == "within_subject_coactivation"
    )
    n_nodes_g = G.number_of_nodes()
    density = nx.density(G) if n_nodes_g >= 2 else 0.0

    return {
        "subject_id":             g_attrs.get("subject_id", "unknown"),
        "n_nodes":                n_nodes,
        "n_edges":                n_edges,
        "graph_density":          round(density, 5),
        "mean_node_activation":   round(sum(act_values) / n_nodes, 5) if n_nodes else 0.0,
        "median_node_activation": round(sorted(act_values)[n_nodes // 2], 5) if n_nodes else 0.0,
        "max_node_activation":    round(max(act_values, default=0.0), 5),
        "total_node_activation":  round(sum(act_values), 5),
        "n_active_domains":       n_active,
        "active_domain_fraction": round(active_fraction, 5),
        "mean_edge_weight":       round(sum(weights) / n_edges, 5) if n_edges else 0.0,
        "max_edge_weight":        round(max(weights, default=0.0), 5),
        "conceptual_edge_count":  conceptual_count,
        "coactivation_edge_count": coact_count,
        "top_domain":             top_domain or "n/a",
        "top_domain_activation":  round(top_activation, 5),
        "baci_score":             g_attrs.get("baci_score", "n/a"),
        "baci_category":          g_attrs.get("baci_category", "n/a"),
    }


def extract_all_graph_level_features(
    graphs: dict[str, nx.Graph],
) -> pd.DataFrame:
    """Extract graph-level features for every participant graph.

    Parameters
    ----------
    graphs:
        Dict mapping subject_id → NetworkX graph.

    Returns
    -------
    DataFrame with one row per participant.
    """
    rows = [extract_graph_level_features(G) for G in graphs.values()]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Node-level features
# ---------------------------------------------------------------------------

def extract_node_level_features(G: nx.Graph) -> pd.DataFrame:
    """Return a DataFrame of node-level features for one participant.

    Features include centrality, activation level, and interpretive flags.
    """
    subject_id = G.graph.get("subject_id", "unknown")
    g_attrs = G.graph

    # Pre-compute centralities (NetworkX handles empty-graph edge cases)
    if G.number_of_nodes() >= 2:
        deg_centrality: dict[str, float] = nx.degree_centrality(G)
        # weighted degree = sum of incident edge weights
        wdeg: dict[str, float] = {
            n: sum(d.get("weight", 1.0) for _, _, d in G.edges(n, data=True))
            for n in G.nodes()
        }
    else:
        deg_centrality = {n: 0.0 for n in G.nodes()}
        wdeg = {n: 0.0 for n in G.nodes()}

    # Identify top domain
    top_domain = g_attrs.get("top_domain", None)
    if top_domain is None:
        # derive it
        best_act = -1.0
        for n, attrs in G.nodes(data=True):
            a = float(attrs.get("activation", 0.0))
            if a > best_act:
                best_act = a
                top_domain = attrs.get("domain", n)

    rows = []
    for node, attrs in G.nodes(data=True):
        activation = float(attrs.get("activation", 0.0))
        domain = attrs.get("domain", node)
        rows.append({
            "subject_id":         subject_id,
            "domain":             domain,
            "activation":         round(activation, 5),
            "activation_level":   attrs.get("activation_level", "n/a"),
            "domain_score":       round(float(attrs.get("domain_score", activation)), 5),
            "degree":             G.degree(node),
            "weighted_degree":    round(float(wdeg.get(node, 0.0)), 5),
            "degree_centrality":  round(float(deg_centrality.get(node, 0.0)), 5),
            "is_active":          activation >= ACTIVE_THRESHOLD,
            "is_top_domain":      domain == top_domain,
            "interpretation":     attrs.get("interpretation", ""),
        })
    return pd.DataFrame(rows)


def extract_all_node_level_features(
    graphs: dict[str, nx.Graph],
) -> pd.DataFrame:
    """Extract node-level features for every participant."""
    frames = [extract_node_level_features(G) for G in graphs.values()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Edge-level features
# ---------------------------------------------------------------------------

def extract_edge_level_features(G: nx.Graph) -> pd.DataFrame:
    """Return a DataFrame of edge-level features for one participant."""
    subject_id = G.graph.get("subject_id", "unknown")

    # Build activation lookup
    act_lookup = {
        node: float(attrs.get("activation", 0.0))
        for node, attrs in G.nodes(data=True)
    }

    rows = []
    for u, v, attrs in G.edges(data=True):
        both_active = (
            act_lookup.get(u, 0.0) >= ACTIVE_THRESHOLD
            and act_lookup.get(v, 0.0) >= ACTIVE_THRESHOLD
        )
        rows.append({
            "subject_id":            subject_id,
            "source":                u,
            "target":                v,
            "edge_type":             attrs.get("edge_type", "n/a"),
            "weight":                round(float(attrs.get("weight", 1.0)), 5),
            "connects_active_domains": both_active,
            "relationship":          attrs.get("relationship", ""),
            "interpretation":        attrs.get(
                "interpretation",
                "Conceptual relationship, not causal proof.",
            ),
        })
    return pd.DataFrame(rows)


def extract_all_edge_level_features(
    graphs: dict[str, nx.Graph],
) -> pd.DataFrame:
    """Extract edge-level features for every participant."""
    frames = [extract_edge_level_features(G) for G in graphs.values()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
