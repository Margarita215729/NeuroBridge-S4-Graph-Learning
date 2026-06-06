"""Phase 4 — Subgraph-level Feature Extraction.

Defines biologically meaningful subgraph templates and computes
per-participant activation statistics for each template.

Templates encode domain clusters that commonly co-participate in
human physiological adaptation (cardiometabolic, immune-metabolic, etc.).

The code is tolerant of missing nodes: if a template domain is absent
from the graph, it is skipped and the available-node count is reported.
No crashes on missing nodes or empty graphs.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import pandas as pd

# ---------------------------------------------------------------------------
# Subgraph template definitions
# Each entry: (template_name, list_of_canonical_domain_names)
# Domain names must match the canonical form used in graph_builder.py
# ---------------------------------------------------------------------------

SUBGRAPH_TEMPLATES: list[tuple[str, list[str]]] = [
    (
        "cardiometabolic",
        [
            "cardiovascular regulation",
            "metabolic regulation",
            "body composition / physical status",
        ],
    ),
    (
        "immune_metabolic",
        [
            "inflammation / immune-adjacent",
            "metabolic regulation",
            "recovery-related markers",
        ],
    ),
    (
        "hematologic_cardiovascular",
        [
            "hematologic / oxygen-carrying",
            "cardiovascular regulation",
            "recovery-related markers",
        ],
    ),
    (
        "sleep_autonomic_recovery",
        [
            "sleep / circadian regulation",
            "autonomic regulation",
            "recovery capacity",
        ],
    ),
    (
        "cognitive_emotional_recovery",
        [
            "cognitive load",
            "emotional regulation",
            "recovery capacity",
            "recovery-related markers",
        ],
    ),
]

_ACTIVE_THRESHOLD = 1.0


def _canonical(name: str) -> str:
    """Lowercase + strip for fuzzy domain matching."""
    return str(name).lower().strip()


def _match_node(node_domain: str, template_domain: str) -> bool:
    """Return True if node_domain matches template_domain (case-insensitive)."""
    return _canonical(node_domain).startswith(_canonical(template_domain)) or \
           _canonical(template_domain).startswith(_canonical(node_domain))


def extract_subgraph_features_for_template(
    G: nx.Graph,
    template_name: str,
    template_domains: list[str],
) -> dict[str, Any]:
    """Compute activation stats for one participant/template pair.

    Handles missing nodes gracefully — computes stats on available nodes only.
    """
    subject_id = G.graph.get("subject_id", "unknown")

    # Build domain → node mapping from the graph
    domain_to_node: dict[str, str] = {}
    for node, attrs in G.nodes(data=True):
        domain = attrs.get("domain", node)
        domain_to_node[_canonical(domain)] = node

    # Find available nodes for this template
    available_nodes: list[str] = []
    for td in template_domains:
        matched = None
        for graph_domain, node in domain_to_node.items():
            if _match_node(graph_domain, td):
                matched = node
                break
        if matched:
            available_nodes.append(matched)

    n_available = len(available_nodes)

    if n_available == 0:
        return {
            "subject_id":               subject_id,
            "subgraph_name":            template_name,
            "available_nodes":          0,
            "n_available_nodes":        0,
            "n_active_nodes":           0,
            "subgraph_activation_mean": float("nan"),
            "subgraph_activation_sum":  float("nan"),
            "subgraph_activation_max":  float("nan"),
            "subgraph_active_fraction": float("nan"),
            "dominant_node":            "n/a",
            "interpretation":           (
                f"No nodes matching the '{template_name}' template "
                "were found in this graph."
            ),
        }

    activations = [float(G.nodes[n].get("activation", 0.0)) for n in available_nodes]
    n_active = sum(1 for a in activations if a >= _ACTIVE_THRESHOLD)
    act_sum = sum(activations)
    act_mean = act_sum / n_available
    act_max = max(activations)
    active_fraction = n_active / n_available
    # dominant node = highest activation
    dom_idx = activations.index(act_max)
    dominant_node = G.nodes[available_nodes[dom_idx]].get("domain", available_nodes[dom_idx])

    if n_active == 0:
        interp = (
            f"The '{template_name}' subgraph shows low activation across all "
            f"{n_available} available domain(s)."
        )
    elif active_fraction >= 0.75:
        interp = (
            f"The '{template_name}' subgraph shows broad activation "
            f"({n_active}/{n_available} domains active)."
        )
    else:
        interp = (
            f"The '{template_name}' subgraph shows partial activation "
            f"({n_active}/{n_available} domains active); "
            f"'{dominant_node}' is dominant."
        )

    return {
        "subject_id":               subject_id,
        "subgraph_name":            template_name,
        "available_nodes":          n_available,
        "n_available_nodes":        n_available,
        "n_active_nodes":           n_active,
        "subgraph_activation_mean": round(act_mean, 5),
        "subgraph_activation_sum":  round(act_sum, 5),
        "subgraph_activation_max":  round(act_max, 5),
        "subgraph_active_fraction": round(active_fraction, 5),
        "dominant_node":            dominant_node,
        "interpretation":           interp,
    }


def extract_subgraph_features(G: nx.Graph) -> pd.DataFrame:
    """Compute subgraph features for all templates for one participant."""
    rows = [
        extract_subgraph_features_for_template(G, name, domains)
        for name, domains in SUBGRAPH_TEMPLATES
    ]
    return pd.DataFrame(rows)


def extract_all_subgraph_features(
    graphs: dict[str, nx.Graph],
) -> pd.DataFrame:
    """Compute subgraph features for all participants and all templates.

    Returns one row per (participant, template).
    """
    frames = [extract_subgraph_features(G) for G in graphs.values()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
