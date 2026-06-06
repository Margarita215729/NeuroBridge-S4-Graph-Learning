"""Phase 1 placeholder graph builder utilities.

Full graph construction will be implemented in Phase 3. This module currently
provides schema-based functions that can be imported and tested.
"""

from __future__ import annotations

import networkx as nx

from .schema import DOMAIN_NODES, CONCEPTUAL_EDGES


def build_empty_schema_graph() -> nx.Graph:
    """Build a conceptual schema graph without participant data.

    Returns
    -------
    networkx.Graph
        Graph containing the Phase 1 domain nodes and conceptual edges.
    """
    graph = nx.Graph()
    for node in DOMAIN_NODES:
        graph.add_node(node, node_type="domain", phase="schema")
    for source, target in CONCEPTUAL_EDGES:
        graph.add_edge(
            source,
            target,
            edge_type="conceptual",
            weight=1.0,
            source="schema",
            confidence="draft",
        )
    return graph
