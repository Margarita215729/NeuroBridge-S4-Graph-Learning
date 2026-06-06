from neurobridge_graph.graph_builder import build_empty_schema_graph
from neurobridge_graph.schema import DOMAIN_NODES


def test_schema_graph_contains_domain_nodes():
    graph = build_empty_schema_graph()
    for node in DOMAIN_NODES:
        assert node in graph.nodes


def test_schema_graph_has_edges():
    graph = build_empty_schema_graph()
    assert graph.number_of_edges() > 0
