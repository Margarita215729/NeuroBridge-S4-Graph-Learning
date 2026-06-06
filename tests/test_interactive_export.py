"""Tests for interactive.py — Phase 3 HTML export."""

import networkx as nx
import pytest

from neurobridge_graph.interactive import (
    export_interactive_graph_html,
    export_all_graphs_html,
    export_index_html,
    _GUARDRAIL,
)
from neurobridge_graph.graph_builder import build_subject_graph
import pandas as pd


def _simple_graph(subject_id: str = "sub_test") -> nx.Graph:
    row = pd.Series({
        "Cardiovascular regulation": 0.48,
        "Metabolic regulation": 1.6,
        "Hematologic / oxygen-carrying capacity": 2.1,
    })
    return build_subject_graph(subject_id, row)


# ---------------------------------------------------------------------------
# export_interactive_graph_html
# ---------------------------------------------------------------------------

def test_export_html_creates_file(tmp_path):
    G = _simple_graph("sub_html")
    out = tmp_path / "graph.html"
    result = export_interactive_graph_html(G, out)
    assert result.exists()
    assert result.suffix == ".html"


def test_export_html_contains_guardrail(tmp_path):
    G = _simple_graph("sub_guardrail")
    out = tmp_path / "graph_gr.html"
    export_interactive_graph_html(G, out)
    content = out.read_text(encoding="utf-8")
    assert "Not diagnosis" in content or "research" in content.lower()


def test_export_html_contains_subject_id(tmp_path):
    G = _simple_graph("subject_xyz")
    out = tmp_path / "graph_sid.html"
    export_interactive_graph_html(G, out)
    content = out.read_text(encoding="utf-8")
    # _participant_label converts underscores to spaces
    assert "subject xyz" in content or "subject_xyz" in content


def test_export_html_nonempty(tmp_path):
    G = _simple_graph()
    out = tmp_path / "nonempty.html"
    export_interactive_graph_html(G, out)
    assert out.stat().st_size > 500


# ---------------------------------------------------------------------------
# export_all_graphs_html
# ---------------------------------------------------------------------------

def test_export_all_graphs_html_count(tmp_path):
    graphs = {
        "sub_a": _simple_graph("sub_a"),
        "sub_b": _simple_graph("sub_b"),
    }
    paths = export_all_graphs_html(graphs, tmp_path / "html")
    assert len(paths) == 2
    for p in paths:
        assert p.exists()


# ---------------------------------------------------------------------------
# export_index_html
# ---------------------------------------------------------------------------

def test_export_index_html_creates_file(tmp_path):
    graphs = {"sub_a": _simple_graph("sub_a")}
    html_dir = tmp_path / "html"
    html_dir.mkdir()
    graph_paths = export_all_graphs_html(graphs, html_dir)
    index_path = export_index_html(graph_paths, html_dir / "index.html", graphs=graphs)
    assert index_path.exists()


def test_export_index_html_links_subjects(tmp_path):
    graphs = {"sub_idx": _simple_graph("sub_idx")}
    html_dir = tmp_path / "html"
    html_dir.mkdir()
    graph_paths = export_all_graphs_html(graphs, html_dir)
    index_path = export_index_html(graph_paths, html_dir / "index.html", graphs=graphs)
    content = index_path.read_text(encoding="utf-8")
    assert "sub_idx" in content


def test_export_index_html_contains_guardrail(tmp_path):
    graphs = {"sub_g": _simple_graph("sub_g")}
    html_dir = tmp_path / "html"
    html_dir.mkdir()
    paths = export_all_graphs_html(graphs, html_dir)
    index = export_index_html(paths, html_dir / "index.html")
    content = index.read_text(encoding="utf-8")
    assert "Not diagnosis" in content or "research" in content.lower()
