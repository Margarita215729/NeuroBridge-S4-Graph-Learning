"""Tests for interactive.py — Phase 3 HTML export."""

import networkx as nx
import pytest

from neurobridge_graph.interactive import (
    export_interactive_graph_html,
    export_all_graphs_html,
    export_index_html,
    create_interactive_index,
    validate_html_graph_output,
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


# ---------------------------------------------------------------------------
# Custom vis-network renderer guarantees
# ---------------------------------------------------------------------------

def test_export_uses_vis_network_not_pyvis(tmp_path):
    G = _simple_graph("sub_vis")
    out = tmp_path / "vis.html"
    export_interactive_graph_html(G, out)
    content = out.read_text(encoding="utf-8")
    assert "new vis.Network(" in content
    # JSON-serialized node/edge arrays drive the render (not pyvis inline blob).
    assert "NB_NODES" in content and "NB_EDGES" in content


def test_export_has_custom_tooltip_div(tmp_path):
    G = _simple_graph("sub_tip")
    out = tmp_path / "tip.html"
    export_interactive_graph_html(G, out)
    content = out.read_text(encoding="utf-8")
    assert 'id="nb-tooltip"' in content
    # Tooltip is populated via textContent, never innerHTML.
    assert "textContent" in content
    assert ".innerHTML" not in content


def test_export_no_raw_br_or_html_tags(tmp_path):
    G = _simple_graph("sub_clean")
    out = tmp_path / "clean.html"
    export_interactive_graph_html(G, out)
    content = out.read_text(encoding="utf-8")
    assert "<br>" not in content
    assert "&lt;br&gt;" not in content
    # No bold/italic tags leaking into the page.
    assert "<b>" not in content and "<i>" not in content


def test_export_tooltip_uses_newlines_in_data(tmp_path):
    G = _simple_graph("sub_nl")
    out = tmp_path / "nl.html"
    export_interactive_graph_html(G, out)
    content = out.read_text(encoding="utf-8")
    # Node tooltips are stored as plain text with escaped newlines in the JSON.
    assert "Domain: " in content
    assert "Guardrail: " in content
    assert "\\nDomain score:" in content


def test_export_node_label_line_break_for_slash_domains(tmp_path):
    G = _simple_graph("sub_label")
    out = tmp_path / "label.html"
    export_interactive_graph_html(G, out)
    content = out.read_text(encoding="utf-8")
    # "Hematologic / oxygen-carrying capacity" -> two-line label with \n.
    assert "Hematologic /\\noxygen-carrying" in content


def test_export_navigation_buttons_disabled(tmp_path):
    G = _simple_graph("sub_nav")
    out = tmp_path / "nav.html"
    export_interactive_graph_html(G, out)
    content = out.read_text(encoding="utf-8")
    assert '"navigationButtons": false' in content


# ---------------------------------------------------------------------------
# create_interactive_index
# ---------------------------------------------------------------------------

def _summary_df(subject_ids):
    return pd.DataFrame([
        {
            "subject_id": sid,
            "top_domain": "Metabolic regulation",
            "max_domain_activation": 1.6,
            "baci_score": 30.0,
            "baci_category": "mild coherence",
        }
        for sid in subject_ids
    ])


def test_create_interactive_index_creates_file(tmp_path):
    graphs = {"sub_a": _simple_graph("sub_a")}
    html_dir = tmp_path / "html"
    paths = export_all_graphs_html(graphs, html_dir)
    idx = create_interactive_index(paths, _summary_df(["sub_a"]), html_dir / "index.html")
    assert idx.exists()


def test_create_interactive_index_links_and_table(tmp_path):
    graphs = {"sub_a": _simple_graph("sub_a"), "sub_b": _simple_graph("sub_b")}
    html_dir = tmp_path / "html"
    paths = export_all_graphs_html(graphs, html_dir)
    idx = create_interactive_index(paths, _summary_df(["sub_a", "sub_b"]),
                                   html_dir / "index.html")
    content = idx.read_text(encoding="utf-8")
    # Links to each subject HTML file.
    assert 'href="subject_graph_sub_a.html"' in content
    assert 'href="subject_graph_sub_b.html"' in content
    # Summary columns are rendered.
    assert "Top domain" in content and "BACI score" in content
    assert "mild coherence" in content


def test_create_interactive_index_notes_cdn_and_png(tmp_path):
    graphs = {"sub_a": _simple_graph("sub_a")}
    html_dir = tmp_path / "html"
    paths = export_all_graphs_html(graphs, html_dir)
    idx = create_interactive_index(paths, _summary_df(["sub_a"]), html_dir / "index.html")
    content = idx.read_text(encoding="utf-8")
    assert "CDN" in content and "internet" in content.lower()
    assert "PNG" in content


def test_create_interactive_index_without_summary(tmp_path):
    graphs = {"sub_a": _simple_graph("sub_a")}
    html_dir = tmp_path / "html"
    paths = export_all_graphs_html(graphs, html_dir)
    idx = create_interactive_index(paths, None, html_dir / "index.html")
    content = idx.read_text(encoding="utf-8")
    # Falls back to a plain link list when no summary is supplied.
    assert 'href="subject_graph_sub_a.html"' in content


# ---------------------------------------------------------------------------
# validate_html_graph_output
# ---------------------------------------------------------------------------

def test_validate_passes_on_real_export(tmp_path):
    G = _simple_graph("sub_valid")
    out = tmp_path / "valid.html"
    export_interactive_graph_html(G, out)
    v = validate_html_graph_output(out)
    assert v["passed"] is True
    assert v["exists"] and v["size_ok"]
    assert v["has_vis_init"] and v["has_node_data"] and v["has_edge_data"]
    assert v["has_custom_tooltip"] and v["clean_tooltips"]
    assert v["notes"] == []


def test_validate_missing_file(tmp_path):
    v = validate_html_graph_output(tmp_path / "does_not_exist.html")
    assert v["passed"] is False
    assert v["exists"] is False
    assert "file not found" in v["notes"]


def test_validate_detects_raw_br_in_data(tmp_path):
    # Hand-craft a file that injects a raw <br> into the node data block.
    bad = (
        '<html><body><div id="nb-tooltip"></div><script>'
        'var NB_NODES = [{"label":"x","tip":"a<br>b"}];'
        'var NB_EDGES = [{"from":"x","to":"y"}];'
        'new vis.Network(c,d,o); el.textContent = "x";'
        'Pseudo-crew participant'
        '</script></body></html>'
    ) + ("<!-- padding -->" * 800)
    p = tmp_path / "bad.html"
    p.write_text(bad, encoding="utf-8")
    v = validate_html_graph_output(p)
    assert v["clean_tooltips"] is False
    assert v["passed"] is False


def test_validate_size_floor(tmp_path):
    p = tmp_path / "tiny.html"
    p.write_text("<html></html>", encoding="utf-8")
    v = validate_html_graph_output(p)
    assert v["size_ok"] is False
    assert v["passed"] is False
