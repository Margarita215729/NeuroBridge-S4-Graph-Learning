"""Interactive HTML graph export for biological adaptation graphs.

Uses pyvis to generate self-contained HTML files that can be opened
locally without an internet connection.
"""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

import networkx as nx

try:
    from pyvis.network import Network as PyvisNetwork
    _PYVIS_AVAILABLE = True
except ImportError:
    _PYVIS_AVAILABLE = False


# Activation → pyvis node colour
_LEVEL_COLORS = {
    "low":      "#AED6F1",
    "mild":     "#A9DFBF",
    "moderate": "#F9E79F",
    "high":     "#F1948A",
}
_DEFAULT_COLOR = "#D5D8DC"

_EDGE_TYPE_COLORS = {
    "conceptual_biological_relationship": "#7F8C8D",
    "within_subject_coactivation":        "#E74C3C",
}
_DEFAULT_EDGE_COLOR = "#AAAAAA"

_GUARDRAIL = (
    "Research interpretation only. "
    "Not diagnosis or treatment guidance."
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _node_size_px(activation: float) -> int:
    """Map activation magnitude to pyvis node size (pixels)."""
    return max(15, min(int(10 + activation * 18), 60))


def _edge_width_px(weight: float) -> float:
    return max(1.0, min(weight * 3.0, 8.0))


def _build_node_title(attrs: dict) -> str:
    """HTML tooltip for a node."""
    lines = [
        f"<b>{attrs.get('domain', attrs.get('subject_id', ''))}</b>",
        f"Domain score: {attrs.get('domain_score', 'n/a')}",
        f"Activation: {attrs.get('activation', 'n/a')}",
        f"Activation level: {attrs.get('activation_level', 'n/a')}",
        f"<i>{attrs.get('interpretation', '')}</i>",
    ]
    return "<br>".join(lines)


def _build_edge_title(attrs: dict) -> str:
    """HTML tooltip for an edge."""
    lines = [
        f"<b>Edge type:</b> {attrs.get('edge_type', 'n/a')}",
        f"<b>Relationship:</b> {attrs.get('relationship', 'n/a')}",
        f"<b>Weight:</b> {attrs.get('weight', 'n/a')}",
        f"<i>{attrs.get('interpretation', '')}</i>",
    ]
    return "<br>".join(lines)


def _fallback_html(G: nx.Graph, output_path: Path, title: str) -> Path:
    """Generate a minimal hand-crafted HTML when pyvis is unavailable."""
    subject_id = G.graph.get("subject_id", "unknown")
    nodes_json = json.dumps(
        [
            {
                "id": n,
                "label": n,
                "title": _build_node_title(G.nodes[n]),
                "color": _LEVEL_COLORS.get(G.nodes[n].get("activation_level", ""), _DEFAULT_COLOR),
                "value": float(G.nodes[n].get("activation", 0.5)),
            }
            for n in G.nodes()
        ]
    )
    edges_json = json.dumps(
        [
            {
                "from": u,
                "to": v,
                "title": _build_edge_title(attrs),
                "color": _EDGE_TYPE_COLORS.get(attrs.get("edge_type", ""), _DEFAULT_EDGE_COLOR),
                "width": _edge_width_px(float(attrs.get("weight", 1.0))),
            }
            for u, v, attrs in G.edges(data=True)
        ]
    )

    html = textwrap.dedent(f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <title>{title}</title>
      <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
      <style>
        body {{ font-family: sans-serif; margin: 0; padding: 10px; background: #f9f9f9; }}
        #network {{ width: 100%; height: 580px; border: 1px solid #ddd; background: white; }}
        .guardrail {{ color: #888; font-size: 0.8em; margin-top: 6px; font-style: italic; }}
        h2 {{ font-size: 1.1em; margin-bottom: 4px; }}
      </style>
    </head>
    <body>
      <h2>{title}</h2>
      <p class="guardrail">{_GUARDRAIL}</p>
      <div id="network"></div>
      <script>
        var nodes = new vis.DataSet({nodes_json});
        var edges = new vis.DataSet({edges_json});
        var container = document.getElementById("network");
        var data = {{ nodes: nodes, edges: edges }};
        var options = {{
          nodes: {{ shape: "dot", scaling: {{ min: 15, max: 60 }} }},
          edges: {{ smooth: true }},
          physics: {{ stabilization: true }}
        }};
        new vis.Network(container, data, options);
      </script>
      <p class="guardrail">Note: Nodes are biological domains. Edges are conceptual or
      co-activation links. Node size reflects domain activation.</p>
    </body>
    </html>
    """).strip()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_interactive_graph_html(
    G: nx.Graph,
    output_path: str | Path,
    title: str | None = None,
) -> Path:
    """Export a subject-level graph as a self-contained interactive HTML file.

    Nodes are draggable; hovering shows full attribute tooltips.
    Uses pyvis when available, falls back to a vis-network CDN template.

    Parameters
    ----------
    G:
        NetworkX graph built by ``build_subject_graph``.
    output_path:
        Destination ``*.html`` path.
    title:
        Page title. Defaults to ``Subject <ID>: Biological Adaptation Graph``.

    Returns
    -------
    Path to the written HTML file.
    """
    out = Path(output_path)
    subject_id = G.graph.get("subject_id", "unknown")
    if title is None:
        title = f"Subject {subject_id}: Biological Adaptation Graph"

    if not _PYVIS_AVAILABLE:
        return _fallback_html(G, out, title)

    net = PyvisNetwork(
        height="600px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#333333",
        notebook=False,
        cdn_resources="in_line",   # embed JS inline → no internet needed
    )
    net.heading = title

    for node, attrs in G.nodes(data=True):
        activation = float(attrs.get("activation", 0.5))
        level = attrs.get("activation_level", "low")
        net.add_node(
            node,
            label=node,
            title=_build_node_title(attrs),
            color=_LEVEL_COLORS.get(level, _DEFAULT_COLOR),
            size=_node_size_px(activation),
        )

    for u, v, attrs in G.edges(data=True):
        etype = attrs.get("edge_type", "")
        net.add_edge(
            u, v,
            title=_build_edge_title(attrs),
            color=_EDGE_TYPE_COLORS.get(etype, _DEFAULT_EDGE_COLOR),
            width=_edge_width_px(float(attrs.get("weight", 1.0))),
        )

    net.set_options("""
    {
      "nodes": { "shape": "dot" },
      "edges": { "smooth": { "type": "continuous" } },
      "physics": { "stabilization": { "iterations": 150 } }
    }
    """)

    out.parent.mkdir(parents=True, exist_ok=True)
    net.save_graph(str(out))

    # Inject guardrail note after <body>
    raw = out.read_text(encoding="utf-8")
    guardrail_div = (
        f'<div style="font-family:sans-serif;font-size:0.82em;color:#888;'
        f'padding:6px 12px;font-style:italic;">{_GUARDRAIL}</div>\n'
    )
    raw = raw.replace("<body>", "<body>\n" + guardrail_div, 1)
    out.write_text(raw, encoding="utf-8")

    return out


def export_all_graphs_html(
    graphs: dict[str, nx.Graph],
    output_dir: str | Path,
) -> list[Path]:
    """Export one HTML file per subject graph.

    Returns list of written paths.
    """
    out_dir = Path(output_dir)
    paths = []
    for subject_id, G in graphs.items():
        safe_id = re.sub(r"[^\w\-]", "_", str(subject_id))
        html_path = out_dir / f"subject_graph_{safe_id}.html"
        export_interactive_graph_html(G, html_path)
        paths.append(html_path)
    return paths


def export_index_html(
    graph_paths: list[Path],
    output_path: str | Path,
    graphs: dict[str, nx.Graph] | None = None,
) -> Path:
    """Generate an index.html linking to all subject graph HTML files.

    Parameters
    ----------
    graph_paths:
        Paths to individual subject HTML files.
    output_path:
        Where to write index.html.
    graphs:
        Optional dict of graphs to extract summary metadata for the index.
    """
    out = Path(output_path)
    rows_html = ""
    for p in sorted(graph_paths):
        sid = p.stem.replace("subject_graph_", "")
        summary = ""
        if graphs and sid in graphs:
            G = graphs[sid]
            baci = G.graph.get("baci_score", "n/a")
            top = G.graph.get("top_domain", "n/a")
            n_active = G.graph.get("n_active_domains", "n/a")
            summary = f"BACI: {baci} | Top domain: {top} | Active domains: {n_active}"
        rows_html += (
            f'<tr><td><a href="{p.name}">{sid}</a></td>'
            f"<td>{summary}</td></tr>\n"
        )

    html = textwrap.dedent(f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <title>NeuroBridge-S4 Graph Learning — Interactive Graph Index</title>
      <style>
        body {{ font-family: sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; }}
        h1 {{ font-size: 1.3em; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
        th {{ background: #f2f2f2; }}
        .guardrail {{ color: #888; font-size: 0.85em; font-style: italic; margin-top: 16px; }}
        a {{ color: #2471A3; }}
      </style>
    </head>
    <body>
      <h1>NeuroBridge-S4 Graph Learning — Interactive Biological Adaptation Graphs</h1>
      <p>
        Each link below opens an interactive HTML graph for one pseudo-crew participant.
        Hover over nodes and edges to inspect attributes.
        Drag nodes to explore the graph layout.
      </p>
      <table>
        <thead><tr><th>Subject ID</th><th>Summary</th></tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
      <p class="guardrail">
        {_GUARDRAIL}<br>
        These graphs are research interpretation artifacts based on processed proxy data
        from NeuroBridge-S4. They do not represent actual Artemis II astronaut data.
      </p>
    </body>
    </html>
    """).strip()

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
