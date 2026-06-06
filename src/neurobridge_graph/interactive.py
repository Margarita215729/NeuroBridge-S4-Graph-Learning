"""Interactive HTML graph export for biological adaptation graphs.

Produces polished, reviewer-facing HTML pages with:
- clean single title (no duplication);
- plain-text tooltips (no raw HTML tags visible);
- activation-aware node colours and sizes;
- visible legend panel;
- guardrail note on every page;
- improved graph canvas size and physics.

Uses pyvis when available; falls back to a vis-network CDN template.
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


# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------

# Activation level → node fill colour (professional, restrained palette)
_LEVEL_COLORS: dict[str, str] = {
    "low":      "#AED6F1",   # muted blue
    "mild":     "#A9DFBF",   # muted green
    "moderate": "#F5CBA7",   # soft orange
    "high":     "#F1948A",   # coral/red
}
_DEFAULT_COLOR = "#D5D8DC"

_EDGE_TYPE_COLORS: dict[str, str] = {
    "conceptual_biological_relationship": "#7F8C8D",   # grey
    "within_subject_coactivation":        "#CB4335",   # deep red
}
_DEFAULT_EDGE_COLOR = "#AAAAAA"

_GUARDRAIL = (
    "Research interpretation only. "
    "Not diagnosis or treatment guidance."
)

# Inline CSS shared across individual graph pages
_PAGE_CSS = """
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    margin: 0; padding: 0; background: #f5f6fa; color: #222;
  }
  .page-header {
    background: #ffffff; border-bottom: 1px solid #dde; padding: 14px 24px 10px;
  }
  .page-header h1 { margin: 0 0 2px; font-size: 1.15em; font-weight: 600; }
  .page-header .subtitle {
    font-size: 0.85em; color: #555; margin: 0 0 6px;
  }
  .guardrail-banner {
    background: #fef9e7; border: 1px solid #f9ca24; border-radius: 4px;
    padding: 5px 12px; font-size: 0.8em; color: #7d6608; margin-top: 6px;
    display: inline-block;
  }
  .content-wrap { display: flex; gap: 12px; padding: 12px 18px; }
  .graph-col { flex: 1 1 auto; min-width: 0; }
  #network-container {
    width: 100%; height: 680px; border: 1px solid #dde;
    background: #ffffff; border-radius: 6px; overflow: hidden;
  }
  .legend-col {
    flex: 0 0 200px; background: #ffffff; border: 1px solid #dde;
    border-radius: 6px; padding: 12px 14px; font-size: 0.82em;
    align-self: flex-start;
  }
  .legend-col h3 { margin: 0 0 8px; font-size: 0.9em; font-weight: 600; }
  .legend-item { display: flex; align-items: center; gap: 7px; margin-bottom: 6px; }
  .legend-dot {
    width: 14px; height: 14px; border-radius: 50%; flex-shrink: 0;
    border: 1px solid #bbb;
  }
  .legend-line {
    width: 20px; height: 3px; flex-shrink: 0;
  }
  .legend-section { margin-top: 10px; font-weight: 600; color: #555; font-size: 0.78em;
    text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 4px; }
  .reviewer-note {
    padding: 8px 18px 14px; font-size: 0.82em; color: #666; font-style: italic;
  }
"""

# Legend HTML block reused across pages
_LEGEND_HTML = """
  <div class="legend-col">
    <h3>Legend</h3>
    <div class="legend-section">Activation level</div>
    <div class="legend-item">
      <div class="legend-dot" style="background:#AED6F1;"></div><span>Low (&lt;&nbsp;0.75)</span>
    </div>
    <div class="legend-item">
      <div class="legend-dot" style="background:#A9DFBF;"></div><span>Mild (0.75&ndash;1.0)</span>
    </div>
    <div class="legend-item">
      <div class="legend-dot" style="background:#F5CBA7;"></div><span>Moderate (1.0&ndash;1.5)</span>
    </div>
    <div class="legend-item">
      <div class="legend-dot" style="background:#F1948A;"></div><span>High (&ge;&nbsp;1.5)</span>
    </div>
    <div class="legend-section">Node size</div>
    <div class="legend-item" style="font-style:italic;color:#555;">
      Larger = higher activation
    </div>
    <div class="legend-section">Edge type</div>
    <div class="legend-item">
      <div class="legend-line" style="background:#7F8C8D;"></div>
      <span>Conceptual relationship</span>
    </div>
    <div class="legend-item">
      <div class="legend-line" style="background:#CB4335;"></div>
      <span>Co-activation</span>
    </div>
    <div style="margin-top:12px;font-size:0.78em;color:#888;">
      Hover nodes &amp; edges to inspect attributes.<br>
      Drag to rearrange.
    </div>
    <div style="margin-top:10px;font-size:0.75em;color:#999;font-style:italic;">
      Not diagnostic.
    </div>
  </div>
"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _participant_label(subject_id: str) -> str:
    """Convert 'Crew 97774' / 'Crew_97774' → 'Pseudo-crew participant 97774'."""
    sid = str(subject_id).replace("_", " ").strip()
    num = re.sub(r"(?i)^crew\s*", "", sid).strip()
    return f"Pseudo-crew participant {num}" if num else sid


def _node_size(activation: float) -> int:
    """Map activation to pyvis node size using the spec formula."""
    return int(18 + 35 * min(float(activation), 2.5) / 2.5)


def _edge_width(weight: float) -> float:
    return max(1.0, min(float(weight) * 2.5, 7.0))


def _fmt(value: object, decimals: int = 3) -> str:
    """Format a numeric value or return 'n/a'."""
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return str(value) if value not in (None, "") else "n/a"


def _node_tooltip_plain(attrs: dict) -> str:
    """Plain-text tooltip — no raw HTML tags, safe for pyvis title attribute."""
    domain = attrs.get("domain", attrs.get("subject_id", ""))
    score = _fmt(attrs.get("domain_score"))
    activation = _fmt(attrs.get("activation"))
    level = attrs.get("activation_level", "n/a")
    interp = attrs.get("interpretation", "")
    lines = [
        f"Domain: {domain}",
        f"Domain score: {score}",
        f"Activation: {activation}",
        f"Activation level: {level}",
    ]
    if interp:
        lines.append(f"Interpretation: {interp}")
    lines.append(f"Guardrail: {_GUARDRAIL}")
    return "\n".join(lines)


def _edge_tooltip_plain(attrs: dict) -> str:
    """Plain-text tooltip for edges."""
    etype = attrs.get("edge_type", "n/a")
    rel = attrs.get("relationship", "n/a")
    weight = _fmt(attrs.get("weight"), 2)
    interp = attrs.get("interpretation", "Conceptual relationship, not causal proof.")
    return (
        f"Edge type: {etype}\n"
        f"Relationship: {rel}\n"
        f"Weight: {weight}\n"
        f"Interpretation: {interp}"
    )


def _page_html(
    network_body: str,
    title: str,
    participant_label: str,
    subject_id: str,
    extra_script: str = "",
) -> str:
    """Wrap network HTML body in the full polished page template."""
    return textwrap.dedent(f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
{_PAGE_CSS}
  </style>
</head>
<body>
  <div class="page-header">
    <h1>{title}</h1>
    <div class="subtitle">
      Interactive graph of reference-relative biological domain activation
      &mdash; {participant_label}
    </div>
    <div class="guardrail-banner">
      &#9888;&nbsp;{_GUARDRAIL}
    </div>
  </div>
  <div class="content-wrap">
    <div class="graph-col">
      <div id="network-container">
{network_body}
      </div>
    </div>
{_LEGEND_HTML}
  </div>
  <div class="reviewer-note">
    This graph represents {participant_label} as a connected biological system.
    Nodes are biological domains; edges are conceptual or co-activation links.
    Node size reflects activation magnitude. Neither edges nor node size imply causality.
  </div>
{extra_script}
</body>
</html>""")


def _fallback_html(G: nx.Graph, output_path: Path, title: str) -> Path:
    """Full polished page using vis-network CDN (no pyvis dependency)."""
    subject_id = str(G.graph.get("subject_id", "unknown"))
    plabel = _participant_label(subject_id)

    nodes_data = [
        {
            "id": n,
            "label": n,
            "title": _node_tooltip_plain(G.nodes[n]),
            "color": {"background": _LEVEL_COLORS.get(G.nodes[n].get("activation_level", ""), _DEFAULT_COLOR),
                      "border": "#999"},
            "size": _node_size(float(G.nodes[n].get("activation", 0.5))),
            "font": {"size": 13, "color": "#222"},
        }
        for n in G.nodes()
    ]
    edges_data = [
        {
            "from": u,
            "to": v,
            "title": _edge_tooltip_plain(attrs),
            "color": {"color": _EDGE_TYPE_COLORS.get(attrs.get("edge_type", ""), _DEFAULT_EDGE_COLOR)},
            "width": _edge_width(float(attrs.get("weight", 1.0))),
            "smooth": {"type": "continuous"},
        }
        for u, v, attrs in G.edges(data=True)
    ]

    nodes_json = json.dumps(nodes_data)
    edges_json = json.dumps(edges_data)

    network_body = f"""
        <div id="vis-graph" style="width:100%;height:680px;"></div>
        <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <script>
          (function() {{
            var nodes = new vis.DataSet({nodes_json});
            var edges = new vis.DataSet({edges_json});
            var container = document.getElementById("vis-graph");
            var options = {{
              nodes: {{ shape: "dot", borderWidth: 1.5 }},
              edges: {{ arrows: {{ to: false }} }},
              physics: {{
                barnesHut: {{
                  gravitationalConstant: -3000,
                  centralGravity: 0.4,
                  springLength: 110,
                  springConstant: 0.04,
                  damping: 0.15
                }},
                stabilization: {{ iterations: 200 }}
              }},
              interaction: {{ hover: true, tooltipDelay: 150 }}
            }};
            new vis.Network(container, {{ nodes: nodes, edges: edges }}, options);
          }})();
        </script>"""

    html = _page_html(network_body, title, plabel, subject_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _build_pyvis_network(G: nx.Graph) -> "PyvisNetwork":
    """Construct a pyvis Network from a subject graph, with plain-text tooltips."""
    net = PyvisNetwork(
        height="680px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#222222",
        notebook=False,
        cdn_resources="in_line",
    )
    # Suppress pyvis's own heading; we render our own title in the wrapper
    net.heading = ""

    for node, attrs in G.nodes(data=True):
        activation = float(attrs.get("activation", 0.5))
        level = attrs.get("activation_level", "low")
        net.add_node(
            node,
            label=node,
            title=_node_tooltip_plain(attrs),
            color={"background": _LEVEL_COLORS.get(level, _DEFAULT_COLOR), "border": "#999"},
            size=_node_size(activation),
            font={"size": 13, "color": "#222"},
        )

    for u, v, attrs in G.edges(data=True):
        etype = attrs.get("edge_type", "")
        net.add_edge(
            u, v,
            title=_edge_tooltip_plain(attrs),
            color={"color": _EDGE_TYPE_COLORS.get(etype, _DEFAULT_EDGE_COLOR)},
            width=_edge_width(float(attrs.get("weight", 1.0))),
            smooth={"type": "continuous"},
        )

    net.set_options(json.dumps({
        "nodes": {"shape": "dot", "borderWidth": 1.5},
        "edges": {"arrows": {"to": False}},
        "physics": {
            "barnesHut": {
                "gravitationalConstant": -3000,
                "centralGravity": 0.4,
                "springLength": 110,
                "springConstant": 0.04,
                "damping": 0.15,
            },
            "stabilization": {"iterations": 200},
        },
        "interaction": {"hover": True, "tooltipDelay": 150},
    }))
    return net


def _extract_pyvis_body(raw_html: str) -> str:
    """Extract only the inner <body>...</body> content from pyvis output."""
    m = re.search(r"<body[^>]*>(.*?)</body>", raw_html, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1)
    # fallback: return everything
    return raw_html


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_interactive_graph_html(
    G: nx.Graph,
    output_path: str | Path,
    title: str | None = None,
) -> Path:
    """Export a subject-level graph as a polished, self-contained HTML page.

    - Single clean title (no duplication).
    - Plain-text tooltips (no raw HTML tags visible to reader).
    - Activation-aware colours and node sizes.
    - Legend panel on every page.
    - Guardrail note on every page.

    Uses pyvis when available; falls back to vis-network CDN template.
    """
    out = Path(output_path)
    subject_id = str(G.graph.get("subject_id", "unknown"))
    plabel = _participant_label(subject_id)

    if title is None:
        title = f"{plabel}: Biological Adaptation Graph"

    if not _PYVIS_AVAILABLE:
        return _fallback_html(G, out, title)

    # --- pyvis path ---
    import tempfile
    net = _build_pyvis_network(G)

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    net.save_graph(str(tmp_path))
    raw = tmp_path.read_text(encoding="utf-8")
    tmp_path.unlink(missing_ok=True)

    # Extract only the body content (avoids duplicate <html>/<head> in wrapper)
    body_content = _extract_pyvis_body(raw)
    # Strip pyvis's own heading element if it snuck in
    body_content = re.sub(
        r'<center>\s*<h1[^>]*>.*?</h1>\s*</center>',
        "",
        body_content,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # Wrap the graph div so it fills the container
    network_body = (
        '<div style="width:100%;height:680px;overflow:hidden;">\n'
        + body_content
        + "\n</div>"
    )

    html = _page_html(network_body, title, plabel, subject_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def export_all_graphs_html(
    graphs: dict[str, nx.Graph],
    output_dir: str | Path,
) -> list[Path]:
    """Export one polished HTML page per subject graph.

    Returns list of written paths. Continues on per-subject errors.
    """
    out_dir = Path(output_dir)
    paths: list[Path] = []
    for subject_id, G in graphs.items():
        safe_id = re.sub(r"[^\w\-]", "_", str(subject_id))
        html_path = out_dir / f"subject_graph_{safe_id}.html"
        try:
            export_interactive_graph_html(G, html_path)
            paths.append(html_path)
        except Exception as exc:  # noqa: BLE001
            print(f"  [interactive] Warning: could not export {subject_id}: {exc}")
    return paths


def export_index_html(
    graph_paths: list[Path],
    output_path: str | Path,
    graphs: dict[str, nx.Graph] | None = None,
) -> Path:
    """Generate an index.html linking to all subject graph HTML files.

    Includes project title, explanation, summary table with participant
    metadata, and guardrail note.
    """
    out = Path(output_path)

    # Build table rows
    rows_html = ""
    for p in sorted(graph_paths):
        safe_id = p.stem.replace("subject_graph_", "")
        plabel = _participant_label(safe_id)
        top_domain = baci = baci_cat = max_act = n_active = "n/a"
        if graphs:
            # Try original key first, then de-underscored variant
            G = graphs.get(safe_id) or graphs.get(safe_id.replace("_", " "))
            if G:
                baci = _fmt(G.graph.get("baci_score"), 3)
                baci_cat = G.graph.get("baci_category", "n/a")
                top_domain = G.graph.get("top_domain", "n/a")
                max_act = _fmt(G.graph.get("max_domain_activation"), 3)
                n_active = str(G.graph.get("n_active_domains", "n/a"))
        rows_html += (
            f'<tr>'
            f'<td><a href="{p.name}">{plabel}</a></td>'
            f"<td>{top_domain}</td>"
            f"<td>{max_act}</td>"
            f"<td>{baci}</td>"
            f"<td>{baci_cat}</td>"
            f'</tr>\n'
        )

    html = textwrap.dedent(f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NeuroBridge-S4 Graph Learning &mdash; Interactive Biological Adaptation Graphs</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
      max-width: 960px; margin: 40px auto; padding: 0 24px;
      background: #f5f6fa; color: #222;
    }}
    .header {{ background: #fff; border: 1px solid #dde; border-radius: 6px;
               padding: 20px 24px 16px; margin-bottom: 20px; }}
    .header h1 {{ margin: 0 0 6px; font-size: 1.25em; }}
    .header p {{ margin: 0 0 8px; font-size: 0.9em; color: #444; }}
    .guardrail-banner {{
      background: #fef9e7; border: 1px solid #f9ca24; border-radius: 4px;
      padding: 6px 14px; font-size: 0.82em; color: #7d6608; display: inline-block;
    }}
    .legend-bar {{
      display: flex; gap: 14px; flex-wrap: wrap; align-items: center;
      background: #fff; border: 1px solid #dde; border-radius: 6px;
      padding: 10px 16px; margin-bottom: 16px; font-size: 0.83em;
    }}
    .legend-dot {{
      width: 12px; height: 12px; border-radius: 50%; display: inline-block;
      border: 1px solid #bbb; margin-right: 4px; vertical-align: middle;
    }}
    table {{ border-collapse: collapse; width: 100%; background: #fff;
             border: 1px solid #dde; border-radius: 6px; overflow: hidden; }}
    th, td {{ border-bottom: 1px solid #eee; padding: 9px 14px; text-align: left;
              font-size: 0.88em; }}
    th {{ background: #f2f2f2; font-weight: 600; }}
    a {{ color: #2471A3; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .footer {{ margin-top: 18px; font-size: 0.8em; color: #888; font-style: italic; }}
  </style>
</head>
<body>
  <div class="header">
    <h1>NeuroBridge-S4 Graph Learning &mdash; Interactive Biological Adaptation Graphs</h1>
    <p>
      Each row links to an interactive HTML graph for one pseudo-crew participant.
      These graphs encode reference-relative biological domain activation as a connected
      graph structure. Hover over nodes and edges to inspect attributes. Drag nodes to
      rearrange the layout.
    </p>
    <div class="guardrail-banner">
      &#9888;&nbsp;{_GUARDRAIL}&nbsp;
      These graphs are research interpretation artifacts based on processed proxy data.
      They do not represent actual Artemis&nbsp;II astronaut data.
    </div>
  </div>
  <div class="legend-bar">
    <strong>Activation levels:</strong>
    <span><span class="legend-dot" style="background:#AED6F1;"></span>Low</span>
    <span><span class="legend-dot" style="background:#A9DFBF;"></span>Mild</span>
    <span><span class="legend-dot" style="background:#F5CBA7;"></span>Moderate</span>
    <span><span class="legend-dot" style="background:#F1948A;"></span>High</span>
    &nbsp;&nbsp;<strong>Node size</strong> = activation magnitude&nbsp;&nbsp;
    <strong>Edges:</strong>
    <span style="border-bottom:2px solid #7F8C8D;padding-bottom:1px;">conceptual</span>
    &nbsp;/&nbsp;
    <span style="border-bottom:2px solid #CB4335;padding-bottom:1px;">co-activation</span>
  </div>
  <table>
    <thead>
      <tr>
        <th>Participant</th>
        <th>Top domain</th>
        <th>Max activation</th>
        <th>BACI score</th>
        <th>BACI category</th>
      </tr>
    </thead>
    <tbody>
{rows_html}
    </tbody>
  </table>
  <div class="footer">
    Graph edges are interpretive links, not causal proof. Node size reflects
    reference-relative domain activation magnitude.
  </div>
</body>
</html>""")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
