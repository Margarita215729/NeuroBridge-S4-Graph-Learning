"""Interactive HTML graph export for biological adaptation graphs.

Custom standalone *vis-network* renderer (v4).

Why this rewrite
----------------
The previous pyvis-injection approach kept producing broken tooltips
(raw ``<br>`` tags shown as literal text), over-wide tooltips that overlaid
the canvas, clipped legends/footers, and unpolished navigation buttons.

This module abandons pyvis entirely and instead:

  1. Serializes NetworkX nodes/edges into plain JSON.
  2. Injects that JSON into a single, controlled, hand-written HTML template
     that loads ``vis-network`` from a CDN.
  3. Renders tooltips through a **custom ``<div>``** whose content is set with
     ``textContent`` (never ``innerHTML``), so no HTML tag can ever appear as
     literal text. Line breaks come from real ``\\n`` characters plus
     ``white-space: pre-wrap`` CSS — never ``<br>``.

The page uses a CSS-grid layout (header / graph + legend / footer) so nothing
is clipped, and ``navigationButtons`` is disabled.

Network requirement
--------------------
``vis-network`` is loaded from a CDN, so a working internet connection is
required the first time an interactive page is opened. Static PNG fallbacks
(generated separately in ``visualization.py``) remain available offline.
"""

from __future__ import annotations

import html as _html_stdlib
import json
import re
from pathlib import Path
from typing import Any

import networkx as nx

try:
    import pandas as pd  # noqa: F401  (only used for typing / optional table)
    _PANDAS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PANDAS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------

# Node fill colour by activation level (spec palette).
_LEVEL_COLORS: dict[str, str] = {
    "low":      "#9ecae1",
    "mild":     "#b7e4c7",
    "moderate": "#fdd49e",
    "high":     "#fb8072",
}
_DEFAULT_NODE_COLOR = "#d9dde1"
_NODE_BORDER = "#6b7785"
_NODE_BORDER_HIGHLIGHT = "#2c3e50"

# Edge colours.
_EDGE_CONCEPTUAL_COLOR = "#90a0ab"   # muted gray
_EDGE_COACTIVATION_COLOR = "#cf6f6a"  # muted red

# vis-network CDN (pinned version for reproducibility).
_VIS_CDN = "https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"

# Short guardrail used inside node tooltips.
_GUARDRAIL_SHORT = "Research interpretation only; not diagnosis."
# Full guardrail used in header badges.
_GUARDRAIL_FULL = "Research interpretation only. Not diagnosis or treatment guidance."
# Backward-compatible alias.
_GUARDRAIL = _GUARDRAIL_FULL

_SUBTITLE = "Interactive graph of reference-relative biological domain activation."

_FOOTER_TEXT = (
    "This graph represents one pseudo-crew participant as a connected "
    "biological system. Edges are interpretive links, not causal proof. "
    "Node size reflects reference-relative activation magnitude."
)

# Human-readable edge-type labels.
_EDGE_TYPE_LABEL: dict[str, str] = {
    "conceptual_biological_relationship": "conceptual biological relationship",
    "within_subject_coactivation":        "within-subject co-activation",
    "conceptual":                         "conceptual biological relationship",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _participant_label(subject_id: str) -> str:
    """'Crew 97774' / 'Crew_97774' -> 'Pseudo-crew participant 97774'."""
    sid = str(subject_id).replace("_", " ").strip()
    num = re.sub(r"(?i)^crew\s*", "", sid).strip()
    return f"Pseudo-crew participant {num}" if num else f"Pseudo-crew participant {sid}"


def _esc(s: object) -> str:
    """HTML-escape a value for safe injection into HTML *text* nodes."""
    return _html_stdlib.escape(str(s))


def _fmt(value: object, d: int = 2) -> str:
    """Format a numeric value to *d* decimals; 'n/a' for missing/non-numeric."""
    if value in (None, ""):
        return "n/a"
    try:
        return f"{float(value):.{d}f}"
    except (TypeError, ValueError):
        return str(value)


def _node_label(domain: str) -> str:
    """Break long domain names onto two lines at the ' / ' separator.

    Examples
    --------
    'Hematologic / oxygen-carrying' -> 'Hematologic /\\noxygen-carrying'
    'Cardiovascular regulation'     -> 'Cardiovascular regulation'  (unchanged)

    The ``\\n`` is interpreted by vis-network as a label line break — it is
    *not* an HTML tag and never appears as literal text.
    """
    d = str(domain)
    if " / " in d:
        head, tail = d.split(" / ", 1)
        return f"{head} /\n{tail}"
    return d


def _node_size(activation: float) -> float:
    """size = 22 + 24 * min(activation, 2.5) / 2.5  (capped at 48)."""
    try:
        a = float(activation)
    except (TypeError, ValueError):
        a = 0.0
    size = 22.0 + 24.0 * min(max(a, 0.0), 2.5) / 2.5
    return round(min(size, 48.0), 1)


def _coactivation_edge_width(weight: float) -> float:
    """Map co-activation weight to a 2.5–4.0 px stroke width."""
    try:
        w = float(weight)
    except (TypeError, ValueError):
        w = 1.0
    frac = min(max((w - 1.0) / 1.5, 0.0), 1.0)
    return round(2.5 + 1.5 * frac, 2)


def _node_tooltip_text(attrs: dict) -> str:
    """Plain-text node tooltip (newline separated, no HTML tags)."""
    domain = attrs.get("domain", attrs.get("subject_id", "domain"))
    domain_score = _fmt(attrs.get("domain_score"), 2)
    activation = _fmt(attrs.get("activation"), 2)
    level = attrs.get("activation_level", "n/a")
    interp = str(attrs.get("interpretation", "")).strip()

    lines = [
        f"Domain: {domain}",
        f"Domain score: {domain_score}",
        f"Activation: {activation}",
        f"Activation level: {level}",
    ]
    if interp:
        lines.append(f"Interpretation: {interp}")
    lines.append(f"Guardrail: {_GUARDRAIL_SHORT}")
    return "\n".join(lines)


def _edge_tooltip_text(attrs: dict) -> str:
    """Plain-text edge tooltip (newline separated, no HTML tags)."""
    etype_raw = str(attrs.get("edge_type", ""))
    etype = _EDGE_TYPE_LABEL.get(etype_raw, etype_raw.replace("_", " ") or "n/a")
    rel = str(attrs.get("relationship", "n/a")).strip() or "n/a"
    weight = _fmt(attrs.get("weight", 1.0), 2)
    interp = str(attrs.get("interpretation", "")).strip()
    if not interp:
        interp = "Conceptual relationship, not causal proof."

    lines = [
        f"Edge type: {etype}",
        f"Relationship: {rel}",
        f"Weight: {weight}",
    ]
    # A conceptual edge can be annotated with co-activation metadata.
    if etype_raw != "within_subject_coactivation" and attrs.get("coactivation"):
        cw = _fmt(attrs.get("coactivation_weight"), 2)
        lines.append(f"Co-activation: yes (within-subject, weight {cw})")
    lines.append(f"Interpretation: {interp}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def _serialize_nodes(G: nx.Graph) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for node, attrs in G.nodes(data=True):
        level = str(attrs.get("activation_level", "low"))
        color = _LEVEL_COLORS.get(level, _DEFAULT_NODE_COLOR)
        activation = attrs.get("activation", 0.0)
        domain = attrs.get("domain", str(node))
        nodes.append({
            "id": str(node),
            "label": _node_label(domain),
            "tip": _node_tooltip_text(attrs),
            "size": _node_size(activation),
            "color": {
                "background": color,
                "border": _NODE_BORDER,
                "highlight": {"background": color, "border": _NODE_BORDER_HIGHLIGHT},
                "hover": {"background": color, "border": _NODE_BORDER_HIGHLIGHT},
            },
        })
    return nodes


def _serialize_edges(G: nx.Graph) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for u, v, attrs in G.edges(data=True):
        etype = str(attrs.get("edge_type", ""))
        is_coact = etype == "within_subject_coactivation"
        if is_coact:
            color = _EDGE_COACTIVATION_COLOR
            width = _coactivation_edge_width(float(attrs.get("weight", 1.0)))
            dashes = True
        else:
            color = _EDGE_CONCEPTUAL_COLOR
            width = 2.0
            dashes = False
        edges.append({
            "from": str(u),
            "to": str(v),
            "tip": _edge_tooltip_text(attrs),
            "color": {"color": color, "highlight": color, "hover": color,
                      "inherit": False},
            "width": width,
            "dashes": dashes,
        })
    return edges


def _network_options() -> dict[str, Any]:
    return {
        "layout": {"improvedLayout": True},
        "physics": {
            "enabled": True,
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
                "gravitationalConstant": -80,
                "centralGravity": 0.012,
                "springLength": 170,
                "springConstant": 0.06,
                "damping": 0.45,
                "avoidOverlap": 0.8,
            },
            "stabilization": {
                "enabled": True,
                "iterations": 800,
                "updateInterval": 25,
            },
        },
        "interaction": {
            "hover": True,
            "tooltipDelay": 80,
            "zoomView": True,
            "dragView": True,
            "dragNodes": True,
            "navigationButtons": False,
            "keyboard": False,
        },
        "nodes": {
            "shape": "dot",
            "borderWidth": 1.5,
            "font": {
                "size": 18,
                "face": "Arial",
                "color": "#1b2733",
                "strokeWidth": 3,
                "strokeColor": "#ffffff",
            },
        },
        "edges": {
            "smooth": {"enabled": True, "type": "dynamic"},
            "color": {"inherit": False},
            "font": {"size": 0},
        },
    }


def _json_for_script(obj: Any) -> str:
    """JSON dump safe to embed inside an inline ``<script>`` block."""
    return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_LEGEND_HTML = """\
<aside class="nb-legend">
  <h2>Legend</h2>

  <div class="nb-legend-section">
    <h3>Activation level</h3>
    <ul>
      <li><span class="nb-dot" style="background:#9ecae1"></span>Low (&lt; 0.75)</li>
      <li><span class="nb-dot" style="background:#b7e4c7"></span>Mild (0.75&ndash;1.0)</li>
      <li><span class="nb-dot" style="background:#fdd49e"></span>Moderate (1.0&ndash;1.5)</li>
      <li><span class="nb-dot" style="background:#fb8072"></span>High (&ge; 1.5)</li>
    </ul>
  </div>

  <div class="nb-legend-section">
    <h3>Node size</h3>
    <p>Larger = higher activation</p>
  </div>

  <div class="nb-legend-section">
    <h3>Edges</h3>
    <ul>
      <li><span class="nb-edge nb-edge-solid"></span>Conceptual relationship</li>
      <li><span class="nb-edge nb-edge-dashed"></span>Co-activation</li>
    </ul>
  </div>

  <div class="nb-legend-section">
    <h3>How to inspect</h3>
    <p>Hover over a node or edge. Drag nodes to rearrange. Scroll to zoom.</p>
  </div>

  <div class="nb-legend-section">
    <h3>Guardrail</h3>
    <p>Not diagnostic.</p>
  </div>
</aside>"""


_PAGE_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <!-- vis-network is loaded from a CDN: an internet connection is required
       the first time this interactive page is opened. A static PNG fallback
       of the same graph is available alongside this file. -->
  <script src="__CDN__"></script>
  <style>
    * { box-sizing: border-box; }
    html, body {
      margin: 0;
      padding: 0;
      height: 100%;
      background: #eef1f5;
      color: #1b2733;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    }
    body {
      display: flex;
      flex-direction: column;
      min-height: 100vh;
    }
    /* ---- Header ---- */
    header.nb-header {
      padding: 16px 24px;
      background: #ffffff;
      border-bottom: 1px solid #d6dce3;
    }
    header.nb-header h1 {
      margin: 0 0 4px;
      font-size: 1.15rem;
      font-weight: 600;
    }
    header.nb-header .nb-subtitle {
      margin: 0 0 8px;
      font-size: 0.85rem;
      color: #56616e;
    }
    .nb-guardrail {
      display: inline-block;
      background: #fff8e1;
      border: 1px solid #f2c94c;
      color: #7a5c00;
      border-radius: 4px;
      padding: 4px 10px;
      font-size: 0.78rem;
    }
    /* ---- Main: graph + legend ---- */
    main.nb-main {
      flex: 1 1 auto;
      display: grid;
      grid-template-columns: 1fr 260px;
      gap: 16px;
      padding: 16px 24px;
      height: calc(100vh - 170px);
      min-height: 0;
    }
    .nb-graph-panel {
      position: relative;
      background: #ffffff;
      border: 1px solid #d6dce3;
      border-radius: 8px;
      overflow: hidden;
      min-height: 620px;
    }
    #nb-network {
      width: 100%;
      height: 100%;
      min-height: 620px;
    }
    .nb-offline {
      position: absolute;
      inset: 0;
      display: none;
      align-items: center;
      justify-content: center;
      text-align: center;
      padding: 24px;
      color: #b03a2e;
      font-size: 0.9rem;
    }
    /* ---- Legend ---- */
    aside.nb-legend {
      background: #ffffff;
      border: 1px solid #d6dce3;
      border-radius: 8px;
      padding: 14px 16px;
      font-size: 0.82rem;
      overflow-y: auto;
    }
    aside.nb-legend h2 {
      margin: 0 0 10px;
      font-size: 0.95rem;
      font-weight: 600;
    }
    .nb-legend-section { margin-bottom: 12px; }
    .nb-legend-section h3 {
      margin: 0 0 5px;
      font-size: 0.8rem;
      font-weight: 600;
      color: #34404d;
    }
    .nb-legend-section p { margin: 0; color: #56616e; line-height: 1.4; }
    aside.nb-legend ul { list-style: none; margin: 0; padding: 0; }
    aside.nb-legend li {
      display: flex;
      align-items: center;
      gap: 8px;
      margin: 3px 0;
      color: #34404d;
    }
    .nb-dot {
      width: 13px; height: 13px;
      border-radius: 50%;
      border: 1px solid #6b7785;
      flex: 0 0 auto;
    }
    .nb-edge {
      width: 26px; height: 0;
      flex: 0 0 auto;
    }
    .nb-edge-solid { border-top: 3px solid #90a0ab; }
    .nb-edge-dashed { border-top: 3px dashed #cf6f6a; }
    /* ---- Footer ---- */
    footer.nb-footer {
      padding: 12px 24px 16px;
      background: #ffffff;
      border-top: 1px solid #d6dce3;
      font-size: 0.78rem;
      color: #56616e;
      font-style: italic;
      line-height: 1.45;
    }
    /* ---- Custom tooltip ---- */
    #nb-tooltip {
      position: fixed;
      display: none;
      z-index: 1000;
      max-width: 320px;
      background: #ffffff;
      border: 1px solid #c3ccd6;
      border-radius: 6px;
      box-shadow: 0 4px 14px rgba(20, 30, 45, 0.18);
      padding: 10px 12px;
      font-size: 0.8rem;
      line-height: 1.4;
      color: #1b2733;
      white-space: pre-wrap;   /* renders real \\n line breaks, wraps long text */
      word-break: break-word;
      pointer-events: none;
    }
    @media (max-width: 760px) {
      main.nb-main { grid-template-columns: 1fr; height: auto; }
    }
  </style>
</head>
<body>
  <header class="nb-header">
    <h1>__TITLE__</h1>
    <p class="nb-subtitle">__SUBTITLE__</p>
    <span class="nb-guardrail">__GUARDRAIL__</span>
  </header>

  <main class="nb-main">
    <div class="nb-graph-panel">
      <div id="nb-network"></div>
      <div class="nb-offline" id="nb-offline">
        Interactive graph could not load vis-network from the CDN.
        Check your internet connection, or open the static PNG figure instead.
      </div>
    </div>
    __LEGEND__
  </main>

  <footer class="nb-footer">__FOOTER__</footer>

  <div id="nb-tooltip" role="tooltip"></div>

  <script>
    (function () {
      "use strict";
      var offline = document.getElementById("nb-offline");
      if (typeof vis === "undefined" || !vis.Network) {
        offline.style.display = "flex";
        return;
      }

      var NB_NODES = __NODES__;
      var NB_EDGES = __EDGES__;
      var options  = __OPTIONS__;

      var nodes = new vis.DataSet(NB_NODES);
      var edges = new vis.DataSet(NB_EDGES);
      var container = document.getElementById("nb-network");
      var network = new vis.Network(container, { nodes: nodes, edges: edges }, options);

      // ---- Custom plain-text tooltip (never innerHTML) ----
      var tooltip = document.getElementById("nb-tooltip");
      var pointer = { x: 0, y: 0 };

      function positionTooltip() {
        var pad = 14;
        var x = pointer.x + pad;
        var y = pointer.y + pad;
        var tw = tooltip.offsetWidth;
        var th = tooltip.offsetHeight;
        if (x + tw > window.innerWidth - 8) { x = pointer.x - tw - pad; }
        if (y + th > window.innerHeight - 8) { y = pointer.y - th - pad; }
        if (x < 8) { x = 8; }
        if (y < 8) { y = 8; }
        tooltip.style.left = x + "px";
        tooltip.style.top = y + "px";
      }

      function showTooltip(text) {
        if (!text) { return; }
        tooltip.textContent = text;          // plain text only
        tooltip.style.display = "block";
        positionTooltip();
      }

      function hideTooltip() {
        tooltip.style.display = "none";
      }

      document.addEventListener("mousemove", function (e) {
        pointer.x = e.clientX;
        pointer.y = e.clientY;
        if (tooltip.style.display === "block") { positionTooltip(); }
      });

      network.on("hoverNode", function (params) {
        var n = nodes.get(params.node);
        if (n) { showTooltip(n.tip); }
      });
      network.on("hoverEdge", function (params) {
        var ed = edges.get(params.edge);
        if (ed) { showTooltip(ed.tip); }
      });
      network.on("blurNode", hideTooltip);
      network.on("blurEdge", hideTooltip);
      network.on("dragStart", hideTooltip);
      network.on("zoom", hideTooltip);
      network.on("dragging", hideTooltip);

      // ---- Fit & freeze after stabilization ----
      network.once("stabilizationIterationsDone", function () {
        network.setOptions({ physics: false });
        network.fit({ animation: { duration: 500, easingFunction: "easeInOutQuad" } });
      });
    })();
  </script>
</body>
</html>"""


def _build_page_html(G: nx.Graph, title: str) -> str:
    nodes = _serialize_nodes(G)
    edges = _serialize_edges(G)
    options = _network_options()

    html = _PAGE_TEMPLATE
    html = html.replace("__TITLE__", _esc(title))
    html = html.replace("__SUBTITLE__", _esc(_SUBTITLE))
    html = html.replace("__GUARDRAIL__", _esc(_GUARDRAIL_FULL))
    html = html.replace("__FOOTER__", _esc(_FOOTER_TEXT))
    html = html.replace("__LEGEND__", _LEGEND_HTML)
    html = html.replace("__CDN__", _VIS_CDN)
    html = html.replace("__NODES__", _json_for_script(nodes))
    html = html.replace("__EDGES__", _json_for_script(edges))
    html = html.replace("__OPTIONS__", _json_for_script(options))
    return html


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_interactive_graph_html(
    G: nx.Graph,
    output_path: "str | Path",
    title: str | None = None,
) -> Path:
    """Generate a complete standalone interactive HTML graph page.

    Serializes the NetworkX graph to JSON and injects it into a controlled
    vis-network HTML template with a custom plain-text tooltip, CSS-grid
    layout (header / graph + legend / footer), and disabled navigation
    buttons. Works locally via ``file://`` (CDN required for vis-network JS).

    Parameters
    ----------
    G:
        Subject-level biological adaptation graph. ``G.graph['subject_id']`` is
        used to build the default title.
    output_path:
        Destination ``.html`` path.
    title:
        Optional explicit page title. Defaults to
        ``"Pseudo-crew participant <ID>: Biological Adaptation Graph"``.

    Returns
    -------
    pathlib.Path
        The written file path.
    """
    out = Path(output_path)
    subject_id = str(G.graph.get("subject_id", "unknown"))
    plabel = _participant_label(subject_id)
    if title is None:
        title = f"{plabel}: Biological Adaptation Graph"

    out.parent.mkdir(parents=True, exist_ok=True)
    html = _build_page_html(G, title)
    out.write_text(html, encoding="utf-8")
    return out


def export_all_graphs_html(
    graphs: "dict[str, nx.Graph]",
    output_dir: "str | Path",
) -> "list[Path]":
    """Export one polished interactive HTML page per subject graph.

    Continues on per-subject errors and prints a warning for each failure.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for subject_id, G in graphs.items():
        safe_id = re.sub(r"[^\w\-]", "_", str(subject_id))
        html_path = out_dir / f"subject_graph_{safe_id}.html"
        try:
            export_interactive_graph_html(G, html_path)
            paths.append(html_path)
            print(f"  [interactive] wrote {html_path.name} "
                  f"({html_path.stat().st_size / 1024:.1f} KB, "
                  f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
        except Exception as exc:  # noqa: BLE001
            print(f"  [interactive] Warning: could not export {subject_id}: {exc}")
    return paths


def create_interactive_index(
    html_files: "list[Path]",
    summary_df: "pd.DataFrame | None",
    output_path: "str | Path",
) -> Path:
    """Generate a polished ``index.html`` linking every participant graph.

    Parameters
    ----------
    html_files:
        Paths to the per-subject interactive HTML files.
    summary_df:
        Optional ``phase3_graph_summary.csv`` DataFrame. When supplied, a
        summary table (top domain, max activation, BACI score/category) is
        rendered and each participant row links to its graph.
    output_path:
        Destination ``index.html`` path.

    Returns
    -------
    pathlib.Path
        The written index path.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Map a normalized subject id -> summary row dict.
    summary_lookup: dict[str, dict] = {}
    have_summary = summary_df is not None and getattr(summary_df, "empty", True) is False
    if have_summary:
        for _, row in summary_df.iterrows():
            key = re.sub(r"[^\w\-]", "_", str(row.get("subject_id", "")))
            summary_lookup[key] = row.to_dict()

    rows_html = ""
    for p in sorted(html_files):
        safe_id = p.stem.replace("subject_graph_", "")
        plabel = _participant_label(safe_id)
        row = summary_lookup.get(safe_id, {})
        top_domain = row.get("top_domain", "n/a")
        max_act = _fmt(row.get("max_domain_activation"), 3)
        baci = _fmt(row.get("baci_score"), 1)
        baci_cat = row.get("baci_category", "n/a")
        rows_html += (
            "    <tr>"
            f'<td><a href="{_esc(p.name)}">{_esc(plabel)}</a></td>'
            f"<td>{_esc(top_domain)}</td>"
            f"<td>{max_act}</td>"
            f"<td>{baci}</td>"
            f"<td>{_esc(baci_cat)}</td>"
            "</tr>\n"
        )

    table_html = (
        "  <table>\n"
        "    <thead><tr>"
        "<th>Participant</th><th>Top domain</th><th>Max activation</th>"
        "<th>BACI score</th><th>BACI category</th>"
        "</tr></thead>\n"
        f"    <tbody>\n{rows_html}    </tbody>\n"
        "  </table>\n"
    ) if rows_html else (
        "  <ul class=\"plain-links\">\n"
        + "".join(
            f'    <li><a href="{_esc(p.name)}">'
            f'{_esc(_participant_label(p.stem.replace("subject_graph_", "")))}</a></li>\n'
            for p in sorted(html_files)
        )
        + "  </ul>\n"
    )

    html = (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
        "  <meta charset=\"UTF-8\">\n"
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "  <title>NeuroBridge-S4 Graph Learning &mdash; Biological Adaptation Graphs</title>\n"
        "  <style>\n"
        "    * { box-sizing: border-box; }\n"
        "    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;\n"
        "           max-width: 960px; margin: 40px auto; padding: 0 24px;\n"
        "           background: #eef1f5; color: #1b2733; }\n"
        "    .hdr { background:#fff; border:1px solid #d6dce3; border-radius:8px;\n"
        "           padding:20px 24px 16px; margin-bottom:18px; }\n"
        "    .hdr h1 { margin:0 0 6px; font-size:1.25rem; }\n"
        "    .hdr p  { margin:0 0 10px; font-size:0.9rem; color:#56616e; line-height:1.45; }\n"
        "    .guardrail { background:#fff8e1; border:1px solid #f2c94c; color:#7a5c00;\n"
        "      border-radius:4px; padding:6px 12px; font-size:0.8rem; display:inline-block; }\n"
        "    table { border-collapse:collapse; width:100%; background:#fff;\n"
        "            border:1px solid #d6dce3; border-radius:8px; overflow:hidden; }\n"
        "    th, td { border-bottom:1px solid #eef1f5; padding:9px 14px;\n"
        "             text-align:left; font-size:0.88rem; }\n"
        "    th { background:#f3f5f8; font-weight:600; }\n"
        "    tr:last-child td { border-bottom:none; }\n"
        "    a  { color:#2471a3; text-decoration:none; }\n"
        "    a:hover { text-decoration:underline; }\n"
        "    .plain-links { list-style:none; padding:0; background:#fff;\n"
        "      border:1px solid #d6dce3; border-radius:8px; }\n"
        "    .plain-links li { padding:9px 14px; border-bottom:1px solid #eef1f5; }\n"
        "    .note { margin-top:16px; font-size:0.8rem; color:#56616e; line-height:1.5; }\n"
        "  </style>\n"
        "</head>\n<body>\n"
        "  <div class=\"hdr\">\n"
        "    <h1>NeuroBridge-S4 Graph Learning &mdash; Interactive Biological Adaptation Graphs</h1>\n"
        "    <p>Each link opens an interactive graph for one pseudo-crew participant, showing\n"
        "    reference-relative biological domain activation. Hover over nodes and edges to inspect\n"
        "    attributes, drag nodes to rearrange, and scroll to zoom.</p>\n"
        f"    <div class=\"guardrail\">{_esc(_GUARDRAIL_FULL)} "
        "These are research interpretation artifacts based on processed proxy data; "
        "they do not represent actual astronaut data.</div>\n"
        "  </div>\n"
        f"{table_html}"
        "  <p class=\"note\">\n"
        "    Interactive pages load the vis-network library from a CDN, so an internet connection\n"
        "    is required the first time a graph is opened. Static PNG fallbacks of every graph are\n"
        "    available in <code>results/figures/</code> for offline review.\n"
        "  </p>\n"
        "  <p class=\"note\">Edges are interpretive links, not causal proof. "
        "Node size reflects reference-relative domain activation magnitude.</p>\n"
        "</body>\n</html>"
    )

    out.write_text(html, encoding="utf-8")
    return out


def export_index_html(
    graph_paths: "list[Path]",
    output_path: "str | Path",
    graphs: "dict[str, nx.Graph] | None" = None,
) -> Path:
    """Backward-compatible wrapper around :func:`create_interactive_index`.

    Builds a minimal summary DataFrame from ``graphs`` (if provided and pandas
    is available) and delegates to :func:`create_interactive_index`.
    """
    summary_df = None
    if graphs and _PANDAS_AVAILABLE:
        import pandas as _pd
        rows = []
        for sid, G in graphs.items():
            rows.append({
                "subject_id": sid,
                "top_domain": G.graph.get("top_domain", "n/a"),
                "max_domain_activation": G.graph.get("max_domain_activation"),
                "baci_score": G.graph.get("baci_score"),
                "baci_category": G.graph.get("baci_category", "n/a"),
            })
        summary_df = _pd.DataFrame(rows)
    return create_interactive_index(list(graph_paths), summary_df, output_path)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_html_graph_output(path: "str | Path") -> dict:
    """Validate a generated interactive HTML graph file.

    Checks (per Phase 3 spec):
      * file exists;
      * file size > 10 KB;
      * contains a ``new vis.Network`` initialization;
      * contains node labels;
      * contains edge data;
      * contains the custom tooltip code (``nb-tooltip`` + ``textContent``);
      * does NOT contain visible raw tooltip tags (``<br>`` or ``&lt;br&gt;``).

    Visual inspection in a browser is still required — rendering quality
    cannot be fully verified from file content alone.
    """
    p = Path(path)
    result: dict = {
        "file":              p.name,
        "exists":            p.exists(),
        "size_kb":           0.0,
        "size_ok":           False,
        "has_vis_init":      False,
        "has_title":         False,
        "has_node_data":     False,
        "has_edge_data":     False,
        "has_custom_tooltip": False,
        "clean_tooltips":    True,
        "passed":            False,
        "notes":             [],
    }

    if not p.exists():
        result["notes"].append("file not found")
        return result

    size = p.stat().st_size
    result["size_kb"] = round(size / 1024, 1)
    result["size_ok"] = size > 10_000

    content = p.read_text(encoding="utf-8")

    result["has_vis_init"] = bool(re.search(r"new\s+vis\.Network\s*\(", content))
    result["has_title"] = (
        "Pseudo-crew participant" in content
        or "Biological Adaptation Graph" in content
    )
    # Node labels: NB_NODES array present and carries a "label" field.
    result["has_node_data"] = ("NB_NODES" in content) and ('"label"' in content)
    # Edge data: NB_EDGES array present and carries a "from" field.
    result["has_edge_data"] = ("NB_EDGES" in content) and ('"from"' in content)
    # Custom tooltip: dedicated div + textContent assignment.
    result["has_custom_tooltip"] = (
        'id="nb-tooltip"' in content and "textContent" in content
    )

    # Raw / escaped tooltip tags must not appear anywhere.
    has_raw_br = bool(re.search(r"&lt;\s*br\s*/?\s*&gt;", content, re.IGNORECASE))
    # A literal <br> is only acceptable in the static offline message, not in
    # any tooltip/data context. We forbid <br> inside the JSON node/edge arrays.
    node_block = re.search(r"NB_NODES\s*=\s*(\[.*?\]);", content, re.DOTALL)
    edge_block = re.search(r"NB_EDGES\s*=\s*(\[.*?\]);", content, re.DOTALL)
    raw_in_data = False
    for blk in (node_block, edge_block):
        if blk and re.search(r"<\s*br\s*/?\s*>|<\s*/?\s*[bi]\s*>", blk.group(1), re.IGNORECASE):
            raw_in_data = True
    result["clean_tooltips"] = not (has_raw_br or raw_in_data)

    if not result["size_ok"]:
        result["notes"].append(f"file too small ({result['size_kb']:.1f} KB)")
    if not result["has_vis_init"]:
        result["notes"].append("new vis.Network initialization not found")
    if not result["has_title"]:
        result["notes"].append("expected title text not found")
    if not result["has_node_data"]:
        result["notes"].append("node label data not found")
    if not result["has_edge_data"]:
        result["notes"].append("edge data not found")
    if not result["has_custom_tooltip"]:
        result["notes"].append("custom tooltip code not found")
    if not result["clean_tooltips"]:
        result["notes"].append("raw HTML tooltip tags detected")

    result["passed"] = all([
        result["exists"],
        result["size_ok"],
        result["has_vis_init"],
        result["has_title"],
        result["has_node_data"],
        result["has_edge_data"],
        result["has_custom_tooltip"],
        result["clean_tooltips"],
    ])
    return result
