"""Interactive HTML graph export for biological adaptation graphs.

Bug fix (v3): previous implementation extracted only ``<body>`` from the
pyvis-generated HTML and discarded the ``<head>``.  The ``<head>`` contains:

  * the entire vis-network JavaScript library (≈700 KB, embedded inline)
  * the ``#mynetwork { height: … }`` CSS rule

Without those, the network canvas had zero height and ``vis.Network`` was
undefined → empty white area.

Correct approach:
  1. Call ``net.save_graph(path)`` which writes a complete, self-contained HTML.
  2. Read that HTML back and **inject** our custom CSS + header / legend / footer
     using string replacement — never removing any ``<head>`` content.
  3. Result: all vis-network JS is preserved; our UI sits above/below the graph.
"""

from __future__ import annotations

import html as _html_stdlib
import json
import re
from pathlib import Path
from typing import Any

import networkx as nx

try:
    from pyvis.network import Network as PyvisNetwork
    _PYVIS_AVAILABLE = True
except ImportError:
    _PYVIS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------

_LEVEL_COLORS: dict[str, str] = {
    "low":      "#AED6F1",   # muted blue
    "mild":     "#A9DFBF",   # muted green
    "moderate": "#F5CBA7",   # soft orange
    "high":     "#F1948A",   # coral/red
}
_DEFAULT_COLOR = "#D5D8DC"

_EDGE_COLORS: dict[str, str] = {
    "conceptual_biological_relationship": "#7F8C8D",
    "within_subject_coactivation":        "#CB4335",
}
_DEFAULT_EDGE_COLOR = "#AAAAAA"

_GUARDRAIL = (
    "Research interpretation only. "
    "Not diagnosis or treatment guidance."
)

# ---------------------------------------------------------------------------
# CSS that we inject into the pyvis <head> — overrides pyvis defaults where
# needed and adds our header/legend/footer classes.
# ---------------------------------------------------------------------------
_INJECTED_CSS = """\
<style type="text/css">
/* NeuroBridge — custom overlay (injected, does not replace pyvis styles) */
html, body {
  margin: 0 !important;
  padding: 0 !important;
  background: #f5f6fa !important;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
}
/* Ensure the network canvas fills its container and has a visible height */
#mynetwork {
  width: 100% !important;
  height: 650px !important;
  border: 1px solid #dde !important;
  background: #ffffff !important;
  display: block !important;
  float: none !important;
  position: relative !important;
}
/* Pyvis card wrapper */
.card { overflow: hidden !important; display: block !important; width: 100% !important; }

/* ---- Custom UI chrome ---- */
.nb-header {
  background: #ffffff;
  border-bottom: 1px solid #dde;
  padding: 13px 22px 11px;
}
.nb-header h1 {
  margin: 0 0 3px;
  font-size: 1.1em;
  font-weight: 600;
  color: #111;
}
.nb-sub { font-size: 0.84em; color: #555; margin: 0 0 7px; }
.nb-guardrail {
  background: #fef9e7;
  border: 1px solid #f9ca24;
  border-radius: 4px;
  padding: 4px 11px;
  font-size: 0.79em;
  color: #7d6608;
  display: inline-block;
}
.nb-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  align-items: center;
  background: #ffffff;
  border-top: 1px solid #dde;
  padding: 8px 18px;
  font-size: 0.82em;
  clear: both;
}
.nb-dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  display: inline-block;
  border: 1px solid #bbb;
  margin-right: 3px;
  vertical-align: middle;
}
.nb-edge-sample {
  display: inline-block;
  width: 22px;
  height: 3px;
  vertical-align: middle;
  margin-right: 3px;
}
.nb-footer {
  padding: 6px 22px 14px;
  font-size: 0.79em;
  color: #777;
  font-style: italic;
  clear: both;
}
</style>
"""


# ---------------------------------------------------------------------------
# pyvis physics / interaction options
# ---------------------------------------------------------------------------
_PYVIS_OPTIONS: dict[str, Any] = {
    "physics": {
        "enabled": True,
        "barnesHut": {
            "gravitationalConstant": -2500,
            "centralGravity": 0.25,
            "springLength": 140,
            "springConstant": 0.04,
            "damping": 0.35,
        },
        "stabilization": {
            "enabled": True,
            "iterations": 200,
            "updateInterval": 25,
        },
    },
    "interaction": {
        "hover": True,
        "tooltipDelay": 100,
        "navigationButtons": True,
        "keyboard": True,
    },
    "nodes": {
        "font": {"size": 15, "face": "Arial"},
    },
    "edges": {
        "smooth": {"enabled": True, "type": "dynamic"},
    },
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _participant_label(subject_id: str) -> str:
    """'Crew 97774' / 'Crew_97774' → 'Pseudo-crew participant 97774'."""
    sid = str(subject_id).replace("_", " ").strip()
    num = re.sub(r"(?i)^crew\s*", "", sid).strip()
    return f"Pseudo-crew participant {num}" if num else sid


def _esc(s: object) -> str:
    """HTML-escape a value for safe injection into HTML text / attributes."""
    return _html_stdlib.escape(str(s))


def _fmt(value: object, d: int = 3) -> str:
    try:
        return f"{float(value):.{d}f}"
    except (TypeError, ValueError):
        return str(value) if value not in (None, "") else "n/a"


def _node_tooltip(attrs: dict) -> str:
    """Tooltip for a node.

    Uses ``<br>`` for line breaks; no bold/italic tags — avoids raw-tag issue.
    vis-network renders the title string as HTML, so ``<br>`` gives clean lines.
    """
    domain = attrs.get("domain", attrs.get("subject_id", ""))
    score = _fmt(attrs.get("domain_score"))
    activation = _fmt(attrs.get("activation"))
    level = attrs.get("activation_level", "n/a")
    interp = attrs.get("interpretation", "")
    parts = [
        f"Domain: {_esc(domain)}",
        f"Domain score: {score}",
        f"Activation: {activation}",
        f"Activation level: {level}",
    ]
    if interp:
        parts.append(f"Interpretation: {_esc(interp)}")
    parts.append(f"Guardrail: {_esc(_GUARDRAIL)}")
    return "<br>".join(parts)


def _edge_tooltip(attrs: dict) -> str:
    """Tooltip for an edge. Uses ``<br>`` only."""
    etype = attrs.get("edge_type", "n/a")
    rel = attrs.get("relationship", "n/a")
    weight = _fmt(attrs.get("weight"), 2)
    interp = attrs.get("interpretation", "Conceptual relationship, not causal proof.")
    return (
        f"Edge type: {_esc(etype)}<br>"
        f"Relationship: {_esc(rel)}<br>"
        f"Weight: {weight}<br>"
        f"Interpretation: {_esc(interp)}"
    )


def _node_size(activation: float) -> int:
    """18 + 35 × min(act, 2.5) / 2.5 — as per spec."""
    return int(18 + 35 * min(float(activation), 2.5) / 2.5)


def _edge_width(weight: float) -> float:
    return max(1.0, min(float(weight) * 2.5, 7.0))


def _build_pyvis_network(G: nx.Graph) -> "PyvisNetwork":
    """Build pyvis Network with correct node/edge attributes.

    ``cdn_resources="in_line"`` embeds vis-network JS directly in the HTML,
    making the output completely self-contained for local ``file://`` viewing.
    """
    net = PyvisNetwork(
        height="650px",
        width="100%",
        bgcolor="#ffffff",
        font_color="#222222",
        notebook=False,
        cdn_resources="in_line",
    )
    net.heading = ""  # suppress pyvis auto-heading; we inject our own

    for node, attrs in G.nodes(data=True):
        activation = float(attrs.get("activation", 0.5))
        level = attrs.get("activation_level", "low")
        color = _LEVEL_COLORS.get(level, _DEFAULT_COLOR)
        net.add_node(
            node,
            label=str(node),
            title=_node_tooltip(attrs),
            color={
                "background": color,
                "border": "#999999",
                "highlight": {"background": color, "border": "#333333"},
            },
            size=_node_size(activation),
            font={"size": 15, "face": "Arial", "color": "#111111"},
            borderWidth=1.5,
            shape="dot",
        )

    for u, v, attrs in G.edges(data=True):
        etype = attrs.get("edge_type", "")
        edge_color = _EDGE_COLORS.get(etype, _DEFAULT_EDGE_COLOR)
        dashes = etype == "within_subject_coactivation"
        net.add_edge(
            u, v,
            title=_edge_tooltip(attrs),
            color={"color": edge_color, "highlight": edge_color},
            width=_edge_width(float(attrs.get("weight", 1.0))),
            dashes=dashes,
            smooth={"enabled": True, "type": "dynamic"},
        )

    net.set_options(json.dumps(_PYVIS_OPTIONS))
    return net


def _inject_into_pyvis_html(raw: str, title: str, plabel: str) -> str:
    """Inject custom UI into pyvis HTML without removing any ``<head>`` content.

    Key rules:
    - Never remove ``<script>`` blocks.
    - Never remove ``#mynetwork`` CSS from the pyvis head.
    - Only insert new content using positional string replacement.
    """
    # 1. Fix or inject <title>
    if re.search(r"<title>[^<]*</title>", raw, re.IGNORECASE):
        raw = re.sub(
            r"<title>[^<]*</title>",
            f"<title>{_esc(title)}</title>",
            raw, count=1, flags=re.IGNORECASE,
        )
    else:
        # pyvis may not generate a <title>; inject one before </head>
        raw = re.sub(
            r"(</head>)",
            f"<title>{_esc(title)}</title>\n\\1",
            raw, count=1, flags=re.IGNORECASE,
        )

    # 2. Inject our CSS into <head> (adds our classes; does NOT override pyvis JS)
    head_end = re.search(r"</head>", raw, re.IGNORECASE)
    if head_end:
        pos = head_end.start()
        raw = raw[:pos] + _INJECTED_CSS + raw[pos:]

    # 3. Remove pyvis auto-heading if it snuck in (e.g. <center><h1>...</h1></center>)
    raw = re.sub(
        r"<center>\s*<h1[^>]*>.*?</h1>\s*</center>",
        "",
        raw, flags=re.DOTALL | re.IGNORECASE,
    )

    # 4. Inject our page header immediately after <body ...> opening tag
    body_open = re.search(r"<body[^>]*>", raw, re.IGNORECASE)
    if body_open:
        pos = body_open.end()
        header_html = (
            "\n"
            f'<div class="nb-header">'
            f"<h1>{_esc(title)}</h1>"
            f'<div class="nb-sub">'
            "Interactive graph of reference-relative biological domain activation"
            f" &mdash; {_esc(plabel)}"
            "</div>"
            f'<span class="nb-guardrail">&#9888;&nbsp;{_esc(_GUARDRAIL)}</span>'
            "</div>\n"
        )
        raw = raw[:pos] + header_html + raw[pos:]

    # 5. Inject legend + noscript fallback immediately before </body>
    body_close = re.search(r"</body>", raw, re.IGNORECASE)
    if body_close:
        pos = body_close.start()
        legend_html = (
            "\n"
            '<div class="nb-legend">'
            "<strong>Activation level:</strong>"
            '<span><span class="nb-dot" style="background:#AED6F1"></span>Low (&lt;&nbsp;0.75)</span>'
            '<span><span class="nb-dot" style="background:#A9DFBF"></span>Mild (0.75&ndash;1.0)</span>'
            '<span><span class="nb-dot" style="background:#F5CBA7"></span>Moderate (1.0&ndash;1.5)</span>'
            '<span><span class="nb-dot" style="background:#F1948A"></span>High (&ge;&nbsp;1.5)</span>'
            "&nbsp;&nbsp;<strong>Node size</strong>&nbsp;=&nbsp;activation"
            "&nbsp;&nbsp;<strong>Edges:</strong>"
            '<span><span class="nb-edge-sample" style="background:#7F8C8D"></span>conceptual</span>'
            '<span><span class="nb-edge-sample" style="background:#CB4335"></span>co-activation</span>'
            "&nbsp;&nbsp;<em>Hover to inspect &middot; Drag to rearrange &middot; Scroll to zoom</em>"
            "</div>"
            "\n"
            f'<div class="nb-footer">'
            f"This graph represents {_esc(plabel)} as a connected biological system."
            " Edges are interpretive links, not causal proof. Not diagnostic."
            "</div>"
            "\n"
            '<noscript>'
            '<p style="color:red;padding:12px">'
            "Interactive graph failed to render. "
            "Please use the PNG figure or check browser JavaScript settings."
            "</p>"
            "</noscript>\n"
        )
        raw = raw[:pos] + legend_html + raw[pos:]

    return raw


def _fallback_html(
    G: nx.Graph,
    output_path: Path,
    title: str,
    plabel: str,
) -> Path:
    """Full vis-network page using CDN (requires internet; used only when pyvis unavailable)."""
    nodes_data = [
        {
            "id": str(n),
            "label": str(n),
            "title": _node_tooltip(G.nodes[n]),
            "color": {
                "background": _LEVEL_COLORS.get(G.nodes[n].get("activation_level", ""), _DEFAULT_COLOR),
                "border": "#999",
            },
            "size": _node_size(float(G.nodes[n].get("activation", 0.5))),
            "font": {"size": 15, "color": "#111"},
            "shape": "dot",
        }
        for n in G.nodes()
    ]
    edges_data = [
        {
            "from": str(u),
            "to": str(v),
            "title": _edge_tooltip(attrs),
            "color": {"color": _EDGE_COLORS.get(attrs.get("edge_type", ""), _DEFAULT_EDGE_COLOR)},
            "width": _edge_width(float(attrs.get("weight", 1.0))),
            "dashes": attrs.get("edge_type", "") == "within_subject_coactivation",
        }
        for u, v, attrs in G.edges(data=True)
    ]

    html = (
        "<!DOCTYPE html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "  <meta charset=\"UTF-8\">\n"
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"  <title>{_esc(title)}</title>\n"
        "  <!-- vis-network from CDN — requires internet connection -->\n"
        "  <script src=\"https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js\"></script>\n"
        + _INJECTED_CSS
        + "  <style>#mynetwork { width:100%; height:650px; border:1px solid #dde; background:#fff; }</style>\n"
        "</head>\n"
        "<body>\n"
        f"  <div class=\"nb-header\"><h1>{_esc(title)}</h1>"
        f"<div class=\"nb-sub\">Interactive graph of reference-relative biological domain activation"
        f" &mdash; {_esc(plabel)}</div>"
        f"<span class=\"nb-guardrail\">&#9888;&nbsp;{_esc(_GUARDRAIL)}</span></div>\n"
        "  <div id=\"mynetwork\"></div>\n"
        "  <script>\n"
        f"    var nodes = new vis.DataSet({json.dumps(nodes_data)});\n"
        f"    var edges = new vis.DataSet({json.dumps(edges_data)});\n"
        "    var container = document.getElementById(\"mynetwork\");\n"
        f"    var options = {json.dumps(_PYVIS_OPTIONS)};\n"
        "    new vis.Network(container, {nodes: nodes, edges: edges}, options);\n"
        "  </script>\n"
        "  <div class=\"nb-legend\">"
        "<strong>Activation:</strong>"
        "<span><span class=\"nb-dot\" style=\"background:#AED6F1\"></span>Low</span>"
        "<span><span class=\"nb-dot\" style=\"background:#A9DFBF\"></span>Mild</span>"
        "<span><span class=\"nb-dot\" style=\"background:#F5CBA7\"></span>Moderate</span>"
        "<span><span class=\"nb-dot\" style=\"background:#F1948A\"></span>High</span>"
        " &nbsp;<em>Hover · Drag · Scroll to zoom</em>"
        "</div>\n"
        f"  <div class=\"nb-footer\">{_esc(plabel)} — {_esc(_GUARDRAIL)}</div>\n"
        "<noscript><p style=\"color:red;padding:10px\">"
        "Interactive graph failed to render. Please use the PNG figure.</p></noscript>\n"
        "</body>\n</html>"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_html_graph_output(path: "str | Path") -> dict:
    """Lightweight validation of a generated HTML graph file.

    Returns a dict with individual check results and an overall ``passed`` flag.
    Does not replace visual inspection, but catches obvious broken exports.
    """
    p = Path(path)
    result: dict = {
        "file":          p.name,
        "exists":        p.exists(),
        "size_kb":       0.0,
        "size_ok":       False,
        "has_vis_init":  False,
        "has_title":     False,
        "has_node_data": False,
        "has_edge_data": False,
        "clean_tooltips": True,
        "passed":        False,
        "notes":         [],
    }

    if not p.exists():
        result["notes"].append("file not found")
        return result

    size = p.stat().st_size
    result["size_kb"] = round(size / 1024, 1)
    result["size_ok"] = size > 10_000

    content = p.read_text(encoding="utf-8")

    # vis.Network initialisation present
    result["has_vis_init"] = bool(re.search(r"vis\.Network\s*\(", content))

    # Our title element present
    result["has_title"] = (
        "Pseudo-crew participant" in content
        or "Biological Adaptation Graph" in content
    )

    # vis.DataSet called at least once (nodes), twice (nodes + edges)
    dataset_count = len(re.findall(r"vis\.DataSet\s*\(", content))
    result["has_node_data"] = dataset_count >= 1
    result["has_edge_data"] = dataset_count >= 2

    # Raw HTML-escaped tooltip tags visible as literal text
    has_raw = bool(re.search(r"&lt;(?:b|i|br)\s*/?\s*&gt;", content))
    result["clean_tooltips"] = not has_raw
    if has_raw:
        result["notes"].append("raw HTML-escaped tags detected in tooltip content")

    if not result["size_ok"]:
        result["notes"].append(f"file too small ({result['size_kb']:.1f} KB)")
    if not result["has_vis_init"]:
        result["notes"].append("vis.Network initialization not found")
    if not result["has_title"]:
        result["notes"].append("expected title text not found")
    if not result["has_node_data"]:
        result["notes"].append("vis.DataSet (nodes) not found")

    result["passed"] = all([
        result["exists"],
        result["size_ok"],
        result["has_vis_init"],
        result["has_title"],
        result["has_node_data"],
        result["clean_tooltips"],
    ])
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_interactive_graph_html(
    G: nx.Graph,
    output_path: "str | Path",
    title: str | None = None,
) -> Path:
    """Export a self-contained interactive HTML graph page.

    Strategy (v3 — injection approach):
    1. Use pyvis (``cdn_resources="in_line"``) to write a complete HTML file
       that embeds the entire vis-network JS library.
    2. Read the file back and **inject** custom CSS, header, legend, and footer
       using positional string replacement — never touching ``<head>`` scripts
       or the ``#mynetwork`` initialization code.
    3. Fall back to a vis-network CDN template if pyvis is unavailable.

    The result works when opened locally via ``file://``.
    """
    out = Path(output_path)
    subject_id = str(G.graph.get("subject_id", "unknown"))
    plabel = _participant_label(subject_id)
    if title is None:
        title = f"{plabel}: Biological Adaptation Graph"

    out.parent.mkdir(parents=True, exist_ok=True)

    if not _PYVIS_AVAILABLE:
        return _fallback_html(G, out, title, plabel)

    try:
        net = _build_pyvis_network(G)
        # Save complete, self-contained HTML (vis-network JS embedded inline)
        net.save_graph(str(out))

        # Sanity check: ensure pyvis wrote a meaningful file
        if out.stat().st_size < 10_000:
            raise RuntimeError(
                f"pyvis output too small ({out.stat().st_size} bytes) — likely failed"
            )

        # Inject custom UI without disturbing any JS
        raw = out.read_text(encoding="utf-8")
        raw = _inject_into_pyvis_html(raw, title, plabel)
        out.write_text(raw, encoding="utf-8")
        return out

    except Exception as exc:
        print(
            f"  [interactive] Warning: pyvis export failed for {subject_id} "
            f"({exc}); using CDN fallback."
        )
        return _fallback_html(G, out, title, plabel)


def export_all_graphs_html(
    graphs: "dict[str, nx.Graph]",
    output_dir: "str | Path",
) -> "list[Path]":
    """Export one polished HTML page per subject graph.

    Continues on per-subject errors; prints a warning for each failure.
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
    graph_paths: "list[Path]",
    output_path: "str | Path",
    graphs: "dict[str, nx.Graph] | None" = None,
) -> Path:
    """Generate index.html linking all subject graph HTML files.

    Includes project title, explanation, participant summary table,
    activation-level legend, and guardrail note.
    """
    out = Path(output_path)

    rows_html = ""
    for p in sorted(graph_paths):
        safe_id = p.stem.replace("subject_graph_", "")
        plabel = _participant_label(safe_id)
        top_domain = baci = baci_cat = max_act = n_active = "n/a"
        if graphs:
            G = graphs.get(safe_id) or graphs.get(safe_id.replace("_", " "))
            if G:
                baci     = _fmt(G.graph.get("baci_score"), 3)
                baci_cat = G.graph.get("baci_category", "n/a")
                top_domain = G.graph.get("top_domain", "n/a")
                max_act  = _fmt(G.graph.get("max_domain_activation"), 3)
                n_active = str(G.graph.get("n_active_domains", "n/a"))
        rows_html += (
            f"<tr>"
            f'<td><a href="{p.name}">{_esc(plabel)}</a></td>'
            f"<td>{_esc(top_domain)}</td>"
            f"<td>{max_act}</td>"
            f"<td>{baci}</td>"
            f"<td>{_esc(baci_cat)}</td>"
            "</tr>\n"
        )

    html = (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
        "  <meta charset=\"UTF-8\">\n"
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "  <title>NeuroBridge-S4 Graph Learning &mdash; Interactive Biological Adaptation Graphs</title>\n"
        "  <style>\n"
        "    * { box-sizing: border-box; }\n"
        "    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;\n"
        "           max-width: 960px; margin: 40px auto; padding: 0 24px;\n"
        "           background: #f5f6fa; color: #222; }\n"
        "    .hdr { background:#fff; border:1px solid #dde; border-radius:6px;\n"
        "           padding:20px 24px 16px; margin-bottom:20px; }\n"
        "    .hdr h1 { margin:0 0 6px; font-size:1.22em; }\n"
        "    .hdr p  { margin:0 0 8px; font-size:0.9em; color:#444; }\n"
        "    .guardrail-banner { background:#fef9e7; border:1px solid #f9ca24;\n"
        "      border-radius:4px; padding:6px 14px; font-size:0.82em; color:#7d6608;\n"
        "      display:inline-block; }\n"
        "    .legend-bar { display:flex; gap:14px; flex-wrap:wrap; align-items:center;\n"
        "      background:#fff; border:1px solid #dde; border-radius:6px;\n"
        "      padding:10px 16px; margin-bottom:16px; font-size:0.83em; }\n"
        "    .ldot { width:12px; height:12px; border-radius:50%; display:inline-block;\n"
        "            border:1px solid #bbb; margin-right:4px; vertical-align:middle; }\n"
        "    table { border-collapse:collapse; width:100%; background:#fff;\n"
        "            border:1px solid #dde; border-radius:6px; overflow:hidden; }\n"
        "    th, td { border-bottom:1px solid #eee; padding:9px 14px;\n"
        "             text-align:left; font-size:0.88em; }\n"
        "    th { background:#f2f2f2; font-weight:600; }\n"
        "    a  { color:#2471A3; text-decoration:none; }\n"
        "    a:hover { text-decoration:underline; }\n"
        "    .footer { margin-top:18px; font-size:0.8em; color:#888; font-style:italic; }\n"
        "  </style>\n"
        "</head>\n<body>\n"
        "  <div class=\"hdr\">\n"
        "    <h1>NeuroBridge-S4 Graph Learning &mdash; Interactive Biological Adaptation Graphs</h1>\n"
        "    <p>Each row links to an interactive HTML graph for one pseudo-crew participant.\n"
        "    Hover over nodes and edges to inspect attributes. Drag nodes to rearrange.</p>\n"
        f"    <div class=\"guardrail-banner\">&#9888;&nbsp;{_esc(_GUARDRAIL)}&nbsp;\n"
        "    These graphs are research interpretation artifacts based on processed proxy data.\n"
        "    They do not represent actual Artemis&nbsp;II astronaut data.</div>\n"
        "  </div>\n"
        "  <div class=\"legend-bar\">\n"
        "    <strong>Activation levels:</strong>\n"
        "    <span><span class=\"ldot\" style=\"background:#AED6F1;\"></span>Low</span>\n"
        "    <span><span class=\"ldot\" style=\"background:#A9DFBF;\"></span>Mild</span>\n"
        "    <span><span class=\"ldot\" style=\"background:#F5CBA7;\"></span>Moderate</span>\n"
        "    <span><span class=\"ldot\" style=\"background:#F1948A;\"></span>High</span>\n"
        "    &nbsp;&nbsp;<strong>Node size</strong> = activation magnitude\n"
        "    &nbsp;&nbsp;<strong>Edges:</strong>\n"
        "    <span style=\"border-bottom:2px solid #7F8C8D;padding-bottom:1px;\">conceptual</span>\n"
        "    &nbsp;/&nbsp;\n"
        "    <span style=\"border-bottom:2px solid #CB4335;padding-bottom:1px;\">co-activation</span>\n"
        "  </div>\n"
        "  <table>\n"
        "    <thead><tr><th>Participant</th><th>Top domain</th>"
        "<th>Max activation</th><th>BACI score</th><th>BACI category</th></tr></thead>\n"
        f"    <tbody>\n{rows_html}    </tbody>\n"
        "  </table>\n"
        "  <div class=\"footer\">Edges are interpretive links, not causal proof. "
        "Node size reflects reference-relative domain activation magnitude.</div>\n"
        "</body>\n</html>"
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out

