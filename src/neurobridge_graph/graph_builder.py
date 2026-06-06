"""Phase 3 biological adaptation graph builder.

Converts per-participant processed NeuroBridge-S4 outputs into structured,
computable NetworkX graphs for downstream graph learning experiments.

Graph design: Hybrid Domain Activation Graph
  - nodes  = biological domains present in the data
  - edges  = conceptual biological relationships  +  within-subject co-activation
  - attrs  = activation level, BACI metadata, guardrail text
"""

from __future__ import annotations

import re
from pathlib import Path

import networkx as nx
import pandas as pd

from .schema import DOMAIN_NODES, CONCEPTUAL_EDGES


# ---------------------------------------------------------------------------
# Guardrail constants
# ---------------------------------------------------------------------------

GUARDRAIL = (
    "Research interpretation only; not diagnosis or treatment guidance."
)
GRAPH_TYPE = "subject_level_biological_adaptation_graph"
SOURCE_PROJECT = "NeuroBridge-S4 Graph Learning"
DATA_SOURCE = "processed NeuroBridge-S4 proxy outputs"

# ---------------------------------------------------------------------------
# Conceptual edge schema
# (canonical lower-case names → resolved at build time via canonical_domain_name)
# ---------------------------------------------------------------------------

CONCEPTUAL_EDGE_SCHEMA: list[tuple[str, str, str]] = [
    # (domain_a, domain_b, relationship_description)
    ("cardiovascular regulation",
     "metabolic regulation",
     "Cardiovascular and metabolic systems share regulatory feedback loops."),
    ("metabolic regulation",
     "body composition / physical status",
     "Metabolic state and body composition are bidirectionally coupled."),
    ("inflammation / immune-adjacent",
     "metabolic regulation",
     "Inflammatory signalling modulates metabolic pathways."),
    ("hematologic / oxygen-carrying",
     "cardiovascular regulation",
     "Oxygen-carrying capacity is tightly coupled to cardiovascular function."),
    ("sleep / circadian regulation",
     "autonomic regulation",
     "Sleep quality and circadian rhythm regulate autonomic tone."),
    ("autonomic regulation",
     "cardiovascular regulation",
     "Autonomic nervous system directly modulates heart rate and vascular tone."),
    ("sleep / circadian regulation",
     "recovery capacity",
     "Sleep is a primary driver of physiological recovery."),
    ("cognitive load",
     "recovery-related markers",
     "Cognitive demand influences physiological recovery markers."),
    ("emotional regulation",
     "cognitive load",
     "Emotional state and cognitive performance share neuroregulatory substrate."),
    ("recovery capacity",
     "inflammation / immune-adjacent",
     "Recovery processes involve inflammatory resolution pathways."),
]

# Future-proof edge types (documented, not yet implemented)
# FUTURE_EDGE_TYPES = [
#     "observed_reference_relationship",
#     "decision_support_relationship",
# ]

# Metadata columns that are NOT domain columns
_META_COL_PATTERNS = re.compile(
    r"(seqn|subject|participant|id|baci|category|crew|notes?|comment)",
    re.IGNORECASE,
)

# Known subject-ID column names (lower-case)
_SUBJECT_ID_CANDIDATES = [
    "seqn",
    "subject_id",
    "participant_id",
    "pseudo-crew member",
    "participant id",
    "id",
]


# ---------------------------------------------------------------------------
# Name normalisation helpers
# ---------------------------------------------------------------------------

def normalize_name(name: str) -> str:
    """Return lower-case, whitespace-collapsed, stripped version of *name*."""
    return re.sub(r"\s+", " ", name.strip().lower())


def canonical_domain_name(name: str) -> str:
    """Map variant spellings to a canonical lower-case domain key.

    This allows domain_scores.csv column names to be matched against the
    conceptual edge schema regardless of minor wording differences.
    """
    n = normalize_name(name)
    # Strip common suffixes that vary between files
    n = re.sub(r"\s*/\s*", " / ", n)  # normalise slash spacing

    # Explicit alias table
    aliases: dict[str, str] = {
        "cardiovascular regulation": "cardiovascular regulation",
        "metabolic regulation": "metabolic regulation",
        "body composition / physical status": "body composition / physical status",
        "body composition": "body composition / physical status",
        "inflammation / immune-adjacent status": "inflammation / immune-adjacent",
        "inflammation / immune-adjacent": "inflammation / immune-adjacent",
        "inflammation": "inflammation / immune-adjacent",
        "hematologic / oxygen-carrying capacity": "hematologic / oxygen-carrying",
        "hematologic / oxygen-carrying": "hematologic / oxygen-carrying",
        "hematologic": "hematologic / oxygen-carrying",
        "recovery-related markers": "recovery-related markers",
        "recovery capacity": "recovery capacity",
        "recovery": "recovery-related markers",
        "sleep / circadian regulation": "sleep / circadian regulation",
        "sleep": "sleep / circadian regulation",
        "autonomic regulation": "autonomic regulation",
        "cognitive load": "cognitive load",
        "emotional regulation": "emotional regulation",
        "monitoring priority": "monitoring priority",
        "countermeasure consideration": "countermeasure consideration",
    }
    return aliases.get(n, n)


# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------

def detect_subject_id_column(df: pd.DataFrame) -> str:
    """Return the name of the subject-ID column in *df*.

    Raises ValueError if no recognisable subject-ID column is found.
    """
    col_lower = {c.lower().strip(): c for c in df.columns}
    for candidate in _SUBJECT_ID_CANDIDATES:
        if candidate in col_lower:
            return col_lower[candidate]
    # Fallback: first column that matches the meta pattern
    for col in df.columns:
        if _META_COL_PATTERNS.search(col):
            return col
    # Last resort: first column
    return df.columns[0]


def detect_domain_columns(df: pd.DataFrame, subject_col: str) -> list[str]:
    """Return list of domain-score column names, excluding *subject_col* and metadata."""
    skip = {subject_col.lower().strip()}
    result = []
    for col in df.columns:
        if col.lower().strip() in skip:
            continue
        if _META_COL_PATTERNS.search(col):
            continue
        result.append(col)
    return result


# ---------------------------------------------------------------------------
# Activation classification
# ---------------------------------------------------------------------------

def classify_activation(value: float) -> str:
    """Return activation-level label for a domain score magnitude.

    Thresholds (mean |z-score|):
        < 0.75        → low
        0.75 – 1.0    → mild
        1.0  – 1.5    → moderate
        ≥ 1.5         → high
    """
    a = abs(value)
    if a < 0.75:
        return "low"
    if a < 1.0:
        return "mild"
    if a < 1.5:
        return "moderate"
    return "high"


def node_interpretation(domain: str, activation: float, activation_level: str) -> str:
    """Return a short plain-language phrase for a node's activation state."""
    if activation_level == "low":
        return (
            f"{domain}: reference-relative signal is low "
            "(within typical reference range)."
        )
    if activation_level == "mild":
        return (
            f"{domain}: mild reference-relative signal "
            "(slightly above typical range; monitor if sustained)."
        )
    if activation_level == "moderate":
        return (
            f"{domain}: moderate domain activation "
            "(notable deviation from reference; warrants review)."
        )
    return (
        f"{domain}: high domain activation "
        "(strong reference-relative signal; prioritise review)."
    )


# ---------------------------------------------------------------------------
# BACI extraction
# ---------------------------------------------------------------------------

def extract_baci_for_subject(
    baci_df: pd.DataFrame | None,
    subject_id: str,
) -> tuple[float | None, str]:
    """Return (baci_score, baci_category) for *subject_id* from *baci_df*.

    Returns (None, 'unknown') if *baci_df* is None or subject not found.
    """
    if baci_df is None:
        return None, "unknown"

    # Detect subject column
    try:
        id_col = detect_subject_id_column(baci_df)
    except Exception:
        return None, "unknown"

    # Normalise comparison
    norm_id = str(subject_id).strip()
    matches = baci_df[baci_df[id_col].astype(str).str.strip() == norm_id]
    if matches.empty:
        return None, "unknown"

    row = matches.iloc[0]
    score = None
    for c in baci_df.columns:
        if "baci" in c.lower() and "categor" not in c.lower():
            try:
                score = float(row[c])
            except (ValueError, TypeError):
                pass
            break

    category = "unknown"
    for c in baci_df.columns:
        if "categor" in c.lower():
            category = str(row[c])
            break

    return score, category


# ---------------------------------------------------------------------------
# Core graph builders
# ---------------------------------------------------------------------------

def build_subject_graph(
    subject_id: str,
    domain_row: pd.Series,
    baci_df: pd.DataFrame | None = None,
    add_coactivation_edges: bool = True,
    activation_threshold: float = 1.0,
) -> nx.Graph:
    """Build a single subject-level biological adaptation graph.

    Parameters
    ----------
    subject_id:
        Identifier string for this subject.
    domain_row:
        Series whose index is domain names and values are domain scores.
    baci_df:
        Optional DataFrame containing BACI scores.
    add_coactivation_edges:
        Whether to add within-subject co-activation edges.
    activation_threshold:
        Minimum activation magnitude for co-activation edges.

    Returns
    -------
    nx.Graph
    """
    G = nx.Graph()

    # ---- nodes ----
    domain_info: dict[str, dict] = {}
    for domain_col, score in domain_row.items():
        try:
            score_f = float(score)
        except (ValueError, TypeError):
            score_f = 0.0
        activation = abs(score_f)
        level = classify_activation(score_f)
        canonical = canonical_domain_name(str(domain_col))
        interp = node_interpretation(str(domain_col), activation, level)
        attrs = {
            "subject_id": subject_id,
            "domain": str(domain_col),
            "domain_score": round(score_f, 6),
            "activation": round(activation, 6),
            "activation_level": level,
            "node_type": "biological_domain",
            "data_source": DATA_SOURCE,
            "interpretation": interp,
        }
        G.add_node(str(domain_col), **attrs)
        domain_info[str(domain_col)] = {"canonical": canonical, "activation": activation, "level": level}

    # ---- conceptual edges ----
    col_to_canonical = {col: domain_info[col]["canonical"] for col in domain_info}
    canonical_to_cols: dict[str, list[str]] = {}
    for col, info in domain_info.items():
        c = info["canonical"]
        canonical_to_cols.setdefault(c, []).append(col)

    for ca, cb, relationship in CONCEPTUAL_EDGE_SCHEMA:
        cols_a = canonical_to_cols.get(ca, [])
        cols_b = canonical_to_cols.get(cb, [])
        for col_a in cols_a:
            for col_b in cols_b:
                if col_a == col_b:
                    continue
                G.add_edge(
                    col_a, col_b,
                    edge_type="conceptual_biological_relationship",
                    relationship=relationship,
                    weight=1.0,
                    source="NeuroBridge-S4 graph schema",
                    interpretation=(
                        "Conceptual biological relationship; not causal proof. "
                        "Flags domains for co-review."
                    ),
                )

    # ---- within-subject co-activation edges ----
    if add_coactivation_edges:
        active_cols = [
            col for col, info in domain_info.items()
            if info["activation"] >= activation_threshold
        ]
        for i, col_a in enumerate(active_cols):
            for col_b in active_cols[i + 1:]:
                act_a = domain_info[col_a]["activation"]
                act_b = domain_info[col_b]["activation"]
                mean_act = round((act_a + act_b) / 2, 6)
                if G.has_edge(col_a, col_b):
                    # Annotate existing edge rather than overwriting
                    G[col_a][col_b]["coactivation"] = True
                    G[col_a][col_b]["coactivation_weight"] = mean_act
                else:
                    G.add_edge(
                        col_a, col_b,
                        edge_type="within_subject_coactivation",
                        relationship=(
                            "Both domains show elevated reference-relative "
                            "activation in this subject."
                        ),
                        weight=mean_act,
                        source=DATA_SOURCE,
                        interpretation=(
                            "Co-activation is not causal proof; "
                            "it flags a pattern for review."
                        ),
                    )

    # ---- graph-level attributes ----
    baci_score, baci_category = extract_baci_for_subject(baci_df, subject_id)

    activations = [info["activation"] for info in domain_info.values()]
    n_active = sum(1 for a in activations if a >= activation_threshold)
    max_act = round(max(activations), 6) if activations else 0.0
    top_domain = max(domain_info, key=lambda c: domain_info[c]["activation"]) if domain_info else ""

    G.graph.update({
        "subject_id": subject_id,
        "baci_score": baci_score,
        "baci_category": baci_category,
        "n_domains": len(domain_info),
        "n_active_domains": n_active,
        "max_domain_activation": max_act,
        "top_domain": top_domain,
        "graph_type": GRAPH_TYPE,
        "source_project": SOURCE_PROJECT,
        "guardrail": GUARDRAIL,
    })

    return G


def build_all_subject_graphs(
    domain_scores: pd.DataFrame,
    baci_df: pd.DataFrame | None = None,
    add_coactivation_edges: bool = True,
    activation_threshold: float = 1.0,
) -> dict[str, nx.Graph]:
    """Build one graph per subject in *domain_scores*.

    Parameters
    ----------
    domain_scores:
        DataFrame with one row per subject; index or a column holds subject IDs.
    baci_df:
        Optional BACI scores DataFrame.

    Returns
    -------
    dict mapping subject_id -> nx.Graph
    """
    graphs: dict[str, nx.Graph] = {}

    # Handle either index-based or column-based subject IDs.
    # If the index is not a default RangeIndex (i.e. it has meaningful labels),
    # treat index values as subject IDs.
    import pandas as _pd
    if not isinstance(domain_scores.index, _pd.RangeIndex):
        id_source = "index"
        domain_cols = list(domain_scores.columns)
    else:
        try:
            id_col = detect_subject_id_column(domain_scores)
            id_source = "column"
            domain_cols = detect_domain_columns(domain_scores, id_col)
        except Exception:
            id_source = "index"
            domain_cols = list(domain_scores.columns)

    for row_label, row in domain_scores.iterrows():
        if id_source == "column":
            subject_id = str(row[id_col])
        else:
            subject_id = str(row_label)
        domain_row = row[domain_cols]
        graphs[subject_id] = build_subject_graph(
            subject_id=subject_id,
            domain_row=domain_row,
            baci_df=baci_df,
            add_coactivation_edges=add_coactivation_edges,
            activation_threshold=activation_threshold,
        )

    return graphs


# ---------------------------------------------------------------------------
# Export utilities
# ---------------------------------------------------------------------------

def export_node_table(graphs: dict[str, nx.Graph]) -> pd.DataFrame:
    """Return a flat DataFrame of all node attributes across all subject graphs."""
    rows = []
    for subject_id, G in graphs.items():
        for node, attrs in G.nodes(data=True):
            row = {"graph_subject_id": subject_id, "node": node}
            row.update(attrs)
            rows.append(row)
    return pd.DataFrame(rows)


def export_edge_table(graphs: dict[str, nx.Graph]) -> pd.DataFrame:
    """Return a flat DataFrame of all edge attributes across all subject graphs."""
    rows = []
    for subject_id, G in graphs.items():
        for u, v, attrs in G.edges(data=True):
            row = {"graph_subject_id": subject_id, "source_node": u, "target_node": v}
            row.update(attrs)
            rows.append(row)
    return pd.DataFrame(rows)


def export_graph_summary(graphs: dict[str, nx.Graph]) -> pd.DataFrame:
    """Return a one-row-per-graph summary DataFrame."""
    rows = []
    for subject_id, G in graphs.items():
        row = {
            "subject_id": subject_id,
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
        }
        row.update({k: v for k, v in G.graph.items() if k != "subject_id"})
        rows.append(row)
    return pd.DataFrame(rows)


def save_graphml_files(
    graphs: dict[str, nx.Graph],
    output_dir: Path,
) -> list[Path]:
    """Save each graph as a GraphML file.

    Returns list of written paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for subject_id, G in graphs.items():
        # GraphML requires string attribute values for node/edge attrs
        H = nx.Graph()
        H.graph.update({k: str(v) if v is not None else "" for k, v in G.graph.items()})
        for node, attrs in G.nodes(data=True):
            H.add_node(node, **{k: str(v) if v is not None else "" for k, v in attrs.items()})
        for u, v, attrs in G.edges(data=True):
            H.add_edge(u, v, **{k: str(v2) if v2 is not None else "" for k, v2 in attrs.items()})
        safe_id = re.sub(r"[^\w\-]", "_", str(subject_id))
        path = output_dir / f"subject_graph_{safe_id}.graphml"
        nx.write_graphml(H, path)
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# Backward-compatible Phase 1 function (kept for existing tests)
# ---------------------------------------------------------------------------

def build_empty_schema_graph() -> nx.Graph:
    """Build a conceptual schema graph without participant data (Phase 1)."""
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
