"""Phase 5 — Hazard-aware graph-feature matrix, scaling, and PCA embedding.

This module assembles the Phase 5 feature matrix by combining:

  1. Phase 4 graph-level features,
  2. Phase 4 subgraph features (optional),
  3. Phase 5 HRP hazard relevance scores (optional),
  4. BACI context (optional).

It then provides transparent baseline tooling — numeric selection, scaling,
and PCA — for placing pseudo-crew biological adaptation graphs into a shared
graph-feature space.

No GNN, no PyTorch. PCA is used as an interpretable visualization of
graph-feature space, not as a classifier or validated manifold.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Graph-level numeric features carried into the Phase 5 matrix.
_GRAPH_LEVEL_FEATURES: list[str] = [
    "mean_node_activation",
    "median_node_activation",
    "max_node_activation",
    "total_node_activation",
    "n_active_domains",
    "active_domain_fraction",
    "graph_density",
    "mean_edge_weight",
    "max_edge_weight",
    "conceptual_edge_count",
    "coactivation_edge_count",
    "top_domain_activation",
]

# Files expected from Phase 4. Required vs optional is enforced in the loader.
_PHASE4_FILES: dict[str, str] = {
    "graph_level_features": "graph_level_features.csv",
    "node_level_features":  "node_level_features.csv",
    "edge_level_features":  "edge_level_features.csv",
    "subgraph_features":    "subgraph_features.csv",
}
_REQUIRED_TABLES = ("graph_level_features", "node_level_features")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_phase4_feature_tables(results_dir: "str | Path") -> dict[str, pd.DataFrame]:
    """Load available Phase 4 feature tables from ``<results_dir>/tables``.

    Parameters
    ----------
    results_dir:
        Path to the ``results`` directory containing a ``tables`` subfolder.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Keys are table names (e.g. ``"graph_level_features"``). Only tables
        that exist on disk are included.

    Raises
    ------
    FileNotFoundError
        If a required table (graph-level or node-level features) is missing.
    """
    tables_dir = Path(results_dir) / "tables"
    loaded: dict[str, pd.DataFrame] = {}
    for key, fname in _PHASE4_FILES.items():
        fpath = tables_dir / fname
        if fpath.exists():
            loaded[key] = pd.read_csv(fpath)

    missing_required = [k for k in _REQUIRED_TABLES if k not in loaded]
    if missing_required:
        raise FileNotFoundError(
            "Required Phase 4 tables not found in "
            f"{tables_dir}: {missing_required}. Run Phase 4 first."
        )
    return loaded


# ---------------------------------------------------------------------------
# Feature matrix assembly
# ---------------------------------------------------------------------------

def _coerce_numeric(series: pd.Series, fill_missing: float) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(fill_missing)


def build_hazard_aware_feature_matrix(
    graph_level_features: pd.DataFrame,
    node_level_features: pd.DataFrame,
    subgraph_features: pd.DataFrame | None = None,
    hazard_relevance_scores: pd.DataFrame | None = None,
    include_baci: bool = True,
    fill_missing: float = 0.0,
) -> pd.DataFrame:
    """Assemble the Phase 5 hazard-aware feature matrix (one row per subject).

    Feature families:
      * graph activation/connectivity features (Phase 4 graph-level),
      * subgraph activation features (Phase 4 subgraph, pivoted to columns),
      * HRP hazard relevance features (``hazard_relevance__<hazard>``),
      * BACI context (``baci_score``) when available.

    Parameters
    ----------
    graph_level_features:
        Phase 4 ``graph_level_features.csv`` (must contain ``subject_id``).
    node_level_features:
        Phase 4 ``node_level_features.csv`` — used to validate the subject set.
    subgraph_features:
        Optional Phase 4 ``subgraph_features.csv`` (long form).
    hazard_relevance_scores:
        Optional Phase 5 hazard relevance scores (long form with
        ``subject_id``, ``hazard``, ``hazard_relevance_score``).
    include_baci:
        Whether to include the ``baci_score`` column.
    fill_missing:
        Value used to fill missing numeric entries.

    Returns
    -------
    pandas.DataFrame
        Wide per-subject feature matrix with ``subject_id`` as the first column.
    """
    if "subject_id" not in graph_level_features.columns:
        raise ValueError("graph_level_features must contain a 'subject_id' column.")

    subjects = list(graph_level_features["subject_id"].astype(str))

    # Cross-check against node-level subject set (informative validation).
    if "subject_id" in node_level_features.columns:
        node_subjects = set(node_level_features["subject_id"].astype(str))
        graph_subjects = set(subjects)
        if node_subjects and graph_subjects and node_subjects != graph_subjects:
            # Not fatal — proceed on the graph-level subject set, which drives
            # the per-subject feature matrix.
            pass

    matrix = pd.DataFrame({"subject_id": subjects})

    # --- graph-level numeric features ---
    for col in _GRAPH_LEVEL_FEATURES:
        if col in graph_level_features.columns:
            matrix[col] = _coerce_numeric(
                graph_level_features[col].reset_index(drop=True), fill_missing
            )

    # --- BACI context ---
    if include_baci and "baci_score" in graph_level_features.columns:
        matrix["baci_score"] = _coerce_numeric(
            graph_level_features["baci_score"].reset_index(drop=True), fill_missing
        )

    # --- subgraph activation features (pivot long -> wide) ---
    if subgraph_features is not None and not subgraph_features.empty:
        if {"subject_id", "subgraph_name", "subgraph_activation_mean"}.issubset(
            subgraph_features.columns
        ):
            piv = subgraph_features.pivot_table(
                index="subject_id",
                columns="subgraph_name",
                values="subgraph_activation_mean",
                aggfunc="mean",
            )
            piv.columns = [f"subgraph_activation_mean__{c}" for c in piv.columns]
            piv = piv.reset_index()
            piv["subject_id"] = piv["subject_id"].astype(str)
            matrix = matrix.merge(piv, on="subject_id", how="left")

    # --- hazard relevance features (pivot long -> wide) ---
    if hazard_relevance_scores is not None and not hazard_relevance_scores.empty:
        if {"subject_id", "hazard", "hazard_relevance_score"}.issubset(
            hazard_relevance_scores.columns
        ):
            hpiv = hazard_relevance_scores.pivot_table(
                index="subject_id",
                columns="hazard",
                values="hazard_relevance_score",
                aggfunc="mean",
                dropna=False,   # keep hazards whose scores are all NaN (zero coverage)
            )
            hpiv.columns = [f"hazard_relevance__{c}" for c in hpiv.columns]
            hpiv = hpiv.reset_index()
            hpiv["subject_id"] = hpiv["subject_id"].astype(str)
            matrix = matrix.merge(hpiv, on="subject_id", how="left")

    # Fill any remaining numeric gaps (e.g. missing subgraph templates,
    # NaN hazard scores from zero domain coverage).
    numeric_cols = [c for c in matrix.columns if c != "subject_id"]
    matrix[numeric_cols] = matrix[numeric_cols].apply(
        lambda s: pd.to_numeric(s, errors="coerce")
    ).fillna(fill_missing)

    return matrix


def select_numeric_features(
    feature_matrix: pd.DataFrame,
    exclude_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Select numeric feature columns from a feature matrix.

    Parameters
    ----------
    feature_matrix:
        Wide per-subject feature matrix.
    exclude_columns:
        Columns to exclude (``subject_id`` is always excluded).

    Returns
    -------
    (X, feature_names)
        ``X`` is the numeric feature DataFrame; ``feature_names`` is the
        ordered list of selected columns.
    """
    exclude = set(exclude_columns or [])
    exclude.add("subject_id")
    numeric = feature_matrix.select_dtypes(include=[np.number])
    feature_names = [c for c in numeric.columns if c not in exclude]
    return numeric[feature_names].copy(), feature_names


def scale_feature_matrix(
    X: pd.DataFrame,
    method: str = "standard",
) -> tuple[pd.DataFrame, object]:
    """Scale a numeric feature matrix.

    Parameters
    ----------
    X:
        Numeric feature DataFrame.
    method:
        ``"standard"`` (z-score) or ``"minmax"``.

    Returns
    -------
    (X_scaled, scaler)
        ``X_scaled`` preserves the index and columns of ``X``. Constant
        (zero-variance) columns are mapped to all-zeros by StandardScaler,
        which is safe for downstream similarity/PCA.
    """
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method!r} (use 'standard' or 'minmax').")

    if X.shape[1] == 0:
        return X.copy(), scaler

    scaled = scaler.fit_transform(X.values)
    X_scaled = pd.DataFrame(scaled, index=X.index, columns=X.columns)
    return X_scaled, scaler


def compute_pca_embedding(
    X_scaled: pd.DataFrame,
    subject_ids: list[str],
    n_components: int = 2,
) -> tuple[pd.DataFrame, object | None]:
    """Compute a PCA embedding of the scaled feature matrix.

    PCA is a transparent visualization of graph-feature space, not a classifier
    or validated manifold. With very small samples the embedding is purely
    illustrative.

    Parameters
    ----------
    X_scaled:
        Scaled numeric feature matrix.
    subject_ids:
        Ordered subject identifiers (one per row of ``X_scaled``).
    n_components:
        Desired number of components (capped at ``min(n_samples, n_features)``).

    Returns
    -------
    (embedding_df, pca)
        ``embedding_df`` has ``subject_id`` plus ``PC1..PCk`` columns. ``pca``
        is the fitted sklearn ``PCA`` object, or ``None`` when an embedding
        cannot be computed (fewer than 2 samples or no features).
    """
    n_samples = X_scaled.shape[0]
    n_features = X_scaled.shape[1]

    if n_samples < 2 or n_features < 1:
        embedding_df = pd.DataFrame({"subject_id": list(subject_ids)})
        embedding_df["note"] = (
            "PCA skipped: requires at least 2 subjects and at least 1 feature."
        )
        return embedding_df, None

    k = min(n_components, n_samples, n_features)
    pca = PCA(n_components=k, random_state=42)
    coords = pca.fit_transform(X_scaled.values)

    embedding_df = pd.DataFrame({"subject_id": list(subject_ids)})
    for i in range(k):
        embedding_df[f"PC{i + 1}"] = coords[:, i]

    return embedding_df, pca
