"""Phase 5 — Graph-feature similarity and distance.

Transparent, baseline structural comparison of pseudo-crew biological
adaptation graphs in the hazard-aware feature space:

  * cosine similarity  = similarity of the graph-feature *pattern*,
  * Euclidean distance = absolute separation in scaled feature space.

Both are structural comparisons of graph-feature vectors. They are **not**
diagnosis, not exposure measurement, and not causal proof. Two participants
being "similar" means their graph-feature patterns are close, not that they
share the same health state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


def _as_matrix(X_scaled: pd.DataFrame, subject_ids: list[str]) -> np.ndarray:
    if X_scaled.shape[0] != len(subject_ids):
        raise ValueError(
            f"Row count ({X_scaled.shape[0]}) does not match number of "
            f"subject_ids ({len(subject_ids)})."
        )
    return X_scaled.values


def compute_cosine_similarity_matrix(
    X_scaled: pd.DataFrame,
    subject_ids: list[str],
) -> pd.DataFrame:
    """Cosine similarity between subjects in scaled feature space.

    Returns a square DataFrame indexed and columned by ``subject_ids``. The
    diagonal is forced to ``1.0`` (a subject is identical to itself, even if
    its scaled feature vector is all zeros).
    """
    mat = _as_matrix(X_scaled, subject_ids)
    if mat.shape[1] == 0:
        sim = np.eye(len(subject_ids))
    else:
        sim = cosine_similarity(mat)
        np.fill_diagonal(sim, 1.0)
    return pd.DataFrame(sim, index=subject_ids, columns=subject_ids)


def compute_euclidean_distance_matrix(
    X_scaled: pd.DataFrame,
    subject_ids: list[str],
) -> pd.DataFrame:
    """Euclidean distance between subjects in scaled feature space.

    Returns a square DataFrame indexed and columned by ``subject_ids`` with a
    zero diagonal.
    """
    mat = _as_matrix(X_scaled, subject_ids)
    if mat.shape[1] == 0:
        dist = np.zeros((len(subject_ids), len(subject_ids)))
    else:
        dist = euclidean_distances(mat)
        np.fill_diagonal(dist, 0.0)
    return pd.DataFrame(dist, index=subject_ids, columns=subject_ids)


def summarize_pairwise_similarity(
    similarity_matrix: pd.DataFrame,
    distance_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize unique subject pairs by similarity and distance.

    Returns
    -------
    pandas.DataFrame
        One row per unordered pair with columns ``subject_a``, ``subject_b``,
        ``cosine_similarity``, ``euclidean_distance``, sorted by descending
        cosine similarity.
    """
    subjects = list(similarity_matrix.index)
    rows: list[dict] = []
    for i in range(len(subjects)):
        for j in range(i + 1, len(subjects)):
            a, b = subjects[i], subjects[j]
            rows.append({
                "subject_a":          a,
                "subject_b":          b,
                "cosine_similarity":  round(float(similarity_matrix.loc[a, b]), 5),
                "euclidean_distance": round(float(distance_matrix.loc[a, b]), 5),
            })
    summary = pd.DataFrame(rows, columns=[
        "subject_a", "subject_b", "cosine_similarity", "euclidean_distance",
    ])
    if not summary.empty:
        summary = summary.sort_values(
            "cosine_similarity", ascending=False
        ).reset_index(drop=True)
    return summary


def identify_most_similar_pair(summary_df: pd.DataFrame) -> dict:
    """Return the most similar subject pair (highest cosine similarity)."""
    if summary_df is None or summary_df.empty:
        return {
            "subject_a": None, "subject_b": None,
            "cosine_similarity": float("nan"),
            "euclidean_distance": float("nan"),
            "note": "No pairs available (fewer than 2 subjects).",
        }
    top = summary_df.sort_values("cosine_similarity", ascending=False).iloc[0]
    return {
        "subject_a":          top["subject_a"],
        "subject_b":          top["subject_b"],
        "cosine_similarity":  float(top["cosine_similarity"]),
        "euclidean_distance": float(top["euclidean_distance"]),
    }


def identify_most_distinct_subject(distance_matrix: pd.DataFrame) -> dict:
    """Return the most distinct subject (largest mean distance to others)."""
    subjects = list(distance_matrix.index)
    if len(subjects) < 2:
        return {
            "subject_id": subjects[0] if subjects else None,
            "mean_distance_to_others": float("nan"),
            "note": "Only one subject; distinctness is undefined.",
        }
    mean_dist = {}
    for s in subjects:
        others = [o for o in subjects if o != s]
        mean_dist[s] = float(distance_matrix.loc[s, others].mean())
    most = max(mean_dist, key=mean_dist.get)
    return {
        "subject_id":              most,
        "mean_distance_to_others": round(mean_dist[most], 5),
    }
