"""Phase 12 — Dataset construction for the PyTorch temporal graph autoencoder.

Phase 12 uses PyTorch for self-supervised representation learning on structured
within-subject biological graph trajectories. The model learns compact latent
representations of baseline-relative trajectory patterns. It does not predict
health outcomes, diagnose conditions, classify mission readiness, measure hazard
exposure, or produce health risk scores.

Phase 12 does not treat features, domains, or timepoints as independent people.
Independent subject count remains small. The learning signal comes from
structured repeated graph states and baseline-relative trajectory segments
within subjects.

This module builds model-ready trajectory feature matrices from the Phase 6-11
output tables. The row unit is a graph-derived trajectory state
(subject_id + timepoint), not a person treated as a population sample.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# (filename without .csv, required|optional)
_INPUT_SPEC: list[tuple[str, str]] = [
    ("longitudinal_node_deltas", "required"),
    ("longitudinal_graph_deltas", "optional"),
    ("longitudinal_hazard_deltas", "optional"),
    ("recovery_metrics", "optional"),
    ("trajectory_node_attribution", "optional"),
    ("trajectory_graph_metric_attribution", "optional"),
    ("trajectory_subgraph_attribution", "optional"),
    ("trajectory_hazard_attribution", "optional"),
    ("recovery_attribution", "optional"),
    ("reference_calibrated_node_delta_scores", "optional"),
    ("reference_calibrated_graph_delta_scores", "optional"),
    ("reference_calibrated_hazard_delta_scores", "optional"),
    ("phase8_reference_calibrated_summary", "optional"),
    ("resilience_state_table", "optional"),
    ("mission_relevance_translation", "optional"),
]

_METADATA_COLS = [
    "trajectory_id", "subject_id", "timepoint", "mission_phase", "time_index",
    "data_type", "resilience_state_label", "dominant_adaptation_mode",
    "confidence_level",
]


def _slug(text: object) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(text).strip().lower())
    return re.sub(r"_+", "_", s).strip("_")


# ---------------------------------------------------------------------------
# Loading and readiness
# ---------------------------------------------------------------------------

def load_phase12_input_tables(
    results_dir: "str | Path" = "results/tables",
) -> dict[str, pd.DataFrame]:
    """Load available Phase 6-11 output tables. Missing optional tables do not fail."""
    results_dir = Path(results_dir)
    tables: dict[str, pd.DataFrame] = {}
    for name, _req in _INPUT_SPEC:
        path = results_dir / f"{name}.csv"
        if path.exists():
            try:
                tables[name] = pd.read_csv(path)
            except Exception:  # noqa: BLE001 - corrupt file treated as absent
                tables[name] = pd.DataFrame()
    return tables


def build_phase12_input_readiness_report(
    tables: dict[str, pd.DataFrame],
    results_dir: "str | Path" = "results/tables",
) -> pd.DataFrame:
    """Report presence/shape of each Phase 12 input table."""
    rows: list[dict] = []
    for name, req in _INPUT_SPEC:
        df = tables.get(name)
        present = isinstance(df, pd.DataFrame) and not df.empty
        if present:
            status, note = "available", "loaded"
        elif name in tables:
            status, note = "empty", "file present but empty"
        else:
            status = "missing"
            note = ("required core input missing" if req == "required"
                    else "optional input not present")
        rows.append({
            "table_name": name,
            "required_or_optional": req,
            "status": status,
            "rows": int(len(df)) if present else 0,
            "columns": int(df.shape[1]) if present else 0,
            "notes": note,
        })
    return pd.DataFrame(rows, columns=[
        "table_name", "required_or_optional", "status", "rows", "columns", "notes"])


# ---------------------------------------------------------------------------
# Feature matrix construction
# ---------------------------------------------------------------------------

def _pivot_family(
    df: pd.DataFrame,
    col: str,
    val: str,
    prefix: str,
    index_cols: list[str],
) -> pd.DataFrame | None:
    if df is None or df.empty or col not in df.columns or val not in df.columns:
        return None
    if not set(index_cols).issubset(df.columns):
        return None
    sub = df[index_cols + [col, val]].copy()
    sub[val] = pd.to_numeric(sub[val], errors="coerce")
    piv = sub.pivot_table(index=index_cols, columns=col, values=val, aggfunc="mean")
    piv.columns = [f"{prefix}{_slug(c)}" for c in piv.columns]
    return piv.reset_index()


def build_trajectory_feature_matrix(
    tables: dict[str, pd.DataFrame],
    fill_missing: float = 0.0,
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Build a graph-derived trajectory feature matrix (one row per subject/timepoint).

    Returns ``(feature_matrix, feature_catalog)``. The feature matrix carries
    metadata columns plus numeric feature columns; the catalog describes each
    feature and its source table.
    """
    node = tables.get("longitudinal_node_deltas")
    if node is None or node.empty or not {"subject_id", "timepoint"}.issubset(node.columns):
        empty = pd.DataFrame(columns=["trajectory_id", "subject_id", "timepoint", "mission_phase"])
        cat = pd.DataFrame(columns=["feature_name", "source_table", "feature_family", "description"])
        empty.attrs["note"] = "No usable longitudinal_node_deltas table; feature matrix is empty."
        return empty, cat

    idx = ["subject_id", "timepoint"]
    meta_cols = [c for c in ("subject_id", "timepoint", "mission_phase", "time_index", "data_type")
                 if c in node.columns]
    base = node[meta_cols].drop_duplicates(subset=idx).reset_index(drop=True)

    catalog: list[dict] = []

    def _merge(piv: pd.DataFrame | None, source: str, family: str, desc: str) -> None:
        nonlocal base
        if piv is None or piv.empty:
            return
        base = base.merge(piv, on=idx, how="left")
        for c in piv.columns:
            if c in idx:
                continue
            catalog.append({"feature_name": c, "source_table": source,
                            "feature_family": family, "description": desc})

    # 1. Domain delta features.
    _merge(_pivot_family(node, "domain", "delta_activation", "domain_delta__", idx),
           "longitudinal_node_deltas", "domain_delta",
           "Baseline-relative domain activation change.")
    # 2. Graph metric delta features.
    _merge(_pivot_family(tables.get("longitudinal_graph_deltas"), "metric",
                         "delta_value", "graph_delta__", idx),
           "longitudinal_graph_deltas", "graph_metric_delta",
           "Baseline-relative graph-metric change.")
    # 3. Hazard-context delta features.
    _merge(_pivot_family(tables.get("longitudinal_hazard_deltas"), "hazard",
                         "delta_hazard_relevance", "hazard_delta__", idx),
           "longitudinal_hazard_deltas", "hazard_context_delta",
           "Baseline-relative HRP hazard-context alignment change (not exposure).")
    # 4. Attribution features (Phase 7).
    _merge(_pivot_family(tables.get("trajectory_node_attribution"), "domain",
                         "contribution_share", "attr_node__", idx),
           "trajectory_node_attribution", "attribution_node",
           "Domain contribution share to the trajectory change.")
    _merge(_pivot_family(tables.get("trajectory_subgraph_attribution"), "subgraph_name",
                         "total_contribution_share", "attr_subgraph__", idx),
           "trajectory_subgraph_attribution", "attribution_subgraph",
           "Biological subgraph contribution share.")
    _merge(_pivot_family(tables.get("trajectory_hazard_attribution"), "hazard",
                         "contribution_share", "attr_hazard__", idx),
           "trajectory_hazard_attribution", "attribution_hazard",
           "Hazard-context contribution share (alignment, not exposure).")
    # 5. Envelope features (Phase 8 summary counts).
    env = tables.get("phase8_reference_calibrated_summary")
    if env is not None and not env.empty and set(idx).issubset(env.columns):
        env_cols = [c for c in ("n_outside_node_envelope", "n_outside_graph_envelope",
                                "n_outside_hazard_envelope") if c in env.columns]
        if env_cols:
            piv = env[idx + env_cols].drop_duplicates(subset=idx).copy()
            piv = piv.rename(columns={c: f"envelope__{c}" for c in env_cols})
            _merge(piv, "phase8_reference_calibrated_summary", "envelope",
                   "Count of features outside the reference-calibrated envelope.")

    # 6. Recovery features (subject-level, broadcast across timepoints).
    rec = tables.get("recovery_metrics")
    if rec is not None and not rec.empty and {"subject_id", "metric"}.issubset(rec.columns):
        for val, pfx, desc in (
            ("recovery_fraction", "recovery_frac__", "Per-metric recovery fraction toward baseline."),
            ("final_delta_from_baseline", "recovery_finaldelta__",
             "Per-metric final delta from personal baseline."),
        ):
            if val not in rec.columns:
                continue
            sub = rec[["subject_id", "metric", val]].copy()
            sub[val] = pd.to_numeric(sub[val], errors="coerce")
            piv = sub.pivot_table(index="subject_id", columns="metric", values=val, aggfunc="mean")
            piv.columns = [f"{pfx}{_slug(c)}" for c in piv.columns]
            piv = piv.reset_index()
            base = base.merge(piv, on="subject_id", how="left")
            for c in piv.columns:
                if c == "subject_id":
                    continue
                catalog.append({"feature_name": c, "source_table": "recovery_metrics",
                                "feature_family": "recovery", "description": desc})

    # trajectory_id + numeric fill.
    base.insert(0, "trajectory_id",
                base["subject_id"].astype(str) + "__" + base["timepoint"].astype(str))
    numeric_cols = [c for c in base.columns if c not in _METADATA_COLS]
    for c in numeric_cols:
        base[c] = pd.to_numeric(base[c], errors="coerce")
    base[numeric_cols] = base[numeric_cols].fillna(fill_missing)

    if "time_index" in base.columns:
        base = base.sort_values(["subject_id", "time_index", "timepoint"]).reset_index(drop=True)
    else:
        base = base.sort_values(["subject_id", "timepoint"]).reset_index(drop=True)

    catalog_df = pd.DataFrame(catalog, columns=[
        "feature_name", "source_table", "feature_family", "description"])
    return base, catalog_df


def encode_resilience_metadata(
    feature_matrix: pd.DataFrame,
    resilience_df: pd.DataFrame | None = None,
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Attach Phase 11 resilience annotations as **metadata** (not model inputs).

    Resilience labels are used for annotation and the consistency view, not as
    learning targets, so they are deliberately kept out of the numeric model
    features (see :func:`select_numeric_model_features`).

    Returns ``(feature_matrix_with_metadata, encoding_catalog)``.
    """
    out = feature_matrix.copy()
    catalog_rows: list[dict] = []

    if (resilience_df is None or resilience_df.empty
            or not {"subject_id", "timepoint"}.issubset(resilience_df.columns)):
        for col in ("resilience_state_label", "dominant_adaptation_mode", "confidence_level"):
            if col not in out.columns:
                out[col] = "not_available"
        return out, pd.DataFrame(catalog_rows, columns=[
            "encoding_name", "source_column", "category", "note"])

    keep = [c for c in ("subject_id", "timepoint", "resilience_state_label",
                        "dominant_adaptation_mode", "confidence_level")
            if c in resilience_df.columns]
    out = out.merge(resilience_df[keep].drop_duplicates(subset=["subject_id", "timepoint"]),
                    on=["subject_id", "timepoint"], how="left")
    for col in ("resilience_state_label", "dominant_adaptation_mode", "confidence_level"):
        if col in out.columns:
            out[col] = out[col].fillna("not_available")
        else:
            out[col] = "not_available"

    for col in ("resilience_state_label", "dominant_adaptation_mode"):
        if col not in out.columns:
            continue
        for cat in sorted(out[col].dropna().astype(str).unique()):
            catalog_rows.append({
                "encoding_name": f"{col}={cat}",
                "source_column": col,
                "category": cat,
                "note": "Phase 11 annotation metadata; excluded from model input features.",
            })
    return out, pd.DataFrame(catalog_rows, columns=[
        "encoding_name", "source_column", "category", "note"])


def select_numeric_model_features(
    feature_matrix: pd.DataFrame,
    metadata_cols: list[str] | None = None,
) -> "tuple[pd.DataFrame, list[str]]":
    """Select numeric model features, excluding metadata and text fields."""
    meta = set(metadata_cols if metadata_cols is not None else _METADATA_COLS)
    feature_names: list[str] = []
    for c in feature_matrix.columns:
        if c in meta:
            continue
        if pd.api.types.is_numeric_dtype(feature_matrix[c]):
            feature_names.append(c)
    X = feature_matrix[feature_names].copy().astype(float)
    return X, feature_names


def scale_model_features(X: pd.DataFrame) -> "tuple[pd.DataFrame, dict]":
    """Standardize numeric features (mean/std). Zero-variance columns map to 0."""
    scaler: dict = {"mean": {}, "std": {}}
    scaled = pd.DataFrame(index=X.index)
    for c in X.columns:
        col = X[c].astype(float)
        mean = float(col.mean())
        std = float(col.std(ddof=0))
        scaler["mean"][c] = mean
        scaler["std"][c] = std
        scaled[c] = (col - mean) / std if std > 1e-12 else 0.0
    return scaled, scaler


def create_masked_training_data(
    X_scaled: pd.DataFrame,
    mask_fraction: float = 0.15,
    random_state: int = 42,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Create masked self-supervised reconstruction data.

    Returns ``(X_masked, X_target, mask_matrix)`` as float32 arrays. Masked
    entries are set to 0.0; ``mask_matrix`` is 1.0 where an entry was masked.
    """
    rng = np.random.default_rng(random_state)
    X = X_scaled.to_numpy(dtype=np.float32)
    mask = (rng.random(X.shape) < float(mask_fraction))
    X_masked = X.copy()
    X_masked[mask] = 0.0
    return X_masked, X, mask.astype(np.float32)


def build_mask_summary(mask_matrix: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Summarize the masking applied (overall + per-feature masked fractions)."""
    if mask_matrix is None or mask_matrix.size == 0:
        return pd.DataFrame(columns=["scope", "name", "masked_fraction", "masked_count", "total"])
    rows = [{
        "scope": "overall", "name": "all_features",
        "masked_fraction": round(float(mask_matrix.mean()), 5),
        "masked_count": int(mask_matrix.sum()),
        "total": int(mask_matrix.size),
    }]
    per_feature = mask_matrix.mean(axis=0)
    counts = mask_matrix.sum(axis=0)
    n_rows = mask_matrix.shape[0]
    for name, frac, cnt in zip(feature_names, per_feature, counts):
        rows.append({
            "scope": "feature", "name": name,
            "masked_fraction": round(float(frac), 5),
            "masked_count": int(cnt), "total": int(n_rows),
        })
    return pd.DataFrame(rows, columns=[
        "scope", "name", "masked_fraction", "masked_count", "total"])


class TrajectoryFeatureDataset(Dataset):
    """PyTorch Dataset for trajectory feature reconstruction.

    Each item is ``(input_tensor, target_tensor)``. When ``X_target`` is not
    provided the target equals the input (plain reconstruction).
    """

    def __init__(self, X_input: np.ndarray, X_target: np.ndarray | None = None):
        # copy=True guarantees writable, contiguous arrays for torch.from_numpy.
        self.X_input = np.array(X_input, dtype=np.float32, copy=True)
        self.X_target = (np.array(X_target, dtype=np.float32, copy=True)
                         if X_target is not None else self.X_input.copy())
        if self.X_input.shape != self.X_target.shape:
            raise ValueError("X_input and X_target must have the same shape.")

    def __len__(self) -> int:
        return int(self.X_input.shape[0])

    def __getitem__(self, idx: int):
        return (torch.from_numpy(self.X_input[idx]),
                torch.from_numpy(self.X_target[idx]))
