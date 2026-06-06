"""Phase 8 — Reference-calibrated trajectory envelope.

Phase 8 does **not** replace the within-subject self-baseline logic of Phases 6
and 7. The primary comparison remains the subject's current state vs that
subject's own baseline. Phase 8 adds a *secondary calibration layer*: it
estimates whether a within-subject baseline-relative graph change is small,
moderate, or unusually large relative to an **expected variability envelope**
derived from reference/analog data.

> The reference envelope does not define whether a person is healthy or
> unhealthy. It calibrates how large a within-subject graph change is relative to
> expected variability in available proxy or analog data.

Envelope exceedance is **not** diagnosis, **not** a health risk score, **not**
treatment guidance, and **not** exposure measurement. Outside-envelope means the
baseline-relative change is larger than expected under the current calibration
data and may be a candidate for expert review.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_DEMO_DATA_TYPE = "schema_demonstration_not_scientific_evidence"

# MAD-to-sigma scaling for a normal distribution (so robust z is comparable to
# a standard z-score).
_MAD_SCALE = 1.4826

# Robust-z thresholds for envelope classification.
_Z_NEAR = 1.5
_Z_OUTSIDE = 2.0

# Minimum reference sample count to trust an envelope.
_MIN_REFERENCE_N = 5

CORE_ENVELOPE_STATEMENT = (
    "The reference envelope does not define whether a person is healthy or "
    "unhealthy. It calibrates how large a within-subject graph change is "
    "relative to expected variability in available proxy or analog data."
)

# Envelope position categories.
WITHIN = "within_expected_envelope"
NEAR = "near_envelope_boundary"
OUTSIDE = "outside_expected_envelope"
INSUFFICIENT = "insufficient_reference"

_REFERENCE_FILES = {
    "reference_longitudinal_deltas":     "reference_longitudinal_deltas.csv",
    "reference_domain_variability":      "reference_domain_variability.csv",
    "reference_graph_feature_variability": "reference_graph_feature_variability.csv",
    "reference_hazard_variability":      "reference_hazard_variability.csv",
}

_ENVELOPE_COLUMNS = [
    "feature", "feature_type", "group", "n_reference", "median_delta",
    "mad_delta", "lower_q", "upper_q", "lower_bound", "upper_bound",
    "envelope_method", "data_type",
]


# ---------------------------------------------------------------------------
# Loading reference inputs
# ---------------------------------------------------------------------------

def load_reference_envelope_inputs(root: "str | Path") -> dict[str, pd.DataFrame]:
    """Load available reference/analog calibration files.

    Parameters
    ----------
    root:
        A directory that may contain reference files directly, or a project root
        whose ``data/processed`` subfolder contains them.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Only files that exist on disk are included. Keys are the logical table
        names (see ``_REFERENCE_FILES``).
    """
    root = Path(root)
    search_dirs = [root, root / "data" / "processed"]
    loaded: dict[str, pd.DataFrame] = {}
    for key, fname in _REFERENCE_FILES.items():
        for d in search_dirs:
            fpath = d / fname
            if fpath.exists():
                loaded[key] = pd.read_csv(fpath)
                break
    return loaded


# ---------------------------------------------------------------------------
# Example (schema-demonstration) envelope
# ---------------------------------------------------------------------------

def create_example_reference_envelope(output_path: "str | Path") -> pd.DataFrame:
    """Create a small schema-only example reference envelope table.

    This is **not scientific evidence**. It exists only to demonstrate the
    calibration workflow and must be replaced with real analog/reference
    variability data. Every row carries
    ``data_type = 'schema_demonstration_not_scientific_evidence'``.

    The envelope is expressed as a precomputed variability summary (median,
    MAD, and 5th/95th-percentile bounds per feature), which
    :func:`build_envelope_from_summary_table` normalizes into the standard
    envelope schema.
    """
    # feature, feature_type, n_reference, median_delta, mad_delta, lower, upper
    rows = [
        # Node / biological domain deltas.
        ("Body composition / physical status", "node", 30, 0.0, 0.05, -0.12, 0.12),
        ("Cardiovascular regulation",          "node", 30, 0.0, 0.05, -0.12, 0.12),
        ("Hematologic / oxygen-carrying",      "node", 30, 0.0, 0.045, -0.11, 0.11),
        ("Inflammation / immune-adjacent",     "node", 30, 0.0, 0.06, -0.14, 0.14),
        ("Metabolic regulation",               "node", 30, 0.0, 0.05, -0.12, 0.12),
        ("Recovery-related markers",           "node", 30, 0.0, 0.05, -0.12, 0.12),
        # Graph-level metric deltas (per-metric scales differ).
        ("mean_node_activation",     "graph", 30, 0.0, 0.03, -0.08, 0.08),
        ("max_node_activation",      "graph", 30, 0.0, 0.04, -0.10, 0.10),
        ("total_node_activation",    "graph", 30, 0.0, 0.20, -0.50, 0.50),
        ("n_active_domains",         "graph", 30, 0.0, 0.50, -1.00, 1.00),
        ("active_domain_fraction",   "graph", 30, 0.0, 0.08, -0.20, 0.20),
        ("graph_density",            "graph", 30, 0.0, 0.05, -0.12, 0.12),
        ("coactivation_edge_count",  "graph", 30, 0.0, 1.00, -2.50, 2.50),
        # Hazard-context relevance deltas.
        ("space_radiation",              "hazard", 30, 0.0, 0.05, -0.12, 0.12),
        ("isolation_and_confinement",    "hazard", 30, 0.0, 0.05, -0.12, 0.12),
        ("distance_from_earth",          "hazard", 30, 0.0, 0.05, -0.12, 0.12),
        ("gravity_fields",               "hazard", 30, 0.0, 0.05, -0.12, 0.12),
        ("hostile_closed_environments",  "hazard", 30, 0.0, 0.05, -0.12, 0.12),
    ]
    df = pd.DataFrame(rows, columns=[
        "feature", "feature_type", "n_reference", "median_delta", "mad_delta",
        "lower_bound", "upper_bound",
    ])
    df["lower_q"] = 0.05
    df["upper_q"] = 0.95
    df["envelope_method"] = "example_summary_quantile_mad"
    df["data_type"] = SCHEMA_DEMO_DATA_TYPE

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


# ---------------------------------------------------------------------------
# Envelope construction
# ---------------------------------------------------------------------------

def build_envelope_from_reference_deltas(
    reference_delta_df: pd.DataFrame,
    feature_col: str = "feature",
    delta_col: str = "delta_value",
    group_cols: list[str] | None = None,
    lower_q: float = 0.05,
    upper_q: float = 0.95,
) -> pd.DataFrame:
    """Build an expected-variability envelope from raw reference delta samples.

    For each feature (and optional group), computes the median delta, the median
    absolute deviation (MAD), and the lower/upper quantile bounds. These define
    the expected variability envelope used to calibrate within-subject deltas.
    """
    required = {feature_col, delta_col}
    missing = required - set(reference_delta_df.columns)
    if missing:
        raise ValueError(f"reference_delta_df missing columns: {missing}")

    keys = [feature_col] + (group_cols or [])
    feature_type = None
    if "feature_type" in reference_delta_df.columns:
        feature_type = "feature_type"

    rows: list[dict] = []
    for key_vals, grp in reference_delta_df.groupby(keys):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        feature = key_vals[0]
        group = "::".join(str(v) for v in key_vals[1:]) if group_cols else np.nan

        deltas = pd.to_numeric(grp[delta_col], errors="coerce").dropna()
        n_ref = int(len(deltas))
        if n_ref == 0:
            median = mad = lb = ub = np.nan
        else:
            median = float(deltas.median())
            mad = float((deltas - median).abs().median())
            lb = float(deltas.quantile(lower_q))
            ub = float(deltas.quantile(upper_q))

        ftype = (str(grp[feature_type].iloc[0])
                 if feature_type is not None and not grp.empty else np.nan)

        rows.append({
            "feature":         feature,
            "feature_type":    ftype,
            "group":           group,
            "n_reference":     n_ref,
            "median_delta":    round(median, 6) if pd.notna(median) else np.nan,
            "mad_delta":       round(mad, 6) if pd.notna(mad) else np.nan,
            "lower_q":         lower_q,
            "upper_q":         upper_q,
            "lower_bound":     round(lb, 6) if pd.notna(lb) else np.nan,
            "upper_bound":     round(ub, 6) if pd.notna(ub) else np.nan,
            "envelope_method": "reference_delta_quantile_mad",
            "data_type":       (str(reference_delta_df["data_type"].iloc[0])
                                if "data_type" in reference_delta_df.columns
                                and not reference_delta_df.empty else "unknown"),
        })

    return pd.DataFrame(rows, columns=_ENVELOPE_COLUMNS)


def build_envelope_from_summary_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Accept a precomputed reference variability summary and normalize columns.

    Recognizes common column aliases and fills any missing standard columns with
    sensible defaults. Bounds are required (or derivable from median +/- a MAD
    multiple); if neither bounds nor MAD are present, bounds become NaN and the
    feature will score as ``insufficient_reference``.
    """
    if summary_df is None or summary_df.empty:
        return pd.DataFrame(columns=_ENVELOPE_COLUMNS)

    df = summary_df.copy()
    aliases = {
        "feature": ["feature", "domain", "metric", "hazard", "name"],
        "feature_type": ["feature_type", "type", "layer"],
        "group": ["group"],
        "n_reference": ["n_reference", "n", "count", "n_ref"],
        "median_delta": ["median_delta", "median", "med"],
        "mad_delta": ["mad_delta", "mad"],
        "lower_q": ["lower_q", "q_low"],
        "upper_q": ["upper_q", "q_high"],
        "lower_bound": ["lower_bound", "lower", "p05", "q05"],
        "upper_bound": ["upper_bound", "upper", "p95", "q95"],
        "envelope_method": ["envelope_method", "method"],
        "data_type": ["data_type"],
    }

    out = pd.DataFrame()
    for std_col, candidates in aliases.items():
        found = next((c for c in candidates if c in df.columns), None)
        if found is not None:
            out[std_col] = df[found]

    if "feature" not in out.columns:
        raise ValueError(
            "summary_df must contain a feature/domain/metric/hazard column."
        )

    # Defaults for missing optional columns.
    if "feature_type" not in out.columns:
        out["feature_type"] = np.nan
    if "group" not in out.columns:
        out["group"] = np.nan
    if "median_delta" not in out.columns:
        out["median_delta"] = 0.0
    if "mad_delta" not in out.columns:
        out["mad_delta"] = np.nan
    if "n_reference" not in out.columns:
        out["n_reference"] = _MIN_REFERENCE_N
    if "lower_q" not in out.columns:
        out["lower_q"] = 0.05
    if "upper_q" not in out.columns:
        out["upper_q"] = 0.95

    # Derive bounds from median +/- scaled MAD when bounds absent.
    if "lower_bound" not in out.columns or "upper_bound" not in out.columns:
        scaled = _MAD_SCALE * out["mad_delta"]
        if "lower_bound" not in out.columns:
            out["lower_bound"] = out["median_delta"] - 2 * scaled
        if "upper_bound" not in out.columns:
            out["upper_bound"] = out["median_delta"] + 2 * scaled

    if "envelope_method" not in out.columns:
        out["envelope_method"] = "summary_table_normalized"
    if "data_type" not in out.columns:
        out["data_type"] = "unknown"

    return out[_ENVELOPE_COLUMNS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Robust z and classification
# ---------------------------------------------------------------------------

def compute_robust_z_score(
    value: float,
    median: float,
    mad: float,
    epsilon: float = 1e-9,
) -> float:
    """Compute a robust z-like score using the median and MAD.

    Uses the standard ``1.4826 * MAD`` scaling so the result is comparable to a
    conventional z-score under approximate normality. Returns ``nan`` if inputs
    are missing.
    """
    if value is None or median is None or mad is None:
        return float("nan")
    if pd.isna(value) or pd.isna(median) or pd.isna(mad):
        return float("nan")
    scale = _MAD_SCALE * float(mad)
    return float((float(value) - float(median)) / (scale + epsilon))


def classify_envelope_position(
    delta_value: float,
    lower_bound: float,
    upper_bound: float,
    robust_z: float | None = None,
) -> str:
    """Classify where a delta falls relative to the reference envelope.

    Returns one of ``within_expected_envelope``, ``near_envelope_boundary``,
    ``outside_expected_envelope``, or ``insufficient_reference``.
    """
    bounds_missing = (
        lower_bound is None or upper_bound is None
        or pd.isna(lower_bound) or pd.isna(upper_bound)
    )
    z_missing = robust_z is None or pd.isna(robust_z)
    if bounds_missing and z_missing:
        return INSUFFICIENT
    if delta_value is None or pd.isna(delta_value):
        return INSUFFICIENT

    az = abs(robust_z) if not z_missing else None

    # Outside: beyond bounds OR robust z at/above the outside threshold.
    outside_bounds = (not bounds_missing) and (
        delta_value < lower_bound or delta_value > upper_bound
    )
    if outside_bounds or (az is not None and az >= _Z_OUTSIDE):
        return OUTSIDE

    # Near boundary: robust z in the near band, or inside but close to a bound.
    if az is not None and az >= _Z_NEAR:
        return NEAR
    if not bounds_missing:
        width = upper_bound - lower_bound
        if width > 0:
            margin = 0.1 * width
            if (delta_value <= lower_bound + margin) or (delta_value >= upper_bound - margin):
                return NEAR

    return WITHIN


def _exceedance(delta_value: float, lower_bound: float, upper_bound: float) -> float:
    """Signed-free distance beyond the nearest bound (0 when inside)."""
    if pd.isna(delta_value) or pd.isna(lower_bound) or pd.isna(upper_bound):
        return float("nan")
    if delta_value > upper_bound:
        return float(delta_value - upper_bound)
    if delta_value < lower_bound:
        return float(lower_bound - delta_value)
    return 0.0


# ---------------------------------------------------------------------------
# Scoring deltas against the envelope
# ---------------------------------------------------------------------------

def _envelope_lookup(envelope_df: pd.DataFrame) -> dict:
    """Index envelope rows by feature name for fast lookup."""
    lookup: dict[str, dict] = {}
    if envelope_df is None or envelope_df.empty:
        return lookup
    for _, row in envelope_df.iterrows():
        lookup[str(row["feature"])] = row.to_dict()
    return lookup


def _score_generic(
    delta_df: pd.DataFrame,
    envelope_df: pd.DataFrame,
    feature_col: str,
    delta_col: str,
    out_feature_name: str,
    layer_label: str,
) -> pd.DataFrame:
    out_cols = [
        "subject_id", "timepoint", "mission_phase", out_feature_name,
        ("delta_activation" if layer_label == "node"
         else ("delta_value" if layer_label == "graph" else "delta_hazard_relevance")),
        "reference_median_delta", "reference_mad_delta", "lower_bound",
        "upper_bound", "robust_z", "envelope_position", "envelope_exceedance",
        "interpretation",
    ]
    if delta_df is None or delta_df.empty:
        return pd.DataFrame(columns=out_cols)

    required = {"subject_id", "timepoint", feature_col, delta_col}
    missing = required - set(delta_df.columns)
    if missing:
        raise ValueError(f"{layer_label} delta_df missing columns: {missing}")

    lookup = _envelope_lookup(envelope_df)
    has_phase = "mission_phase" in delta_df.columns
    delta_out_col = out_cols[4]

    rows: list[dict] = []
    for _, r in delta_df.iterrows():
        feature = r[feature_col]
        delta_val = pd.to_numeric(pd.Series([r[delta_col]]), errors="coerce").iloc[0]
        env = lookup.get(str(feature))

        if env is None:
            median = mad = lb = ub = np.nan
            n_ref = 0
        else:
            median = env.get("median_delta", np.nan)
            mad = env.get("mad_delta", np.nan)
            lb = env.get("lower_bound", np.nan)
            ub = env.get("upper_bound", np.nan)
            n_ref = env.get("n_reference", 0)

        insufficient = env is None or (pd.notna(n_ref) and float(n_ref) < _MIN_REFERENCE_N)
        if insufficient:
            position = INSUFFICIENT
            rz = float("nan")
            exceed = float("nan")
        else:
            rz = compute_robust_z_score(delta_val, median, mad)
            position = classify_envelope_position(delta_val, lb, ub, rz)
            exceed = _exceedance(delta_val, lb, ub)

        row = {
            "subject_id":             r["subject_id"],
            "timepoint":              r["timepoint"],
            "mission_phase":          r.get("mission_phase", "unknown") if has_phase else "unknown",
            out_feature_name:         feature,
            delta_out_col:            round(float(delta_val), 5) if pd.notna(delta_val) else np.nan,
            "reference_median_delta": round(float(median), 5) if pd.notna(median) else np.nan,
            "reference_mad_delta":    round(float(mad), 5) if pd.notna(mad) else np.nan,
            "lower_bound":            round(float(lb), 5) if pd.notna(lb) else np.nan,
            "upper_bound":            round(float(ub), 5) if pd.notna(ub) else np.nan,
            "robust_z":               round(float(rz), 4) if pd.notna(rz) else np.nan,
            "envelope_position":      position,
            "envelope_exceedance":    round(float(exceed), 5) if pd.notna(exceed) else np.nan,
        }
        row["interpretation"] = _interpret(row, layer_label, out_feature_name)
        rows.append(row)

    return pd.DataFrame(rows, columns=out_cols)


def _interpret(row: dict, layer_label: str, feature_name_col: str) -> str:
    feature = row[feature_name_col]
    position = row["envelope_position"]
    layer_word = {"node": "domain", "graph": "graph metric", "hazard": "hazard-context"}[layer_label]
    guard = (
        "This is a baseline-relative change relative to expected variability, "
        "not diagnosis, risk scoring, or exposure measurement."
    )
    if position == OUTSIDE:
        return (
            f"The {layer_word} delta for {feature} is outside the expected "
            "variability envelope. This identifies a baseline-relative change "
            f"that may deserve expert review. {guard}"
        )
    if position == NEAR:
        return (
            f"The {layer_word} delta for {feature} is near the expected "
            f"variability envelope boundary. {guard}"
        )
    if position == INSUFFICIENT:
        return (
            f"Insufficient reference calibration data for {feature}; the envelope "
            "position cannot be determined under the current calibration data."
        )
    return (
        f"The {layer_word} delta for {feature} remains within the expected "
        "variability envelope under the current calibration data."
    )


def score_node_deltas_against_envelope(
    node_delta_df: pd.DataFrame,
    envelope_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare node/domain deltas against the reference-calibrated envelope."""
    return _score_generic(
        node_delta_df, envelope_df,
        feature_col="domain", delta_col="delta_activation",
        out_feature_name="domain", layer_label="node",
    )


def score_graph_deltas_against_envelope(
    graph_delta_df: pd.DataFrame,
    envelope_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare graph metric deltas against the reference-calibrated envelope."""
    return _score_generic(
        graph_delta_df, envelope_df,
        feature_col="metric", delta_col="delta_value",
        out_feature_name="metric", layer_label="graph",
    )


def score_hazard_deltas_against_envelope(
    hazard_delta_df: pd.DataFrame,
    envelope_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare hazard-context deltas against the reference-calibrated envelope.

    Guardrail: this does not measure hazard exposure or risk. It calibrates
    whether the hazard-context mapping changed more than expected under
    reference variability.
    """
    return _score_generic(
        hazard_delta_df, envelope_df,
        feature_col="hazard", delta_col="delta_hazard_relevance",
        out_feature_name="hazard", layer_label="hazard",
    )


# ---------------------------------------------------------------------------
# Phase 8 summary
# ---------------------------------------------------------------------------

def _top_outside(scores: pd.DataFrame, feature_col: str) -> str:
    if scores is None or scores.empty:
        return "n/a"
    outside = scores[scores["envelope_position"] == OUTSIDE]
    if outside.empty:
        return "n/a"
    outside = outside.copy()
    outside["_rank"] = outside["envelope_exceedance"].fillna(0).abs()
    return str(outside.sort_values("_rank", ascending=False).iloc[0][feature_col])


def build_phase8_envelope_summary(
    node_scores: pd.DataFrame,
    graph_scores: pd.DataFrame,
    hazard_scores: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build one summary row per subject/timepoint of envelope exceedance."""
    out_cols = [
        "subject_id", "timepoint", "mission_phase",
        "n_outside_node_envelope", "n_outside_graph_envelope",
        "n_outside_hazard_envelope", "top_outside_domain",
        "top_outside_graph_metric", "top_outside_hazard_context",
        "overall_envelope_flag", "interpretation",
    ]

    frames = [df for df in [node_scores, graph_scores, hazard_scores]
              if df is not None and not df.empty]
    if not frames:
        return pd.DataFrame(columns=out_cols)

    keys = pd.concat(
        [f[["subject_id", "timepoint", "mission_phase"]] for f in frames],
        ignore_index=True,
    ).drop_duplicates().reset_index(drop=True)

    rows: list[dict] = []
    for _, key in keys.iterrows():
        sid, tp, phase = key["subject_id"], key["timepoint"], key["mission_phase"]

        def _slice(df):
            if df is None or df.empty:
                return df
            return df[(df["subject_id"] == sid) & (df["timepoint"] == tp)]

        n_node = n_graph = n_haz = 0
        positions: list[str] = []

        ns = _slice(node_scores)
        if ns is not None and not ns.empty:
            n_node = int((ns["envelope_position"] == OUTSIDE).sum())
            positions += list(ns["envelope_position"])
        gs = _slice(graph_scores)
        if gs is not None and not gs.empty:
            n_graph = int((gs["envelope_position"] == OUTSIDE).sum())
            positions += list(gs["envelope_position"])
        hs = _slice(hazard_scores)
        if hs is not None and not hs.empty:
            n_haz = int((hs["envelope_position"] == OUTSIDE).sum())
            positions += list(hs["envelope_position"])

        total_outside = n_node + n_graph + n_haz
        any_near = NEAR in positions
        only_insufficient = len(positions) > 0 and all(p == INSUFFICIENT for p in positions)

        if only_insufficient:
            flag = INSUFFICIENT
        elif total_outside > 0:
            flag = OUTSIDE
        elif any_near:
            flag = NEAR
        else:
            flag = WITHIN

        if flag == OUTSIDE:
            interp = (
                f"At {tp} ({phase}), {total_outside} baseline-relative delta(s) fall "
                "outside the expected variability envelope; candidate for expert "
                "review. Outside-envelope does not mean disease, danger, or health "
                "risk."
            )
        elif flag == NEAR:
            interp = (
                f"At {tp} ({phase}), one or more deltas are near the envelope "
                "boundary but none are clearly outside under the current calibration."
            )
        elif flag == INSUFFICIENT:
            interp = (
                f"At {tp} ({phase}), reference calibration data are insufficient to "
                "score the envelope position."
            )
        else:
            interp = (
                f"At {tp} ({phase}), all scored deltas remain within the expected "
                "variability envelope under the current calibration data."
            )

        rows.append({
            "subject_id":                 sid,
            "timepoint":                  tp,
            "mission_phase":              phase,
            "n_outside_node_envelope":    n_node,
            "n_outside_graph_envelope":   n_graph,
            "n_outside_hazard_envelope":  n_haz,
            "top_outside_domain":         _top_outside(ns, "domain"),
            "top_outside_graph_metric":   _top_outside(gs, "metric"),
            "top_outside_hazard_context": _top_outside(hs, "hazard") if hs is not None else "n/a",
            "overall_envelope_flag":      flag,
            "interpretation":             interp,
        })

    summary = pd.DataFrame(rows, columns=out_cols)
    return summary.sort_values(["subject_id", "timepoint"]).reset_index(drop=True)
