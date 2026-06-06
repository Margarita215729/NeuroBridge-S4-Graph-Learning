"""Phase 10 — HRP-like data adapters.

Transforms raw or semi-structured HRP-like longitudinal data streams
(biomarker, sleep/activity, cognitive, questionnaire, environmental, and generic
wide/long tables) into the graph-ready self-baseline domain-score schema used by
the NeuroBridge-S4 longitudinal trajectory pipeline (Phase 6).

Pipeline::

    raw HRP-like streams -> schema validation -> variable-to-domain mapping
    -> self-baseline transformation -> domain-level longitudinal scores
    -> graph-ready NeuroBridge-S4 inputs

Phase 10 does not interpret health status. It validates and transforms HRP-like
longitudinal data streams into a graph-ready self-baseline schema for downstream
NeuroBridge-S4 analysis. It does not diagnose, score risk, infer exposure, or
recommend treatment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from neurobridge_graph.data_validation import (
    build_input_readiness_report,
    detect_standard_columns,
    detect_table_format,
    measurement_columns,
)
from neurobridge_graph.data_validation import (
    _detect_long_columns as _detect_long_columns,  # noqa: F401 (internal reuse)
)
from neurobridge_graph.domain_mapping import (
    UNMAPPED_DOMAIN,
    build_domain_coverage_report,
    get_default_variable_domain_mapping,
    map_variable_to_domain,
    map_variables_dataframe,
    normalize_variable_name,
)

SCHEMA_TEMPLATE_DATA_TYPE = "schema_template_not_scientific_evidence"
GENERATED_INPUT_DATA_TYPE = "adapter_generated_from_provided_input"

STANDARDIZED_COLUMNS = [
    "subject_id", "timepoint", "mission_phase", "time_index",
    "variable_name", "value", "unit", "data_stream", "source_table",
]

VARIABLE_DELTA_COLUMNS = [
    "subject_id", "timepoint", "mission_phase", "time_index",
    "variable_name", "value", "unit", "data_stream",
    "baseline_value", "delta_from_baseline", "percent_change_from_baseline",
    "baseline_assumption",
]

DOMAIN_SCORE_LONG_COLUMNS = [
    "subject_id", "timepoint", "mission_phase", "time_index", "domain",
    "domain_score", "domain_score_method", "available_variable_count",
    "missing_variable_count", "data_streams_used", "data_quality_note",
]

SUPPORTED_AGGREGATIONS = ("mean_delta", "mean_abs_delta", "median_delta", "mean_z_delta")

GUARDRAIL = (
    "This adapter validates and transforms HRP-like longitudinal data into "
    "graph-ready domain scores. It does not diagnose, score risk, infer "
    "exposure, or recommend treatment."
)


# ---------------------------------------------------------------------------
# 1. Templates
# ---------------------------------------------------------------------------

_TEMPLATE_ROWS: dict[str, list[dict]] = {
    "longitudinal_wide_template.csv": [
        {"subject_id": "Demo_Crew_01", "timepoint": "T0_baseline", "mission_phase": "baseline",
         "time_index": 0, "heart_rate": 62, "hrv_rmssd": 65, "sleep_duration": 7.5,
         "reaction_time": 250, "crp": 1.0, "glucose": 90},
        {"subject_id": "Demo_Crew_01", "timepoint": "T2_inflight", "mission_phase": "inflight",
         "time_index": 2, "heart_rate": 70, "hrv_rmssd": 52, "sleep_duration": 6.2,
         "reaction_time": 285, "crp": 1.8, "glucose": 102},
    ],
    "longitudinal_long_template.csv": [
        {"subject_id": "Demo_Crew_01", "timepoint": "T0_baseline", "mission_phase": "baseline",
         "time_index": 0, "variable_name": "heart_rate", "value": 62, "unit": "bpm",
         "data_stream": "vitals"},
        {"subject_id": "Demo_Crew_01", "timepoint": "T2_inflight", "mission_phase": "inflight",
         "time_index": 2, "variable_name": "heart_rate", "value": 70, "unit": "bpm",
         "data_stream": "vitals"},
    ],
    "biomarkers_template.csv": [
        {"subject_id": "Demo_Crew_01", "timepoint": "T0_baseline", "mission_phase": "baseline",
         "time_index": 0, "crp": 1.0, "glucose": 90, "hemoglobin": 14.5,
         "hematocrit": 43, "cholesterol": 180},
        {"subject_id": "Demo_Crew_01", "timepoint": "T2_inflight", "mission_phase": "inflight",
         "time_index": 2, "crp": 1.8, "glucose": 102, "hemoglobin": 13.8,
         "hematocrit": 41, "cholesterol": 195},
    ],
    "sleep_activity_template.csv": [
        {"subject_id": "Demo_Crew_01", "timepoint": "T0_baseline", "mission_phase": "baseline",
         "time_index": 0, "sleep_duration": 7.5, "sleep_efficiency": 92,
         "wake_after_sleep_onset": 20, "steps": 9000, "resting_hr": 58},
        {"subject_id": "Demo_Crew_01", "timepoint": "T2_inflight", "mission_phase": "inflight",
         "time_index": 2, "sleep_duration": 6.2, "sleep_efficiency": 85,
         "wake_after_sleep_onset": 38, "steps": 6500, "resting_hr": 64},
    ],
    "cognitive_tests_template.csv": [
        {"subject_id": "Demo_Crew_01", "timepoint": "T0_baseline", "mission_phase": "baseline",
         "time_index": 0, "reaction_time": 250, "accuracy": 97, "cognitive_score": 88},
        {"subject_id": "Demo_Crew_01", "timepoint": "T2_inflight", "mission_phase": "inflight",
         "time_index": 2, "reaction_time": 285, "accuracy": 94, "cognitive_score": 82},
    ],
    "questionnaires_template.csv": [
        {"subject_id": "Demo_Crew_01", "timepoint": "T0_baseline", "mission_phase": "baseline",
         "time_index": 0, "fatigue_score": 2, "stress_score": 3, "mood_score": 8,
         "recovery_score": 8},
        {"subject_id": "Demo_Crew_01", "timepoint": "T2_inflight", "mission_phase": "inflight",
         "time_index": 2, "fatigue_score": 5, "stress_score": 6, "mood_score": 6,
         "recovery_score": 5},
    ],
    "environmental_context_template.csv": [
        {"subject_id": "Demo_Crew_01", "timepoint": "T0_baseline", "mission_phase": "baseline",
         "time_index": 0, "co2": 600, "temperature": 22.0, "humidity": 45,
         "noise": 50, "light_exposure": 300},
        {"subject_id": "Demo_Crew_01", "timepoint": "T2_inflight", "mission_phase": "inflight",
         "time_index": 2, "co2": 2200, "temperature": 24.5, "humidity": 38,
         "noise": 62, "light_exposure": 180},
    ],
}


def create_data_templates(output_dir: "str | Path" = "data/templates") -> list[Path]:
    """Write the HRP-like input CSV templates with illustrative example rows.

    All template rows are marked ``data_type =
    schema_template_not_scientific_evidence``. Returns the written paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fname, rows in _TEMPLATE_ROWS.items():
        df = pd.DataFrame(rows)
        df["data_type"] = SCHEMA_TEMPLATE_DATA_TYPE
        path = out_dir / fname
        df.to_csv(path, index=False)
        written.append(path)
    return written


# ---------------------------------------------------------------------------
# Index canonicalization helpers
# ---------------------------------------------------------------------------

def _derive_time_index(df: pd.DataFrame, subj: str, tp: str) -> pd.Series:
    """Derive a per-subject time index from timepoint order of appearance."""
    out = pd.Series(index=df.index, dtype=float)
    for _, grp in df.groupby(subj, sort=False):
        uniq = list(dict.fromkeys(grp[tp].tolist()))
        order = {t: i for i, t in enumerate(uniq)}
        out.loc[grp.index] = grp[tp].map(order).astype(float)
    return out


def _canonical_index_frame(df: pd.DataFrame) -> "tuple[pd.DataFrame, list[str]]":
    """Return a frame with canonical index columns and a list of assumption notes.

    Returns ``(None, notes)`` when subject_id or timepoint cannot be detected.
    """
    std = detect_standard_columns(df)
    notes: list[str] = []
    if std["subject_id"] is None or std["timepoint"] is None:
        return None, ["subject_id or timepoint column not detected"]

    out = pd.DataFrame(index=df.index)
    out["subject_id"] = df[std["subject_id"]].astype(str)
    out["timepoint"] = df[std["timepoint"]].astype(str)

    if std["mission_phase"] is not None:
        out["mission_phase"] = df[std["mission_phase"]].astype(str)
    else:
        out["mission_phase"] = "unknown"
        notes.append("mission_phase not detected; defaulted to 'unknown'")

    if std["time_index"] is not None:
        out["time_index"] = pd.to_numeric(df[std["time_index"]], errors="coerce")
        if out["time_index"].isna().any():
            derived = _derive_time_index(out, "subject_id", "timepoint")
            out["time_index"] = out["time_index"].fillna(derived)
            notes.append("some time_index values were derived from timepoint order")
    else:
        out["time_index"] = _derive_time_index(out, "subject_id", "timepoint")
        notes.append("time_index not detected; derived from timepoint order")

    return out, notes


def _data_stream_for(variable: str, mapping_df: pd.DataFrame) -> str:
    info = map_variable_to_domain(variable, mapping_df)
    return info["data_stream"]


def _standardization_report(
    table_name: str,
    fmt: str,
    rows_in: int,
    rows_out: int,
    n_variables: int,
    notes: list[str],
) -> pd.DataFrame:
    return pd.DataFrame([{
        "source_table":      table_name,
        "detected_format":   fmt,
        "rows_in":           rows_in,
        "standardized_rows": rows_out,
        "variable_count":    n_variables,
        "assumptions":       "; ".join(notes) if notes else "none",
    }])


# ---------------------------------------------------------------------------
# 2 & 3. Standardization
# ---------------------------------------------------------------------------

def standardize_wide_longitudinal_table(
    df: pd.DataFrame,
    table_name: str = "wide_input",
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Convert a wide subject/timepoint table into the standardized variable table.

    Returns ``(standardized_df, report_df)``. ``standardized_df`` has columns
    :data:`STANDARDIZED_COLUMNS`.
    """
    empty = pd.DataFrame(columns=STANDARDIZED_COLUMNS)
    if df is None or df.empty:
        return empty, _standardization_report(table_name, "unknown", 0, 0, 0,
                                               ["empty table"])

    idx, notes = _canonical_index_frame(df)
    if idx is None:
        return empty, _standardization_report(
            table_name, detect_table_format(df), len(df), 0, 0, notes)

    meas_cols = measurement_columns(df)
    if not meas_cols:
        return empty, _standardization_report(
            table_name, "wide_longitudinal", len(df), 0, 0,
            notes + ["no measurement columns detected"])

    wide = idx.copy()
    for c in meas_cols:
        wide[c] = df[c].values

    melted = wide.melt(
        id_vars=["subject_id", "timepoint", "mission_phase", "time_index"],
        value_vars=meas_cols,
        var_name="variable_name",
        value_name="value",
    )
    melted["value"] = pd.to_numeric(melted["value"], errors="coerce")
    melted = melted.dropna(subset=["value"]).reset_index(drop=True)

    mapping = get_default_variable_domain_mapping()
    stream_lookup = {v: _data_stream_for(v, mapping) for v in meas_cols}
    melted["unit"] = "unknown"
    melted["data_stream"] = melted["variable_name"].map(stream_lookup).fillna("unspecified")
    melted["source_table"] = table_name

    standardized = melted[STANDARDIZED_COLUMNS].copy()
    report = _standardization_report(
        table_name, "wide_longitudinal", len(df), len(standardized),
        len(meas_cols), notes)
    return standardized, report


def standardize_long_longitudinal_table(
    df: pd.DataFrame,
    table_name: str = "long_input",
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Normalize a long-format table into the standardized variable table.

    Returns ``(standardized_df, report_df)``.
    """
    empty = pd.DataFrame(columns=STANDARDIZED_COLUMNS)
    if df is None or df.empty:
        return empty, _standardization_report(table_name, "unknown", 0, 0, 0,
                                               ["empty table"])

    idx, notes = _canonical_index_frame(df)
    long_cols = _detect_long_columns(df)
    if idx is None or long_cols["variable_name"] is None or long_cols["value"] is None:
        return empty, _standardization_report(
            table_name, detect_table_format(df), len(df), 0, 0,
            (notes or []) + ["missing variable_name/value columns"])

    out = idx.copy()
    out["variable_name"] = df[long_cols["variable_name"]].astype(str).values
    out["value"] = pd.to_numeric(df[long_cols["value"]], errors="coerce").values
    out = out.dropna(subset=["value"]).reset_index(drop=True)

    if long_cols["unit"] is not None:
        unit_series = df[long_cols["unit"]].astype(str)
        out["unit"] = unit_series.reindex(out.index).fillna("unknown").values \
            if len(unit_series) == len(out) else "unknown"
    if "unit" not in out.columns:
        out["unit"] = "unknown"

    mapping = get_default_variable_domain_mapping()
    if long_cols["data_stream"] is not None and len(df) == len(out):
        out["data_stream"] = df[long_cols["data_stream"]].astype(str).values
    else:
        out["data_stream"] = out["variable_name"].apply(
            lambda v: _data_stream_for(v, mapping))
    out["source_table"] = table_name

    standardized = out[STANDARDIZED_COLUMNS].copy()
    n_vars = standardized["variable_name"].nunique()
    report = _standardization_report(
        table_name, "long_longitudinal", len(df), len(standardized), n_vars, notes)
    return standardized, report


# ---------------------------------------------------------------------------
# 4. Combine
# ---------------------------------------------------------------------------

def combine_standardized_streams(
    standardized_tables: list[pd.DataFrame],
) -> pd.DataFrame:
    """Combine standardized variable tables and de-duplicate where possible.

    Exact duplicate rows are dropped, then duplicates on
    (subject_id, timepoint, variable_name) keep the first occurrence.
    """
    frames = [t for t in standardized_tables
              if t is not None and not t.empty]
    if not frames:
        return pd.DataFrame(columns=STANDARDIZED_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates().reset_index(drop=True)
    combined = combined.drop_duplicates(
        subset=["subject_id", "timepoint", "variable_name"], keep="first"
    ).reset_index(drop=True)
    return combined[STANDARDIZED_COLUMNS]


UNIT_CONVERSION_STATUSES = (
    "already_standard", "not_provided", "unsupported_conversion", "not_applied",
)


def standardize_units_if_known(
    df: pd.DataFrame,
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Placeholder for unit standardization.

    Current behavior (intentionally conservative):

    - does **not** perform broad biomedical unit conversion;
    - records a ``unit_conversion_status`` per row;
    - marks units that already match the mapping's ``expected_unit`` as
      ``already_standard``;
    - marks rows without a usable unit as ``not_provided``;
    - marks mismatched units as ``unsupported_conversion`` (left untransformed);
    - otherwise marks ``not_applied``.

    Returns ``(df_with_status, unit_conversion_report)``. Values are never
    silently transformed.
    """
    if df is None or df.empty or "variable_name" not in df.columns:
        out = (df.copy() if df is not None else pd.DataFrame())
        if isinstance(out, pd.DataFrame) and "variable_name" in out.columns:
            out["unit_conversion_status"] = "not_applied"
        report = pd.DataFrame(columns=[
            "variable_name", "unit", "expected_unit", "unit_conversion_status", "count"])
        return out, report

    mapping = get_default_variable_domain_mapping()
    expected_by_norm = {
        normalize_variable_name(r["canonical_variable"]): str(r["expected_unit"])
        for _, r in mapping.iterrows()
    }

    out = df.copy()
    units = out["unit"].astype(str) if "unit" in out.columns else pd.Series(
        ["unknown"] * len(out), index=out.index)

    def _status(var: str, unit: str) -> str:
        u = str(unit).strip().lower()
        if u in ("", "unknown", "n/a", "nan", "none"):
            return "not_provided"
        expected = expected_by_norm.get(normalize_variable_name(var))
        if expected is None:
            return "not_applied"
        if u == str(expected).strip().lower():
            return "already_standard"
        return "unsupported_conversion"

    out["unit_conversion_status"] = [
        _status(v, u) for v, u in zip(out["variable_name"].astype(str), units)
    ]

    rep = out.copy()
    rep["expected_unit"] = rep["variable_name"].astype(str).map(
        lambda v: expected_by_norm.get(normalize_variable_name(v), "unknown"))
    report = (rep.groupby(
        ["variable_name", "unit", "expected_unit", "unit_conversion_status"],
        dropna=False).size().reset_index(name="count"))
    return out, report


# ---------------------------------------------------------------------------
# 5. Baseline-relative variable deltas
# ---------------------------------------------------------------------------

def compute_variable_baseline_deltas(
    standardized_df: pd.DataFrame,
    baseline_phase: str = "baseline",
) -> pd.DataFrame:
    """Compute self-baseline deltas for each subject/variable.

    For each (subject, variable) the baseline value is taken from the
    ``baseline_phase`` rows when present, otherwise from the earliest
    ``time_index`` (recorded as an assumption). Returns columns
    :data:`VARIABLE_DELTA_COLUMNS`.
    """
    if standardized_df is None or standardized_df.empty:
        return pd.DataFrame(columns=VARIABLE_DELTA_COLUMNS)

    df = standardized_df.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    rows: list[dict] = []
    for (subject_id, variable), grp in df.groupby(["subject_id", "variable_name"], sort=False):
        grp = grp.sort_values("time_index")
        baseline_rows = grp[grp["mission_phase"].astype(str).str.lower() == baseline_phase.lower()]
        if not baseline_rows.empty:
            baseline_value = float(baseline_rows.iloc[0]["value"])
            assumption = f"baseline_phase='{baseline_phase}'"
        else:
            baseline_value = float(grp.iloc[0]["value"])
            assumption = "earliest_time_index_used_as_baseline"

        for _, r in grp.iterrows():
            val = float(r["value"])
            delta = round(val - baseline_value, 6)
            if baseline_value not in (0.0,) and not np.isclose(baseline_value, 0.0):
                pct = round((delta / baseline_value) * 100.0, 5)
            else:
                pct = float("nan")
            rows.append({
                "subject_id":                   subject_id,
                "timepoint":                    r["timepoint"],
                "mission_phase":                r["mission_phase"],
                "time_index":                   r["time_index"],
                "variable_name":                variable,
                "value":                        val,
                "unit":                         r.get("unit", "unknown"),
                "data_stream":                  r.get("data_stream", "unspecified"),
                "baseline_value":               round(baseline_value, 6),
                "delta_from_baseline":          delta,
                "percent_change_from_baseline": pct,
                "baseline_assumption":          assumption,
            })

    return pd.DataFrame(rows, columns=VARIABLE_DELTA_COLUMNS)


# ---------------------------------------------------------------------------
# 6. Domain scores
# ---------------------------------------------------------------------------

def _aggregate(deltas: np.ndarray, z_values: "np.ndarray | None", aggregation: str):
    """Aggregate per-variable deltas into one domain score; returns (score, method)."""
    deltas = deltas[~np.isnan(deltas)]
    if aggregation == "mean_delta":
        return (round(float(np.mean(deltas)), 6), "mean_delta") if deltas.size else (float("nan"), "mean_delta")
    if aggregation == "median_delta":
        return (round(float(np.median(deltas)), 6), "median_delta") if deltas.size else (float("nan"), "median_delta")
    if aggregation == "mean_z_delta":
        if z_values is not None:
            z = z_values[~np.isnan(z_values)]
            if z.size:
                return round(float(np.mean(z)), 6), "mean_z_delta"
        # Fall back to mean_abs_delta when robust repeated-measure z is unavailable.
        if deltas.size:
            return round(float(np.mean(np.abs(deltas))), 6), "mean_abs_delta_fallback"
        return float("nan"), "mean_z_delta"
    # default
    return (round(float(np.mean(np.abs(deltas))), 6), "mean_abs_delta") if deltas.size else (float("nan"), "mean_abs_delta")


def build_domain_scores_from_variables(
    variable_delta_df: pd.DataFrame,
    mapping_df: pd.DataFrame | None = None,
    aggregation: str = "mean_abs_delta",
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Aggregate baseline-relative variable deltas into per-domain scores.

    Returns ``(domain_scores_long, variable_domain_mapping_report)``.
    ``domain_scores_long`` has columns :data:`DOMAIN_SCORE_LONG_COLUMNS`.
    """
    if aggregation not in SUPPORTED_AGGREGATIONS:
        aggregation = "mean_abs_delta"

    empty_scores = pd.DataFrame(columns=DOMAIN_SCORE_LONG_COLUMNS)
    if variable_delta_df is None or variable_delta_df.empty:
        return empty_scores, pd.DataFrame()

    mapping = get_default_variable_domain_mapping() if mapping_df is None else mapping_df
    variables = list(variable_delta_df["variable_name"].astype(str).unique())
    mapping_report = map_variables_dataframe(variables, mapping)

    # variable -> (domain, data_stream)
    var_lookup: dict[str, tuple[str, str]] = {}
    for _, r in mapping_report.iterrows():
        var_lookup[normalize_variable_name(r["variable"])] = (r["domain"], r["data_stream"])

    df = variable_delta_df.copy()
    df["_norm"] = df["variable_name"].apply(normalize_variable_name)
    df["domain"] = df["_norm"].map(lambda v: var_lookup.get(v, (UNMAPPED_DOMAIN, "unspecified"))[0])
    df["data_stream"] = df["_norm"].map(lambda v: var_lookup.get(v, (UNMAPPED_DOMAIN, "unspecified"))[1])
    df["delta_from_baseline"] = pd.to_numeric(df["delta_from_baseline"], errors="coerce")

    mapped = df[df["domain"] != UNMAPPED_DOMAIN].copy()
    if mapped.empty:
        return empty_scores, mapping_report

    # Per (subject, variable) std of value for z-scoring (repeated-measure based).
    value_std: dict[tuple[str, str], float] = {}
    if aggregation == "mean_z_delta":
        for (sid, var), grp in mapped.groupby(["subject_id", "_norm"]):
            vals = pd.to_numeric(grp["value"], errors="coerce").dropna()
            std = float(vals.std(ddof=1)) if vals.shape[0] >= 2 else float("nan")
            value_std[(sid, var)] = std if (std and std > 0) else float("nan")

    # Variables seen per (subject, domain) across all timepoints (for missing count).
    subj_domain_vars: dict[tuple[str, str], set[str]] = {}
    for (sid, dom), grp in mapped.groupby(["subject_id", "domain"]):
        subj_domain_vars[(sid, dom)] = set(grp["_norm"].unique())

    rows: list[dict] = []
    group_cols = ["subject_id", "timepoint", "mission_phase", "time_index", "domain"]
    for (sid, tp, phase, tidx, dom), grp in mapped.groupby(group_cols, sort=False):
        deltas = grp["delta_from_baseline"].to_numpy(dtype=float)
        z_values = None
        if aggregation == "mean_z_delta":
            zs = []
            for _, rr in grp.iterrows():
                std = value_std.get((sid, rr["_norm"]), float("nan"))
                d = rr["delta_from_baseline"]
                zs.append((d / std) if (std and not np.isnan(std)) else float("nan"))
            z_values = np.array(zs, dtype=float)

        score, method = _aggregate(deltas, z_values, aggregation)
        available = int(np.sum(~np.isnan(deltas)))
        expected = len(subj_domain_vars.get((sid, dom), set()))
        missing = max(expected - available, 0)
        streams = sorted(grp["data_stream"].astype(str).unique())

        if available == 0:
            note = "No valid variable deltas at this timepoint for this domain."
        elif missing > 0:
            note = (f"{available}/{expected} domain variables available at this "
                    f"timepoint (self-baseline aggregation).")
        else:
            note = f"All {available} domain variables available (self-baseline aggregation)."
        if method == "mean_abs_delta_fallback":
            note += (" z-score unavailable from repeated data; used mean absolute "
                     "delta as a transparent fallback.")

        rows.append({
            "subject_id":             sid,
            "timepoint":              tp,
            "mission_phase":          phase,
            "time_index":             tidx,
            "domain":                 dom,
            "domain_score":           score,
            "domain_score_method":    method,
            "available_variable_count": available,
            "missing_variable_count": missing,
            "data_streams_used":      "; ".join(streams) if streams else "none",
            "data_quality_note":      note,
        })

    domain_scores_long = pd.DataFrame(rows, columns=DOMAIN_SCORE_LONG_COLUMNS)
    return domain_scores_long, mapping_report


# ---------------------------------------------------------------------------
# 7. Pivot to wide graph-ready format
# ---------------------------------------------------------------------------

def pivot_domain_scores_wide(domain_scores_long: pd.DataFrame) -> pd.DataFrame:
    """Convert the long domain-score table into a Phase-6-compatible wide table.

    Index columns: ``subject_id``, ``timepoint``, ``mission_phase``,
    ``time_index``; remaining columns are biological domains.
    """
    base_cols = ["subject_id", "timepoint", "mission_phase", "time_index"]
    if domain_scores_long is None or domain_scores_long.empty:
        return pd.DataFrame(columns=base_cols)

    wide = domain_scores_long.pivot_table(
        index=base_cols,
        columns="domain",
        values="domain_score",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    if "time_index" in wide.columns:
        wide = wide.sort_values(["subject_id", "time_index"]).reset_index(drop=True)
    return wide


# ---------------------------------------------------------------------------
# 8. Full pipeline
# ---------------------------------------------------------------------------

def _standardize_one(df: pd.DataFrame, table_name: str) -> "tuple[pd.DataFrame, pd.DataFrame]":
    fmt = detect_table_format(df)
    if fmt == "long_longitudinal":
        return standardize_long_longitudinal_table(df, table_name)
    # Default to wide handling for wide_longitudinal and best-effort unknown.
    return standardize_wide_longitudinal_table(df, table_name)


def run_adapter_pipeline(
    input_paths: "list[str | Path]",
    output_dir: "str | Path" = "results/tables",
    templates_dir: "str | Path" = "data/templates",
) -> dict[str, Path]:
    """Run the full HRP-like data adapter pipeline.

    Loads input CSVs (or schema templates when no real inputs exist), validates,
    standardizes, maps variables to domains, computes self-baseline deltas,
    builds domain scores, and exports reports plus graph-ready outputs.

    Returns a dict mapping logical output name to written path.
    """
    output_dir = Path(output_dir)
    reports_dir = output_dir.parent / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Resolve inputs / provenance.
    resolved = [Path(p) for p in (input_paths or []) if Path(p).exists()]
    using_templates = len(resolved) == 0
    if using_templates:
        create_data_templates(templates_dir)
        resolved = sorted(Path(templates_dir).glob("*.csv"))
        provenance_note = (
            "No real HRP-like input files were found. Schema templates were used "
            "to demonstrate adapter mechanics only. Template data are not "
            "scientific evidence."
        )
        generated_data_type = SCHEMA_TEMPLATE_DATA_TYPE
    else:
        provenance_note = (
            f"{len(resolved)} HRP-like input file(s) provided and processed."
        )
        generated_data_type = GENERATED_INPUT_DATA_TYPE

    # Load + standardize each input.
    loaded_tables: dict[str, pd.DataFrame] = {}
    standardized_list: list[pd.DataFrame] = []
    for path in resolved:
        try:
            raw = pd.read_csv(path)
        except Exception:  # noqa: BLE001
            continue
        loaded_tables[path.name] = raw
        std_df, _ = _standardize_one(raw, path.stem)
        if not std_df.empty:
            standardized_list.append(std_df)

    readiness = build_input_readiness_report(loaded_tables)
    standardized = combine_standardized_streams(standardized_list)
    standardized, unit_report = standardize_units_if_known(standardized)
    variable_deltas = compute_variable_baseline_deltas(standardized)
    domain_scores_long, mapping_report = build_domain_scores_from_variables(
        variable_deltas, aggregation="mean_abs_delta")
    coverage_report = build_domain_coverage_report(mapping_report)
    domain_scores_wide = pivot_domain_scores_wide(domain_scores_long)

    # Graph-ready generated table = wide + provenance data_type column.
    generated = domain_scores_wide.copy()
    if not generated.empty:
        insert_at = min(4, generated.shape[1])
        generated.insert(insert_at, "data_type", generated_data_type)

    # Export.
    outputs: dict[str, Path] = {}

    def _write(name: str, df: pd.DataFrame, subdir: Path = output_dir) -> None:
        path = subdir / name
        df.to_csv(path, index=False)
        outputs[name] = path

    _write("adapter_input_readiness_report.csv", readiness)
    _write("standardized_longitudinal_variables.csv", standardized)
    _write("variable_baseline_deltas.csv", variable_deltas)
    _write("variable_domain_mapping_report.csv", mapping_report)
    _write("domain_coverage_report.csv", coverage_report)
    _write("adapter_domain_scores_long.csv", domain_scores_long)
    _write("adapter_domain_scores_wide.csv", domain_scores_wide)
    _write("adapter_generated_longitudinal_domain_scores.csv", generated)
    _write("adapter_unit_conversion_report.csv", unit_report)

    # Plain-language report.
    from neurobridge_graph.adapter_reporting import generate_adapter_report
    report_text = generate_adapter_report(
        readiness, mapping_report, coverage_report, domain_scores_long,
        data_provenance_note=provenance_note)
    report_path = reports_dir / "phase10_data_adapter_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    outputs["phase10_data_adapter_report.txt"] = report_path

    return outputs
