"""Phase 10 — Validation for HRP-like longitudinal input tables.

This module validates and characterizes raw or semi-structured HRP-like
longitudinal data streams before they are transformed into graph-ready domain
scores. It detects table format (wide vs long), reconciles common column-name
variants, and produces readiness/missingness reports.

Phase 10 does not interpret health status. It validates and transforms HRP-like
longitudinal data streams into a graph-ready self-baseline schema for downstream
NeuroBridge-S4 analysis. It does not diagnose, score risk, infer exposure, or
recommend treatment.
"""

from __future__ import annotations

import re
from pathlib import Path  # noqa: F401 - part of the documented public signature surface

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column-name variants for the four standard longitudinal index columns.
# ---------------------------------------------------------------------------

_SUBJECT_VARIANTS = {"subject_id", "subject", "participant_id", "participant",
                     "crew_id", "id"}
_TIMEPOINT_VARIANTS = {"timepoint", "time_point", "visit", "session",
                       "measurement_time"}
_PHASE_VARIANTS = {"mission_phase", "phase", "period"}
_TIME_INDEX_VARIANTS = {"time_index", "day", "mission_day", "elapsed_day", "order"}

# Long-format signal columns.
_VARIABLE_NAME_VARIANTS = {"variable_name", "variable", "measure", "parameter"}
_VALUE_VARIANTS = {"value", "measurement", "result", "reading"}
_UNIT_VARIANTS = {"unit", "units"}
_DATA_STREAM_VARIANTS = {"data_stream", "stream", "modality", "source_stream"}

# Metadata columns that are never treated as measurement variables.
_NON_VARIABLE_COLS = (
    _SUBJECT_VARIANTS | _TIMEPOINT_VARIANTS | _PHASE_VARIANTS
    | _TIME_INDEX_VARIANTS | {"data_type", "source_table"}
)


def normalize_column_name(name: str) -> str:
    """Normalize a column name for robust matching.

    Lowercases, trims, and converts any run of non-alphanumeric characters to a
    single underscore (e.g. ``"Heart Rate (bpm)"`` -> ``"heart_rate_bpm"``,
    ``"Mission-Phase"`` -> ``"mission_phase"``).
    """
    s = str(name).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def _normalized_lookup(df: pd.DataFrame) -> dict[str, str]:
    """Map normalized column name -> first original column with that norm."""
    lookup: dict[str, str] = {}
    for col in df.columns:
        norm = normalize_column_name(col)
        lookup.setdefault(norm, col)
    return lookup


def detect_standard_columns(df: pd.DataFrame) -> dict:
    """Detect the subject/timepoint/phase/time-index columns.

    Returns a dict mapping each canonical role
    (``subject_id``, ``timepoint``, ``mission_phase``, ``time_index``) to the
    original column name in ``df`` when found, otherwise ``None``.
    """
    lookup = _normalized_lookup(df)

    def _find(variants: set[str]) -> "str | None":
        for norm, original in lookup.items():
            if norm in variants:
                return original
        return None

    return {
        "subject_id":    _find(_SUBJECT_VARIANTS),
        "timepoint":     _find(_TIMEPOINT_VARIANTS),
        "mission_phase": _find(_PHASE_VARIANTS),
        "time_index":    _find(_TIME_INDEX_VARIANTS),
    }


def _detect_long_columns(df: pd.DataFrame) -> dict:
    """Detect long-format variable_name/value/unit/data_stream columns."""
    lookup = _normalized_lookup(df)

    def _find(variants: set[str]) -> "str | None":
        for norm, original in lookup.items():
            if norm in variants:
                return original
        return None

    return {
        "variable_name": _find(_VARIABLE_NAME_VARIANTS),
        "value":         _find(_VALUE_VARIANTS),
        "unit":          _find(_UNIT_VARIANTS),
        "data_stream":   _find(_DATA_STREAM_VARIANTS),
    }


def detect_table_format(df: pd.DataFrame) -> str:
    """Classify a table as ``wide_longitudinal``, ``long_longitudinal`` or ``unknown``.

    Wide format has subject/time columns plus multiple measurement (variable)
    columns. Long format has subject/time columns plus ``variable_name`` and
    ``value`` columns.
    """
    if df is None or df.shape[1] == 0:
        return "unknown"

    std = detect_standard_columns(df)
    has_subject = std["subject_id"] is not None
    has_time = std["timepoint"] is not None

    long_cols = _detect_long_columns(df)
    if has_subject and has_time and long_cols["variable_name"] and long_cols["value"]:
        return "long_longitudinal"

    # Count measurement columns (anything not a recognized index/metadata col).
    measurement_cols = [
        c for c in df.columns
        if normalize_column_name(c) not in _NON_VARIABLE_COLS
    ]
    if has_subject and has_time and len(measurement_cols) >= 1:
        return "wide_longitudinal"

    return "unknown"


def measurement_columns(df: pd.DataFrame) -> list[str]:
    """Return the wide-format measurement (variable) columns of ``df``."""
    return [
        c for c in df.columns
        if normalize_column_name(c) not in _NON_VARIABLE_COLS
    ]


def validate_required_columns(df: pd.DataFrame, required: list[str]) -> pd.DataFrame:
    """Validate presence of required columns (variant-aware for standard roles).

    Returns a report with columns ``required_column``, ``status``
    (``present``/``missing``), and ``matched_column``.
    """
    std = detect_standard_columns(df)
    lookup = _normalized_lookup(df)
    rows: list[dict] = []
    for req in required:
        norm = normalize_column_name(req)
        matched = None
        if norm in ("subject_id", "timepoint", "mission_phase", "time_index"):
            matched = std.get(norm)
        if matched is None:
            matched = lookup.get(norm)
        rows.append({
            "required_column": req,
            "status":          "present" if matched is not None else "missing",
            "matched_column":  matched if matched is not None else "n/a",
        })
    return pd.DataFrame(rows, columns=["required_column", "status", "matched_column"])


def validate_longitudinal_structure(df: pd.DataFrame) -> pd.DataFrame:
    """Run structural checks on a longitudinal input table.

    Returns a report with columns ``check``, ``status`` (``ok``/``warning``/
    ``fail``), and ``detail``.
    """
    rows: list[dict] = []

    def _add(check: str, status: str, detail: str) -> None:
        rows.append({"check": check, "status": status, "detail": detail})

    std = detect_standard_columns(df)
    fmt = detect_table_format(df)

    # Required subject/time structure.
    _add("subject_column_present",
         "ok" if std["subject_id"] else "fail",
         f"matched '{std['subject_id']}'" if std["subject_id"]
         else "no subject column detected")
    _add("timepoint_column_present",
         "ok" if std["timepoint"] else "fail",
         f"matched '{std['timepoint']}'" if std["timepoint"]
         else "no timepoint column detected")
    _add("mission_phase_present",
         "ok" if std["mission_phase"] else "warning",
         f"matched '{std['mission_phase']}'" if std["mission_phase"]
         else "no mission_phase column; will default to 'unknown'")
    _add("time_index_present",
         "ok" if std["time_index"] else "warning",
         f"matched '{std['time_index']}'" if std["time_index"]
         else "no time_index column; will be derived from timepoint order")

    subj = std["subject_id"]
    tp = std["timepoint"]

    # Missing subject IDs / timepoints.
    if subj:
        n_missing = int(df[subj].isna().sum())
        _add("missing_subject_ids",
             "ok" if n_missing == 0 else "warning",
             f"{n_missing} rows with missing subject id")
    if tp:
        n_missing = int(df[tp].isna().sum())
        _add("missing_timepoints",
             "ok" if n_missing == 0 else "warning",
             f"{n_missing} rows with missing timepoint")

    # Duplicate rows on the natural key.
    if fmt == "long_longitudinal":
        long_cols = _detect_long_columns(df)
        key = [c for c in (subj, tp, long_cols["variable_name"]) if c]
    else:
        key = [c for c in (subj, tp) if c]
    if key:
        n_dup = int(df.duplicated(subset=key).sum())
        _add("duplicate_key_rows",
             "ok" if n_dup == 0 else "warning",
             f"{n_dup} duplicate rows on {key}")

    # Nonnumeric values in measurement columns.
    if fmt == "long_longitudinal":
        long_cols = _detect_long_columns(df)
        val_col = long_cols["value"]
        if val_col:
            coerced = pd.to_numeric(df[val_col], errors="coerce")
            n_bad = int(coerced.isna().sum() - df[val_col].isna().sum())
            _add("numeric_measurements",
                 "ok" if n_bad <= 0 else "warning",
                 f"{max(n_bad, 0)} nonnumeric values in '{val_col}'")
    else:
        meas = measurement_columns(df)
        bad_cols = []
        for c in meas:
            coerced = pd.to_numeric(df[c], errors="coerce")
            extra_nan = int(coerced.isna().sum() - df[c].isna().sum())
            if extra_nan > 0:
                bad_cols.append(c)
        _add("numeric_measurements",
             "ok" if not bad_cols else "warning",
             "all measurement columns numeric" if not bad_cols
             else f"nonnumeric values in: {', '.join(bad_cols)}")

    # Sorted time_index availability.
    if std["time_index"]:
        coerced = pd.to_numeric(df[std["time_index"]], errors="coerce")
        _add("time_index_sortable",
             "ok" if coerced.notna().any() else "warning",
             "time_index is numeric and sortable" if coerced.notna().any()
             else "time_index present but not numeric")
    else:
        _add("time_index_sortable", "warning",
             "no time_index; ordering will rely on timepoint appearance")

    return pd.DataFrame(rows, columns=["check", "status", "detail"])


def summarize_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-column missingness (``column``, ``dtype``, ``n_missing``, ``missing_fraction``)."""
    n = len(df)
    rows: list[dict] = []
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        rows.append({
            "column":           col,
            "dtype":            str(df[col].dtype),
            "n_missing":        n_missing,
            "missing_fraction": round(n_missing / n, 5) if n else 0.0,
        })
    return pd.DataFrame(rows, columns=["column", "dtype", "n_missing", "missing_fraction"])


def build_input_readiness_report(
    input_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Summarize readiness for a set of named input tables.

    Returns one row per table with columns: ``table_name``, ``detected_format``,
    ``rows``, ``columns``, ``subject_count``, ``timepoint_count``,
    ``required_columns_status``, ``missingness_summary``, ``notes``.
    """
    rows: list[dict] = []
    for name, df in input_tables.items():
        if df is None or df.empty:
            rows.append({
                "table_name":               name,
                "detected_format":          "unknown",
                "rows":                     0,
                "columns":                  0,
                "subject_count":            0,
                "timepoint_count":          0,
                "required_columns_status":  "empty table",
                "missingness_summary":      "n/a",
                "notes":                    "table is empty",
            })
            continue

        fmt = detect_table_format(df)
        std = detect_standard_columns(df)
        subj = std["subject_id"]
        tp = std["timepoint"]
        subject_count = int(df[subj].nunique()) if subj else 0
        timepoint_count = int(df[tp].nunique()) if tp else 0

        req_report = validate_required_columns(df, ["subject_id", "timepoint"])
        missing_req = req_report.loc[req_report["status"] == "missing", "required_column"].tolist()
        req_status = "ok" if not missing_req else f"missing: {', '.join(missing_req)}"

        miss = summarize_missingness(df)
        n_cols_missing = int((miss["n_missing"] > 0).sum())
        overall_missing = round(miss["missing_fraction"].mean(), 5) if not miss.empty else 0.0
        miss_summary = (f"{n_cols_missing} columns with missing values "
                        f"(mean missing fraction {overall_missing})")

        notes = []
        if fmt == "unknown":
            notes.append("format not recognized; check subject/time columns")
        if std["mission_phase"] is None:
            notes.append("mission_phase not detected (will default to 'unknown')")
        if std["time_index"] is None:
            notes.append("time_index not detected (will be derived)")
        notes_text = "; ".join(notes) if notes else "ok"

        rows.append({
            "table_name":               name,
            "detected_format":          fmt,
            "rows":                     int(len(df)),
            "columns":                  int(df.shape[1]),
            "subject_count":            subject_count,
            "timepoint_count":          timepoint_count,
            "required_columns_status":  req_status,
            "missingness_summary":      miss_summary,
            "notes":                    notes_text,
        })

    return pd.DataFrame(rows, columns=[
        "table_name", "detected_format", "rows", "columns", "subject_count",
        "timepoint_count", "required_columns_status", "missingness_summary", "notes",
    ])
