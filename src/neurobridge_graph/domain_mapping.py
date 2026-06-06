"""Phase 10 — Variable-to-biological-domain mapping.

Maps HRP-like measurement variable names (biomarkers, sleep/activity, cognitive,
questionnaire, environmental) onto the canonical NeuroBridge-S4 biological
domains. The mapping is an interpretation scaffold for downstream graph
analysis. It is approximate, extensible, and explicitly reported.

This mapping does not diagnose, score risk, infer exposure, or recommend
treatment. Domain assignment is a structural scaffold, not a clinical judgment.
"""

from __future__ import annotations

import re

import pandas as pd

# ---------------------------------------------------------------------------
# Canonical biological domains (lowercase, matching the downstream pipeline).
# ---------------------------------------------------------------------------

CANONICAL_DOMAINS: list[str] = [
    "cardiovascular regulation",
    "metabolic regulation",
    "body composition / physical status",
    "inflammation / immune-adjacent status",
    "hematologic / oxygen-carrying capacity",
    "recovery-related markers",
    "sleep / circadian regulation",
    "autonomic regulation",
    "cognitive load",
    "emotional regulation",
    "recovery capacity",
    "environmental context",
]

UNMAPPED_DOMAIN = "unmapped"

GUARDRAIL = (
    "Variable-to-domain mapping is an interpretation scaffold for downstream "
    "analysis. It does not diagnose, score risk, infer exposure, or recommend "
    "treatment."
)

# Some variables are intentionally left unmapped by default; documented here so
# the choice is explicit and surfaced in reports rather than silently dropped.
INTENTIONALLY_UNMAPPED_NOTE: dict[str, str] = {
    "steps": (
        "Steps are intentionally left unmapped by default because their "
        "interpretation depends on protocol context, mission phase, workload, "
        "exercise prescription, and activity constraints. Users may map steps "
        "explicitly in project-specific configurations."
    ),
}

# ---------------------------------------------------------------------------
# Default mapping specification: canonical_variable -> attributes.
# `data_stream` groups variables by acquisition modality.
# `direction_hint` is a non-clinical note about how a raw value tends to move;
# it is informational only and does not drive any scoring.
# ---------------------------------------------------------------------------

_MAPPING_SPEC: list[dict] = [
    # cardiovascular regulation (biomarker / vitals)
    {"canonical_variable": "systolic_bp", "domain": "cardiovascular regulation",
     "data_stream": "vitals", "expected_unit": "mmHg", "direction_hint": "context"},
    {"canonical_variable": "diastolic_bp", "domain": "cardiovascular regulation",
     "data_stream": "vitals", "expected_unit": "mmHg", "direction_hint": "context"},
    {"canonical_variable": "heart_rate", "domain": "cardiovascular regulation",
     "data_stream": "vitals", "expected_unit": "bpm", "direction_hint": "context"},
    {"canonical_variable": "resting_hr", "domain": "cardiovascular regulation",
     "data_stream": "vitals", "expected_unit": "bpm", "direction_hint": "context"},
    # autonomic regulation (HRV)
    {"canonical_variable": "hrv", "domain": "autonomic regulation",
     "data_stream": "autonomic", "expected_unit": "ms", "direction_hint": "context"},
    {"canonical_variable": "rmssd", "domain": "autonomic regulation",
     "data_stream": "autonomic", "expected_unit": "ms", "direction_hint": "context"},
    {"canonical_variable": "sdnn", "domain": "autonomic regulation",
     "data_stream": "autonomic", "expected_unit": "ms", "direction_hint": "context"},
    # metabolic regulation
    {"canonical_variable": "glucose", "domain": "metabolic regulation",
     "data_stream": "biomarker", "expected_unit": "mg/dL", "direction_hint": "context"},
    {"canonical_variable": "insulin", "domain": "metabolic regulation",
     "data_stream": "biomarker", "expected_unit": "uIU/mL", "direction_hint": "context"},
    {"canonical_variable": "hba1c", "domain": "metabolic regulation",
     "data_stream": "biomarker", "expected_unit": "%", "direction_hint": "context"},
    {"canonical_variable": "triglycerides", "domain": "metabolic regulation",
     "data_stream": "biomarker", "expected_unit": "mg/dL", "direction_hint": "context"},
    {"canonical_variable": "cholesterol", "domain": "metabolic regulation",
     "data_stream": "biomarker", "expected_unit": "mg/dL", "direction_hint": "context"},
    # body composition / physical status
    {"canonical_variable": "bmi", "domain": "body composition / physical status",
     "data_stream": "anthropometric", "expected_unit": "kg/m^2", "direction_hint": "context"},
    {"canonical_variable": "weight", "domain": "body composition / physical status",
     "data_stream": "anthropometric", "expected_unit": "kg", "direction_hint": "context"},
    {"canonical_variable": "body_fat_percent", "domain": "body composition / physical status",
     "data_stream": "anthropometric", "expected_unit": "%", "direction_hint": "context"},
    {"canonical_variable": "lean_mass", "domain": "body composition / physical status",
     "data_stream": "anthropometric", "expected_unit": "kg", "direction_hint": "context"},
    # inflammation / immune-adjacent status
    {"canonical_variable": "crp", "domain": "inflammation / immune-adjacent status",
     "data_stream": "biomarker", "expected_unit": "mg/L", "direction_hint": "context"},
    {"canonical_variable": "il6", "domain": "inflammation / immune-adjacent status",
     "data_stream": "biomarker", "expected_unit": "pg/mL", "direction_hint": "context"},
    {"canonical_variable": "white_blood_cell_count", "domain": "inflammation / immune-adjacent status",
     "data_stream": "biomarker", "expected_unit": "10^9/L", "direction_hint": "context"},
    {"canonical_variable": "cytokine", "domain": "inflammation / immune-adjacent status",
     "data_stream": "biomarker", "expected_unit": "pg/mL", "direction_hint": "context"},
    # hematologic / oxygen-carrying capacity
    {"canonical_variable": "hemoglobin", "domain": "hematologic / oxygen-carrying capacity",
     "data_stream": "biomarker", "expected_unit": "g/dL", "direction_hint": "context"},
    {"canonical_variable": "hematocrit", "domain": "hematologic / oxygen-carrying capacity",
     "data_stream": "biomarker", "expected_unit": "%", "direction_hint": "context"},
    {"canonical_variable": "rbc", "domain": "hematologic / oxygen-carrying capacity",
     "data_stream": "biomarker", "expected_unit": "10^12/L", "direction_hint": "context"},
    {"canonical_variable": "ferritin", "domain": "hematologic / oxygen-carrying capacity",
     "data_stream": "biomarker", "expected_unit": "ng/mL", "direction_hint": "context"},
    # sleep / circadian regulation
    {"canonical_variable": "sleep_duration", "domain": "sleep / circadian regulation",
     "data_stream": "sleep_activity", "expected_unit": "hours", "direction_hint": "context"},
    {"canonical_variable": "sleep_efficiency", "domain": "sleep / circadian regulation",
     "data_stream": "sleep_activity", "expected_unit": "%", "direction_hint": "context"},
    {"canonical_variable": "sleep_latency", "domain": "sleep / circadian regulation",
     "data_stream": "sleep_activity", "expected_unit": "min", "direction_hint": "context"},
    {"canonical_variable": "wake_after_sleep_onset", "domain": "sleep / circadian regulation",
     "data_stream": "sleep_activity", "expected_unit": "min", "direction_hint": "context"},
    # cognitive load
    {"canonical_variable": "reaction_time", "domain": "cognitive load",
     "data_stream": "cognitive", "expected_unit": "ms", "direction_hint": "context"},
    {"canonical_variable": "accuracy", "domain": "cognitive load",
     "data_stream": "cognitive", "expected_unit": "%", "direction_hint": "context"},
    {"canonical_variable": "psychomotor_vigilance", "domain": "cognitive load",
     "data_stream": "cognitive", "expected_unit": "score", "direction_hint": "context"},
    {"canonical_variable": "cognitive_score", "domain": "cognitive load",
     "data_stream": "cognitive", "expected_unit": "score", "direction_hint": "context"},
    # emotional regulation
    {"canonical_variable": "stress_score", "domain": "emotional regulation",
     "data_stream": "questionnaire", "expected_unit": "score", "direction_hint": "context"},
    {"canonical_variable": "mood_score", "domain": "emotional regulation",
     "data_stream": "questionnaire", "expected_unit": "score", "direction_hint": "context"},
    {"canonical_variable": "anxiety_score", "domain": "emotional regulation",
     "data_stream": "questionnaire", "expected_unit": "score", "direction_hint": "context"},
    {"canonical_variable": "affect_score", "domain": "emotional regulation",
     "data_stream": "questionnaire", "expected_unit": "score", "direction_hint": "context"},
    # recovery capacity (self-assessed composite capacity to recover)
    {"canonical_variable": "recovery_score", "domain": "recovery capacity",
     "data_stream": "questionnaire", "expected_unit": "score", "direction_hint": "context"},
    {"canonical_variable": "perceived_recovery", "domain": "recovery capacity",
     "data_stream": "questionnaire", "expected_unit": "score", "direction_hint": "context"},
    # recovery-related markers (observed recovery / strain-state indicators)
    {"canonical_variable": "fatigue_score", "domain": "recovery-related markers",
     "data_stream": "questionnaire", "expected_unit": "score", "direction_hint": "context"},
    {"canonical_variable": "soreness_score", "domain": "recovery-related markers",
     "data_stream": "questionnaire", "expected_unit": "score", "direction_hint": "context"},
    {"canonical_variable": "muscle_soreness", "domain": "recovery-related markers",
     "data_stream": "questionnaire", "expected_unit": "score", "direction_hint": "context"},
    {"canonical_variable": "readiness_score", "domain": "recovery-related markers",
     "data_stream": "questionnaire", "expected_unit": "score", "direction_hint": "context"},
    {"canonical_variable": "sleep_recovery_score", "domain": "recovery-related markers",
     "data_stream": "questionnaire", "expected_unit": "score", "direction_hint": "context"},
    {"canonical_variable": "restoration_score", "domain": "recovery-related markers",
     "data_stream": "questionnaire", "expected_unit": "score", "direction_hint": "context"},
    # environmental context
    {"canonical_variable": "co2", "domain": "environmental context",
     "data_stream": "environmental", "expected_unit": "ppm", "direction_hint": "context"},
    {"canonical_variable": "temperature", "domain": "environmental context",
     "data_stream": "environmental", "expected_unit": "C", "direction_hint": "context"},
    {"canonical_variable": "humidity", "domain": "environmental context",
     "data_stream": "environmental", "expected_unit": "%", "direction_hint": "context"},
    {"canonical_variable": "noise", "domain": "environmental context",
     "data_stream": "environmental", "expected_unit": "dB", "direction_hint": "context"},
    {"canonical_variable": "light_exposure", "domain": "environmental context",
     "data_stream": "environmental", "expected_unit": "lux", "direction_hint": "context"},
]


def normalize_variable_name(name: str) -> str:
    """Normalize a measurement variable name for robust matching.

    Lowercases, trims, and converts runs of non-alphanumeric characters to a
    single underscore (e.g. ``"HRV (RMSSD)"`` -> ``"hrv_rmssd"``).
    """
    s = str(name).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def _row_interpretation(canonical_variable: str, domain: str) -> str:
    return (
        f"'{canonical_variable}' is mapped to the '{domain}' biological domain "
        "as an interpretation scaffold. This is not a clinical assignment and "
        "does not diagnose, score risk, infer exposure, or recommend treatment."
    )


def get_default_variable_domain_mapping() -> pd.DataFrame:
    """Return the default variable → biological-domain mapping.

    Columns: ``variable_pattern`` (normalized matching token),
    ``canonical_variable``, ``domain``, ``data_stream``, ``expected_unit``,
    ``direction_hint``, ``interpretation_note``.
    """
    rows: list[dict] = []
    for spec in _MAPPING_SPEC:
        cv = spec["canonical_variable"]
        rows.append({
            "variable_pattern":    normalize_variable_name(cv),
            "canonical_variable":  cv,
            "domain":              spec["domain"],
            "data_stream":         spec["data_stream"],
            "expected_unit":       spec["expected_unit"],
            "direction_hint":      spec["direction_hint"],
            "interpretation_note": _row_interpretation(cv, spec["domain"]),
        })
    return pd.DataFrame(rows, columns=[
        "variable_pattern", "canonical_variable", "domain", "data_stream",
        "expected_unit", "direction_hint", "interpretation_note",
    ])


def map_variable_to_domain(
    variable_name: str,
    mapping_df: pd.DataFrame | None = None,
) -> dict:
    """Map one variable name to a biological domain.

    Matching strategy (case/format-insensitive):

    1. Exact match on the normalized variable name.
    2. Token-substring match: a mapping pattern is contained in the variable
       (or vice versa), preferring the longest pattern to avoid spurious hits.

    Returns a dict with ``canonical_variable``, ``domain``, ``data_stream``,
    ``expected_unit``, ``matched_pattern``, ``mapping_status``
    (``mapped``/``unmapped``).
    """
    mapping = get_default_variable_domain_mapping() if mapping_df is None else mapping_df
    norm = normalize_variable_name(variable_name)

    unmapped = {
        "canonical_variable": normalize_variable_name(variable_name),
        "domain":             UNMAPPED_DOMAIN,
        "data_stream":        "unspecified",
        "expected_unit":      "unknown",
        "matched_pattern":    "n/a",
        "mapping_status":     "unmapped",
    }
    if not norm:
        return unmapped

    # 1. Exact normalized match.
    exact = mapping[mapping["variable_pattern"] == norm]
    if not exact.empty:
        r = exact.iloc[0]
        return {
            "canonical_variable": r["canonical_variable"],
            "domain":             r["domain"],
            "data_stream":        r["data_stream"],
            "expected_unit":      r["expected_unit"],
            "matched_pattern":    r["variable_pattern"],
            "mapping_status":     "mapped",
        }

    # 2. Substring match, longest pattern first.
    candidates = sorted(
        mapping["variable_pattern"].tolist(), key=len, reverse=True)
    for pattern in candidates:
        if not pattern:
            continue
        if pattern in norm or norm in pattern:
            r = mapping[mapping["variable_pattern"] == pattern].iloc[0]
            return {
                "canonical_variable": r["canonical_variable"],
                "domain":             r["domain"],
                "data_stream":        r["data_stream"],
                "expected_unit":      r["expected_unit"],
                "matched_pattern":    pattern,
                "mapping_status":     "mapped",
            }

    return unmapped


def map_variables_dataframe(
    variables: list[str],
    mapping_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return a mapping report for a list of variable names.

    Columns: ``variable``, ``normalized_variable``, ``canonical_variable``,
    ``domain``, ``data_stream``, ``expected_unit``, ``matched_pattern``,
    ``mapping_status``.
    """
    mapping = get_default_variable_domain_mapping() if mapping_df is None else mapping_df
    seen: set[str] = set()
    rows: list[dict] = []
    for var in variables:
        norm = normalize_variable_name(var)
        if norm in seen:
            continue
        seen.add(norm)
        m = map_variable_to_domain(var, mapping)
        rows.append({
            "variable":            var,
            "normalized_variable": norm,
            "canonical_variable":  m["canonical_variable"],
            "domain":              m["domain"],
            "data_stream":         m["data_stream"],
            "expected_unit":       m["expected_unit"],
            "matched_pattern":     m["matched_pattern"],
            "mapping_status":      m["mapping_status"],
        })
    return pd.DataFrame(rows, columns=[
        "variable", "normalized_variable", "canonical_variable", "domain",
        "data_stream", "expected_unit", "matched_pattern", "mapping_status",
    ])


def build_domain_coverage_report(mapped_variables_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize domain coverage from a variable mapping report.

    Returns one row per canonical domain plus one ``unmapped`` row when
    unmapped variables exist. Columns: ``domain``, ``mapped_variable_count``,
    ``variables``, ``data_streams``, ``coverage_status``.
    """
    rows: list[dict] = []
    df = mapped_variables_df if mapped_variables_df is not None else pd.DataFrame()

    for domain in CANONICAL_DOMAINS:
        if df.empty or "domain" not in df.columns:
            sub = pd.DataFrame()
        else:
            sub = df[df["domain"] == domain]
        variables = sorted(sub["variable"].astype(str).unique()) if not sub.empty else []
        streams = (sorted(sub["data_stream"].astype(str).unique())
                   if not sub.empty else [])
        rows.append({
            "domain":                domain,
            "mapped_variable_count": len(variables),
            "variables":             "; ".join(variables) if variables else "none",
            "data_streams":          "; ".join(streams) if streams else "none",
            "coverage_status":       "covered" if variables else "absent",
        })

    if not df.empty and "mapping_status" in df.columns:
        unmapped = df[df["mapping_status"] == "unmapped"]
        if not unmapped.empty:
            variables = sorted(unmapped["variable"].astype(str).unique())
            rows.append({
                "domain":                UNMAPPED_DOMAIN,
                "mapped_variable_count": len(variables),
                "variables":             "; ".join(variables),
                "data_streams":          "unspecified",
                "coverage_status":       "unmapped_variables_present",
            })

    return pd.DataFrame(rows, columns=[
        "domain", "mapped_variable_count", "variables", "data_streams",
        "coverage_status",
    ])
