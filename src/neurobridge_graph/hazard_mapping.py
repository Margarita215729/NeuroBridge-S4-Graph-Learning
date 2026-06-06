"""Phase 5 — NASA HRP five-hazard context mapping.

This module adds a *hazard-context* interpretation layer on top of the
biological adaptation graphs. It maps biological domains to NASA Human
Research Program (HRP) five human spaceflight hazard categories and computes
per-participant *hazard relevance scores*.

Scientific positioning (read carefully):

    NeuroBridge-S4 connects individual biological adaptation patterns to
    NASA's five human spaceflight hazard categories without claiming actual
    exposure, diagnosis, or causal proof.

A ``hazard_relevance_score`` is **not**:
  * a measured exposure score,
  * a health risk score,
  * a diagnosis or prediction.

It is a transparent, weighted mapping from *activated biological domains* to
HRP hazard categories, designed to help reviewers see which spaceflight
hazard contexts may be most relevant for closer monitoring. Domain coverage is
limited by the proxy dataset and is reported explicitly rather than hidden.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Hazard categories
# ---------------------------------------------------------------------------

HAZARD_CANONICAL: list[str] = [
    "space_radiation",
    "isolation_and_confinement",
    "distance_from_earth",
    "gravity_fields",
    "hostile_closed_environments",
]

HAZARD_DISPLAY_NAMES: dict[str, str] = {
    "space_radiation":             "Space Radiation",
    "isolation_and_confinement":   "Isolation and Confinement",
    "distance_from_earth":         "Distance from Earth",
    "gravity_fields":              "Gravity Fields",
    "hostile_closed_environments": "Hostile / Closed Environments",
}

# Special framing for the "Distance from Earth" hazard. It is not primarily a
# biological domain; it raises the operational value of autonomous monitoring.
DISTANCE_FROM_EARTH_NOTE: str = (
    "Distance from Earth is treated as an autonomy and delayed-support "
    "context. Hazard relevance does not imply exposure; it indicates that a "
    "biological graph pattern may be more operationally important when "
    "real-time Earth support is limited."
)

# Core positioning sentence reused across documentation and notebook.
CORE_POSITIONING_SENTENCE: str = (
    "NeuroBridge-S4 connects individual biological adaptation patterns to "
    "NASA's five human spaceflight hazard categories without claiming actual "
    "exposure, diagnosis, or causal proof."
)

# ---------------------------------------------------------------------------
# Domain → hazard conceptual relevance weights (0.0–1.0).
# This is a conceptual HRP relevance mapping, NOT measured exposure.
# Domain names are written in their canonical (long) form; node-data variants
# are reconciled by ``normalize_domain_name``.
# ---------------------------------------------------------------------------

_HAZARD_DOMAIN_WEIGHTS: dict[str, dict[str, float]] = {
    "space_radiation": {
        "inflammation / immune-adjacent status": 0.8,
        "hematologic / oxygen-carrying capacity": 0.7,
        "cardiovascular regulation": 0.5,
        "recovery-related markers": 0.6,
        "cognitive load": 0.4,
    },
    "isolation_and_confinement": {
        "sleep / circadian regulation": 0.9,
        "autonomic regulation": 0.8,
        "emotional regulation": 0.9,
        "cognitive load": 0.8,
        "inflammation / immune-adjacent status": 0.4,
        "recovery capacity": 0.5,
    },
    "distance_from_earth": {
        "recovery capacity": 0.8,
        "cognitive load": 0.7,
        "emotional regulation": 0.6,
        "autonomic regulation": 0.6,
        "cardiovascular regulation": 0.4,
    },
    "gravity_fields": {
        "body composition / physical status": 0.9,
        "cardiovascular regulation": 0.8,
        "metabolic regulation": 0.7,
        "hematologic / oxygen-carrying capacity": 0.6,
        "recovery-related markers": 0.6,
        "autonomic regulation": 0.5,
    },
    "hostile_closed_environments": {
        "inflammation / immune-adjacent status": 0.8,
        "recovery-related markers": 0.7,
        "sleep / circadian regulation": 0.6,
        "autonomic regulation": 0.5,
        "metabolic regulation": 0.5,
        "emotional regulation": 0.5,
    },
}

# Reconcile node-data domain spellings with the canonical mapping forms.
_DOMAIN_ALIASES: dict[str, str] = {
    "inflammation / immune-adjacent": "inflammation / immune-adjacent status",
    "hematologic / oxygen-carrying": "hematologic / oxygen-carrying capacity",
}

_GUARDRAIL = (
    "Hazard-context mapping: biological domains and graph patterns are mapped "
    "to NASA HRP's five human spaceflight hazard categories as interpretation "
    "context, not as exposure measurement or causal attribution."
)


# ---------------------------------------------------------------------------
# Domain name normalization
# ---------------------------------------------------------------------------

def normalize_domain_name(name: str) -> str:
    """Normalize a biological domain name to its canonical mapping form.

    Lowercases, trims, collapses whitespace, standardizes ``/`` spacing, and
    applies a small alias table so that node-data spellings (e.g.
    ``"Inflammation / immune-adjacent"``) match the canonical hazard-mapping
    spelling (``"inflammation / immune-adjacent status"``).

    Parameters
    ----------
    name:
        Raw domain name from any source.

    Returns
    -------
    str
        Canonical lowercase domain key.
    """
    s = str(name).strip().lower()
    s = re.sub(r"\s*/\s*", " / ", s)   # normalize slash spacing
    s = re.sub(r"\s+", " ", s)         # collapse internal whitespace
    return _DOMAIN_ALIASES.get(s, s)


# ---------------------------------------------------------------------------
# Default mapping table
# ---------------------------------------------------------------------------

def _row_interpretation(hazard_canonical: str, domain: str, weight: float) -> str:
    display = HAZARD_DISPLAY_NAMES[hazard_canonical]
    return (
        f"'{domain}' is conceptually relevant to {display} "
        f"(HRP relevance weight {weight:.1f}). Interpretation context only; "
        "not exposure measurement or causal proof."
    )


def get_default_hazard_domain_mapping() -> pd.DataFrame:
    """Return the default biological-domain → HRP-hazard relevance mapping.

    Returns
    -------
    pandas.DataFrame
        One row per (hazard, domain) with columns:
        ``hazard`` (canonical), ``hazard_display``, ``domain`` (canonical),
        ``weight`` (0.0–1.0), and ``interpretation``.
    """
    rows: list[dict] = []
    for hazard in HAZARD_CANONICAL:
        for domain, weight in _HAZARD_DOMAIN_WEIGHTS[hazard].items():
            canonical_domain = normalize_domain_name(domain)
            rows.append({
                "hazard":         hazard,
                "hazard_display": HAZARD_DISPLAY_NAMES[hazard],
                "domain":         canonical_domain,
                "weight":         float(weight),
                "interpretation": _row_interpretation(hazard, canonical_domain, weight),
            })
    return pd.DataFrame(rows, columns=[
        "hazard", "hazard_display", "domain", "weight", "interpretation",
    ])


def export_hazard_domain_mapping(
    output_path: "str | Path",
    hazard_mapping: pd.DataFrame | None = None,
) -> Path:
    """Save the hazard-domain mapping to CSV.

    Parameters
    ----------
    output_path:
        Destination ``.csv`` path.
    hazard_mapping:
        Optional mapping DataFrame; defaults to
        :func:`get_default_hazard_domain_mapping`.

    Returns
    -------
    pathlib.Path
        The written path.
    """
    mapping = get_default_hazard_domain_mapping() if hazard_mapping is None else hazard_mapping
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(out, index=False)
    return out


# ---------------------------------------------------------------------------
# Hazard relevance scoring
# ---------------------------------------------------------------------------

def _available_domains(node_level_features: pd.DataFrame) -> set[str]:
    """Set of canonical domain names present in the node-level feature table."""
    if "domain" not in node_level_features.columns:
        raise ValueError(
            "node_level_features must contain a 'domain' column; "
            f"found columns: {list(node_level_features.columns)}"
        )
    return {normalize_domain_name(d) for d in node_level_features["domain"].unique()}


def interpret_hazard_score(
    hazard_name: str,
    score: float,
    coverage_fraction: float,
) -> str:
    """Return a plain-language interpretation of one hazard relevance score.

    Parameters
    ----------
    hazard_name:
        Canonical or display hazard name.
    score:
        Hazard relevance score (mean reference-relative activation weighted by
        HRP relevance). ``NaN`` when no mapped domains are available.
    coverage_fraction:
        Fraction of mapped domains that are present in the proxy dataset.

    Returns
    -------
    str
        Reviewer-facing interpretation that avoids exposure/diagnosis language.
    """
    display = HAZARD_DISPLAY_NAMES.get(hazard_name, hazard_name)

    if score is None or (isinstance(score, float) and np.isnan(score)) or coverage_fraction == 0:
        return (
            f"{display}: no mapped biological domains are present in the "
            "current proxy dataset, so hazard-context relevance cannot be "
            "estimated. This is a domain-coverage limitation, not a finding."
        )

    if score < 0.75:
        band = "low"
    elif score < 1.0:
        band = "mild"
    elif score < 1.5:
        band = "moderate"
    else:
        band = "high"

    text = (
        f"{display}: {band} hazard-context relevance ({score:.2f}) from the "
        "mapped biological domains available. This is a monitoring-relevant "
        "pattern in graph-feature space, not exposure measurement, diagnosis, "
        "or causal proof."
    )
    if coverage_fraction < 0.5:
        text += (
            f" Interpretation is domain-coverage-limited "
            f"(only {coverage_fraction * 100:.0f}% of mapped domains available)."
        )
    return text


def compute_hazard_relevance_scores(
    node_level_features: pd.DataFrame,
    hazard_mapping: pd.DataFrame | None = None,
    activation_col: str = "activation",
    subject_col: str = "subject_id",
) -> pd.DataFrame:
    """Compute per-subject, per-hazard relevance scores.

    For each subject and hazard::

        hazard_relevance_score = sum(activation * weight) / sum(weight)

    over the mapped domains that are actually available for that subject.

    Parameters
    ----------
    node_level_features:
        Long table with at least ``subject_col``, ``domain``, and
        ``activation_col`` columns (Phase 4 ``node_level_features.csv``).
    hazard_mapping:
        Optional mapping; defaults to :func:`get_default_hazard_domain_mapping`.
    activation_col:
        Name of the activation column (default ``"activation"``).
    subject_col:
        Name of the subject id column (default ``"subject_id"``).

    Returns
    -------
    pandas.DataFrame
        One row per (subject, hazard) with columns: ``subject_id``,
        ``hazard``, ``hazard_display``, ``hazard_relevance_score``,
        ``available_domain_count``, ``expected_domain_count``,
        ``coverage_fraction``, ``coverage_note``, ``top_contributing_domain``,
        ``interpretation``.
    """
    for col in (subject_col, "domain", activation_col):
        if col not in node_level_features.columns:
            raise ValueError(
                f"node_level_features must contain a '{col}' column; "
                f"found columns: {list(node_level_features.columns)}"
            )

    mapping = get_default_hazard_domain_mapping() if hazard_mapping is None else hazard_mapping

    rows: list[dict] = []
    subjects = list(node_level_features[subject_col].unique())

    for subject_id in subjects:
        sub = node_level_features[node_level_features[subject_col] == subject_id]
        # Canonical domain -> activation for this subject.
        act_by_domain: dict[str, float] = {}
        for _, r in sub.iterrows():
            dom = normalize_domain_name(r["domain"])
            try:
                act_by_domain[dom] = float(r[activation_col])
            except (TypeError, ValueError):
                act_by_domain[dom] = 0.0

        for hazard in HAZARD_CANONICAL:
            haz_rows = mapping[mapping["hazard"] == hazard]
            expected_count = int(len(haz_rows))

            num = 0.0
            den = 0.0
            available = 0
            best_domain = "n/a"
            best_contrib = -np.inf
            for _, mr in haz_rows.iterrows():
                dom = normalize_domain_name(mr["domain"])
                weight = float(mr["weight"])
                if dom in act_by_domain:
                    activation = act_by_domain[dom]
                    contrib = activation * weight
                    num += contrib
                    den += weight
                    available += 1
                    if contrib > best_contrib:
                        best_contrib = contrib
                        best_domain = dom

            coverage_fraction = (available / expected_count) if expected_count else 0.0

            if available == 0 or den == 0:
                score = float("nan")
                coverage_note = "No mapped domains available in current proxy dataset."
                best_domain = "n/a"
            else:
                score = round(num / den, 5)
                coverage_note = (
                    f"{available}/{expected_count} mapped domains available "
                    f"(coverage {coverage_fraction * 100:.0f}%)."
                )

            rows.append({
                "subject_id":              subject_id,
                "hazard":                  hazard,
                "hazard_display":          HAZARD_DISPLAY_NAMES[hazard],
                "hazard_relevance_score":  score,
                "available_domain_count":  available,
                "expected_domain_count":   expected_count,
                "coverage_fraction":       round(coverage_fraction, 5),
                "coverage_note":           coverage_note,
                "top_contributing_domain": best_domain,
                "interpretation":          interpret_hazard_score(
                    hazard, score, coverage_fraction),
            })

    return pd.DataFrame(rows, columns=[
        "subject_id", "hazard", "hazard_display", "hazard_relevance_score",
        "available_domain_count", "expected_domain_count", "coverage_fraction",
        "coverage_note", "top_contributing_domain", "interpretation",
    ])


def compute_hazard_coverage(
    node_level_features: pd.DataFrame,
    hazard_mapping: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute dataset-level domain coverage for each hazard.

    Coverage reflects which mapped domains exist in the current proxy dataset.
    Missing domains reduce coverage and are reported explicitly.

    Returns
    -------
    pandas.DataFrame
        One row per hazard with columns: ``hazard``, ``hazard_display``,
        ``expected_domain_count``, ``available_domain_count``,
        ``coverage_fraction``, ``available_domains``, ``missing_domains``,
        ``coverage_note``.
    """
    mapping = get_default_hazard_domain_mapping() if hazard_mapping is None else hazard_mapping
    present = _available_domains(node_level_features)

    rows: list[dict] = []
    for hazard in HAZARD_CANONICAL:
        haz_rows = mapping[mapping["hazard"] == hazard]
        mapped_domains = [normalize_domain_name(d) for d in haz_rows["domain"]]
        expected = len(mapped_domains)
        available_domains = [d for d in mapped_domains if d in present]
        missing_domains = [d for d in mapped_domains if d not in present]
        available = len(available_domains)
        coverage_fraction = (available / expected) if expected else 0.0

        if available == 0:
            note = "No mapped domains available in current proxy dataset."
        elif missing_domains:
            note = (
                f"{available}/{expected} mapped domains available; "
                f"missing: {', '.join(missing_domains)}."
            )
        else:
            note = f"Full mapped-domain coverage ({available}/{expected})."

        rows.append({
            "hazard":                 hazard,
            "hazard_display":         HAZARD_DISPLAY_NAMES[hazard],
            "expected_domain_count":  expected,
            "available_domain_count": available,
            "coverage_fraction":      round(coverage_fraction, 5),
            "available_domains":      "; ".join(available_domains) if available_domains else "none",
            "missing_domains":        "; ".join(missing_domains) if missing_domains else "none",
            "coverage_note":          note,
        })

    return pd.DataFrame(rows, columns=[
        "hazard", "hazard_display", "expected_domain_count",
        "available_domain_count", "coverage_fraction",
        "available_domains", "missing_domains", "coverage_note",
    ])
