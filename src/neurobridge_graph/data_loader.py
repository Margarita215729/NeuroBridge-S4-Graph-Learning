"""Loaders for processed NeuroBridge-S4 output CSV files.

Each loader:
  - resolves the path relative to this package (../../data/processed/)
    but also accepts an explicit *data_dir* override;
  - validates that every expected column is present;
  - raises a descriptive FileNotFoundError or ValueError on problems;
  - returns a pandas.DataFrame.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

# Mapping: filename -> required columns (subset that must exist)
_SCHEMA: dict[str, list[str]] = {
    "pseudo_crew.csv": ["SEQN", "RIDAGEYR", "RIAGENDR", "BMXBMI"],
    "deviation_scores.csv": [],          # row index = participant, no fixed named cols
    "domain_scores.csv": [],             # same: domain names are dynamic
    "baci_scores.csv": ["crew", "BACI", "category"],
    "baci_sensitivity.csv": ["participant", "threshold", "BACI", "category"],
    "crew_level_summary.csv": ["Participant ID", "BACI score", "BACI category"],
}


def _load(
    filename: str,
    data_dir: Path | str | None = None,
    index_col: int | str | None = None,
) -> pd.DataFrame:
    """Generic loader with path resolution, existence check, and column validation."""
    base = Path(data_dir) if data_dir is not None else _DEFAULT_DATA_DIR
    path = base / filename

    if not path.exists():
        raise FileNotFoundError(
            f"\n  ✗  Expected file not found: {path}\n"
            f"  Make sure you have run Phase 2 data import or set data_dir correctly.\n"
            f"  Default directory: {_DEFAULT_DATA_DIR}"
        )

    df = pd.read_csv(path, index_col=index_col)

    required = _SCHEMA.get(filename, [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"\n  ✗  File '{filename}' is missing required column(s): {missing}\n"
            f"  Found columns: {list(df.columns)}"
        )

    return df


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_pseudo_crew(data_dir: Path | str | None = None) -> pd.DataFrame:
    """Return raw proxy-crew biomarker data (pseudo_crew.csv).

    Index: default integer index.
    Key columns: SEQN, RIDAGEYR, RIAGENDR, BMXBMI, BPXSY1, ...
    """
    return _load("pseudo_crew.csv", data_dir=data_dir)


def load_deviation_scores(data_dir: Path | str | None = None) -> pd.DataFrame:
    """Return per-participant z-score deviations from NHANES reference (deviation_scores.csv).

    Index: participant ID (Crew XXXXX).
    Columns: one per biomarker.
    """
    return _load("deviation_scores.csv", data_dir=data_dir, index_col=0)


def load_domain_scores(data_dir: Path | str | None = None) -> pd.DataFrame:
    """Return mean absolute deviation aggregated by biological domain (domain_scores.csv).

    Index: participant ID.
    Columns: one per domain (Cardiovascular regulation, Metabolic regulation, ...).
    """
    return _load("domain_scores.csv", data_dir=data_dir, index_col=0)


def load_baci_scores(data_dir: Path | str | None = None) -> pd.DataFrame:
    """Return BACI (Biological Adaptation Coherence Index) scores (baci_scores.csv).

    Columns: crew, BACI, category, shifted_domains.
    """
    return _load("baci_scores.csv", data_dir=data_dir)


def load_baci_sensitivity(data_dir: Path | str | None = None) -> pd.DataFrame:
    """Return BACI sensitivity analysis across thresholds (baci_sensitivity.csv).

    Columns: participant, threshold, BACI, category.
    """
    return _load("baci_sensitivity.csv", data_dir=data_dir)


def load_crew_level_summary(data_dir: Path | str | None = None) -> pd.DataFrame:
    """Return the full crew-level summary with monitoring priorities (crew_level_summary.csv).

    Columns: Participant ID, Top biological domain, BACI score, BACI category, ...
    """
    return _load("crew_level_summary.csv", data_dir=data_dir)


def load_all(data_dir: Path | str | None = None) -> dict[str, pd.DataFrame]:
    """Load all six processed outputs and return them as a named dict.

    Returns
    -------
    dict with keys:
        'pseudo_crew', 'deviation_scores', 'domain_scores',
        'baci_scores', 'baci_sensitivity', 'crew_level_summary'
    """
    loaders = {
        "pseudo_crew": load_pseudo_crew,
        "deviation_scores": load_deviation_scores,
        "domain_scores": load_domain_scores,
        "baci_scores": load_baci_scores,
        "baci_sensitivity": load_baci_sensitivity,
        "crew_level_summary": load_crew_level_summary,
    }
    return {name: fn(data_dir=data_dir) for name, fn in loaders.items()}
