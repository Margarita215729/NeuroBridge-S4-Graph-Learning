"""Phase 10 — Plain-language reporting for the HRP-like data adapter.

Generates a reviewer-facing text report describing what the adapter found,
which variables mapped to which biological domains, domain coverage,
transformations applied, and the graph-ready output. The report is descriptive
only.

Phase 10 does not interpret health status. It validates and transforms HRP-like
longitudinal data streams into a graph-ready self-baseline schema for downstream
NeuroBridge-S4 analysis.
"""

from __future__ import annotations

import pandas as pd

from neurobridge_graph.domain_mapping import UNMAPPED_DOMAIN

GUARDRAIL_SENTENCE = (
    "This adapter validates and transforms HRP-like longitudinal data into "
    "graph-ready domain scores. It does not diagnose, score risk, infer "
    "exposure, or recommend treatment."
)

CORE_STATEMENT = (
    "Phase 10 does not interpret health status. It validates and transforms "
    "HRP-like longitudinal data streams into a graph-ready self-baseline schema "
    "for downstream NeuroBridge-S4 analysis."
)


def _section(title: str) -> str:
    return f"\n{title}\n{'-' * len(title)}\n"


def generate_adapter_report(
    readiness_report: pd.DataFrame,
    mapping_report: pd.DataFrame,
    coverage_report: pd.DataFrame,
    domain_scores_long: pd.DataFrame,
    data_provenance_note: str | None = None,
) -> str:
    """Generate the plain-language Phase 10 adapter report text."""
    lines: list[str] = []
    lines.append("NeuroBridge-S4 — Phase 10 HRP-Like Data Adapter Report")
    lines.append("=" * 56)
    lines.append("")
    lines.append(CORE_STATEMENT)

    # Data source / provenance.
    lines.append(_section("Data source"))
    lines.append(data_provenance_note or "Input provenance not specified.")

    # Input tables.
    lines.append(_section("Input tables"))
    if readiness_report is not None and not readiness_report.empty:
        for _, r in readiness_report.iterrows():
            lines.append(
                f"- {r['table_name']}: format={r['detected_format']}, "
                f"rows={r['rows']}, columns={r['columns']}, "
                f"subjects={r['subject_count']}, timepoints={r['timepoint_count']}, "
                f"required={r['required_columns_status']}")
    else:
        lines.append("- No input tables were loaded.")

    # Detected formats summary.
    lines.append(_section("Detected formats"))
    if readiness_report is not None and not readiness_report.empty:
        fmt_counts = readiness_report["detected_format"].value_counts().to_dict()
        for fmt, n in fmt_counts.items():
            lines.append(f"- {fmt}: {n} table(s)")
    else:
        lines.append("- none")

    # Subject / timepoint counts.
    lines.append(_section("Subject and timepoint counts"))
    if domain_scores_long is not None and not domain_scores_long.empty:
        n_subj = domain_scores_long["subject_id"].nunique()
        n_tp = domain_scores_long["timepoint"].nunique()
        lines.append(f"- subjects with domain scores: {n_subj}")
        lines.append(f"- distinct timepoints: {n_tp}")
    else:
        lines.append("- no domain scores were produced")

    # Variables mapped / unmapped.
    lines.append(_section("Variable-to-domain mapping"))
    if mapping_report is not None and not mapping_report.empty:
        mapped = mapping_report[mapping_report["mapping_status"] == "mapped"]
        unmapped = mapping_report[mapping_report["mapping_status"] == "unmapped"]
        lines.append(f"- variables mapped to domains: {len(mapped)}")
        lines.append(f"- variables unmapped: {len(unmapped)}")
        if not unmapped.empty:
            lines.append("  unmapped variables: "
                         + ", ".join(sorted(unmapped["variable"].astype(str))))
    else:
        lines.append("- no variables were mapped")

    # Domain coverage.
    lines.append(_section("Domain coverage"))
    if coverage_report is not None and not coverage_report.empty:
        for _, r in coverage_report.iterrows():
            if r["domain"] == UNMAPPED_DOMAIN:
                continue
            lines.append(
                f"- {r['domain']}: {r['mapped_variable_count']} variable(s) "
                f"[{r['coverage_status']}]")
    else:
        lines.append("- no coverage information")

    # Missingness note.
    lines.append(_section("Missingness"))
    if readiness_report is not None and not readiness_report.empty:
        for _, r in readiness_report.iterrows():
            lines.append(f"- {r['table_name']}: {r['missingness_summary']}")
    else:
        lines.append("- not assessed")

    # Transformations applied.
    lines.append(_section("Transformations applied"))
    lines.append("- schema validation and format detection (wide / long)")
    lines.append("- column-name reconciliation to the standard longitudinal schema")
    lines.append("- standardization into a long variable table")
    lines.append("- self-baseline transformation (delta and percent change from "
                 "personal baseline)")
    lines.append("- variable-to-domain mapping")
    lines.append("- per-domain aggregation into graph-ready domain scores "
                 "(mean absolute self-baseline delta)")

    # Graph-ready output.
    lines.append(_section("Graph-ready output"))
    lines.append("- adapter_generated_longitudinal_domain_scores.csv "
                 "(wide, Phase-6 compatible)")
    lines.append("- adapter_domain_scores_long.csv (long domain scores)")
    lines.append("- adapter_domain_scores_wide.csv (wide domain scores)")

    # Limitations.
    lines.append(_section("Limitations"))
    lines.append("- variable-to-domain mapping is approximate and extensible")
    lines.append("- template/example data are not scientific evidence")
    lines.append("- unit conversion is limited unless explicitly implemented")
    lines.append("- domain scores depend on which variables are available")
    lines.append("- missing data can reduce domain coverage")
    lines.append("- no clinical interpretation is performed")

    # Guardrails.
    lines.append(_section("Guardrails"))
    lines.append(GUARDRAIL_SENTENCE)

    return "\n".join(lines) + "\n"
