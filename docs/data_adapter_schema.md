# HRP-Like Data Adapter Schema (Phase 10)

> Phase 10 does not interpret health status. It validates and transforms
> HRP-like longitudinal data streams into a graph-ready self-baseline schema for
> downstream NeuroBridge-S4 analysis. It does not diagnose, score risk, infer
> exposure, or recommend treatment.

The adapter layer accepts raw or semi-structured HRP-like longitudinal data and
produces graph-ready domain scores compatible with the Phase 6 within-subject
longitudinal graph trajectory pipeline.

## Accepted input formats

The adapter detects two longitudinal table formats automatically:

### Wide longitudinal

One row per subject/timepoint, one column per measurement variable.

```text
subject_id, timepoint, mission_phase, time_index, <variable_1>, <variable_2>, ..., data_type
```

### Long longitudinal

One row per subject/timepoint/variable.

```text
subject_id, timepoint, mission_phase, time_index, variable_name, value, unit, data_stream, data_type
```

### Column-name reconciliation

Standard index columns are detected from common variants (case/format
insensitive):

| Canonical role  | Accepted variants                                              |
|-----------------|----------------------------------------------------------------|
| `subject_id`    | subject_id, subject, participant_id, participant, crew_id, id  |
| `timepoint`     | timepoint, time_point, visit, session, measurement_time        |
| `mission_phase` | mission_phase, phase, period                                   |
| `time_index`    | time_index, day, mission_day, elapsed_day, order               |

## Required vs optional columns

- **Required:** `subject_id`, `timepoint` (a variant of each).
- **Recommended:** `mission_phase` (defaults to `unknown` when absent),
  `time_index` (derived from timepoint order when absent — recorded as an
  assumption).
- **Long format also requires:** `variable_name` and `value`. `unit` and
  `data_stream` are optional (inferred from the mapping when absent).
- **`data_type`** marks provenance; template rows use
  `schema_template_not_scientific_evidence`.

## Standardized variable schema

All inputs are normalized to one long table:

```text
subject_id, timepoint, mission_phase, time_index,
variable_name, value, unit, data_stream, source_table
```

## Self-baseline transformation

For each subject and variable, a baseline value is taken from the `baseline`
mission phase when present, otherwise from the earliest `time_index` (recorded
as `earliest_time_index_used_as_baseline`). Outputs:

```text
... baseline_value, delta_from_baseline, percent_change_from_baseline, baseline_assumption
```

## Unit standardization (placeholder)

Phase 10 includes a unit-standardization placeholder, but broad biomedical unit
conversion is not yet implemented. Units are tracked and unsupported conversions
are reported rather than silently transformed. `standardize_units_if_known`
adds a `unit_conversion_status` per row — one of `already_standard`,
`not_provided`, `unsupported_conversion`, `not_applied` — and the pipeline emits
`adapter_unit_conversion_report.csv`.

## Output domain-score schema (long)

```text
subject_id, timepoint, mission_phase, time_index, domain,
domain_score, domain_score_method, available_variable_count,
missing_variable_count, data_streams_used, data_quality_note
```

Supported aggregation methods: `mean_delta`, `mean_abs_delta` (default),
`median_delta`, `mean_z_delta`. `mean_z_delta` uses per-subject repeated-measure
standard deviation; when robust repeated data are unavailable it transparently
falls back to mean absolute delta (recorded in `domain_score_method` and
`data_quality_note`).

## Graph-ready (wide) format

The long domain scores are pivoted to a Phase-6-compatible wide table:

```text
subject_id, timepoint, mission_phase, time_index, data_type, <domain_1>, <domain_2>, ...
```

The domain columns are the canonical NeuroBridge-S4 biological domains (see
`docs/domain_mapping.md`). This file
(`adapter_generated_longitudinal_domain_scores.csv`) can be passed directly to
the Phase 6 longitudinal graph trajectory code.

## Pipeline outputs

`run_adapter_pipeline` writes:

```text
results/tables/adapter_input_readiness_report.csv
results/tables/standardized_longitudinal_variables.csv
results/tables/variable_baseline_deltas.csv
results/tables/variable_domain_mapping_report.csv
results/tables/domain_coverage_report.csv
results/tables/adapter_domain_scores_long.csv
results/tables/adapter_domain_scores_wide.csv
results/tables/adapter_generated_longitudinal_domain_scores.csv
results/reports/phase10_data_adapter_report.txt
```

## Example

```python
from neurobridge_graph.data_adapters import run_adapter_pipeline

# With real inputs in data/hrp_like_inputs/, or [] to use schema templates.
outputs = run_adapter_pipeline([], output_dir="results/tables",
                               templates_dir="data/templates")
```

## Guardrails

The adapter validates and transforms data only. It does not diagnose, score
risk, infer exposure, or recommend treatment. Template/example data are schema
illustrations and are not scientific evidence.
