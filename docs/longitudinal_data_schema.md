# Longitudinal Data Schema

## Purpose

Phase 6 requires a **within-subject longitudinal table** where each row represents one
subject at one timepoint. The primary analysis compares each timepoint's biological
adaptation graph to that subject's **personal baseline**.

> The primary signal is within-subject change from the individual's own baseline.
> Population reference data are used only to calibrate scale, estimate noise,
> contextualize rarity, and stabilize feature geometry.

## Expected table format

```text
subject_id | timepoint | mission_phase | time_index | domain_1 | domain_2 | ...
```

### Required columns

| Column | Description |
|---|---|
| `subject_id` | Participant identifier |
| `timepoint` | Unique label for this observation (e.g. `T0_baseline`, `T2_inflight`) |
| `mission_phase` | Mission phase category (see below) |
| `time_index` | Integer ordering key (0 = earliest) |

### Recommended mission phases

```text
baseline | pre_mission | inflight | postflight | recovery
```

Baseline identification: prefer `mission_phase == 'baseline'`; otherwise use the earliest
`time_index`.

### Optional metadata columns

| Column | Description |
|---|---|
| `data_type` | `schema_demonstration_not_scientific_evidence` for pipeline test data |

### Domain columns (numeric)

Recommended domains (code tolerates missing columns):

```text
Cardiovascular regulation
Metabolic regulation
Body composition / physical status
Inflammation / immune-adjacent
Hematologic / oxygen-carrying
Recovery-related markers
Sleep / circadian regulation
Autonomic regulation
Cognitive load
Emotional regulation
Recovery capacity
```

## File locations

| Path | Purpose |
|---|---|
| `data/processed/longitudinal_domain_scores.csv` | Real longitudinal data (if available) |
| `data/examples/example_longitudinal_domain_scores.csv` | Schema demonstration only |

## Schema demonstration data

If no real longitudinal file exists, the Phase 6 notebook creates an example table marked:

```text
data_type = schema_demonstration_not_scientific_evidence
```

This data tests the trajectory pipeline only. It must **not** be presented as scientific
evidence or actual astronaut data.

## Conceptual model

```text
subject_id + timepoint → biological adaptation graph
subject trajectory   → sequence of graphs over time
trajectory signal      → delta from personal baseline

G_baseline → G_inflight → G_postflight → G_recovery
DeltaGraph(t) = Graph(t) - Graph(baseline)
```

## Output tables (Phase 6)

| File | Content |
|---|---|
| `longitudinal_node_deltas.csv` | Per-domain activation deltas from baseline |
| `longitudinal_graph_deltas.csv` | Graph-level metric deltas |
| `longitudinal_hazard_deltas.csv` | HRP hazard-context relevance deltas |
| `longitudinal_trajectory_summary.csv` | Per-subject/timepoint shift summary |
| `recovery_metrics.csv` | Recovery slope, fraction, time-to-baseline-like state |
