# Biological Adaptation Graph Schema

## Purpose

A biological adaptation graph represents each participant at a single timepoint as a connected
biological system. In Phase 6, graphs are extended into **within-subject longitudinal
trajectories**: a sequence of graphs per subject, with the primary signal being change from
personal baseline.

Instead of storing biomedical measurements as isolated columns, the graph links biological
domains, deviations, plausible relationships, observed co-variation, and monitoring-relevant
interpretation.

## Core graph object

Each participant graph is a weighted attributed graph:

```text
G = (V, E, X, W)
```

Where:

- `V` = biological and functional nodes;
- `E` = relationships between nodes;
- `X` = node attributes and feature vectors;
- `W` = edge weights and relationship metadata.

## Node types

Initial node set:

| Node | Meaning |
|---|---|
| Cardiovascular regulation | Blood pressure, cardiovascular markers, circulation-related deviations. |
| Metabolic regulation | Glucose, lipids, energy metabolism, metabolic strain. |
| Inflammation / immune-adjacent status | CRP, CBC-derived markers, immune/inflammatory proxies. |
| Body composition / physical status | BMI, weight, anthropometrics, physical status markers. |
| Hematologic / oxygen-carrying capacity | Hemoglobin, hematocrit, RBC-related markers. |
| Sleep / circadian regulation | Sleep, actigraphy, circadian proxy features in later phases. |
| Autonomic regulation | HRV, resting heart rate, autonomic proxy features in later phases. |
| Cognitive load | Cognitive performance or workload proxy features in later phases. |
| Emotional regulation | Evidence-based psychological indicators and validated self-report context. |
| Recovery capacity | Return-to-baseline, recovery slope, resilience-related features. |
| Monitoring priority | Decision-support node connecting patterns to follow-up needs. |
| Countermeasure consideration | Non-treatment support category for HRP-style interpretation. |

## Node attributes

Each node can include:

| Attribute | Description |
|---|---|
| `domain_score` | Mean absolute reference deviation for the biological domain. |
| `signed_deviation` | Directional deviation where biologically meaningful. |
| `percentile_rank` | Reference-relative percentile. |
| `missingness` | Missing-data flag or missingness rate. |
| `baci_contribution` | Contribution to Biological Adaptation Coherence Index. |
| `data_source` | Source dataset or measurement stream. |
| `confidence_level` | Confidence based on data quality and domain coverage. |
| `phase` | Baseline/exposure/recovery phase in future longitudinal data. |

## Edge types

| Edge type | Meaning |
|---|---|
| `conceptual_biological_relationship` | Literature-plausible or mechanistic relationship between two biological domains (Phase 3, implemented). |
| `within_subject_coactivation` | Both domains show activation ≥ threshold in this subject (Phase 3, implemented). |
| `observed_reference_relationship` | Data-derived relationship from reference population (Phase 4+, future). |
| `decision_support_relationship` | Biological pattern maps to monitoring priority or countermeasure category (future). |
| `hazard_context_relationship` | Biological domain maps to a NASA HRP hazard category as interpretation context (Phase 5 mapping; future graph-expanded extension). |

## Edge attributes (Phase 3)

| Attribute | Description |
|---|---|
| `edge_type` | One of the types above. |
| `relationship` | Plain-language description of the relationship. |
| `weight` | Relationship strength (1.0 for conceptual; mean activation for co-activation). |
| `source` | NeuroBridge-S4 graph schema or processed proxy outputs. |
| `interpretation` | Guardrail-aware plain-language phrase. |
| `coactivation` | Boolean — True if a conceptual edge also shows co-activation. |
| `coactivation_weight` | Mean activation when co-activation is annotated on a conceptual edge. |

## Graph-level attributes (Phase 3)

| Attribute | Value |
|---|---|
| `subject_id` | Participant identifier |
| `baci_score` | BACI from baci_scores.csv |
| `baci_category` | Coherence category string |
| `n_domains` | Number of domain nodes |
| `n_active_domains` | Domains with activation ≥ 1.0 |
| `max_domain_activation` | Highest activation value |
| `top_domain` | Domain with highest activation |
| `graph_type` | `subject_level_biological_adaptation_graph` |
| `source_project` | `NeuroBridge-S4 Graph Learning` |
| `guardrail` | `Research interpretation only; not diagnosis or treatment guidance.` |

## Phase 3 conceptual edge schema

Implemented edges (both endpoints must be present in domain_scores.csv):

| Domain A | Domain B | Biological rationale |
|---|---|---|
| Cardiovascular regulation | Metabolic regulation | Shared regulatory feedback loops |
| Metabolic regulation | Body composition / physical status | Bidirectional coupling |
| Inflammation / immune-adjacent | Metabolic regulation | Inflammatory signalling modulates metabolism |
| Hematologic / oxygen-carrying | Cardiovascular regulation | Oxygen capacity coupled to cardiac function |
| Sleep / circadian regulation | Autonomic regulation | Sleep regulates autonomic tone |
| Autonomic regulation | Cardiovascular regulation | Autonomic modulation of heart rate |
| Sleep / circadian regulation | Recovery capacity | Sleep drives physiological recovery |
| Cognitive load | Recovery-related markers | Cognitive demand influences recovery |
| Emotional regulation | Cognitive load | Shared neuroregulatory substrate |
| Recovery capacity | Inflammation / immune-adjacent | Recovery involves inflammatory resolution |

## Interactive visualization outputs (Phase 3)

- `results/html/subject_graph_<ID>.html` — one interactive HTML per subject
- `results/html/index.html` — index page linking all subject graphs

HTML files are self-contained (no internet required when using pyvis inline CDN).
Nodes are draggable; hover tooltips show domain score, activation level, and interpretation.

## Phase 5 hazard-context mapping

Phase 5 adds a NASA HRP five-hazard context layer linking biological domains to hazard categories.

Hazard categories (canonical names):

```text
space_radiation
isolation_and_confinement
distance_from_earth
gravity_fields
hostile_closed_environments
```

The domain → hazard relevance mapping (weights in `[0, 1]`) is defined in
`src/neurobridge_graph/hazard_mapping.py` and exported to
`results/tables/hazard_domain_mapping.csv` with columns `hazard`, `hazard_display`, `domain`,
`weight`, `interpretation`.

**Implementation note:** the current Phase 5 implementation uses **feature-level hazard mapping**
— hazard relevance scores are computed per subject and added as `hazard_relevance__<hazard>`
feature columns. It does **not** add hazard nodes into each participant graph. A future extension
could expand the graph itself with **hazard context nodes** connected to domain nodes via
`hazard_context_relationship` edges; that graph-expanded form is not implemented yet.

This is a conceptual HRP relevance mapping for interpretation context — not exposure measurement,
diagnosis, or causal attribution.

## Phase 6 — Longitudinal graph trajectories

Phase 6 extends the single-timepoint graph to a **within-subject trajectory**:

```text
subject_id + timepoint → biological adaptation graph
G_baseline → G_inflight → G_postflight → G_recovery
DeltaGraph(t) = Graph(t) - Graph(baseline)
```

Graph-level metadata for longitudinal graphs:

| Attribute | Value |
|---|---|
| `timepoint` | Observation label |
| `mission_phase` | baseline / pre_mission / inflight / postflight / recovery |
| `time_index` | Integer ordering |
| `baseline_timepoint` | Personal baseline reference |
| `data_type` | `schema_demonstration_not_scientific_evidence` for pipeline test data |
| `graph_type` | `longitudinal_biological_adaptation_graph` |

See `docs/longitudinal_data_schema.md` for input format and output tables.

## Future extensions

- `measured_variable` nodes connecting individual biomarkers to domain nodes
- `observed_reference_relationship` edges from reference population data
- `decision_support_relationship` edges connecting domain patterns to monitoring priorities
- hazard context nodes + `hazard_context_relationship` edges (graph-expanded HRP hazard layer)
- temporal edge types linking graphs across timepoints within a subject trajectory
