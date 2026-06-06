# Biological Adaptation Graph Schema

## Purpose

A biological adaptation graph represents each participant as a connected biological system.

Instead of storing biomedical measurements as isolated columns, the graph links biological domains, deviations, plausible relationships, observed co-variation, and monitoring-relevant interpretation.

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
| Conceptual biological relationship | Literature-plausible or mechanistic relationship. |
| Observed correlation | Data-derived correlation in the reference population. |
| Co-deviation relationship | Two domains show coordinated deviation in a subject. |
| Temporal coupling | Two domains change together over time in longitudinal data. |
| Decision-support relationship | Biological pattern maps to monitoring priority or countermeasure category. |

## Edge attributes

| Attribute | Description |
|---|---|
| `edge_type` | Conceptual, observed, co-deviation, temporal, or decision-support. |
| `weight` | Relationship strength. |
| `source` | Literature, reference data, subject data, or expert schema. |
| `directional` | Whether edge direction is meaningful. |
| `confidence` | Confidence in relationship interpretation. |

## Initial conceptual edges

Examples:

- Sleep / circadian regulation ↔ Autonomic regulation
- Sleep / circadian regulation ↔ Cognitive load
- Autonomic regulation ↔ HPA/stress biology proxy
- Immune / inflammatory status ↔ Fatigue / recovery proxy
- Metabolic regulation ↔ Fatigue / recovery proxy
- Cognitive load ↔ Emotional regulation
- Recovery capacity ↔ Sleep / circadian regulation
- Recovery capacity ↔ Autonomic regulation
- Monitoring priority → Follow-up data stream
- Monitoring priority → Countermeasure consideration

## Interpretation rules

1. Conceptual edges are not causal proof.
2. Observed edges are not medical claims.
3. A graph pattern is a signal for human review, not diagnosis.
4. Node activation should be interpreted relative to data quality and missingness.
5. Graph novelty should be interpreted as reference-relative unusualness, not disease.

## Future longitudinal extension

In longitudinal data, each subject becomes a sequence of graphs:

```text
G_baseline → G_exposure → G_recovery
```

Future temporal features:

- graph delta from baseline;
- edge activation changes;
- persistent active subgraphs;
- recovery slope;
- return-to-baseline time;
- temporal BACI.
