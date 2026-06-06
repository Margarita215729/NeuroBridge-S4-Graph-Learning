# HRP and Space-Health Relevance

## Why this matters

Human spaceflight research often involves very small samples and high-dimensional measurements. Artemis-style human research data may include biomarkers, sleep/activity signals, immune markers, cognitive indicators, self-report, and recovery measurements.

Conventional population-level statistics are limited when n is extremely small. NeuroBridge-S4 Graph Learning addresses this by tracking **within-subject longitudinal graph trajectories** — how each crew member's biological adaptation graph changes from their own baseline across mission and recovery phases.

> The method is designed for n=4 contexts by avoiding underpowered group inference and focusing
> on individualized graph trajectories.

Generic healthy-cohort comparison is insufficient: operationally meaningful changes may occur
within population-normal ranges, while stable individual baselines may differ from generic norms.

## Potential relevance to HRP-style workflows

The framework can support:

- individual adaptation profiles;
- cross-system biological coherence maps;
- reference-relative graph novelty scores;
- monitoring-priority summaries;
- follow-up data stream recommendations;
- transparent reviewer-facing explanations.

## Relation to Artemis-style study streams

| Study stream | Possible graph layer |
|---|---|
| Standard Measures | Biomarker, cardiovascular, functional, metabolic nodes |
| ARCHeR | Sleep, activity, behavioral, self-report, social context nodes |
| Immune Biomarkers | Stress-immune, inflammation, immune-cell, latent-virus-related nodes |

## NASA HRP five human spaceflight hazards

NASA's Human Research Program organizes human spaceflight risk around five major hazards:

1. **Space Radiation** — ionizing radiation beyond Earth's protective magnetosphere.
2. **Isolation and Confinement** — long-duration confinement with limited social contact.
3. **Distance from Earth** — delayed communication and limited real-time medical support.
4. **Gravity Fields** — transitions between gravity environments and microgravity adaptation.
5. **Hostile / Closed Environments** — engineered closed life-support habitats.

Phase 5 of this project adds a **hazard-context mapping** layer that translates biological graph
patterns into HRP hazard-context relevance scores.

> NeuroBridge-S4 connects individual biological adaptation patterns to NASA's five human
> spaceflight hazard categories without claiming actual exposure, diagnosis, or causal proof.

A **hazard relevance score** is not a measured exposure score and not a health risk score. It is a
transparent mapping from activated biological domains to HRP hazard categories, designed to help
reviewers interpret which spaceflight hazard contexts may be most relevant for closer monitoring.

## Why distance from Earth increases the value of autonomous monitoring

Distance from Earth is special among the five hazards. It is not primarily a biological domain;
it is an operational context. As communication delay grows and real-time clinical support becomes
unavailable, the value of **autonomous, interpretable, on-board monitoring** rises sharply.

A transparent graph-based pattern that flags which biological systems warrant closer review is
therefore more operationally valuable precisely when Earth support is limited — even though the
pattern itself measures no exposure. Phase 5 encodes this by treating distance-from-Earth hazard
relevance as an autonomy / delayed-support interpretation context.

## Why individualized baselines matter in small crew missions

Human spaceflight crews are extremely small. Population-level statistics are weak at n=4, but each
participant carries rich multimodal information that changes across mission phases. The primary
monitoring question is not "Is this astronaut normal?" but "How has this astronaut's biological
adaptation graph changed from their own baseline?"

We should not train on n=4. We should track individualized graph trajectories and use reference
cohorts only for calibration (scale, noise, rarity, feature geometry).

## Why recovery trajectories matter after mission phases

Post-mission and recovery-phase graph deltas reveal whether biological adaptation patterns return
toward personal baseline. Recovery slope, recovery fraction, and time-to-baseline-like state are
monitoring-relevant trajectory metrics — not diagnosis or health outcome prediction.

## Additional Artemis II data streams that would improve hazard coverage

The current NHANES-derived proxy demonstration has limited coverage for some hazard categories,
especially Isolation and Confinement and the Distance-from-Earth autonomy context. The following
additional data streams would strengthen hazard-context coverage:

| Missing / weak domain | Hazard contexts it strengthens | Example Artemis II-style stream |
|---|---|---|
| Sleep / circadian regulation | Isolation and Confinement, Hostile / Closed Environments | Actigraphy, sleep logs |
| Autonomic regulation | Isolation, Gravity Fields, Hostile environments | HRV, resting heart rate |
| Emotional regulation | Isolation and Confinement, Distance from Earth | Validated self-report context |
| Cognitive load | Space Radiation, Isolation, Distance from Earth | Cognitive performance batteries |
| Recovery capacity | Distance from Earth, Isolation | Return-to-baseline / recovery slope |

The coverage report (`results/tables/hazard_coverage_report.csv`) makes these gaps explicit: low
coverage indicates missing proxy data streams, **not** absence of hazard relevance.

## Why attribution matters for HRP

Small-crew monitoring requires **explainable evidence**, not just a number. A single trajectory
delta tells HRP reviewers that a graph changed, but not what changed or whether it recovered.

- HRP needs more than a score: it needs to know which biological systems moved.
- Attribution supports expert review by showing which domains and subgraphs drove the
  baseline-relative shift and whether they returned toward baseline, persisted, or overshot.
- Hazard-context alignment helps translate biological graph shifts into HRP operational
  language (the five hazards) **without** claiming exposure or causal effect.

Phase 7 attribution therefore turns a within-subject trajectory into a transparent evidence
trail: baseline-relative change → domain contributors → subgraph contributors → hazard-context
alignment → recovery behavior → plain-language explanation. This is hazard-context alignment
and monitoring-relevant interpretation, never exposure attribution or diagnosis.

## Why reference-calibrated envelopes matter for HRP

A within-subject trajectory tells HRP reviewers *that* a graph changed and *what* drove it, but
not whether the magnitude is unusual. A reference-calibrated variability envelope adds that
missing calibration:

- Astronauts may show operationally meaningful within-subject shifts that still sit **inside
  population-normal ranges** — a population-normal check would miss them.
- Envelope calibration helps reviewers **avoid overreacting** to normal biological/measurement
  variability.
- Envelope calibration helps **identify unusually large self-baseline shifts** that warrant
  closer expert review.
- This is especially useful when **crews are too small for group inference**: the envelope
  borrows variability structure from reference/analog data without making the cohort the
  endpoint.

The envelope is a calibration layer, never a health-status verdict. Outside-envelope means a
baseline-relative change is larger than expected under the current calibration data — a
candidate for expert review, not diagnosis, a risk score, or an exposure measurement.

## Why a review dashboard matters for HRP

HRP-style human spaceflight studies involve very few subjects, many longitudinal measurements,
and reviewers who are domain experts rather than programmers. A local longitudinal review
dashboard supports that workflow:

- it lets a reviewer move through **within-subject trajectories, attribution, hazard-context
  alignment, and reference-calibrated envelope status** for one subject/timepoint in a single
  coherent view;
- it surfaces **monitoring-relevant patterns** (large self-baseline shifts, persistent
  non-recovery) as candidates for **expert review support**, not as automated outputs;
- it keeps the **personal baseline** as the primary comparison and the reference envelope as a
  secondary calibration layer, consistent with the rest of the pipeline.

The dashboard is a local research-review prototype. It is not a clinical monitoring system, not
diagnosis, not treatment guidance, not exposure measurement, and not health risk scoring.

## Why HRP-like data adapters matter

HRP human research data are likely **multimodal**: biomarkers, sleep/activity wearables,
cognitive test batteries, questionnaires, and environmental/habitat context arrive as separate
tables in inconsistent formats. A reusable adapter layer makes the framework practical for that
reality:

- HRP-like data are **multimodal and heterogeneous**; the adapter standardizes wide and long
  streams into one validated schema;
- adapters make the framework **extensible** — new variables and data streams map onto the
  canonical biological domains without changing the trajectory pipeline;
- they **separate ingestion from interpretation**, so validation and self-baseline transformation
  happen before any graph analysis;
- they **improve reproducibility and reviewability** by emitting explicit readiness, mapping, and
  coverage reports alongside the graph-ready output.

The adapter validates and transforms data. It does not diagnose, score risk, infer exposure, or
recommend treatment.

## Important distinction

This repository is not an official NASA project and does not include actual Artemis II data. It is an independent research prototype inspired by small-N human spaceflight data challenges.
