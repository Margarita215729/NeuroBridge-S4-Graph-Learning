# Methodology

## Overview

NeuroBridge-S4 Graph Learning extends the original NeuroBridge-S4 methodology by converting participant-level biomedical measurements into graph objects.

The method is designed for small-N human research settings where population-level statistical power may be weak, but each participant contains rich multimodal information.

## Method stages

1. Load processed biomedical/domain outputs.
2. Map measurements into biological domains.
3. Construct one biological adaptation graph per participant.
4. Compute node-level and graph-level features.
5. Embed graphs into a reference graph space.
6. Detect reference-relative graph novelty.
7. Generate explainable subgraph profiles for human review.

## Why graph representation matters

Tables break a human system into columns. Graph representation helps assemble those columns back into a biological and operational system.

A participant is not just a row in a dataset. In human adaptation research, sleep, autonomic regulation, inflammation, metabolic regulation, cognitive load, and recovery may interact as a system.

Graph representation preserves this relational structure.

## Why this is appropriate for small-N data

The method does not train a supervised model on n=4.

Instead, it learns a reference graph space from larger public proxy datasets and interprets small-N target subjects relative to that reference space.

This supports:

- individualized interpretation;
- reference-relative novelty detection;
- coherent subgraph identification;
- reviewer-friendly explanations.

## Modeling hierarchy

1. Transparent graph features
2. Similarity and distance analysis
3. Reference-relative novelty detection
4. Explainable subgraph profiles
5. Optional self-supervised graph learning
6. Future longitudinal graph trajectories

## Responsible interpretation

The project is not a diagnostic system. It is a research framework for signal triage and graph-based interpretation.
