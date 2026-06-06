# Variable-to-Domain Mapping (Phase 10)

> Variable-to-domain mapping is an interpretation scaffold for downstream
> analysis. It does not diagnose, score risk, infer exposure, or recommend
> treatment.

The adapter maps HRP-like measurement variables onto the canonical
NeuroBridge-S4 biological domains. The mapping is approximate, transparent, and
extensible.

## Canonical biological domains

```text
cardiovascular regulation
metabolic regulation
body composition / physical status
inflammation / immune-adjacent status
hematologic / oxygen-carrying capacity
recovery-related markers
sleep / circadian regulation
autonomic regulation
cognitive load
emotional regulation
recovery capacity
environmental context
```

## Default variable patterns and data streams

| Domain                                   | Example variables                                              | Data stream    |
|------------------------------------------|----------------------------------------------------------------|----------------|
| cardiovascular regulation                | systolic_bp, diastolic_bp, heart_rate, resting_hr              | vitals         |
| autonomic regulation                     | hrv, rmssd, sdnn                                               | autonomic      |
| metabolic regulation                     | glucose, insulin, hba1c, triglycerides, cholesterol           | biomarker      |
| body composition / physical status       | bmi, weight, body_fat_percent, lean_mass                      | anthropometric |
| inflammation / immune-adjacent status    | crp, il6, white_blood_cell_count, cytokine                    | biomarker      |
| hematologic / oxygen-carrying capacity    | hemoglobin, hematocrit, rbc, ferritin                         | biomarker      |
| sleep / circadian regulation             | sleep_duration, sleep_efficiency, sleep_latency, wake_after_sleep_onset | sleep_activity |
| cognitive load                           | reaction_time, accuracy, psychomotor_vigilance, cognitive_score | cognitive    |
| emotional regulation                     | stress_score, mood_score, anxiety_score, affect_score         | questionnaire  |
| recovery capacity                        | fatigue_score, soreness_score, recovery_score, readiness_score | questionnaire |
| environmental context                    | co2, temperature, humidity, noise, light_exposure             | environmental  |

`recovery-related markers` is part of the canonical domain set (shared with the
single-timepoint pipeline) but has no default adapter variable patterns yet; it
is reported as `absent` until variables are added.

## Matching strategy

1. Variable names are normalized (lowercased; non-alphanumeric runs become a
   single underscore), e.g. `"HRV (RMSSD)"` -> `hrv_rmssd`.
2. An exact normalized match wins.
3. Otherwise a token-substring match is used, preferring the longest pattern, so
   e.g. `hrv_rmssd` maps to autonomic regulation via the `rmssd` pattern.
4. Variables with no match are labeled `unmapped` and surfaced in the mapping
   and coverage reports rather than silently dropped.

## Expected units and direction hints

Each mapping row carries an `expected_unit` and a non-clinical `direction_hint`
(`context`). These are informational only and do **not** drive any scoring; the
default domain score is the mean absolute self-baseline delta.

## Limitations

- The mapping is approximate and intended as a structural scaffold, not a
  clinical assignment.
- Some variables (e.g. activity `steps`) are intentionally left unmapped until a
  suitable domain is defined.
- Units are not converted automatically; `expected_unit` documents the assumed
  unit only.
- Domain coverage depends on which variables are present in the input data.

## How to extend the mapping

`get_default_variable_domain_mapping()` returns a DataFrame with columns
`variable_pattern, canonical_variable, domain, data_stream, expected_unit,
direction_hint, interpretation_note`. To extend:

1. Build the default mapping DataFrame.
2. Append rows for new variables (use a normalized `variable_pattern`).
3. Pass the extended DataFrame as `mapping_df=...` to `map_variable_to_domain`,
   `map_variables_dataframe`, or `build_domain_scores_from_variables`.

New domains must use the canonical domain spellings above so the graph-ready
output stays compatible with the downstream pipeline.
