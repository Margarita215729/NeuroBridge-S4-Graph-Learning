"""Phase 11 — Transparent rule engine for operational resilience interpretation.

Translates Phase 6-10 evidence (within-subject graph deltas, attribution,
reference-calibrated envelope status, recovery/persistence, HRP hazard-context
alignment, and data coverage) into an *adaptive resilience state* and a
*dominant adaptation mode* using explicit, inspectable rules.

Operational resilience interpretation is a research-review layer. It describes
baseline-relative adaptation patterns for expert interpretation. It is not
diagnosis, treatment guidance, health risk scoring, exposure measurement, or an
operational medical decision.
"""

from __future__ import annotations

from pathlib import Path  # noqa: F401 - part of the documented public signature surface

import numpy as np  # noqa: F401 - available for downstream numeric helpers
import pandas as pd

GUARDRAIL = (
    "Operational resilience interpretation is a research-review layer. It "
    "describes baseline-relative adaptation patterns for expert interpretation. "
    "It is not diagnosis, treatment guidance, health risk scoring, exposure "
    "measurement, or an operational medical decision."
)

RESILIENCE_STATES: dict[str, str] = {
    "stable_compensated":              "Stable compensated trajectory",
    "localized_adaptive_shift":        "Localized adaptive shift",
    "distributed_adaptive_load":       "Distributed adaptive load",
    "systemic_strain_pattern":         "Systemic strain pattern",
    "persistent_displacement":         "Persistent displacement",
    "recovery_lag_pattern":            "Recovery lag pattern",
    "multi_domain_instability":        "Multi-domain instability",
    "coverage_limited_interpretation": "Coverage-limited interpretation",
}

DOMINANT_ADAPTATION_MODES: dict[str, str] = {
    "cardiometabolic_recovery_dominant":     "Cardiometabolic-recovery dominant",
    "immune_metabolic_dominant":             "Immune-metabolic dominant",
    "hematologic_cardiovascular_dominant":   "Hematologic-cardiovascular dominant",
    "sleep_autonomic_recovery_dominant":     "Sleep-autonomic-recovery dominant",
    "cognitive_emotional_recovery_dominant": "Cognitive-emotional-recovery dominant",
    "hazard_context_dominant":               "HRP hazard-context dominant",
    "multi_subgraph_distributed":            "Multi-subgraph distributed",
    "coverage_limited":                      "Coverage-limited",
}

# Phase 7 subgraph_name -> adaptation mode key.
_SUBGRAPH_TO_MODE: dict[str, str] = {
    "cardiometabolic":              "cardiometabolic_recovery_dominant",
    "immune_metabolic":             "immune_metabolic_dominant",
    "hematologic_cardiovascular":   "hematologic_cardiovascular_dominant",
    "sleep_autonomic_recovery":     "sleep_autonomic_recovery_dominant",
    "cognitive_emotional_recovery": "cognitive_emotional_recovery_dominant",
}

DEFAULT_THRESHOLDS: dict[str, float] = {
    "high_contribution_share":          0.35,
    "distributed_min_contributors":     3,
    "outside_envelope_count_moderate":  2,
    "outside_envelope_count_high":      4,
    "low_recovery_fraction":            0.4,
    "partial_recovery_fraction":        0.75,
    "low_coverage_fraction":            0.4,
    "min_meaningful_share":             0.05,
    "min_subgraph_share":               0.1,
    "min_domains_for_interpretation":   3,
}

# Mission phases where persistence (still displaced) is meaningful.
_POST_SHIFT_PHASES = {"postflight", "recovery"}


def _t(thresholds: dict | None, key: str) -> float:
    return float((thresholds or DEFAULT_THRESHOLDS).get(key, DEFAULT_THRESHOLDS[key]))


# ---------------------------------------------------------------------------
# Coverage assessment
# ---------------------------------------------------------------------------

def evaluate_coverage_limitations(
    coverage_report: pd.DataFrame | None,
    subject_id: str | None = None,
    timepoint: str | None = None,
    thresholds: dict | None = None,
) -> dict:
    """Assess whether interpretation should be coverage-limited.

    Uses the Phase 10 domain coverage report when available. Returns a dict with
    ``coverage_limited`` (bool), ``n_covered_domains``, ``n_total_domains``,
    ``coverage_fraction``, ``missing_domains`` (list), and ``coverage_note``.
    """
    min_domains = int(_t(thresholds, "min_domains_for_interpretation"))
    out = {
        "coverage_limited":   False,
        "n_covered_domains":  0,
        "n_total_domains":    0,
        "coverage_fraction":  float("nan"),
        "missing_domains":    [],
        "coverage_note":      "Coverage report unavailable; coverage not assessed.",
    }
    if coverage_report is None or coverage_report.empty or "domain" not in coverage_report.columns:
        # No coverage info: do not force coverage-limited, but note it.
        return out

    cov = coverage_report[coverage_report["domain"] != "unmapped"]
    total = int(len(cov))
    if "coverage_status" in cov.columns:
        covered_mask = cov["coverage_status"].astype(str) == "covered"
    elif "mapped_variable_count" in cov.columns:
        covered_mask = cov["mapped_variable_count"] > 0
    else:
        covered_mask = pd.Series([False] * total, index=cov.index)
    covered = int(covered_mask.sum())
    missing = sorted(cov.loc[~covered_mask, "domain"].astype(str).tolist())
    frac = (covered / total) if total else float("nan")

    limited = (covered < min_domains) or (
        not np.isnan(frac) and frac < _t(thresholds, "low_coverage_fraction"))
    out.update({
        "coverage_limited":  bool(limited),
        "n_covered_domains": covered,
        "n_total_domains":   total,
        "coverage_fraction": round(frac, 5) if not np.isnan(frac) else float("nan"),
        "missing_domains":   missing,
        "coverage_note": (
            f"{covered}/{total} biological domains covered"
            + (f"; coverage-limited (< {min_domains} domains or low fraction)."
               if limited else ".")
        ),
    })
    return out


# ---------------------------------------------------------------------------
# Evidence collection
# ---------------------------------------------------------------------------

def _filter_st(df: pd.DataFrame | None, subject_id: str, timepoint: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df
    if "subject_id" in out.columns:
        out = out[out["subject_id"].astype(str) == str(subject_id)]
    if "timepoint" in out.columns:
        out = out[out["timepoint"].astype(str) == str(timepoint)]
    return out


def _filter_subject(df: pd.DataFrame | None, subject_id: str) -> pd.DataFrame:
    if df is None or df.empty or "subject_id" not in df.columns:
        return pd.DataFrame()
    return df[df["subject_id"].astype(str) == str(subject_id)]


def derive_resilience_evidence(
    subject_id: str,
    timepoint: str,
    node_attr: pd.DataFrame | None,
    graph_attr: pd.DataFrame | None,
    subgraph_attr: pd.DataFrame | None,
    hazard_attr: pd.DataFrame | None,
    recovery_attr: pd.DataFrame | None,
    envelope_summary: pd.DataFrame | None,
    node_envelope: pd.DataFrame | None,
    graph_envelope: pd.DataFrame | None,
    hazard_envelope: pd.DataFrame | None,
    coverage_report: pd.DataFrame | None,
    thresholds: dict | None = None,
) -> dict:
    """Collect transparent evidence for one subject/timepoint.

    Returns a dict consumed by :func:`classify_resilience_state` and
    :func:`build_evidence_chain`.
    """
    min_share = _t(thresholds, "min_meaningful_share")
    min_sg_share = _t(thresholds, "min_subgraph_share")

    ev: dict = {
        "subject_id":   str(subject_id),
        "timepoint":    str(timepoint),
        "mission_phase": "unknown",
        "top_domain_contributor": "n/a",
        "top_domain_share": 0.0,
        "n_node_contributors": 0,
        "n_increasing": 0,
        "n_decreasing": 0,
        "direction_mixed": False,
        "top_graph_metric_contributor": "n/a",
        "top_graph_share": 0.0,
        "top_subgraph_contributor": "n/a",
        "top_subgraph_share": 0.0,
        "n_subgraphs_involved": 0,
        "top_hazard_context": "n/a",
        "top_hazard_share": 0.0,
        "n_outside_node": 0,
        "n_outside_graph": 0,
        "n_outside_hazard": 0,
        "total_outside": 0,
        "overall_envelope_flag": "n/a",
        "recovery_categories": [],
        "min_recovery_fraction": float("nan"),
        "persistent": False,
        "recovery_lag": False,
    }

    # Node attribution.
    nd = _filter_st(node_attr, subject_id, timepoint)
    if not nd.empty:
        if "mission_phase" in nd.columns:
            ev["mission_phase"] = str(nd.iloc[0]["mission_phase"])
        if "contribution_share" in nd.columns:
            ranked = nd.sort_values("contribution_share", ascending=False)
            top = ranked.iloc[0]
            ev["top_domain_contributor"] = str(top.get("domain", "n/a"))
            ev["top_domain_share"] = float(top.get("contribution_share", 0.0) or 0.0)
            ev["n_node_contributors"] = int((nd["contribution_share"] >= min_share).sum())
        if "direction" in nd.columns:
            ev["n_increasing"] = int((nd["direction"].astype(str) == "increase").sum())
            ev["n_decreasing"] = int((nd["direction"].astype(str) == "decrease").sum())
            ev["direction_mixed"] = ev["n_increasing"] >= 2 and ev["n_decreasing"] >= 2

    # Graph metric attribution.
    gm = _filter_st(graph_attr, subject_id, timepoint)
    if not gm.empty and "contribution_share" in gm.columns:
        top = gm.sort_values("contribution_share", ascending=False).iloc[0]
        ev["top_graph_metric_contributor"] = str(top.get("metric", "n/a"))
        ev["top_graph_share"] = float(top.get("contribution_share", 0.0) or 0.0)

    # Subgraph attribution.
    sg = _filter_st(subgraph_attr, subject_id, timepoint)
    if not sg.empty:
        if "n_available_domains" in sg.columns:
            sg = sg[sg["n_available_domains"] > 0]
    if not sg.empty and "total_contribution_share" in sg.columns:
        ranked = sg.sort_values("total_contribution_share", ascending=False)
        top = ranked.iloc[0]
        ev["top_subgraph_contributor"] = str(top.get("subgraph_name", "n/a"))
        ev["top_subgraph_share"] = float(top.get("total_contribution_share", 0.0) or 0.0)
        ev["n_subgraphs_involved"] = int((sg["total_contribution_share"] >= min_sg_share).sum())

    # Hazard-context attribution.
    hz = _filter_st(hazard_attr, subject_id, timepoint)
    if not hz.empty and "contribution_share" in hz.columns:
        top = hz.sort_values("contribution_share", ascending=False).iloc[0]
        ev["top_hazard_context"] = str(top.get("hazard", "n/a"))
        ev["top_hazard_share"] = float(top.get("contribution_share", 0.0) or 0.0)

    # Envelope summary / scores.
    es = _filter_st(envelope_summary, subject_id, timepoint)
    if not es.empty:
        row = es.iloc[0]
        ev["overall_envelope_flag"] = str(row.get("overall_envelope_flag", "n/a"))
        ev["n_outside_node"] = int(row.get("n_outside_node_envelope", 0) or 0)
        ev["n_outside_graph"] = int(row.get("n_outside_graph_envelope", 0) or 0)
        ev["n_outside_hazard"] = int(row.get("n_outside_hazard_envelope", 0) or 0)
    else:
        # Derive outside counts from per-feature envelope score tables if present.
        for tbl, key in ((node_envelope, "n_outside_node"),
                         (graph_envelope, "n_outside_graph"),
                         (hazard_envelope, "n_outside_hazard")):
            sub = _filter_st(tbl, subject_id, timepoint)
            if not sub.empty and "envelope_position" in sub.columns:
                ev[key] = int((sub["envelope_position"].astype(str)
                               == "outside_expected_envelope").sum())
    ev["total_outside"] = ev["n_outside_node"] + ev["n_outside_graph"] + ev["n_outside_hazard"]

    # Recovery (subject-level).
    rec = _filter_subject(recovery_attr, subject_id)
    if not rec.empty and "recovery_category" in rec.columns:
        ev["recovery_categories"] = sorted(rec["recovery_category"].astype(str).unique())
        if "recovery_fraction" in rec.columns:
            fr = pd.to_numeric(rec["recovery_fraction"], errors="coerce").dropna()
            ev["min_recovery_fraction"] = round(float(fr.min()), 5) if not fr.empty else float("nan")
        cats = set(ev["recovery_categories"])
        ev["persistent"] = "persistent_shift" in cats
        ev["recovery_lag"] = ("partial_recovery" in cats) and ("returned_near_baseline" not in cats)

    # Coverage.
    cov = evaluate_coverage_limitations(coverage_report, subject_id, timepoint, thresholds)
    ev["coverage"] = cov

    return ev


# ---------------------------------------------------------------------------
# Evidence chain
# ---------------------------------------------------------------------------

def build_evidence_chain(evidence: dict) -> list[str]:
    """Convert an evidence dict into ordered, plain-language evidence bullets."""
    chain: list[str] = []

    if evidence.get("top_domain_contributor", "n/a") not in ("n/a", None):
        chain.append(
            f"Top baseline-relative domain contributor: "
            f"{evidence['top_domain_contributor']} "
            f"(contribution share {evidence.get('top_domain_share', 0.0):.2f}).")

    if evidence.get("top_graph_metric_contributor", "n/a") not in ("n/a", None):
        chain.append(
            f"Leading graph-metric displacement: "
            f"{evidence['top_graph_metric_contributor']} "
            f"(share {evidence.get('top_graph_share', 0.0):.2f}).")

    if evidence.get("top_subgraph_contributor", "n/a") not in ("n/a", None):
        chain.append(
            f"Dominant biological subgraph: {evidence['top_subgraph_contributor']} "
            f"(share {evidence.get('top_subgraph_share', 0.0):.2f}); "
            f"{evidence.get('n_subgraphs_involved', 0)} subgraph(s) involved.")

    total_outside = evidence.get("total_outside", 0)
    chain.append(
        f"Reference-calibrated envelope: {total_outside} feature(s) outside the "
        f"expected envelope (node {evidence.get('n_outside_node', 0)}, "
        f"graph {evidence.get('n_outside_graph', 0)}, "
        f"hazard {evidence.get('n_outside_hazard', 0)}); overall flag "
        f"'{evidence.get('overall_envelope_flag', 'n/a')}'.")

    if evidence.get("top_hazard_context", "n/a") not in ("n/a", None):
        chain.append(
            f"HRP hazard-context alignment: {str(evidence['top_hazard_context']).replace('_', ' ')} "
            f"(share {evidence.get('top_hazard_share', 0.0):.2f}); alignment context only, "
            "not exposure measurement.")

    if evidence.get("recovery_categories"):
        cats = ", ".join(evidence["recovery_categories"])
        chain.append(
            f"Recovery / persistence: categories observed = [{cats}]; "
            f"minimum recovery fraction = {evidence.get('min_recovery_fraction', float('nan'))}.")

    cov = evidence.get("coverage", {})
    if cov:
        chain.append(f"Data coverage: {cov.get('coverage_note', 'not assessed')}")

    if evidence.get("direction_mixed"):
        chain.append(
            f"Direction of change is mixed across domains "
            f"({evidence.get('n_increasing', 0)} increasing, "
            f"{evidence.get('n_decreasing', 0)} decreasing).")

    return chain


# ---------------------------------------------------------------------------
# Resilience state classification
# ---------------------------------------------------------------------------

def classify_resilience_state(
    evidence: dict,
    thresholds: dict | None = None,
) -> dict:
    """Classify the adaptive resilience state from collected evidence.

    Returns ``resilience_state``, ``resilience_state_label``,
    ``confidence_level`` (``high``/``moderate``/``low``/``coverage_limited``),
    ``rule_triggers`` (list), and ``interpretation``.
    """
    out_high = int(_t(thresholds, "outside_envelope_count_high"))
    out_mod = int(_t(thresholds, "outside_envelope_count_moderate"))
    high_share = _t(thresholds, "high_contribution_share")
    distrib_min = int(_t(thresholds, "distributed_min_contributors"))

    triggers: list[str] = []
    cov = evidence.get("coverage", {})
    total_outside = int(evidence.get("total_outside", 0))
    n_node_contrib = int(evidence.get("n_node_contributors", 0))
    n_subgraphs = int(evidence.get("n_subgraphs_involved", 0))
    top_domain_share = float(evidence.get("top_domain_share", 0.0) or 0.0)
    top_hazard_share = float(evidence.get("top_hazard_share", 0.0) or 0.0)
    phase = str(evidence.get("mission_phase", "unknown")).lower()
    persistent = bool(evidence.get("persistent", False))
    recovery_lag = bool(evidence.get("recovery_lag", False))
    direction_mixed = bool(evidence.get("direction_mixed", False))

    # Priority cascade (most constraining first).
    state = None

    # 1. Coverage-limited interpretation.
    if cov.get("coverage_limited", False):
        state = "coverage_limited_interpretation"
        triggers.append("data coverage below interpretation threshold")

    # 2. Systemic strain pattern.
    if state is None and (
        total_outside >= out_high
        or (evidence.get("n_outside_node", 0) >= out_mod and n_subgraphs >= 2
            and top_hazard_share >= high_share and not _recovery_established(evidence))
    ):
        state = "systemic_strain_pattern"
        triggers.append(
            f"high graph displacement (outside-envelope features={total_outside}, "
            f"subgraphs involved={n_subgraphs})")

    # 3. Persistent displacement.
    if state is None and phase in _POST_SHIFT_PHASES and (
        persistent or (total_outside >= out_mod)
    ):
        state = "persistent_displacement"
        triggers.append(
            f"graph remains displaced in {phase} phase "
            f"(persistent recovery category={persistent}, outside-envelope={total_outside})")

    # 4. Recovery lag pattern.
    if state is None and recovery_lag:
        state = "recovery_lag_pattern"
        triggers.append("partial/delayed return toward baseline in recovery metrics")

    # 5. Multi-domain instability.
    if state is None and direction_mixed and n_subgraphs >= 2:
        state = "multi_domain_instability"
        triggers.append("mixed-direction deltas across multiple domains/subgraphs")

    # 6. Distributed adaptive load.
    if state is None and (
        n_node_contrib >= distrib_min and n_subgraphs >= 2 and total_outside >= 1
    ):
        state = "distributed_adaptive_load"
        triggers.append(
            f"broad cross-system load (node contributors={n_node_contrib}, "
            f"subgraphs={n_subgraphs})")

    # 7. Localized adaptive shift.
    if state is None and (
        total_outside >= 1 or top_domain_share >= high_share or n_node_contrib >= 1
    ):
        state = "localized_adaptive_shift"
        triggers.append(
            f"localized shift (top-domain share={top_domain_share:.2f}, "
            f"outside-envelope={total_outside})")

    # 8. Stable compensated (default).
    if state is None:
        state = "stable_compensated"
        triggers.append("low baseline-relative displacement and no envelope exceedance")

    # Confidence.
    if state == "coverage_limited_interpretation":
        confidence = "coverage_limited"
    else:
        has_envelope = evidence.get("overall_envelope_flag", "n/a") != "n/a"
        has_attr = evidence.get("top_subgraph_contributor", "n/a") != "n/a"
        cov_frac = cov.get("coverage_fraction", float("nan"))
        if has_envelope and has_attr and (isinstance(cov_frac, float) and not np.isnan(cov_frac)
                                          and cov_frac >= 0.6):
            confidence = "high"
        elif has_envelope or has_attr:
            confidence = "moderate"
        else:
            confidence = "low"

    label = RESILIENCE_STATES[state]
    interpretation = _state_interpretation(state, evidence)

    return {
        "resilience_state":       state,
        "resilience_state_label": label,
        "confidence_level":       confidence,
        "rule_triggers":          triggers,
        "interpretation":         interpretation,
    }


def _recovery_established(evidence: dict) -> bool:
    cats = set(evidence.get("recovery_categories", []))
    if not cats:
        return False
    return cats == {"returned_near_baseline"} or (
        "returned_near_baseline" in cats and "persistent_shift" not in cats
        and "partial_recovery" not in cats)


_STATE_MEANINGS: dict[str, str] = {
    "stable_compensated":
        "The graph remains close to personal baseline under current data coverage.",
    "localized_adaptive_shift":
        "A localized biological domain or subgraph shift is present without clear "
        "broad-system graph displacement.",
    "distributed_adaptive_load":
        "The graph shows a broader cross-system adaptive load pattern.",
    "systemic_strain_pattern":
        "The graph trajectory shows a stronger multi-system strain-like pattern "
        "requiring expert review.",
    "persistent_displacement":
        "The graph remains shifted from personal baseline after the main "
        "mission-phase shift.",
    "recovery_lag_pattern":
        "Recovery trajectory suggests delayed return toward baseline in selected "
        "graph features, domains, or subgraphs.",
    "multi_domain_instability":
        "The trajectory appears unstable across multiple domains under current "
        "data coverage.",
    "coverage_limited_interpretation":
        "Interpretation is limited by insufficient data coverage.",
}


def _state_interpretation(state: str, evidence: dict) -> str:
    meaning = _STATE_MEANINGS.get(state, "")
    return (
        f"{RESILIENCE_STATES[state]}: {meaning} This is a baseline-relative "
        "adaptation pattern for expert review, not diagnosis, treatment guidance, "
        "health risk scoring, exposure measurement, or an operational medical "
        "decision."
    )


# ---------------------------------------------------------------------------
# Dominant adaptation mode
# ---------------------------------------------------------------------------

def classify_dominant_adaptation_mode(
    subgraph_attr: pd.DataFrame | None,
    hazard_attr: pd.DataFrame | None,
    coverage_report: pd.DataFrame | None,
    subject_id: str,
    timepoint: str,
    thresholds: dict | None = None,
) -> dict:
    """Identify the dominant adaptation mode for one subject/timepoint.

    Returns ``dominant_adaptation_mode``, ``dominant_adaptation_mode_label``,
    ``mode_basis`` (short rationale), and ``modes`` (list of one or more keys).
    """
    high_share = _t(thresholds, "high_contribution_share")
    min_sg_share = _t(thresholds, "min_subgraph_share")

    cov = evaluate_coverage_limitations(coverage_report, subject_id, timepoint, thresholds)
    if cov.get("coverage_limited", False):
        return _mode_result(["coverage_limited"], "data coverage below interpretation threshold")

    sg = _filter_st(subgraph_attr, subject_id, timepoint)
    if not sg.empty and "n_available_domains" in sg.columns:
        sg = sg[sg["n_available_domains"] > 0]

    if sg.empty or "total_contribution_share" not in sg.columns:
        # Fall back to hazard-context dominance when subgraph attribution absent.
        hz = _filter_st(hazard_attr, subject_id, timepoint)
        if not hz.empty and "contribution_share" in hz.columns:
            top = hz.sort_values("contribution_share", ascending=False).iloc[0]
            if float(top.get("contribution_share", 0.0) or 0.0) >= high_share:
                return _mode_result(["hazard_context_dominant"],
                                    f"hazard-context alignment dominant "
                                    f"({top.get('hazard', 'n/a')})")
        return _mode_result(["coverage_limited"],
                            "no subgraph attribution available for this selection")

    ranked = sg.sort_values("total_contribution_share", ascending=False)
    involved = ranked[ranked["total_contribution_share"] >= min_sg_share]

    # Multiple comparably-involved subgraphs -> distributed.
    if len(involved) >= 2:
        top_share = float(ranked.iloc[0]["total_contribution_share"])
        second_share = float(ranked.iloc[1]["total_contribution_share"])
        if second_share >= 0.5 * top_share:
            return _mode_result(["multi_subgraph_distributed"],
                                f"{len(involved)} subgraphs comparably involved")

    top = ranked.iloc[0]
    top_name = str(top.get("subgraph_name", ""))
    mode = _SUBGRAPH_TO_MODE.get(top_name)

    # Strong hazard-context alignment can dominate.
    hz = _filter_st(hazard_attr, subject_id, timepoint)
    hazard_dominant = False
    if not hz.empty and "contribution_share" in hz.columns:
        top_hz = hz.sort_values("contribution_share", ascending=False).iloc[0]
        if float(top_hz.get("contribution_share", 0.0) or 0.0) >= high_share:
            hazard_dominant = True

    modes: list[str] = []
    if mode:
        modes.append(mode)
    if hazard_dominant:
        modes.append("hazard_context_dominant")
    if not modes:
        modes = ["multi_subgraph_distributed"]

    basis = f"dominant subgraph '{top_name}'"
    if hazard_dominant:
        basis += " with strong hazard-context alignment"
    return _mode_result(modes, basis)


def _mode_result(modes: list[str], basis: str) -> dict:
    primary = modes[0]
    return {
        "dominant_adaptation_mode":       primary,
        "dominant_adaptation_mode_label": DOMINANT_ADAPTATION_MODES.get(primary, primary),
        "modes":                          modes,
        "mode_basis":                     basis,
    }
