"""Phase 9 — Reusable reviewer-facing text blocks for the dashboard.

Centralizing the copy here keeps the guardrail language consistent across the
app and lets it be checked by tests without importing Streamlit.
"""

from __future__ import annotations

DASHBOARD_GUARDRAIL = (
    "This dashboard is a local research-review prototype. It is not a clinical "
    "monitoring system, not diagnosis, not treatment guidance, not exposure "
    "measurement, and not health risk scoring."
)

DASHBOARD_TITLE = "NeuroBridge-S4 Longitudinal Review Dashboard"

DASHBOARD_SUBTITLE = (
    "Within-subject biological adaptation graph trajectories with HRP "
    "hazard-context and reference-calibrated envelope review."
)

HEADER_GUARDRAIL_SHORT = (
    "Local research-review prototype. Not clinical monitoring. Not diagnosis. "
    "Not treatment guidance. Not exposure measurement. Not health risk scoring."
)


def get_dashboard_intro_text() -> str:
    """Short plain-language description of what the dashboard does."""
    return (
        "This dashboard turns the Phase 6\u20138 outputs into a reviewer-facing "
        "interface. For a selected subject and timepoint it shows what changed "
        "relative to personal baseline, what drove the change, how it maps to "
        "HRP hazard contexts, and whether the change exceeds the current "
        "reference-calibrated variability envelope. It supports expert review; "
        "it does not make decisions."
    )


def get_guardrail_text() -> str:
    """The full guardrail statement."""
    return DASHBOARD_GUARDRAIL


def explain_self_baseline() -> str:
    """Explain the within-subject self-baseline comparison."""
    return (
        "The primary signal is within-subject change from each individual's own "
        "personal baseline. A baseline-relative delta is the difference between a "
        "timepoint and that subject's baseline graph \u2014 not a comparison "
        "against a healthy population."
    )


def explain_reference_envelope() -> str:
    """Explain the reference-calibrated variability envelope."""
    return (
        "The reference envelope does not define whether a person is healthy or "
        "unhealthy. It calibrates how large a within-subject graph change is "
        "relative to expected variability in available proxy or analog data. "
        "Outside-envelope means a baseline-relative change is larger than "
        "expected under the current calibration data \u2014 a candidate for "
        "expert review, not diagnosis or risk scoring."
    )


def explain_hazard_context() -> str:
    """Explain HRP hazard-context alignment."""
    return (
        "Hazard-context alignment maps biological graph shifts onto NASA HRP "
        "hazard categories as operational interpretation context. It is "
        "hazard-context alignment, not exposure measurement and not a causal "
        "effect."
    )


def explain_attribution() -> str:
    """Explain trajectory attribution."""
    return (
        "Attribution decomposes the baseline-relative graph change into "
        "transparent contribution shares (absolute delta / total absolute "
        "delta) across biological domains, graph metrics, subgraphs, and "
        "hazard contexts. It identifies monitoring-relevant graph-shift "
        "contributors for expert review, not diagnosis or causal proof."
    )


def explain_recovery() -> str:
    """Explain recovery behavior categories."""
    return (
        "Recovery behavior describes whether each metric returned near personal "
        "baseline, partially recovered, remained shifted (persistent), overshot/"
        "reversed, or had insufficient data. Recovery persistence is a "
        "monitoring-relevant pattern, not diagnosis."
    )
