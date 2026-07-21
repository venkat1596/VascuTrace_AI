"""Transparent classical (non-neural) baseline detectors (VascuTrace Phase 5).

Research prototype. Trained and evaluated using simulated vascular-like
abnormalities, not confirmed human post-angioplasty lesions.

Re-exports the public surface of :mod:`src.vascutrace.baselines.threshold`.
"""

from src.vascutrace.baselines.threshold import (
    RESEARCH_PROTOTYPE_WARNING,
    CONNECTIVITY_26,
    UNTOUCHED_MEAN_COMPONENT_CEILING,
    BaselineCase,
    DetectionMode,
    FreezeResult,
    PositiveCaseOutcome,
    ThresholdSweepPoint,
    UntouchedCaseOutcome,
    aggregate_f2,
    compute_detection_map,
    evaluate_positive_case,
    evaluate_positive_case_at_threshold,
    evaluate_untouched_case,
    evaluate_untouched_case_at_threshold,
    f2_score,
    freeze_threshold,
    label_components,
    match_components_to_source,
)

__all__ = [
    "RESEARCH_PROTOTYPE_WARNING",
    "CONNECTIVITY_26",
    "UNTOUCHED_MEAN_COMPONENT_CEILING",
    "BaselineCase",
    "DetectionMode",
    "FreezeResult",
    "PositiveCaseOutcome",
    "ThresholdSweepPoint",
    "UntouchedCaseOutcome",
    "aggregate_f2",
    "compute_detection_map",
    "evaluate_positive_case",
    "evaluate_positive_case_at_threshold",
    "evaluate_untouched_case",
    "evaluate_untouched_case_at_threshold",
    "f2_score",
    "freeze_threshold",
    "label_components",
    "match_components_to_source",
]
