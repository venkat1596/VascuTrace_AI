"""Deterministic PET quantification engine (VascuTrace Phase 4).

Research prototype. Trained and evaluated using simulated vascular-like
abnormalities, not confirmed human post-angioplasty lesions.

Re-exports the public surface of :mod:`src.vascutrace.quantification.measure`.
"""

from src.vascutrace.quantification.measure import (
    RESEARCH_PROTOTYPE_WARNING,
    Measurement,
    QCFlags,
    QuantificationErrorCode,
    QuantificationResult,
    quantify_target,
)

__all__ = [
    "RESEARCH_PROTOTYPE_WARNING",
    "Measurement",
    "QCFlags",
    "QuantificationErrorCode",
    "QuantificationResult",
    "quantify_target",
]
