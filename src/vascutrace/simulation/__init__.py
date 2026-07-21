"""VascuTrace physics-informed synthetic-source simulation package (Phase 3).

Research prototype. Trained and evaluated using simulated vascular-like
abnormalities, not confirmed human post-angioplasty lesions.
"""

from src.vascutrace.simulation.anomaly import (
    RESEARCH_PROTOTYPE_WARNING,
    AchievedActivitySummary,
    AnomalySimulationParams,
    SimulationError,
    SimulationErrorCode,
    SimulationProvenance,
    SimulationResult,
    fwhm_mm_to_sigma_voxels,
    shift_ct_array,
    simulate_vascular_anomaly,
)

__all__ = [
    "RESEARCH_PROTOTYPE_WARNING",
    "AchievedActivitySummary",
    "AnomalySimulationParams",
    "SimulationError",
    "SimulationErrorCode",
    "SimulationProvenance",
    "SimulationResult",
    "fwhm_mm_to_sigma_voxels",
    "shift_ct_array",
    "simulate_vascular_anomaly",
]
