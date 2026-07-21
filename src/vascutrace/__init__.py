"""VascuTrace imaging/ML package (real pipeline).

This ``src/vascutrace`` package holds the real imaging and machine-learning
pipeline (geometry, data, simulation, quantification, baselines, ml, evaluation).
The legacy root ``vascutrace`` package remains the labeled reference/mock
product slice until real inference is promoted; the two must not be confused.

Research prototype. Trained and evaluated using simulated vascular-like
abnormalities, not confirmed human post-angioplasty lesions.
"""

from src.vascutrace.geometry import RESEARCH_PROTOTYPE_WARNING

__all__ = ["RESEARCH_PROTOTYPE_WARNING", "__version__"]
__version__ = "0.1.0"
