"""Transparent 26-connectivity threshold baseline detector (VascuTrace Phase 5).

RESEARCH_PROTOTYPE_WARNING
---------------------------------------------------------------------------
Research prototype. Trained and evaluated using simulated vascular-like
abnormalities, not confirmed human post-angioplasty lesions.
---------------------------------------------------------------------------

This module is the classical (non-neural), fully-transparent detector
VascuTrace ships as its default until/unless a neural model is promoted
(``evaluation.md``, "Learned-model promotion": promotion requires the
Siamese model to beat this baseline's validation sensitivity by >= 0.05
while matching or improving its untouched-validation mean component count
-- otherwise this module *is* the shipped demo default). It establishes NO
clinical claim: only controlled image-domain synthetic-source detectability
against a frozen, auditable, non-learned rule. It is pure, side-effect-free
numeric code -- no I/O, no network, no CUDA.

Implementation notes
============================================================================
1. Detection map (:func:`compute_detection_map`)
    The primary mode, ``DetectionMode.ASYMMETRY``, detects unilateral
    abnormal uptake by *bilateral asymmetry*: ``diff = pet - reflect(pet)``,
    where ``reflect`` maps every voxel through the caller-supplied
    *physical* mirror affine (never ``np.flip``/array-index mirroring --
    see "Physical reflection, not `np.flip`" below), then only the positive
    (ipsilateral-excess) side is kept (``np.maximum(diff, 0.0)``): a target
    voxel whose own SUV exceeds its mirrored contralateral SUV is
    "detectable excess"; a voxel that is *less* active than its mirror
    contributes nothing to the map (clamped to 0, not left negative), so
    thresholding the map at ``T > 0`` can only ever select excess-side
    voxels. A second mode, ``DetectionMode.SUV``, is a plain-SUV-threshold
    ablation: the raw PET array is returned unchanged, with no bilateral
    comparison at all -- provided so an experiment can quantify how much of
    the primary mode's performance actually comes from the asymmetry
    comparison versus a naive absolute-uptake threshold.

2. Physical reflection, not ``np.flip``
    ``_reflect_scalar_field`` performs the same *physical-affine* reflection
    technique already used, and already tested, elsewhere in this project --
    duplicated here rather than imported, per this project's own explicit
    convention for small private cross-module primitives (see
    ``simulation/anomaly.py``'s ``_apply_affine_points`` docstring: "those
    functions are private ... and this project's existing convention ... is
    a tiny independent copy per module rather than a private cross-module
    import"). Concretely, for every voxel of the output grid it walks
    voxel -> world (``geometry.affine``) -> reflected world (the caller's
    validated Householder mirror affine) -> fractional input-voxel
    coordinate (inverse ``geometry.affine``), then trilinearly interpolates
    the source field there (``scipy.ndimage.map_coordinates``, ``order=1``,
    ``mode="constant"``, ``cval=0.0`` outside the field of view) -- the same
    output-voxel -> world -> input-voxel walk
    ``geometry.resample_ct_to_pet``/``geometry._output_voxel_grid_input_indices``
    uses for cross-grid resampling, applied here on a *single* grid (no
    RAS<->LPS bridge needed: the reflection affine is already expressed in
    the PET grid's own physical space, exactly as
    ``quantification.measure.quantify_target`` uses it for its
    contralateral-ROI sampling). The one deliberate difference from that
    ``quantify_target`` usage: it samples sparsely, at *nearest*-voxel
    precision, restricted to a known mask; this module needs a value at
    *every* voxel (there is no mask yet -- finding one is the point), so it
    interpolates trilinearly over the whole grid instead. The mirror-affine
    *validation* (``_validate_reflection_affine``) is likewise a
    deliberately-duplicated copy of
    ``quantification.measure._validate_reflection_affine`` -- identical
    checks and tolerances (symmetric, involutive, orientation-reversing,
    and the eigenvalue-multiplicity check that rejects a full 3-axis point
    inversion ``-I``), same rationale.

3. Thresholding and 26-connected components (:func:`label_components`)
    The detection map is thresholded strictly (``detection_map > T``) and
    labeled with a FROZEN full 3x3x3 all-ones structuring element
    (``CONNECTIVITY_26``) -- 26-connectivity: two voxels that touch only at
    a shared corner or edge (not just a face) belong to the same component.
    This matches both this project's own quantification engine
    (``quantification.measure._CONNECTIVITY_26``, used there for the
    ``fragmented``/``component_count`` QC flags) and the evaluation
    contract's frozen 3-D component rule.

4. One-source matching and the FROZEN tie rule (:func:`match_components_to_source`)
    Each positive case carries exactly one ground-truth synthetic source
    (a binary mask, already thresholded at ``source_fraction >= 0.5`` by
    ``simulation.anomaly.simulate_vascular_anomaly``). A candidate component
    "matches" the source only when at least one of its voxels intersects
    that mask -- verbatim
    ``evaluation.md``: "A component matches it only when at least one
    component voxel intersects the binary supervision mask
    (``source_fraction >= 0.5``)." When more than one component intersects,
    the FROZEN, deterministic tie-break (also verbatim ``evaluation.md``) is:

        (1) greatest intersection voxel count,
        (2) greatest *component score* -- this module's score is the
            component's own peak (max) detection-map value, the natural
            per-component analogue of this project's frozen scan-level
            "abnormality score (for example, the maximum frozen candidate
            score)" (``evaluation.md``, "Metrics and terminology"); the
            specification that specified this module left the exact score
            definition to be "defined and frozen" here, so this choice is
            an explicit, cited design decision, not a re-derivation of
            ``evaluation.md`` from memory,
        (3) lexicographically smallest native voxel index -- implemented as
            "lowest ``scipy.ndimage.label`` label id". These two are
            equivalent for this project's installed SciPy: ``label``
            assigns ids by increasing C-order (row-major) raster-scan
            position, i.e. the component containing the lexicographically
            smallest ``(i, j, k)`` voxel index is always assigned the
            lowest label id -- verified empirically against this
            environment's installed SciPy before relying on it (the same
            "verify before relying on it" discipline ``geometry.py`` and
            ``simulation/anomaly.py`` already apply to their own SciPy/NumPy
            assumptions).

    Every component that is not the winner -- whether it overlapped the
    source and lost the tie, or never overlapped it at all -- is
    "unmatched": counted, reported, and visible (never silently dropped),
    and contributes to ``PositiveCaseOutcome.fp`` (see next section).

5. TP/FN/FP and the exact, FROZEN F2 (:func:`f2_score`, :func:`aggregate_f2`)
    Per positive case: ``tp = 1`` if the source was matched, else
    ``fn = 1`` (mutually exclusive, exactly one of the two per case, since
    there is exactly one source); ``fp`` is the count of unmatched
    components for that case (0, 1, or more). ``f2_score`` is the exact,
    FROZEN formula ``F2 = 5*TP / (5*TP + 4*FN + FP)`` (weights fixed; a
    degenerate ``TP = FN = FP = 0`` -- reachable only from an empty case
    set, never from a single well-formed case -- returns ``0.0`` by
    explicit convention, matching the usual precision/recall "no signal, no
    error" convention; this is a resolved, documented default, not a
    silent NaN).

    **Scope note vs. the deeper evaluation contract (flagged, not
    silent):** ``evaluation.md`` also names this same
    ``F2 = 5*TP/(5*TP+4*FN+FP)`` formula but, in its fuller atlas-integrated
    setting, defines ``FP`` at the *scan* level over pooled sham+untouched
    *negative* scans ("FP as a negative scan with at least one component"),
    not at the component level from *positive*-case extra detections. This
    implementation's own explicit contract (points 3-4) and its required-tests list
    ("unmatched components counted as FP and remain visible"; the worked
    example "TP=1, FN=0, FP=2 -> 5/7") instead define FP at the *component*
    level, computed only from positive cases -- with untouched-case
    activity handled by a wholly separate mechanism, the ceiling gate in
    (6) below, never folded into F2's FP term. This module implements the
    implementation's explicit, test-plan-backed definition (the narrower, standalone
    baseline this phase asks for); ``evaluation.md``'s scan-level,
    sham-inclusive F2 belongs to the later, fuller atlas/promotion phases
    (P7/P8), which this CPU-only, generated-fixture-only phase does not
    build (no "sham" case concept exists here at all). Flagging this
    explicitly for reviewer awareness rather than silently picking one.
    Also out of this implementation's scope (so also not implemented here):
    ``evaluation.md``'s "minimum physical component volume" gate, applied
    alongside the score threshold when forming components, and its
    corresponding 4th freeze-tie-break level ("larger minimum component
    volume") -- this module has no such parameter to tie-break on.

6. Validation-only freeze (:func:`freeze_threshold`)
    Sweeps a set of candidate thresholds (explicit, or an internally-built
    default grid derived from the *validation positive* cases' own positive
    detection-map values -- never from test data). For each candidate ``T``,
    computes the pooled positive-case F2 (5) and, independently, the mean
    number of 26-connected components detected on the *untouched* validation
    scans. A candidate is feasible only when that untouched mean is
    ``<= 0.25`` per scan -- verbatim ``evaluation.md``: "A candidate is
    feasible only when the mean number of components on untouched
    validation scans is at most 0.25 per scan." Among feasible candidates,
    the frozen threshold maximizes F2; ties are broken, in order, exactly as
    ``evaluation.md`` specifies ("Among feasible candidates, maximize F2.
    Break exact ties by lower untouched mean component count, then higher
    score threshold[...]"): (1) higher F2, (2) lower untouched mean
    component count, (3) higher threshold value (more conservative). (The
    4th-level tie-break ``evaluation.md`` names, "then larger minimum
    component volume," is inapplicable here -- see the scope note in (5).)

7. No-promotion fallback (contract point 6)
    If *no* candidate threshold satisfies the ceiling, :func:`freeze_threshold`
    returns an explicit, structured ``FreezeResult(feasible=False, ...)``
    carrying a machine-readable ``reason`` and the full sweep record for
    audit -- never a threshold that violates the ceiling, and never a
    silent substitute value. This matches ``evaluation.md`` verbatim: "If no
    candidate is feasible, record the gate as infeasible, do not relax it or
    promote a method[...]."

8. Untouched-scan reporting (contract point 7)
    Detections on untouched (healthy) validation/control scans are reported
    only as ``UntouchedCaseOutcome.component_count`` (an "activation/
    component count") -- this module has no field, method, or docstring
    anywhere that calls an untouched-scan detection a "false positive",
    matching both this implementation's explicit contract and ``evaluation.md``
    ("Untouched healthy-control scans: ... do not report clinical
    false-positive rate, specificity, or NPV").
============================================================================
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from scipy.ndimage import label
from scipy.ndimage import maximum as _ndi_component_maximum
from scipy.ndimage import map_coordinates

from src.vascutrace.geometry import RESEARCH_PROTOTYPE_WARNING, GridGeometry

__all__ = [
    "RESEARCH_PROTOTYPE_WARNING",
    "CONNECTIVITY_26",
    "UNTOUCHED_MEAN_COMPONENT_CEILING",
    "DetectionMode",
    "BaselineCase",
    "PositiveCaseOutcome",
    "UntouchedCaseOutcome",
    "ThresholdSweepPoint",
    "FreezeResult",
    "compute_detection_map",
    "label_components",
    "match_components_to_source",
    "evaluate_positive_case_at_threshold",
    "evaluate_untouched_case_at_threshold",
    "evaluate_positive_case",
    "evaluate_untouched_case",
    "f2_score",
    "aggregate_f2",
    "freeze_threshold",
]

# ---------------------------------------------------------------------------
# Numerical policy constants
# ---------------------------------------------------------------------------

# FROZEN 26-connectivity structuring element (full 3x3x3, all-ones): matches
# `quantification.measure._CONNECTIVITY_26` and evaluation.md's "Form 3-D
# predicted components with 26-connectivity" rule.
CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=np.int64)

# FROZEN untouched-validation feasibility ceiling: "the mean number of
# components on untouched validation scans is at most 0.25 per scan"
# (evaluation.md, "Components, matching, and threshold selection").
UNTOUCHED_MEAN_COMPONENT_CEILING = 0.25

# FROZEN F2 weights (beta=2: recall weighted 4x more than precision's FP
# penalty). evaluation.md / this implementation: `F2 = 5*TP / (5*TP + 4*FN + FP)`.
_F2_TP_WEIGHT = 5.0
_F2_FN_WEIGHT = 4.0
_F2_FP_WEIGHT = 1.0

# Reflection-affine validation tolerances: identical to
# `quantification.measure`'s own tolerances for the identical check,
# duplicated per this project's per-module-copy convention (see module
# docstring, section 2).
_REFLECTION_SYMMETRY_ATOL = 1e-6
_REFLECTION_INVOLUTION_ATOL = 1e-6
_REFLECTION_EIGENVALUE_ATOL = 1e-6

# Default candidate-threshold sweep grid: percentiles of the pooled positive
# (nonzero) detection-map values across the validation positive cases, used
# only when the caller does not supply an explicit `candidate_thresholds`.
_DEFAULT_SWEEP_PERCENTILES = tuple(float(p) for p in np.linspace(1.0, 99.0, 25))


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


class DetectionMode(StrEnum):
    """Which detection map :func:`compute_detection_map` builds.

    ``ASYMMETRY`` (the primary/default mode) is the bilateral-asymmetry
    detector (contract point 1). ``SUV`` is a plain-SUV-threshold ablation
    mode: no bilateral comparison at all, provided only to let an experiment
    quantify the asymmetry comparison's contribution.
    """

    ASYMMETRY = "asymmetry"
    SUV = "suv"


@dataclass(frozen=True, slots=True, eq=False)
class BaselineCase:
    """One PET SUVbw case on a validated grid, with the physical reflection
    affine needed for ``DetectionMode.ASYMMETRY``.

    ``ground_truth_mask`` is the single frozen synthetic-source mask
    (contract point 3) for a *positive* case, or ``None`` for an
    *untouched* (healthy-control) case -- untouched cases have no source to
    match against by construction.

    ``eq``/``__hash__`` are left at Python's default identity-based
    behavior (matching ``GridGeometry``/``SimulationResult`` elsewhere in
    this project) because ndarray-field tuple equality raises rather than
    doing anything useful.
    """

    pet_suvbw: np.ndarray = field(repr=False)
    geometry: GridGeometry
    reflection_affine: np.ndarray = field(repr=False)
    ground_truth_mask: np.ndarray | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class PositiveCaseOutcome:
    """TP/FN/FP and matching detail for one positive (synthetic-source)
    case at one threshold. Exactly one of ``tp``/``fn`` is 1 (contract
    point 3: exactly one ground-truth source per positive case).
    """

    tp: int
    fn: int
    fp: int
    component_count: int
    matched_label: int | None
    unmatched_labels: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class UntouchedCaseOutcome:
    """Detections on one untouched (healthy-control) case at one threshold.

    Reported only as an activation/component count (contract point 7) --
    never as a "false positive": there is no ``fp`` field on this record,
    by design.
    """

    component_count: int
    component_labels: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ThresholdSweepPoint:
    """One row of :func:`freeze_threshold`'s full sweep record, for audit."""

    threshold: float
    positive_f2: float
    positive_tp: int
    positive_fn: int
    positive_fp: int
    untouched_mean_components: float
    # Proportion of untouched cases with >= 1 component -- report-only,
    # distinct from the ceiling-gated mean count (evaluation.md: "This is
    # distinct from untouched-scan activation rate ... report both.").
    untouched_activation_rate: float
    ceiling_satisfied: bool


@dataclass(frozen=True, slots=True)
class FreezeResult:
    """The frozen threshold + validation-only sweep record, or an explicit
    structured no-feasible-threshold result (contract point 6) when no
    candidate satisfies the untouched-mean-component ceiling. The ceiling is
    never relaxed to force a number: ``feasible=False`` always pairs with
    every ``frozen_*`` field being ``None``.
    """

    feasible: bool
    frozen_threshold: float | None
    frozen_f2: float | None
    frozen_untouched_mean_components: float | None
    frozen_untouched_activation_rate: float | None
    reason: str | None
    swept_thresholds: tuple[ThresholdSweepPoint, ...]


# ---------------------------------------------------------------------------
# Shared numeric helpers (deliberately-duplicated small primitives -- see
# module docstring, section 2, for the project convention this follows)
# ---------------------------------------------------------------------------


def _apply_affine(affine: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous affine to an ``(N, 3)`` array of points."""
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    homogeneous = np.concatenate([pts, ones], axis=1)
    return (np.asarray(affine, dtype=np.float64) @ homogeneous.T).T[:, :3]


def _validate_scalar_field(pet_suvbw: np.ndarray, geometry: GridGeometry) -> np.ndarray:
    pet = np.asarray(pet_suvbw, dtype=np.float64)
    if pet.shape != tuple(geometry.shape):
        raise ValueError("pet_suvbw shape must match geometry.shape")
    return pet


def _validate_reflection_affine(reflection_affine: np.ndarray) -> np.ndarray:
    """Validate a genuine physical single-plane mirror (Householder
    reflection): symmetric linear part, involutive (``L @ L == I``),
    orientation-reversing (``det < 0``), and exactly one eigenvalue ``~= -1``
    with the other two ``~= +1`` (the check that rejects a full 3-axis point
    inversion ``-I``, which also satisfies the first three alone). See
    module docstring, section 2.
    """
    reflection = np.asarray(reflection_affine, dtype=np.float64)
    if reflection.shape != (4, 4):
        raise ValueError("reflection_affine must be a 4x4 matrix")
    if not np.all(np.isfinite(reflection)):
        raise ValueError("reflection_affine must be finite")
    if not np.array_equal(reflection[3], np.array([0.0, 0.0, 0.0, 1.0])):
        raise ValueError(
            "reflection_affine must be a homogeneous affine (last row [0,0,0,1])"
        )
    linear = reflection[:3, :3]
    if not np.allclose(linear, linear.T, atol=_REFLECTION_SYMMETRY_ATOL):
        raise ValueError(
            "reflection_affine linear part must be symmetric "
            "(a physical mirror/Householder matrix)"
        )
    if not np.allclose(linear @ linear, np.eye(3), atol=_REFLECTION_INVOLUTION_ATOL):
        raise ValueError(
            "reflection_affine linear part must be an involution (L @ L == I)"
        )
    if np.linalg.det(linear) >= 0:
        raise ValueError("reflection_affine must be orientation-reversing (det < 0)")
    eigenvalues = np.linalg.eigvalsh(linear)
    num_negative_one = int(
        np.sum(np.isclose(eigenvalues, -1.0, atol=_REFLECTION_EIGENVALUE_ATOL))
    )
    num_positive_one = int(
        np.sum(np.isclose(eigenvalues, 1.0, atol=_REFLECTION_EIGENVALUE_ATOL))
    )
    if num_negative_one != 1 or num_positive_one != 2:
        raise ValueError(
            "reflection_affine must be a single-plane mirror: its linear part "
            "must have exactly one eigenvalue ~= -1 and two eigenvalues ~= +1"
        )
    return reflection


def _reflect_scalar_field(
    field_values: np.ndarray, geometry: GridGeometry, reflection: np.ndarray
) -> np.ndarray:
    """Physically reflect ``field_values`` (shaped ``geometry.shape``)
    through the already-validated mirror affine ``reflection``, sampled
    (trilinearly) back onto ``geometry``'s own grid. See module docstring,
    section 2.
    """
    shape = geometry.shape
    grids = np.meshgrid(*[np.arange(n, dtype=np.float64) for n in shape], indexing="ij")
    voxel_idx = np.stack([g.reshape(-1) for g in grids], axis=1)
    world = _apply_affine(geometry.affine, voxel_idx)
    reflected_world = _apply_affine(reflection, world)
    inverse_affine = np.linalg.inv(np.asarray(geometry.affine, dtype=np.float64))
    reflected_voxel = _apply_affine(inverse_affine, reflected_world)
    coordinates = reflected_voxel.T.reshape(3, *shape)
    reflected = map_coordinates(
        np.asarray(field_values, dtype=np.float64),
        coordinates,
        order=1,
        mode="constant",
        cval=0.0,
    )
    return reflected.reshape(shape)


# ---------------------------------------------------------------------------
# Public API: detection map
# ---------------------------------------------------------------------------


def compute_detection_map(
    pet_suvbw: np.ndarray,
    geometry: GridGeometry,
    reflection_affine: np.ndarray,
    *,
    mode: DetectionMode = DetectionMode.ASYMMETRY,
) -> np.ndarray:
    """Build a per-voxel detection map from a PET SUVbw crop.

    ``ASYMMETRY`` (default): ``diff = pet - reflect(pet)``, positive
    (ipsilateral-excess) side only (``np.maximum(diff, 0.0)``); ``reflect``
    is the *physical*-affine reflection in :func:`_reflect_scalar_field`
    (never ``np.flip``). ``SUV``: returns the raw ``pet_suvbw`` array
    unchanged (a plain-SUV-threshold ablation; ``reflection_affine`` is
    accepted for interface uniformity but not used in this mode).

    Returns a ``float32`` array shaped ``geometry.shape``.
    """
    pet = _validate_scalar_field(pet_suvbw, geometry)
    if mode is DetectionMode.SUV:
        return pet.astype(np.float32)
    if mode is not DetectionMode.ASYMMETRY:
        raise ValueError(f"unsupported DetectionMode: {mode!r}")
    reflection = _validate_reflection_affine(reflection_affine)
    reflected = _reflect_scalar_field(pet, geometry, reflection)
    positive_excess = np.maximum(pet - reflected, 0.0)
    return positive_excess.astype(np.float32)


# ---------------------------------------------------------------------------
# Public API: thresholding + FROZEN 26-connectivity
# ---------------------------------------------------------------------------


def label_components(
    detection_map: np.ndarray, threshold: float
) -> tuple[np.ndarray, int]:
    """Threshold ``detection_map`` strictly (``> threshold``) and label
    connected components with the FROZEN 26-connectivity structuring
    element (:data:`CONNECTIVITY_26`). Returns ``(labeled_map, component_count)``,
    the exact ``scipy.ndimage.label`` return shape/semantics.
    """
    binary = np.asarray(detection_map) > float(threshold)
    labeled_map, component_count = label(binary, structure=CONNECTIVITY_26)
    return labeled_map, int(component_count)


# ---------------------------------------------------------------------------
# Public API: one-source matching + FROZEN tie rule
# ---------------------------------------------------------------------------


def match_components_to_source(
    detection_map: np.ndarray,
    labeled_map: np.ndarray,
    component_count: int,
    ground_truth_mask: np.ndarray,
) -> tuple[int | None, tuple[int, ...]]:
    """FROZEN one-source matching + tie rule (module docstring, section 4).

    Returns ``(matched_label, unmatched_labels)``: ``matched_label`` is the
    winning component's label id, or ``None`` if no component intersects
    ``ground_truth_mask`` (a source-detection miss). ``unmatched_labels``
    always lists every other detected component (never silently dropped;
    contract point 3) -- when ``matched_label`` is ``None`` this is *every*
    detected component; otherwise it is every component except the winner.
    """
    if component_count == 0:
        return None, ()

    labels = np.arange(1, component_count + 1)
    gt_bool = np.asarray(ground_truth_mask, dtype=bool)
    if gt_bool.any():
        overlaps = np.bincount(
            labeled_map[gt_bool].ravel(), minlength=component_count + 1
        )[1:]
    else:
        overlaps = np.zeros(component_count, dtype=np.int64)

    overlapping_labels = labels[overlaps > 0]
    if overlapping_labels.size == 0:
        return None, tuple(int(x) for x in labels)

    peak_scores = np.asarray(
        _ndi_component_maximum(detection_map, labeled_map, index=overlapping_labels),
        dtype=np.float64,
    )
    overlap_for = {int(lbl): int(overlaps[lbl - 1]) for lbl in overlapping_labels}
    score_for = dict(
        zip((int(x) for x in overlapping_labels), (float(s) for s in peak_scores))
    )
    # Tie order: (1) greatest overlap, (2) greatest component peak score,
    # (3) lowest label id (== lexicographically smallest native voxel index
    # for this project's installed SciPy; see module docstring, section 4).
    matched = min(
        (int(x) for x in overlapping_labels),
        key=lambda lbl: (-overlap_for[lbl], -score_for[lbl], lbl),
    )
    unmatched = tuple(int(x) for x in labels if x != matched)
    return matched, unmatched


# ---------------------------------------------------------------------------
# Public API: per-case outcomes
# ---------------------------------------------------------------------------


def evaluate_positive_case_at_threshold(
    detection_map: np.ndarray, ground_truth_mask: np.ndarray, threshold: float
) -> PositiveCaseOutcome:
    """TP/FN/FP for one positive case's already-built ``detection_map`` at
    one ``threshold``. ``tp=1`` iff the source was matched, else ``fn=1``;
    ``fp`` is the count of unmatched (visible, counted) components.
    """
    labeled_map, component_count = label_components(detection_map, threshold)
    matched, unmatched = match_components_to_source(
        detection_map, labeled_map, component_count, ground_truth_mask
    )
    tp = 1 if matched is not None else 0
    fn = 0 if matched is not None else 1
    return PositiveCaseOutcome(
        tp=tp,
        fn=fn,
        fp=len(unmatched),
        component_count=component_count,
        matched_label=matched,
        unmatched_labels=unmatched,
    )


def evaluate_untouched_case_at_threshold(
    detection_map: np.ndarray, threshold: float
) -> UntouchedCaseOutcome:
    """Activation/component count for one untouched case's already-built
    ``detection_map`` at one ``threshold`` (contract point 7: never "FP").
    """
    _labeled_map, component_count = label_components(detection_map, threshold)
    return UntouchedCaseOutcome(
        component_count=component_count,
        component_labels=tuple(range(1, component_count + 1)),
    )


def evaluate_positive_case(
    case: BaselineCase,
    threshold: float,
    *,
    mode: DetectionMode = DetectionMode.ASYMMETRY,
) -> PositiveCaseOutcome:
    """:func:`evaluate_positive_case_at_threshold`, computing the detection
    map from ``case`` first. Requires ``case.ground_truth_mask``.
    """
    if case.ground_truth_mask is None:
        raise ValueError(
            "evaluate_positive_case requires case.ground_truth_mask "
            "(this is a positive-case-only operation)"
        )
    detection_map = compute_detection_map(
        case.pet_suvbw, case.geometry, case.reflection_affine, mode=mode
    )
    return evaluate_positive_case_at_threshold(
        detection_map, case.ground_truth_mask, threshold
    )


def evaluate_untouched_case(
    case: BaselineCase,
    threshold: float,
    *,
    mode: DetectionMode = DetectionMode.ASYMMETRY,
) -> UntouchedCaseOutcome:
    """:func:`evaluate_untouched_case_at_threshold`, computing the
    detection map from ``case`` first.
    """
    detection_map = compute_detection_map(
        case.pet_suvbw, case.geometry, case.reflection_affine, mode=mode
    )
    return evaluate_untouched_case_at_threshold(detection_map, threshold)


# ---------------------------------------------------------------------------
# Public API: exact, FROZEN F2
# ---------------------------------------------------------------------------


def f2_score(tp: int, fn: int, fp: int) -> float:
    """Exact, FROZEN ``F2 = 5*TP / (5*TP + 4*FN + FP)`` (module docstring,
    section 5). The degenerate ``TP=FN=FP=0`` denominator returns ``0.0``
    by explicit convention (never a NaN/silent substitute) -- unreachable
    from any single well-formed positive case (exactly one of TP/FN is
    always 1), only from an empty aggregate.
    """
    numerator = _F2_TP_WEIGHT * tp
    denominator = _F2_TP_WEIGHT * tp + _F2_FN_WEIGHT * fn + _F2_FP_WEIGHT * fp
    if denominator == 0.0:
        return 0.0
    return float(numerator / denominator)


def aggregate_f2(outcomes: Sequence[PositiveCaseOutcome]) -> float:
    """Pooled F2 (:func:`f2_score`) over a set of :class:`PositiveCaseOutcome`."""
    tp = sum(o.tp for o in outcomes)
    fn = sum(o.fn for o in outcomes)
    fp = sum(o.fp for o in outcomes)
    return f2_score(tp, fn, fp)


# ---------------------------------------------------------------------------
# Public API: validation-only freeze
# ---------------------------------------------------------------------------


def _default_threshold_grid(detection_maps: Iterable[np.ndarray]) -> tuple[float, ...]:
    """Percentile-based default candidate-threshold grid, pooled from the
    positive (nonzero) values of the given detection maps. See module
    docstring, section 6.
    """
    pooled = [np.asarray(dm)[np.asarray(dm) > 0.0].ravel() for dm in detection_maps]
    pooled = [p for p in pooled if p.size > 0]
    if not pooled:
        return ()
    pooled_values = np.concatenate(pooled).astype(np.float64)
    percentile_values = np.percentile(pooled_values, _DEFAULT_SWEEP_PERCENTILES)
    return tuple(sorted({float(np.round(v, 8)) for v in percentile_values}))


def freeze_threshold(
    validation_positive_cases: Sequence[BaselineCase],
    validation_untouched_cases: Sequence[BaselineCase],
    *,
    candidate_thresholds: Sequence[float] | None = None,
    mode: DetectionMode = DetectionMode.ASYMMETRY,
    ceiling: float = UNTOUCHED_MEAN_COMPONENT_CEILING,
) -> FreezeResult:
    """Sweep candidate thresholds on VALIDATION ONLY and freeze the
    F2-maximizing ``T`` subject to the untouched mean-component-count
    ceiling (module docstring, section 6). If no candidate is feasible,
    returns a structured ``FreezeResult(feasible=False, ...)`` -- the
    ceiling is never relaxed to force a number (module docstring, section 7).

    ``candidate_thresholds``, if omitted, is built internally from the
    validation *positive* cases only (never from a caller's test data).
    """
    if len(validation_positive_cases) == 0:
        raise ValueError(
            "freeze_threshold requires at least one validation_positive_cases entry"
        )
    if len(validation_untouched_cases) == 0:
        raise ValueError(
            "freeze_threshold requires at least one validation_untouched_cases entry"
        )
    for case in validation_positive_cases:
        if case.ground_truth_mask is None:
            raise ValueError(
                "every validation_positive_cases entry needs ground_truth_mask"
            )

    positive_maps = [
        compute_detection_map(c.pet_suvbw, c.geometry, c.reflection_affine, mode=mode)
        for c in validation_positive_cases
    ]
    positive_gt_masks = [c.ground_truth_mask for c in validation_positive_cases]
    untouched_maps = [
        compute_detection_map(c.pet_suvbw, c.geometry, c.reflection_affine, mode=mode)
        for c in validation_untouched_cases
    ]

    thresholds = (
        tuple(sorted({float(t) for t in candidate_thresholds}))
        if candidate_thresholds is not None
        else _default_threshold_grid(positive_maps)
    )
    if not thresholds:
        return FreezeResult(
            feasible=False,
            frozen_threshold=None,
            frozen_f2=None,
            frozen_untouched_mean_components=None,
            frozen_untouched_activation_rate=None,
            reason="NO_CANDIDATE_THRESHOLDS",
            swept_thresholds=(),
        )

    sweep: list[ThresholdSweepPoint] = []
    for t in thresholds:
        positive_outcomes = [
            evaluate_positive_case_at_threshold(dm, gt, t)
            for dm, gt in zip(positive_maps, positive_gt_masks)
        ]
        tp = sum(o.tp for o in positive_outcomes)
        fn = sum(o.fn for o in positive_outcomes)
        fp = sum(o.fp for o in positive_outcomes)
        positive_f2 = f2_score(tp, fn, fp)

        untouched_outcomes = [
            evaluate_untouched_case_at_threshold(dm, t) for dm in untouched_maps
        ]
        counts = [o.component_count for o in untouched_outcomes]
        mean_components = float(np.mean(counts))
        activation_rate = float(np.mean([1.0 if c > 0 else 0.0 for c in counts]))
        ceiling_satisfied = mean_components <= ceiling

        sweep.append(
            ThresholdSweepPoint(
                threshold=t,
                positive_f2=positive_f2,
                positive_tp=tp,
                positive_fn=fn,
                positive_fp=fp,
                untouched_mean_components=mean_components,
                untouched_activation_rate=activation_rate,
                ceiling_satisfied=ceiling_satisfied,
            )
        )

    feasible_points = [p for p in sweep if p.ceiling_satisfied]
    if not feasible_points:
        return FreezeResult(
            feasible=False,
            frozen_threshold=None,
            frozen_f2=None,
            frozen_untouched_mean_components=None,
            frozen_untouched_activation_rate=None,
            reason=(
                "NO_FEASIBLE_THRESHOLD: every swept threshold's untouched "
                f"mean-component-count exceeds the frozen ceiling ({ceiling}); "
                "the ceiling was NOT relaxed"
            ),
            swept_thresholds=tuple(sweep),
        )

    # FROZEN freeze tie-break (module docstring, section 6, verbatim
    # evaluation.md order): (1) higher F2, (2) lower untouched mean
    # component count, (3) higher threshold. Thresholds are deduplicated
    # above, so this 3-key order is always a total order (no 4th key
    # needed).
    best = min(
        feasible_points,
        key=lambda p: (-p.positive_f2, p.untouched_mean_components, -p.threshold),
    )
    return FreezeResult(
        feasible=True,
        frozen_threshold=best.threshold,
        frozen_f2=best.positive_f2,
        frozen_untouched_mean_components=best.untouched_mean_components,
        frozen_untouched_activation_rate=best.untouched_activation_rate,
        reason=None,
        swept_thresholds=tuple(sweep),
    )
