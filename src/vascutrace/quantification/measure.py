"""Deterministic PET SUV quantification engine (VascuTrace Phase 4).

RESEARCH_PROTOTYPE_WARNING
---------------------------------------------------------------------------
Research prototype. Trained and evaluated using simulated vascular-like
abnormalities, not confirmed human post-angioplasty lesions.
---------------------------------------------------------------------------

This module computes deterministic SUV statistics from a raw native PET
SUVbw volume and an integer/boolean target mask. It is pure, side-effect-free
numeric code -- no I/O, no network, no CUDA -- so it works identically over a
ground-truth synthetic mask, a predicted DL mask, or a manual debugging mask.
Every measurement here is a code output; an LLM (or any other caller) must
never invent, "eyeball", or otherwise rewrite a number this module produces.

Raw vs. normalized separation
==============================================================================
This engine measures ONLY the array it is given. There is no clip/scale
logic anywhere in this module: no ``[0, 10]/10`` PET-network normalization,
no implicit rescaling. Callers must pass the raw native SUVbw array, never
the ``[0, 10]``-clipped network copy. Because the engine has no normalization code path, the
guarantee "measures the raw array you pass and does not silently normalize"
holds by construction, not by a runtime detection heuristic that could be
fooled -- see the test suite's dedicated raw-vs-normalized-array case.

Typed structured nulls
==============================================================================
Every measurement that can be invalid or empty (SUVmax/mean, contralateral
SUVmax/mean, asymmetry index, target-to-background ratio, longitudinal
extent, metabolic volume) is returned as a :class:`Measurement`: either
``Measurement(value=<float>, reason=None)`` for a valid result, or
``Measurement(value=None, reason="<CODE>")`` for an invalid one. Never 0,
never NaN, never a silently dropped field, never an epsilon-derived extreme
substitute standing in for "invalid". See :class:`QuantificationErrorCode`
for the exhaustive reason vocabulary.

Formula provenance
==============================================================================
The core formulas (SUVmax/mean over a valid mask, determinant-based mL
volume, the contralateral-denominator floor gate, and the exact asymmetry
index / target-to-background-ratio expressions) follow the project's frozen
deterministic-quantification contract:

- ``suvmax`` / ``suvmean``: max/mean native SUV inside a valid (nonempty)
  target mask. A mask with zero voxels is invalid (``EMPTY_MASK``). If any
  selected target SUV is nonfinite, both SUV measurements are null with
  ``NONFINITE_TARGET`` -- nonfinite voxels are never silently dropped from
  the reduction.
- ``metabolic_volume_ml``: ``voxel_count * abs(det(affine[:3, :3])) / 1000``,
  reusing the PET affine's determinant (P1's :class:`GridGeometry`) as the
  sole source of voxel volume -- never a hardcoded spacing assumption.
- Contralateral measurements sample the same physical mask reflected through
  a caller-supplied physical-space reflection affine (never ``np.flip`` or
  array-index mirroring -- see "Laterality, reflection, and corridors" in the
  same contract file). All reflected voxel locations must lie inside the PET
  grid; any out-of-domain location invalidates the whole contralateral ROI
  (``CONTRALATERAL_OUT_OF_DOMAIN``), matching "do not silently drop voxels".
- ``asymmetry_index = (target_mean - contra_mean) / (contra_mean + 1e-6)``
  and ``target_to_background_ratio = target_mean / (contra_mean + 1e-6)``
  (imaging-physics.md line 127 names this ``target_to_contralateral_ratio``
  and its numerator is explicitly ``target_mean``, not ``target_max`` --
  both ratios share the same SUVmean numerator/denominator shape), computed
  only after the contralateral-mean floor gate (``contra_mean > 1e-6``)
  passes; otherwise both are null with
  ``INVALID_CONTRALATERAL_DENOMINATOR``. The floor gate and the ``+1e-6``
  smoothing term are two distinct steps, per the contract, not one.

Field naming in :class:`QuantificationResult` follows this implementation's explicit
contract (``target_suvmax``, ``target_suvmean``, ``metabolic_volume_ml``,
``longitudinal_extent_mm``, ``contralateral_suvmax``, ``contralateral_suvmean``,
``asymmetry_index``, ``target_to_background_ratio``, ``laterality``) rather
than the shorter names used in the prose contract (``suvmax``,
``mask_volume_ml``, ``target_to_contralateral_ratio``); the underlying
formulas are identical -- ``target_to_background_ratio`` is computed from
``target_suvmean`` (not ``target_suvmax``), matching the frozen contract
verbatim.

Laterality and the reflection plane
==============================================================================
``laterality`` is derived algebraically from the same
``reflection_affine`` used for the contralateral ROI, never from array
display orientation. A physical mirror transform ``T(p) = p - 2 * (dot(n, p)
- offset) * n`` has a symmetric, involutive, orientation-reversing linear
part ``L = I - 2 * n @ n.T`` (eigenvalues ``-1`` once, ``+1`` twice); the
plane normal ``n`` is recovered as ``L``'s eigenvector for its most-negative
eigenvalue, sign-fixed to point toward RAS ``+R`` (matching "Orient every
left-to-right pair vector toward RAS +R" in the project's imaging-physics
contract), and the plane offset is recovered from the affine's translation
component via ``offset = dot(translation, n) / 2``. Every occupied target
voxel's signed distance to the plane then determines "left" (all negative),
"right" (all positive), "bilateral" (mixed sign, or all within tolerance of
the plane), or "none" (empty mask).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from scipy.ndimage import label

from src.vascutrace.geometry import RESEARCH_PROTOTYPE_WARNING, GridGeometry

__all__ = [
    "RESEARCH_PROTOTYPE_WARNING",
    "QuantificationErrorCode",
    "Measurement",
    "QCFlags",
    "QuantificationResult",
    "quantify_target",
]

# ---------------------------------------------------------------------------
# Numerical policy constants
# ---------------------------------------------------------------------------

# Contralateral-mean validity gate (strict: must exceed, not merely reach).
_CONTRALATERAL_DENOMINATOR_FLOOR = 1e-6
# Smoothing term added to a *validated* (above-floor) denominator, per the
# frozen imaging-physics.md formula -- distinct from the floor gate above.
_DENOMINATOR_EPSILON = 1e-6
# A mask smaller than a full 3x3x3 voxel neighbourhood is flagged as a
# partial-volume risk when no caller-supplied blur FWHM is available to
# compare against directly.
_PARTIAL_VOLUME_MIN_VOXELS = 27
# Tolerances for validating a caller-supplied reflection affine is a genuine
# single-plane physical mirror (Householder reflection): linear part
# symmetric, its own inverse (an involution), and -- critically -- exactly
# one eigenvalue ~= -1 with the other two ~= +1 (a *rank-1* mirror; without
# this multiplicity check, symmetric + involutive + det<0 alone is also
# satisfied by a full 3-axis point inversion (-I), which is not a mirror).
_REFLECTION_SYMMETRY_ATOL = 1e-6
_REFLECTION_INVOLUTION_ATOL = 1e-6
_REFLECTION_EIGENVALUE_ATOL = 1e-6
# A signed distance to the reflection plane within this tolerance (mm) is
# treated as "on the plane" for laterality purposes.
_LATERALITY_SIGN_TOL_MM = 1e-6
# 26-connectivity structuring element (3x3x3, all-ones) for fragmentation QC.
_CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=np.int64)


# ---------------------------------------------------------------------------
# Error taxonomy
# ---------------------------------------------------------------------------


class QuantificationErrorCode(StrEnum):
    """Typed, code-only reasons a :class:`Measurement` is a structured null."""

    EMPTY_MASK = "EMPTY_MASK"
    NONFINITE_TARGET = "NONFINITE_TARGET"
    CONTRALATERAL_OUT_OF_DOMAIN = "CONTRALATERAL_OUT_OF_DOMAIN"
    NONFINITE_CONTRALATERAL = "NONFINITE_CONTRALATERAL"
    INVALID_CONTRALATERAL_DENOMINATOR = "INVALID_CONTRALATERAL_DENOMINATOR"


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Measurement:
    """A typed, structured measurement result.

    Either a valid float (``value`` set, ``reason`` ``None``) or a typed
    structured null (``value`` ``None``, ``reason`` one of
    :class:`QuantificationErrorCode`'s values). Never 0/NaN/omission stands
    in for an invalid result -- a legitimately-zero *valid* measurement
    (e.g. ``asymmetry_index == 0.0``) is represented as
    ``Measurement(value=0.0, reason=None)``, which is distinguishable from
    ``Measurement(value=None, reason=...)`` by construction.
    """

    value: float | None
    reason: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.value is not None


@dataclass(frozen=True, slots=True)
class QCFlags:
    """Deterministic QC flags. Always defined -- even when every SUV-derived
    :class:`Measurement` above is a structured null.

    ``empty_mask`` and the geometry-only flags (``boundary_touching``,
    ``fragmented``, ``component_count``, ``partial_volume_risk``,
    ``misregistration_risk_placeholder``) depend only on the mask/geometry,
    never on SUV intensity values. ``invalid_region`` is the one flag that
    *does* look at intensity: it is ``True`` for an empty mask (zero voxels)
    **or** for a nonempty mask whose selected SUV values are *entirely*
    nonfinite -- a genuinely distinct, stronger condition than
    ``NONFINITE_TARGET`` (which nulls the SUV measurements the instant *any*
    selected voxel is nonfinite, per the "do not silently drop voxels"
    contract). A region with one bad voxel among otherwise-good ones has
    null SUV measurements but ``invalid_region == False``; a region that is
    entirely unusable has ``invalid_region == True``.
    """

    empty_mask: bool
    invalid_region: bool
    boundary_touching: bool
    fragmented: bool
    component_count: int
    partial_volume_risk: bool
    misregistration_risk_placeholder: bool


@dataclass(frozen=True, slots=True)
class QuantificationResult:
    """The full deterministic measurement bundle for one target mask."""

    target_suvmax: Measurement
    target_suvmean: Measurement
    metabolic_volume_ml: Measurement
    longitudinal_extent_mm: Measurement
    contralateral_suvmax: Measurement
    contralateral_suvmean: Measurement
    asymmetry_index: Measurement
    target_to_background_ratio: Measurement
    laterality: str
    qc: QCFlags


# ---------------------------------------------------------------------------
# Shared numeric helpers
# ---------------------------------------------------------------------------


def _apply_affine(affine: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous affine to an ``(N, 3)`` array of points."""
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    homogeneous = np.concatenate([pts, ones], axis=1)
    return (affine @ homogeneous.T).T[:, :3]


def _validate_scalar_field(suvbw: np.ndarray, geometry: GridGeometry) -> np.ndarray:
    suv = np.asarray(suvbw, dtype=np.float64)
    if suv.shape != tuple(geometry.shape):
        raise ValueError("suvbw shape must match geometry.shape")
    return suv


def _validate_and_binarize_mask(mask: np.ndarray, geometry: GridGeometry) -> np.ndarray:
    msk = np.asarray(mask)
    if msk.shape != tuple(geometry.shape):
        raise ValueError("mask shape must match geometry.shape")
    if not (np.issubdtype(msk.dtype, np.integer) or msk.dtype == np.bool_):
        raise TypeError("mask must be an integer- or boolean-labeled array")
    return msk != 0


def _validate_reflection_affine(reflection_affine: np.ndarray) -> np.ndarray:
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
    # Symmetric + involutive + det<0 alone is not sufficient: a full 3-axis
    # point inversion (-I) also satisfies all three (det((-I)) == -1) but is
    # not a single-plane mirror. A genuine Householder mirror's eigenvalues
    # are exactly {-1 (once), +1 (twice)} -- reject anything else (e.g. -I's
    # {-1, -1, -1}).
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


def _validate_centerline_direction(direction: np.ndarray) -> np.ndarray:
    vec = np.asarray(direction, dtype=np.float64)
    if vec.shape != (3,):
        raise ValueError("centerline_direction must be shape (3,)")
    if not np.all(np.isfinite(vec)):
        raise ValueError("centerline_direction must be finite")
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        raise ValueError("centerline_direction must be nonzero")
    return vec / norm


def _reflection_plane_normal_and_offset(
    reflection: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Recover the mirror plane's unit normal (oriented toward RAS +R) and
    scalar offset from a validated reflection affine's linear/translation
    parts. See the module docstring, "Laterality and the reflection plane".
    """
    linear = reflection[:3, :3]
    translation = reflection[:3, 3]
    eigenvalues, eigenvectors = np.linalg.eigh(linear)
    idx = int(np.argmin(eigenvalues))  # eigenvalue closest to -1
    normal = eigenvectors[:, idx]
    normal = normal / np.linalg.norm(normal)
    if np.dot(normal, np.array([1.0, 0.0, 0.0])) < 0:
        normal = -normal
    offset = float(np.dot(translation, normal) / 2.0)
    return normal, offset


def _principal_axis(world_points: np.ndarray) -> np.ndarray:
    """The dominant PCA direction of a set of physical points, via SVD of
    the mean-centered coordinates. Degenerate inputs (a single point, or
    several coincident points) have no well-defined direction but also a
    trivially-zero projected span, so any unit axis is returned for them.
    """
    centered = world_points - world_points.mean(axis=0, keepdims=True)
    if world_points.shape[0] < 2:
        return np.array([1.0, 0.0, 0.0])
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0])
    return axis / norm


def _laterality_from_signed_distances(distances: np.ndarray) -> str:
    positive = bool(np.any(distances > _LATERALITY_SIGN_TOL_MM))
    negative = bool(np.any(distances < -_LATERALITY_SIGN_TOL_MM))
    if positive and negative:
        return "bilateral"
    if positive:
        return "right"
    if negative:
        return "left"
    return "bilateral"  # every point is on the plane, within tolerance


def _connected_components(mask_bool: np.ndarray) -> tuple[int, bool]:
    """26-connectivity component count and the fragmentation flag."""
    _labeled, count = label(mask_bool, structure=_CONNECTIVITY_26)
    return int(count), count > 1


def _boundary_touching(mask_bool: np.ndarray) -> bool:
    return bool(
        mask_bool[0, :, :].any()
        or mask_bool[-1, :, :].any()
        or mask_bool[:, 0, :].any()
        or mask_bool[:, -1, :].any()
        or mask_bool[:, :, 0].any()
        or mask_bool[:, :, -1].any()
    )


def _partial_volume_risk(
    voxel_count: int, longitudinal_extent_mm: Measurement, blur_fwhm_mm: float | None
) -> bool:
    if voxel_count == 0:
        return False
    if blur_fwhm_mm is not None and longitudinal_extent_mm.value is not None:
        return bool(longitudinal_extent_mm.value < blur_fwhm_mm)
    return voxel_count < _PARTIAL_VOLUME_MIN_VOXELS


def _asymmetry_index(
    target_suvmean: Measurement, contralateral_suvmean: Measurement
) -> Measurement:
    if contralateral_suvmean.value is None:
        return Measurement(None, contralateral_suvmean.reason)
    if target_suvmean.value is None:
        return Measurement(None, target_suvmean.reason)
    if contralateral_suvmean.value <= _CONTRALATERAL_DENOMINATOR_FLOOR:
        return Measurement(
            None, QuantificationErrorCode.INVALID_CONTRALATERAL_DENOMINATOR.value
        )
    denominator = contralateral_suvmean.value + _DENOMINATOR_EPSILON
    return Measurement(
        (target_suvmean.value - contralateral_suvmean.value) / denominator
    )


def _target_to_background_ratio(
    target_suvmean: Measurement, contralateral_suvmean: Measurement
) -> Measurement:
    # Numerator is target_suvmean, per the frozen contract
    # (target_to_contralateral_ratio = target_mean / (contra_mean + 1e-6)),
    # not target_suvmax.
    if contralateral_suvmean.value is None:
        return Measurement(None, contralateral_suvmean.reason)
    if target_suvmean.value is None:
        return Measurement(None, target_suvmean.reason)
    if contralateral_suvmean.value <= _CONTRALATERAL_DENOMINATOR_FLOOR:
        return Measurement(
            None, QuantificationErrorCode.INVALID_CONTRALATERAL_DENOMINATOR.value
        )
    denominator = contralateral_suvmean.value + _DENOMINATOR_EPSILON
    return Measurement(target_suvmean.value / denominator)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def quantify_target(
    suvbw: np.ndarray,
    mask: np.ndarray,
    geometry: GridGeometry,
    reflection_affine: np.ndarray,
    *,
    centerline_direction: np.ndarray | None = None,
    blur_fwhm_mm: float | None = None,
    registration_shift_mm: float = 0.0,
) -> QuantificationResult:
    """Compute deterministic SUV measurements for one target mask.

    Parameters
    ----------
    suvbw:
        Raw native SUVbw array, shaped ``geometry.shape``. Never a
        normalized/clipped copy (see module docstring). No whole-array
        finiteness check is performed here -- only values selected by
        ``mask`` (target) or its reflection (contralateral) are checked.
    mask:
        Integer- or boolean-labeled array, shaped ``geometry.shape``.
        Nonzero (or ``True``) voxels are the target region; works
        identically for a ground-truth, predicted, or manual mask.
    geometry:
        The PET grid's validated :class:`GridGeometry` (P1). Supplies the
        voxel-\N{RIGHTWARDS ARROW}world affine and shape used for every
        physical-space computation below.
    reflection_affine:
        A ``(4, 4)`` physical-space mirror affine (e.g. through the subject
        mid-sagittal plane). Must be a genuine Householder reflection
        (symmetric, involutive, orientation-reversing linear part) -- see
        :func:`_validate_reflection_affine`. Used, in physical space, both
        to build the contralateral ROI and to derive ``laterality``; never
        ``np.flip`` or array-index mirroring.
    centerline_direction:
        Optional ``(3,)`` world-space direction to project onto for
        ``longitudinal_extent_mm``. When omitted, the direction is the
        first principal component (PCA) of the mask's own occupied-voxel
        physical coordinates.
    blur_fwhm_mm:
        Optional declared post-blur FWHM (mm) used to sharpen the
        partial-volume-risk QC rule (see :class:`QCFlags`). When omitted, a
        voxel-count-only heuristic is used instead.
    registration_shift_mm:
        A caller-declared/known PET/CT registration shift magnitude (mm).
        This engine does not compute a registration residual itself (that
        needs CT alignment data outside this module's scope); the
        resulting ``misregistration_risk_placeholder`` flag is a
        deterministic pass-through (``> 0.0``), not a measured signal.

    Returns
    -------
    QuantificationResult
    """
    suv = _validate_scalar_field(suvbw, geometry)
    mask_bool = _validate_and_binarize_mask(mask, geometry)
    reflection = _validate_reflection_affine(reflection_affine)
    normal, offset = _reflection_plane_normal_and_offset(reflection)
    axis = (
        _validate_centerline_direction(centerline_direction)
        if centerline_direction is not None
        else None
    )

    voxel_count = int(mask_bool.sum())
    component_count, fragmented = _connected_components(mask_bool)
    boundary_touching = _boundary_touching(mask_bool)
    misregistration_risk_placeholder = registration_shift_mm > 0.0

    if voxel_count == 0:
        null = Measurement(None, QuantificationErrorCode.EMPTY_MASK.value)
        qc = QCFlags(
            empty_mask=True,
            invalid_region=True,
            boundary_touching=False,
            fragmented=False,
            component_count=0,
            partial_volume_risk=False,
            misregistration_risk_placeholder=misregistration_risk_placeholder,
        )
        return QuantificationResult(
            target_suvmax=null,
            target_suvmean=null,
            metabolic_volume_ml=null,
            longitudinal_extent_mm=null,
            contralateral_suvmax=null,
            contralateral_suvmean=null,
            asymmetry_index=null,
            target_to_background_ratio=null,
            laterality="none",
            qc=qc,
        )

    affine_arr = np.asarray(geometry.affine, dtype=np.float64)
    voxel_indices = np.argwhere(mask_bool)
    world_points = _apply_affine(affine_arr, voxel_indices)

    target_values = suv[mask_bool]
    target_finite_mask = np.isfinite(target_values)
    if np.all(target_finite_mask):
        target_suvmax = Measurement(float(np.max(target_values)))
        target_suvmean = Measurement(float(np.mean(target_values)))
    else:
        nonfinite_reason = QuantificationErrorCode.NONFINITE_TARGET.value
        target_suvmax = Measurement(None, nonfinite_reason)
        target_suvmean = Measurement(None, nonfinite_reason)
    # A distinct signal from NONFINITE_TARGET above: NONFINITE_TARGET nulls
    # the SUV measurements the instant *any* selected voxel is nonfinite
    # (per the frozen "do not silently drop voxels" contract), but a
    # nonempty region whose values are *entirely* nonfinite is a stronger,
    # independently-meaningful condition -- the whole region is unusable
    # data, not merely contaminated by a stray bad voxel.
    invalid_region = not bool(np.any(target_finite_mask))

    voxel_mm3 = abs(float(np.linalg.det(affine_arr[:3, :3])))
    metabolic_volume_ml = Measurement(voxel_count * voxel_mm3 / 1000.0)

    projection_axis = axis if axis is not None else _principal_axis(world_points)
    projections = world_points @ projection_axis
    longitudinal_extent_mm = Measurement(float(projections.max() - projections.min()))

    signed_distances = world_points @ normal - offset
    laterality = _laterality_from_signed_distances(signed_distances)

    reflected_world = _apply_affine(reflection, world_points)
    inverse_affine = np.linalg.inv(affine_arr)
    reflected_voxel_float = _apply_affine(inverse_affine, reflected_world)
    reflected_voxel_round = np.rint(reflected_voxel_float).astype(np.int64)
    shape_arr = np.array(geometry.shape, dtype=np.int64)
    in_bounds = np.all(
        (reflected_voxel_round >= 0) & (reflected_voxel_round < shape_arr), axis=1
    )
    if not np.all(in_bounds):
        out_of_domain_reason = QuantificationErrorCode.CONTRALATERAL_OUT_OF_DOMAIN.value
        contralateral_suvmax = Measurement(None, out_of_domain_reason)
        contralateral_suvmean = Measurement(None, out_of_domain_reason)
    else:
        unique_voxels = np.unique(reflected_voxel_round, axis=0)
        contralateral_values = suv[
            unique_voxels[:, 0], unique_voxels[:, 1], unique_voxels[:, 2]
        ]
        if np.all(np.isfinite(contralateral_values)):
            contralateral_suvmax = Measurement(float(np.max(contralateral_values)))
            contralateral_suvmean = Measurement(float(np.mean(contralateral_values)))
        else:
            nonfinite_contra_reason = (
                QuantificationErrorCode.NONFINITE_CONTRALATERAL.value
            )
            contralateral_suvmax = Measurement(None, nonfinite_contra_reason)
            contralateral_suvmean = Measurement(None, nonfinite_contra_reason)

    asymmetry_index = _asymmetry_index(target_suvmean, contralateral_suvmean)
    target_to_background_ratio = _target_to_background_ratio(
        target_suvmean, contralateral_suvmean
    )

    qc = QCFlags(
        empty_mask=False,
        invalid_region=invalid_region,
        boundary_touching=boundary_touching,
        fragmented=fragmented,
        component_count=component_count,
        partial_volume_risk=_partial_volume_risk(
            voxel_count, longitudinal_extent_mm, blur_fwhm_mm
        ),
        misregistration_risk_placeholder=misregistration_risk_placeholder,
    )

    return QuantificationResult(
        target_suvmax=target_suvmax,
        target_suvmean=target_suvmean,
        metabolic_volume_ml=metabolic_volume_ml,
        longitudinal_extent_mm=longitudinal_extent_mm,
        contralateral_suvmax=contralateral_suvmax,
        contralateral_suvmean=contralateral_suvmean,
        asymmetry_index=asymmetry_index,
        target_to_background_ratio=target_to_background_ratio,
        laterality=laterality,
        qc=qc,
    )
