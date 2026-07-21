"""Bilateral iliac-corridor crop + reflection-plane pipeline (VascuTrace
Phase 2 data pipeline).

RESEARCH_PROTOTYPE_WARNING
---------------------------------------------------------------------------
Research prototype. Trained and evaluated using simulated vascular-like
abnormalities, not confirmed human post-angioplasty lesions.
---------------------------------------------------------------------------

This module computes, per session, a fixed-size bilateral iliac-corridor
crop and a physical reflection (mirror) affine, and assembles them into a
:class:`src.vascutrace.data.contract.CropBundle`. It is split into two
layers:

1. **Pure geometry helpers** (bounding-box/margin/crop-window math, per
   -slice label-centroid pairing, plane fitting, Householder-affine
   construction) that operate only on plain ``numpy`` arrays/affines --
   no file I/O, no ``nibabel``. These are exercised directly by
   generated-fixture tests that need no ``Data/`` access.
2. **Session-level orchestration** (:func:`compute_session_crop`,
   :func:`run_crop_pipeline`) that loads a session's PET/CT/segmentation
   NIfTI files, canonicalizes them via ``src.vascutrace.geometry``
   (never array-index overlay or ``np.flip``), and calls the pure helpers
   above. Only this layer touches ``Data/`` and is marked ``local_data`` in
   tests.

Crop-anchor and margin policy (frozen; certified in
``docs/p2_data_fitness_2026-07-16.md``, "Interpretation")
==============================================================================
The crop's bounding box is anchored on the **iliac-only** union (MOOSE
labels 7=left/8=right in the ``Cardiac`` group), expanded by a 15 mm
physical margin, then centered into the fixed 144 x 80 x 144 PET-voxel crop
(``src.vascutrace.data.contract.FIXED_CROP_SHAPE``). Femur (labels 5/6 in
``Peripheral-Bones``) is used **only** as an additional source of paired
centerline points for the reflection-plane fit (below) -- never as a crop
-bounding-box anchor. The fitness note found combining iliac+femur into the
bbox anchor roughly quintuples crop voxel volume and nearly doubles the
Z-extent (MOOSE's femur label spans the whole bone to the knee, not a
proximal sub-region) for no additional vascular target (no femoral-artery
label exists in this archive at all) -- so iliac-only is the correct anchor,
consistent with the project scientific contract, which uses MOOSE iliac
masks as crop anchors.

Reflection-plane method -- a resolved discrepancy (flag for reconciliation)
==============================================================================
This module fits the reflection plane by **per-axial-slice** L/R centerline
-point pairing (iliac + femur combined), ``normal = mean(R - L)`` direction
(oriented toward RAS +R), ``point = mean of pair midpoints`` -- exactly the
method this implementation specifies and that the certified fitness note's
Interpretation section explicitly recommends ("fit the plane from paired
per-slice centerline points ... using normal = mean(R-L) direction and
point = mean of pair midpoints").

This is a **deliberate, flagged departure** from the intended scientific
method, which specifies resampling the two
centerlines to **paired normalized arc-length locations**, a **robust mean**
of the L-to-R vectors (not a plain mean), a **median** (not mean) of pair
-midpoint projections for the offset, and **iliac-only** anchors (not
iliac+femur). Both methods share the same normal-orientation convention
(toward RAS +R) and the same Householder-affine construction; they differ
only in *which* points are paired and *how* the normal/offset are
aggregated (mean/per-slice vs. robust-mean-median/arc-length-resampled).

This module follows the implementation's and the fitness note's method because both
are the explicit, dated (2026-07-16), certified spec for *this* deliverable
-- the fitness note's own Interpretation section reasons through this exact
choice (per-slice pairing needs no extra I/O beyond what the crop bbox
already loads; the arc-length-resampled alternative was not implemented or
validated in that pass either). The discrepancy between the two documents is
real and is flagged in this module's docstring and in this implementation's documentation
for whoever owns the shared imaging-physics contract to reconcile -- it is
not silently resolved by this module choosing one silently.
============================================================================
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import nibabel as nib
import numpy as np

from src.vascutrace.data.contract import (
    FIXED_CROP_SHAPE,
    make_crop_bundle,
    save_crop_bundle,
    validate_reflection_affine,
)
from src.vascutrace.data.ingest import DatasetManifest, SessionPaths
from src.vascutrace.geometry import (
    CanonicalVolume,
    GridGeometry,
    canonicalize_nifti_to_ras,
    resample_ct_to_pet,
    resample_mask_to_pet,
    restore_mask_to_original_pet,
)

__all__ = [
    "RESEARCH_PROTOTYPE_WARNING",
    "ILIAC_LABELS",
    "FEMUR_LABELS",
    "DEFAULT_CROP_MARGIN_MM",
    "REFLECTION_RESIDUAL_QC_THRESHOLD_MM",
    "AXIAL_AXIS",
    "ReflectionFitErrorCode",
    "ReflectionFitError",
    "CropErrorCode",
    "CropError",
    "label_union_bbox_voxels",
    "bbox_world_corners",
    "expand_world_bbox",
    "world_bbox_to_voxel_bbox",
    "center_crop_window",
    "extract_fixed_crop",
    "paired_centerline_points",
    "fit_reflection_plane",
    "reflection_plane_residual_mm",
    "build_reflection_affine",
    "CropSessionOutcome",
    "CropPipelineReport",
    "compute_session_crop",
    "run_crop_pipeline",
]

RESEARCH_PROTOTYPE_WARNING = (
    "Research prototype. Trained and evaluated using simulated vascular-like "
    "abnormalities, not confirmed human post-angioplasty lesions."
)

ILIAC_LABELS: tuple[int, int] = (7, 8)  # left, right (Cardiac group)
FEMUR_LABELS: tuple[int, int] = (5, 6)  # left, right (Peripheral-Bones group)
DEFAULT_CROP_MARGIN_MM = 15.0
REFLECTION_RESIDUAL_QC_THRESHOLD_MM = 8.4  # cohort 95th percentile, fitness note Q5/Q6
AXIAL_AXIS = 2  # S/I axis; matches contract.AXIAL_ADJACENT_SLICE_AXIS (verified in crops.py tests)


class ReflectionFitErrorCode(StrEnum):
    NO_PAIRED_POINTS = "NO_PAIRED_POINTS"
    DEGENERATE_NORMAL = "DEGENERATE_NORMAL"


class ReflectionFitError(ValueError):
    def __init__(self, code: ReflectionFitErrorCode) -> None:
        super().__init__(code.value)
        self.code = code


class CropErrorCode(StrEnum):
    ILIAC_LABELS_NOT_FOUND = "ILIAC_LABELS_NOT_FOUND"


class CropError(ValueError):
    def __init__(self, code: CropErrorCode) -> None:
        super().__init__(code.value)
        self.code = code


# ---------------------------------------------------------------------------
# Pure geometry helpers (fixture-testable, no I/O)
# ---------------------------------------------------------------------------


def _apply_affine(affine: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    homogeneous = np.concatenate([pts, ones], axis=1)
    return (np.asarray(affine, dtype=np.float64) @ homogeneous.T).T[:, :3]


def label_union_bbox_voxels(
    mask: np.ndarray, labels: Sequence[int]
) -> tuple[np.ndarray, np.ndarray] | None:
    """Inclusive integer voxel-index bounding box of the union of ``labels``
    in ``mask``, as ``(min_idx, max_idx)``. Returns ``None`` if no voxel in
    ``mask`` carries any of ``labels``.
    """
    selector = np.isin(mask, np.asarray(labels))
    if not np.any(selector):
        return None
    indices = np.argwhere(selector)
    return indices.min(axis=0), indices.max(axis=0)


def bbox_world_corners(
    min_idx: np.ndarray, max_idx: np.ndarray, affine: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """World-space (RAS+ mm) bounding box spanned by the 8 voxel-index
    corners of an inclusive integer bbox, mapped through ``affine``.
    """
    corners_idx = np.array(
        [
            [i, j, k]
            for i in (min_idx[0], max_idx[0])
            for j in (min_idx[1], max_idx[1])
            for k in (min_idx[2], max_idx[2])
        ],
        dtype=np.float64,
    )
    world = _apply_affine(affine, corners_idx)
    return world.min(axis=0), world.max(axis=0)


def expand_world_bbox(
    min_world: np.ndarray, max_world: np.ndarray, margin_mm: float
) -> tuple[np.ndarray, np.ndarray]:
    margin = np.full(3, float(margin_mm))
    return np.asarray(min_world) - margin, np.asarray(max_world) + margin


def world_bbox_to_voxel_bbox(
    min_world: np.ndarray, max_world: np.ndarray, affine: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a physical (RAS+ mm) bbox to an inclusive integer voxel-index
    bbox in the grid described by ``affine``, guaranteeing full containment
    (floor the min corner, ceil the max corner across all 8 world corners --
    correct even under an oblique/rotated affine, though this project's
    grids are axis-aligned).
    """
    inverse = np.linalg.inv(np.asarray(affine, dtype=np.float64))
    corners_world = np.array(
        [
            [x, y, z]
            for x in (min_world[0], max_world[0])
            for y in (min_world[1], max_world[1])
            for z in (min_world[2], max_world[2])
        ],
        dtype=np.float64,
    )
    voxel = _apply_affine(inverse, corners_world)
    min_idx = np.floor(voxel.min(axis=0)).astype(np.int64)
    max_idx = np.ceil(voxel.max(axis=0)).astype(np.int64)
    return min_idx, max_idx


def center_crop_window(
    min_idx: np.ndarray, max_idx: np.ndarray, fixed_shape: Sequence[int]
) -> tuple[np.ndarray, tuple[bool, bool, bool]]:
    """Integer crop-start index (in the target grid) centering
    ``fixed_shape`` on the ``[min_idx, max_idx]`` bbox, plus a per-axis flag
    for whether the (padded) bbox itself exceeds ``fixed_shape`` on that
    axis (a genuine corridor-clipping event, distinct from FOV clipping --
    see :func:`extract_fixed_crop`).
    """
    fixed = np.asarray(fixed_shape, dtype=np.int64)
    min_idx = np.asarray(min_idx, dtype=np.int64)
    max_idx = np.asarray(max_idx, dtype=np.int64)
    bbox_extent = max_idx - min_idx + 1
    exceeds = tuple(bool(e) for e in (bbox_extent > fixed))
    center = (min_idx.astype(np.float64) + max_idx.astype(np.float64)) / 2.0
    crop_start = np.round(center - fixed / 2.0).astype(np.int64)
    return crop_start, exceeds


def extract_fixed_crop(
    volume: np.ndarray, crop_start: np.ndarray, fixed_shape: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a ``fixed_shape`` window starting at ``crop_start`` from
    ``volume``, zero-padding (and marking ``valid=0``) any part of the
    window that falls outside ``volume``'s own bounds. Returns
    ``(cropped_array, valid_mask)`` where ``valid_mask`` is ``uint8``,
    ``1`` inside the source field of view.
    """
    fixed = tuple(int(s) for s in fixed_shape)
    crop_start = np.asarray(crop_start, dtype=np.int64)
    out = np.zeros(fixed, dtype=volume.dtype)
    valid = np.zeros(fixed, dtype=np.uint8)

    volume_shape = np.asarray(volume.shape, dtype=np.int64)
    src_start = np.maximum(crop_start, 0)
    src_end = np.minimum(crop_start + np.asarray(fixed, dtype=np.int64), volume_shape)
    if np.any(src_end <= src_start):
        return out, valid  # crop window entirely outside the volume

    dst_start = src_start - crop_start
    dst_end = dst_start + (src_end - src_start)

    src_slices = tuple(
        slice(int(a), int(b)) for a, b in zip(src_start, src_end, strict=True)
    )
    dst_slices = tuple(
        slice(int(a), int(b)) for a, b in zip(dst_start, dst_end, strict=True)
    )

    out[dst_slices] = volume[src_slices]
    valid[dst_slices] = 1
    return out, valid


def _apply_affine_point(affine: np.ndarray, point_voxel: np.ndarray) -> np.ndarray:
    return _apply_affine(affine, point_voxel.reshape(1, 3))[0]


def paired_centerline_points(
    mask: np.ndarray,
    left_label: int,
    right_label: int,
    affine: np.ndarray,
    *,
    axial_axis: int = AXIAL_AXIS,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per-slice (along ``axial_axis``) physical centroid pairing for one
    left/right label pair. A slice index contributes a pair only when both
    labels are present in it. Returns a list of
    ``(left_world_point, right_world_point)`` physical (mm) pairs. See
    module docstring, "Reflection-plane method", for the pairing policy
    this implements.
    """
    affine_arr = np.asarray(affine, dtype=np.float64)
    left_idx = np.argwhere(mask == left_label)
    right_idx = np.argwhere(mask == right_label)
    if left_idx.size == 0 or right_idx.size == 0:
        return []

    left_slices = left_idx[:, axial_axis]
    right_slices = right_idx[:, axial_axis]
    common = np.intersect1d(left_slices, right_slices)

    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for slice_index in common:
        left_centroid_voxel = left_idx[left_slices == slice_index].mean(axis=0)
        right_centroid_voxel = right_idx[right_slices == slice_index].mean(axis=0)
        pairs.append(
            (
                _apply_affine_point(affine_arr, left_centroid_voxel),
                _apply_affine_point(affine_arr, right_centroid_voxel),
            )
        )
    return pairs


def fit_reflection_plane(
    paired_points: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the subject mid-sagittal reflection plane:
    ``normal = mean(R - L)`` (unit-normalized, oriented toward RAS +R),
    ``point = mean of pair midpoints``. See module docstring, "Reflection
    -plane method".
    """
    if not paired_points:
        raise ReflectionFitError(ReflectionFitErrorCode.NO_PAIRED_POINTS)
    lefts = np.array([p[0] for p in paired_points], dtype=np.float64)
    rights = np.array([p[1] for p in paired_points], dtype=np.float64)

    mean_diff = (rights - lefts).mean(axis=0)
    norm = float(np.linalg.norm(mean_diff))
    if norm < 1e-9:
        raise ReflectionFitError(ReflectionFitErrorCode.DEGENERATE_NORMAL)
    normal = mean_diff / norm
    if normal[0] < 0.0:  # orient toward RAS +R, matching measure.py's convention
        normal = -normal

    point = ((lefts + rights) / 2.0).mean(axis=0)
    return normal, point


def reflection_plane_residual_mm(
    paired_points: Sequence[tuple[np.ndarray, np.ndarray]],
    normal: np.ndarray,
    point: np.ndarray,
) -> float:
    """Mean absolute left/right asymmetry residual after the best-fit plane
    (fitness note Q5): for each pair, ``signed_L + signed_R`` (zero for a
    perfect mirror), averaged in absolute value over all pairs.
    """
    if not paired_points:
        raise ReflectionFitError(ReflectionFitErrorCode.NO_PAIRED_POINTS)
    lefts = np.array([p[0] for p in paired_points], dtype=np.float64)
    rights = np.array([p[1] for p in paired_points], dtype=np.float64)
    signed_left = (lefts - point) @ normal
    signed_right = (rights - point) @ normal
    residual_per_pair = np.abs(signed_left + signed_right)
    return float(residual_per_pair.mean())


def build_reflection_affine(normal: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Householder mirror affine reflecting through the plane
    ``{y : dot(n, y - p0) = 0}``: ``x' = x - 2*(dot(n, x - p0))*n``. Linear
    part ``I - 2*outer(n, n)`` is symmetric, involutive, and
    orientation-reversing (``det == -1``) by construction -- see the
    algebraic derivation in this module's implementation documentation and the runtime
    re-verification in :func:`src.vascutrace.data.contract.
    validate_reflection_affine`, which every caller of this function must
    run before trusting the result.
    """
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    p0 = np.asarray(point, dtype=np.float64)
    linear = np.eye(3) - 2.0 * np.outer(n, n)
    translation = 2.0 * float(np.dot(n, p0)) * n
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = linear
    affine[:3, 3] = translation
    return affine


# ---------------------------------------------------------------------------
# Session-level orchestration (touches Data/, local_data only)
# ---------------------------------------------------------------------------


def _grid_geometry_world_bounds(
    affine: np.ndarray, shape: tuple[int, int, int]
) -> tuple[np.ndarray, np.ndarray]:
    corner_indices = np.array(
        [
            [i, j, k]
            for i in (0, shape[0] - 1)
            for j in (0, shape[1] - 1)
            for k in (0, shape[2] - 1)
        ],
        dtype=np.float64,
    )
    world = _apply_affine(affine, corner_indices)
    return world.min(axis=0), world.max(axis=0)


def _translated_grid_geometry(
    base: GridGeometry, crop_start: np.ndarray, fixed_shape: tuple[int, int, int]
) -> GridGeometry:
    """A :class:`GridGeometry` for the fixed-size crop window: same
    rotation/spacing as ``base`` (a plain axis-aligned translation of the
    grid origin to ``crop_start``), shaped ``fixed_shape``. Used as the
    output grid handed to ``resample_ct_to_pet`` (P1) so CT resampling
    happens directly onto the small crop grid rather than the full PET
    volume.
    """
    base_affine = np.asarray(base.affine, dtype=np.float64)
    origin_world = base_affine @ np.array([*crop_start, 1.0])
    new_affine = base_affine.copy()
    new_affine[:3, 3] = origin_world[:3]
    world_min, world_max = _grid_geometry_world_bounds(new_affine, fixed_shape)
    return GridGeometry(
        shape=(int(fixed_shape[0]), int(fixed_shape[1]), int(fixed_shape[2])),
        affine=new_affine,
        spacing=base.spacing,
        units=base.units,
        world_bounds_min=world_min,
        world_bounds_max=world_max,
    )


def _session_geometry_sidecar(pet_canonical: CanonicalVolume):
    """Reuse geometry.py's own sidecar machinery
    (``restore_mask_to_original_pet``) to obtain a real ``GeometrySidecar``
    fingerprint for this session's PET canonicalization. The sidecar
    depends only on ``pet_canonical``'s shapes and affines (see
    ``geometry._sidecar_for``), never on mask *content*, so a cheap all-zero
    dummy mask of the canonical shape is sufficient and correct -- not a
    workaround.
    """
    dummy = np.zeros(pet_canonical.geometry.shape, dtype=np.uint8)
    _restored, sidecar = restore_mask_to_original_pet(dummy, pet_canonical)
    return sidecar


def compute_session_crop(
    session: SessionPaths,
    *,
    fixed_shape: tuple[int, int, int] = FIXED_CROP_SHAPE,
    margin_mm: float = DEFAULT_CROP_MARGIN_MM,
):
    """Compute one session's base :class:`~src.vascutrace.data.contract.
    CropBundle`. Loads PET/CT/Cardiac/Peripheral-Bones NIfTI files,
    canonicalizes each to RAS+ (P1), computes the iliac-anchored crop
    window and the reflection plane, resamples CT (and the iliac label
    mask, labels 7/8 only) onto the crop's own PET-aligned grid, and
    returns a fully hashed, validated :class:`CropBundle`. Raises
    :class:`CropError` if no iliac voxels are found, or
    :class:`ReflectionFitError` if the reflection plane cannot be fit (see
    module docstring).
    """
    pet_canonical = canonicalize_nifti_to_ras(nib.load(session.pet))
    ct_canonical = canonicalize_nifti_to_ras(nib.load(session.ct))
    cardiac_canonical = canonicalize_nifti_to_ras(
        nib.load(session.segmentation("Cardiac"))
    )
    peripheral_canonical = canonicalize_nifti_to_ras(
        nib.load(session.segmentation("Peripheral-Bones"))
    )

    cardiac_data = np.asarray(cardiac_canonical.data)
    peripheral_data = np.asarray(peripheral_canonical.data)

    bbox = label_union_bbox_voxels(cardiac_data, ILIAC_LABELS)
    if bbox is None:
        raise CropError(CropErrorCode.ILIAC_LABELS_NOT_FOUND)
    min_idx, max_idx = bbox

    cardiac_affine = np.asarray(cardiac_canonical.geometry.affine, dtype=np.float64)
    min_world, max_world = bbox_world_corners(min_idx, max_idx, cardiac_affine)
    min_world, max_world = expand_world_bbox(min_world, max_world, margin_mm)

    pet_affine = np.asarray(pet_canonical.geometry.affine, dtype=np.float64)
    pmin_idx, pmax_idx = world_bbox_to_voxel_bbox(min_world, max_world, pet_affine)
    crop_start, exceeds = center_crop_window(pmin_idx, pmax_idx, fixed_shape)

    pet_data = np.asarray(pet_canonical.data, dtype=np.float32)
    pet_crop, valid_mask = extract_fixed_crop(pet_data, crop_start, fixed_shape)

    crop_geometry = _translated_grid_geometry(
        pet_canonical.geometry, crop_start, fixed_shape
    )
    ct_crop = np.asarray(
        resample_ct_to_pet(ct_canonical, crop_geometry), dtype=np.float32
    )

    # Iliac label mask: resample the (native-grid) Cardiac segmentation onto
    # the exact same crop grid as ct_crop (same crop_geometry -> identical
    # crop origin/shape), via P1's resample_mask_to_pet (nearest-neighbor,
    # never array-index overlay). The Cardiac MOOSE group can carry other
    # structures (heart, aorta, pulmonary artery) beyond the iliac arteries
    # -- zero out everything except the two iliac label values so only
    # {0, ILIAC_LABEL_LEFT, ILIAC_LABEL_RIGHT} ever appear, matching
    # contract.py's stored-label contract.
    cardiac_crop_full = np.asarray(
        resample_mask_to_pet(cardiac_canonical, crop_geometry), dtype=np.uint8
    )
    iliac_label_mask = np.where(
        np.isin(cardiac_crop_full, ILIAC_LABELS), cardiac_crop_full, 0
    ).astype(np.uint8)

    iliac_pairs = paired_centerline_points(
        cardiac_data, ILIAC_LABELS[0], ILIAC_LABELS[1], cardiac_affine
    )
    peripheral_affine = np.asarray(
        peripheral_canonical.geometry.affine, dtype=np.float64
    )
    femur_pairs = paired_centerline_points(
        peripheral_data, FEMUR_LABELS[0], FEMUR_LABELS[1], peripheral_affine
    )
    all_pairs = iliac_pairs + femur_pairs
    normal, point = fit_reflection_plane(all_pairs)
    residual_mm = reflection_plane_residual_mm(all_pairs, normal, point)
    reflection_affine = build_reflection_affine(normal, point)
    validate_reflection_affine(reflection_affine)  # defense-in-depth self-check

    geometry_sidecar = _session_geometry_sidecar(pet_canonical)

    return make_crop_bundle(
        subject=session.subject,
        session=session.session,
        pet_suvbw=pet_crop,
        ct_hu=ct_crop,
        valid_pet_mask=valid_mask,
        iliac_label_mask=iliac_label_mask,
        reflection_affine=reflection_affine,
        crop_to_pet_canonical_affine=np.asarray(crop_geometry.affine, dtype=np.float64),
        crop_origin_voxel=tuple(int(v) for v in crop_start),
        original_voxel_from_pet_canonical_voxel=np.asarray(
            pet_canonical.original_voxel_from_canonical_voxel, dtype=np.float64
        ),
        geometry_sidecar=geometry_sidecar,
        reflection_residual_mm=residual_mm,
        reflection_qc_flag=residual_mm > REFLECTION_RESIDUAL_QC_THRESHOLD_MM,
        bbox_exceeds_fixed_crop=exceeds,
        crop_margin_mm=margin_mm,
        pet_spacing_mm=pet_canonical.geometry.spacing,
        paired_point_count=len(all_pairs),
    )


@dataclass(frozen=True, slots=True)
class CropSessionOutcome:
    subject: str
    session: str
    bundle_path: Path
    valid_pet_mask_coverage: float
    reflection_residual_mm: float
    reflection_qc_flag: bool
    bbox_exceeds_fixed_crop: tuple[bool, bool, bool]


@dataclass(frozen=True, slots=True)
class CropPipelineReport:
    outcomes: tuple[CropSessionOutcome, ...]

    @property
    def qc_flagged_count(self) -> int:
        return sum(1 for o in self.outcomes if o.reflection_qc_flag)

    @property
    def bbox_exceeded_count(self) -> int:
        return sum(1 for o in self.outcomes if any(o.bbox_exceeds_fixed_crop))

    @property
    def mean_valid_coverage(self) -> float:
        if not self.outcomes:
            return float("nan")
        return float(np.mean([o.valid_pet_mask_coverage for o in self.outcomes]))

    @property
    def min_valid_coverage(self) -> float:
        if not self.outcomes:
            return float("nan")
        return float(np.min([o.valid_pet_mask_coverage for o in self.outcomes]))


def run_crop_pipeline(
    manifest: DatasetManifest,
    output_root: Path,
    *,
    sessions: Sequence[SessionPaths] | None = None,
    margin_mm: float = DEFAULT_CROP_MARGIN_MM,
) -> CropPipelineReport:
    """Build and save a crop bundle for each of ``sessions`` (or every
    session in ``manifest`` if omitted), returning an aggregate
    :class:`CropPipelineReport`.
    """
    target_sessions = sessions if sessions is not None else manifest.sessions
    outcomes: list[CropSessionOutcome] = []
    for session in target_sessions:
        bundle = compute_session_crop(session, margin_mm=margin_mm)
        directory = save_crop_bundle(bundle, output_root)
        coverage = float(np.mean(bundle.valid_pet_mask))
        outcomes.append(
            CropSessionOutcome(
                subject=session.subject,
                session=session.session,
                bundle_path=directory,
                valid_pet_mask_coverage=coverage,
                reflection_residual_mm=bundle.reflection_residual_mm,
                reflection_qc_flag=bundle.reflection_qc_flag,
                bbox_exceeds_fixed_crop=bundle.bbox_exceeds_fixed_crop,
            )
        )
    return CropPipelineReport(tuple(outcomes))
