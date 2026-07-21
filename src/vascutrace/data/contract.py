"""Versioned bilateral-corridor crop/tensor schema (VascuTrace Phase 2).

RESEARCH_PROTOTYPE_WARNING
---------------------------------------------------------------------------
Research prototype. Trained and evaluated using simulated vascular-like
abnormalities, not confirmed human post-angioplasty lesions.
---------------------------------------------------------------------------

This module owns the on-disk **base-crop bundle** contract that
``src.vascutrace.data.crops`` produces and that a later P6 dataset builder
consumes. It is deliberately narrow:

1. :class:`CropBundle` -- the frozen base-crop schema (raw PET/CT crops, a
   valid-FOV mask, the cropped iliac label mask, a physical reflection
   affine, and the transforms/hashes needed to trace a crop back to its
   native PET grid). :func:`make_crop_bundle` validates and hashes a
   bundle; :func:`save_crop_bundle`/:func:`load_crop_bundle` are the
   deterministic, integrity-checked read/write pair.
2. ``NETWORK_TENSOR_FIELDS`` -- documentation only (see "Downstream network
   tensor contract" below) of the tensor fields a *separate*, not-yet-built
   dataset builder assembles from one or more base-crop bundles for P6. This
   module does not compute clipped/normalized network views or insert a
   synthetic target; it only freezes the *shape* of that downstream contract
   so P6 can be designed against a stable interface today.
3. :func:`reflect_volume` -- the one piece of downstream-view logic frozen
   here rather than left to the dataset builder, because it is pure physical
   -space geometry (not normalization policy): given a base crop and its
   ``reflection_affine``, produce the physically mirrored ("contralateral")
   view of that crop. **Never** ``np.flip`` -- see the function docstring.

Frozen spatial contract
==============================================================================
Every base-crop array is defined on a **fixed 144 x 80 x 144 PET-voxel
grid**, axes ``(X, Y, Z)`` = (R/L, A/P, S/I) in the session's own
RAS+-canonicalized PET frame (see ``src.vascutrace.geometry``). This size
comes from the certified data-fitness note
(``docs/p2_data_fitness_2026-07-16.md``, "Interpretation"): the observed
cohort maximum iliac-only union bbox at a 15 mm physical margin is
129 x 70 x 131 PET voxels; rounding each axis up to the next multiple of 8
gives a zero-headroom minimum of 136 x 72 x 136, and one further multiple-of
-8 step gives **144 x 80 x 144** (~238 x 132 x 288 mm) as a headroom option
for a subject outside this (48-subject, single-scanner, healthy) cohort.
Verified empirically in this pass: RAS+-canonicalizing this cohort's PET
grids preserves axis order (LAS -> RAS+ is a pure axis-0 sign flip for this
scanner/protocol), so axis 2 (S/I, 2.0 mm spacing) is unambiguously the
"axial" direction referred to below.

Axial (adjacent-slice) axis and K
==============================================================================
``AXIAL_ADJACENT_SLICE_AXIS = 2`` -- the S/I (superior/inferior) axis of the
fixed crop, at the PET's native 2.0 mm spacing. This is the axis a 2.5D
network stacks neighboring slices along: for a predicted axial slice at
index ``z``, the network input is the ``K``-slice slab
``[z - K//2, ..., z, ..., z + K//2]`` (pseudo-channel axis in
``[batch, adjacent_slice, H, W]``), and the target is the single center
slice (``[batch, 1, H, W]``), as required by the frozen P6 tensor schema. This is also
the same axis :func:`src.vascutrace.data.crops.paired_centerline_points`
pairs left/right label centroids *per slice* along when fitting the
reflection plane -- one axial-slice concept used consistently in both
places.

``ADJACENT_SLICE_COUNT_K = 5`` (frozen here, in the implementation's allowed [3, 7]
range, required odd for a well-defined center slice):
- K=5 spans 4 x 2.0 mm = 8 mm of physical S/I context around the predicted
  slice (+/-4 mm) -- comparable to a common/external iliac artery's
  cross-sectional diameter (roughly 8-12 mm), giving the network enough
  adjacent-slice context to resolve a genuinely tubular 3D structure from a
  2.5D input.
- K=3 (+/-2 mm) risks under-using axial context for that same tubular
  structure; K=7 (+/-6 mm) pulls in more compute (40% more channels than
  K=5) for context that, given corridor tortuosity, may extend past the
  locally relevant vessel segment.
- This is a design judgment call, not a value derived from the scanner's
  reconstruction point-spread function (which this project has not measured
  independently -- see the fitness note's open question #1 on the 15 mm
  crop margin, which has the same caveat). Flagged for confirmation by
  whoever owns PET reconstruction-physics judgment calls.

Downstream network tensor contract (documentation only)
==============================================================================
``NETWORK_TENSOR_FIELDS`` documents, but does not implement, the P6 input
contract a downstream dataset builder assembles per training example from
one base-crop bundle:
``pet_left_suvbw`` (the as-cropped, network-normalized PET view: clip
``[0, 10]``, divide by 10), ``pet_right_suvbw`` (the same crop physically
reflected through ``reflection_affine`` via :func:`reflect_volume`, then
normalized identically), ``pet_diff`` (``pet_left_suvbw - pet_right_suvbw``),
``ct_left_hu``/``ct_right_hu`` (network-normalized CT: clip
``[-1000, 1000]``, divide by 1000, as-cropped and reflected respectively),
``valid_pet_mask`` (carried through from the base bundle unchanged), and
``target_mask`` (the P3 simulator's synthetic-lesion supervision mask for
the center slice -- absent from every base-crop bundle this module
produces; base crops are healthy corridors only). Raw (non-normalized)
``pet_suvbw``/``ct_hu`` are retained separately in the base bundle for
quantification (P4), as required by the project scientific contract. The base bundle's own
``iliac_label_mask`` field (below) is what lets that downstream builder
place a synthetic lesion *on the real vessel* rather than at an arbitrary
crop-space location.

Schema versions
==============================================================================
- ``p2-crop-v1`` (superseded): raw PET/CT crops, ``valid_pet_mask``,
  ``reflection_affine``, crop->original transforms, hashes. No vessel
  -location field -- a P6 dataset builder had no way to place a synthetic
  lesion on the actual iliac vessel rather than an arbitrary crop-space
  location.
- ``p2-crop-v2`` (current): **additive** -- adds ``iliac_label_mask``
  (below). Every other field, and the crop geometry/reflection/split that
  produced them, is unchanged from v1. :func:`load_crop_bundle` only
  accepts the current ``CROP_SCHEMA_VERSION``; a v1 bundle on disk must be
  regenerated, not silently upgraded (the loader fails closed with
  ``SCHEMA_VERSION_MISMATCH`` rather than guessing a migration).

``iliac_label_mask`` field
==============================================================================
``uint8``, shape :data:`FIXED_CROP_SHAPE`, on the exact same crop grid as
``pet_suvbw``/``ct_hu`` (same ``crop_to_pet_canonical_affine``, same
``crop_origin_voxel``): the session's MOOSE iliac labels (``ILIAC_LABEL_LEFT
= 7``, ``ILIAC_LABEL_RIGHT = 8``), resampled from their native segmentation
grid onto the crop's grid via
:func:`src.vascutrace.geometry.resample_mask_to_pet` (nearest-neighbor --
never array-index overlay), with every voxel not carrying one of those two
label values zeroed out (the source ``Cardiac`` MOOSE group can carry other
structures -- heart, aorta, pulmonary artery -- that are not part of this
project's iliac corridor and must not leak into this field).
:func:`make_crop_bundle`/:func:`load_crop_bundle` both enforce that only
``{0, 7, 8}`` appear. This is the field a P6 dataset builder reads to place
a synthetic lesion on the real vessel centerline instead of an arbitrary
crop-space location.
============================================================================
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import map_coordinates

from src.vascutrace.geometry import GeometrySidecar

__all__ = [
    "RESEARCH_PROTOTYPE_WARNING",
    "CROP_SCHEMA_VERSION",
    "FIXED_CROP_SHAPE",
    "AXIAL_ADJACENT_SLICE_AXIS",
    "ADJACENT_SLICE_COUNT_K",
    "ILIAC_LABEL_LEFT",
    "ILIAC_LABEL_RIGHT",
    "TensorFieldSpec",
    "NETWORK_TENSOR_FIELDS",
    "CropBundle",
    "CropIntegrityErrorCode",
    "CropIntegrityError",
    "compute_bundle_hashes",
    "make_crop_bundle",
    "save_crop_bundle",
    "load_crop_bundle",
    "bundle_directory",
    "validate_reflection_affine",
    "reflect_volume",
]

RESEARCH_PROTOTYPE_WARNING = (
    "Research prototype. Trained and evaluated using simulated vascular-like "
    "abnormalities, not confirmed human post-angioplasty lesions."
)

# ---------------------------------------------------------------------------
# Frozen schema constants
# ---------------------------------------------------------------------------

CROP_SCHEMA_VERSION = "p2-crop-v2"
FIXED_CROP_SHAPE: tuple[int, int, int] = (
    144,
    80,
    144,
)  # (X=R/L, Y=A/P, Z=S/I) PET voxels
AXIAL_ADJACENT_SLICE_AXIS = 2  # Z (S/I), 2.0 mm native PET spacing
ADJACENT_SLICE_COUNT_K = 5  # frozen; see module docstring for justification

# iliac_label_mask label semantics (MOOSE Cardiac group; see fitness note
# Q2/Q4). Any other value appearing in a stored iliac_label_mask is a
# contract violation (see _VALID_ILIAC_LABEL_MASK_VALUES below).
ILIAC_LABEL_LEFT = 7
ILIAC_LABEL_RIGHT = 8
_VALID_ILIAC_LABEL_MASK_VALUES = frozenset({0, ILIAC_LABEL_LEFT, ILIAC_LABEL_RIGHT})

# Tolerances for validating a reflection affine is a genuine physical mirror
# (Householder reflection): linear part symmetric and its own inverse (an
# involution), orientation-reversing. Matches
# ``src.vascutrace.quantification.measure._validate_reflection_affine``
# exactly so a bundle produced by this pipeline is guaranteed consumable by
# that (downstream, P4) validator without surprises.
_REFLECTION_SYMMETRY_ATOL = 1e-6
_REFLECTION_INVOLUTION_ATOL = 1e-6


@dataclass(frozen=True, slots=True)
class TensorFieldSpec:
    """Documentation-only description of one downstream network tensor
    field. See module docstring, "Downstream network tensor contract".
    """

    dtype: str
    shape: str
    description: str


NETWORK_TENSOR_FIELDS: dict[str, TensorFieldSpec] = {
    "pet_left_suvbw": TensorFieldSpec(
        dtype="float32",
        shape="(K, H, W)",
        description=(
            "Network-normalized PET SUVbw slab (clip [0, 10], divide by 10), "
            "K adjacent axial slices centered on the predicted slice, "
            "as-cropped (unreflected) view."
        ),
    ),
    "pet_right_suvbw": TensorFieldSpec(
        dtype="float32",
        shape="(K, H, W)",
        description=(
            "Same slab, physically reflected through the session's "
            "reflection_affine via reflect_volume (never np.flip), then "
            "normalized identically to pet_left_suvbw -- the contralateral "
            "comparison view."
        ),
    ),
    "pet_diff": TensorFieldSpec(
        dtype="float32",
        shape="(K, H, W)",
        description="pet_left_suvbw - pet_right_suvbw; highlights bilateral asymmetry.",
    ),
    "ct_left_hu": TensorFieldSpec(
        dtype="float32",
        shape="(K, H, W)",
        description=(
            "Network-normalized CT HU slab (clip [-1000, 1000], divide by "
            "1000), as-cropped view."
        ),
    ),
    "ct_right_hu": TensorFieldSpec(
        dtype="float32",
        shape="(K, H, W)",
        description="Same slab, physically reflected (as pet_right_suvbw).",
    ),
    "valid_pet_mask": TensorFieldSpec(
        dtype="uint8",
        shape="(K, H, W)",
        description=(
            "1 = voxel was inside the original PET field of view; carried "
            "through unchanged from the base CropBundle."
        ),
    ),
    "target_mask": TensorFieldSpec(
        dtype="uint8",
        shape="(1, H, W)",
        description=(
            "Synthetic-lesion supervision mask for the center slice, "
            "inserted by the P3 simulator's downstream dataset builder. "
            "Absent from every base CropBundle this module produces -- base "
            "crops are healthy corridors only."
        ),
    ),
}


# ---------------------------------------------------------------------------
# Error taxonomy
# ---------------------------------------------------------------------------


class CropIntegrityErrorCode(StrEnum):
    """Typed, code-only failure reasons for the crop-bundle contract."""

    SCHEMA_VERSION_MISMATCH = "SCHEMA_VERSION_MISMATCH"
    SHAPE_MISMATCH = "SHAPE_MISMATCH"
    HASH_MISMATCH = "HASH_MISMATCH"
    MISSING_BUNDLE_FILE = "MISSING_BUNDLE_FILE"
    INVALID_REFLECTION_AFFINE = "INVALID_REFLECTION_AFFINE"
    INVALID_ILIAC_LABEL_MASK = "INVALID_ILIAC_LABEL_MASK"


class CropIntegrityError(ValueError):
    """Raised for any crop-bundle contract violation.

    The exception message is exactly the failing
    :class:`CropIntegrityErrorCode` value -- never a path, array value, or
    subject/session identifier -- matching the geometry contract's (P1)
    fail-closed, code-only error style.
    """

    def __init__(self, code: CropIntegrityErrorCode) -> None:
        super().__init__(code.value)
        self.code = code


# ---------------------------------------------------------------------------
# The base-crop bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class CropBundle:
    """One session's frozen base-crop bundle.

    Raw (non-normalized) arrays only -- no ``[0, 10]``/``[-1000, 1000]``
    network clipping here (see module docstring). ``eq``/``__hash__`` are
    left at Python's default identity-based behavior (not dataclass
    -generated), matching ``src.vascutrace.geometry.GridGeometry``, because
    tuple-wise ``==`` across ndarray fields raises ``ValueError`` rather than
    doing anything useful.
    """

    schema_version: str
    subject: str
    session: str

    pet_suvbw: np.ndarray = field(repr=False)  # float32, FIXED_CROP_SHAPE
    ct_hu: np.ndarray = field(repr=False)  # float32, FIXED_CROP_SHAPE
    valid_pet_mask: np.ndarray = field(repr=False)  # uint8, FIXED_CROP_SHAPE
    iliac_label_mask: np.ndarray = field(
        repr=False
    )  # uint8, FIXED_CROP_SHAPE, values {0,7,8}

    reflection_affine: np.ndarray = field(repr=False)  # float64 (4, 4)

    # Crop -> original-PET tracing.
    crop_to_pet_canonical_affine: np.ndarray = field(repr=False)  # float64 (4, 4)
    crop_origin_voxel: tuple[int, int, int]
    original_voxel_from_pet_canonical_voxel: np.ndarray = field(
        repr=False
    )  # float64 (4, 4)
    geometry_sidecar: GeometrySidecar

    # QC / provenance.
    reflection_residual_mm: float
    reflection_qc_flag: bool
    bbox_exceeds_fixed_crop: tuple[bool, bool, bool]
    crop_margin_mm: float
    pet_spacing_mm: tuple[float, float, float]
    paired_point_count: int

    hashes: dict[str, str]


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def compute_bundle_hashes(bundle: CropBundle) -> dict[str, str]:
    """SHA-256 hashes of every array field plus a canonical-JSON hash of the
    scalar/tuple params, used both to freeze a bundle at save time and to
    re-verify it at load time (see :func:`load_crop_bundle`).
    """
    params_payload = json.dumps(
        {
            "schema_version": bundle.schema_version,
            "fixed_shape": list(FIXED_CROP_SHAPE),
            "crop_origin_voxel": list(bundle.crop_origin_voxel),
            "crop_margin_mm": bundle.crop_margin_mm,
            "pet_spacing_mm": list(bundle.pet_spacing_mm),
        },
        sort_keys=True,
    ).encode("utf-8")
    return {
        "pet_suvbw": _array_sha256(bundle.pet_suvbw),
        "ct_hu": _array_sha256(bundle.ct_hu),
        "valid_pet_mask": _array_sha256(bundle.valid_pet_mask),
        "iliac_label_mask": _array_sha256(bundle.iliac_label_mask),
        "reflection_affine": _array_sha256(bundle.reflection_affine),
        "crop_to_pet_canonical_affine": _array_sha256(
            bundle.crop_to_pet_canonical_affine
        ),
        "original_voxel_from_pet_canonical_voxel": _array_sha256(
            bundle.original_voxel_from_pet_canonical_voxel
        ),
        "params": hashlib.sha256(params_payload).hexdigest(),
    }


def make_crop_bundle(
    *,
    subject: str,
    session: str,
    pet_suvbw: np.ndarray,
    ct_hu: np.ndarray,
    valid_pet_mask: np.ndarray,
    iliac_label_mask: np.ndarray,
    reflection_affine: np.ndarray,
    crop_to_pet_canonical_affine: np.ndarray,
    crop_origin_voxel: tuple[int, int, int],
    original_voxel_from_pet_canonical_voxel: np.ndarray,
    geometry_sidecar: GeometrySidecar,
    reflection_residual_mm: float,
    reflection_qc_flag: bool,
    bbox_exceeds_fixed_crop: tuple[bool, bool, bool],
    crop_margin_mm: float,
    pet_spacing_mm: tuple[float, float, float],
    paired_point_count: int,
) -> CropBundle:
    """Validate, hash, and freeze a :class:`CropBundle`. The sole
    construction path used by ``src.vascutrace.data.crops`` -- callers never
    build a ``CropBundle`` directly, so hashes can never drift from content.
    """
    pet_arr = np.asarray(pet_suvbw, dtype=np.float32)
    ct_arr = np.asarray(ct_hu, dtype=np.float32)
    valid_arr = np.asarray(valid_pet_mask, dtype=np.uint8)
    iliac_arr = np.asarray(iliac_label_mask, dtype=np.uint8)
    for arr in (pet_arr, ct_arr, valid_arr, iliac_arr):
        if arr.shape != FIXED_CROP_SHAPE:
            raise CropIntegrityError(CropIntegrityErrorCode.SHAPE_MISMATCH)
    if not set(np.unique(iliac_arr).tolist()) <= _VALID_ILIAC_LABEL_MASK_VALUES:
        raise CropIntegrityError(CropIntegrityErrorCode.INVALID_ILIAC_LABEL_MASK)

    reflection = validate_reflection_affine(reflection_affine)

    bundle = CropBundle(
        schema_version=CROP_SCHEMA_VERSION,
        subject=subject,
        session=session,
        pet_suvbw=pet_arr,
        ct_hu=ct_arr,
        valid_pet_mask=valid_arr,
        iliac_label_mask=iliac_arr,
        reflection_affine=reflection,
        crop_to_pet_canonical_affine=np.asarray(
            crop_to_pet_canonical_affine, dtype=np.float64
        ),
        crop_origin_voxel=tuple(int(v) for v in crop_origin_voxel),
        original_voxel_from_pet_canonical_voxel=np.asarray(
            original_voxel_from_pet_canonical_voxel, dtype=np.float64
        ),
        geometry_sidecar=geometry_sidecar,
        reflection_residual_mm=float(reflection_residual_mm),
        reflection_qc_flag=bool(reflection_qc_flag),
        bbox_exceeds_fixed_crop=tuple(bool(v) for v in bbox_exceeds_fixed_crop),
        crop_margin_mm=float(crop_margin_mm),
        pet_spacing_mm=tuple(float(v) for v in pet_spacing_mm),
        paired_point_count=int(paired_point_count),
        hashes={},
    )
    return replace(bundle, hashes=compute_bundle_hashes(bundle))


# ---------------------------------------------------------------------------
# Deterministic save/load
# ---------------------------------------------------------------------------


def bundle_directory(
    output_root: Path, schema_version: str, subject: str, session: str
) -> Path:
    return Path(output_root) / "crops" / schema_version / subject / session


def save_crop_bundle(bundle: CropBundle, output_root: Path) -> Path:
    """Write ``bundle`` to ``<output_root>/crops/<schema_version>/<subject>/
    <session>/`` as ``bundle.npz`` (arrays) + ``bundle.json`` (metadata,
    geometry sidecar, hashes). ``output_root`` is caller-supplied and MUST be
    a gitignored local path -- this module does not enforce that (it has no
    knowledge of ``.gitignore``), the caller does.
    """
    directory = bundle_directory(
        output_root, bundle.schema_version, bundle.subject, bundle.session
    )
    directory.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        directory / "bundle.npz",
        pet_suvbw=bundle.pet_suvbw,
        ct_hu=bundle.ct_hu,
        valid_pet_mask=bundle.valid_pet_mask,
        iliac_label_mask=bundle.iliac_label_mask,
        reflection_affine=bundle.reflection_affine,
        crop_to_pet_canonical_affine=bundle.crop_to_pet_canonical_affine,
        original_voxel_from_pet_canonical_voxel=bundle.original_voxel_from_pet_canonical_voxel,
    )

    meta: dict[str, Any] = {
        "schema_version": bundle.schema_version,
        "subject": bundle.subject,
        "session": bundle.session,
        "fixed_shape": list(FIXED_CROP_SHAPE),
        "crop_origin_voxel": list(bundle.crop_origin_voxel),
        "geometry_sidecar": {
            "canonical_shape": list(bundle.geometry_sidecar.canonical_shape),
            "original_shape": list(bundle.geometry_sidecar.original_shape),
            "canonical_affine_sha256": bundle.geometry_sidecar.canonical_affine_sha256,
            "original_affine_sha256": bundle.geometry_sidecar.original_affine_sha256,
            "original_voxel_from_canonical_voxel_sha256": (
                bundle.geometry_sidecar.original_voxel_from_canonical_voxel_sha256
            ),
        },
        "reflection_residual_mm": bundle.reflection_residual_mm,
        "reflection_qc_flag": bundle.reflection_qc_flag,
        "bbox_exceeds_fixed_crop": list(bundle.bbox_exceeds_fixed_crop),
        "crop_margin_mm": bundle.crop_margin_mm,
        "pet_spacing_mm": list(bundle.pet_spacing_mm),
        "paired_point_count": bundle.paired_point_count,
        "hashes": bundle.hashes,
    }
    (directory / "bundle.json").write_text(json.dumps(meta, indent=2) + "\n")
    return directory


def load_crop_bundle(directory: Path) -> CropBundle:
    """Read a bundle written by :func:`save_crop_bundle`, verify its schema
    version, shapes, and every content hash against a fresh recomputation,
    and return the reconstructed :class:`CropBundle`. Raises
    :class:`CropIntegrityError` on any mismatch -- this is the pipeline's
    integrity gate against a truncated write, a hand-edited sidecar, or a
    bit-flipped array.
    """
    directory = Path(directory)
    meta_path = directory / "bundle.json"
    arrays_path = directory / "bundle.npz"
    if not meta_path.is_file() or not arrays_path.is_file():
        raise CropIntegrityError(CropIntegrityErrorCode.MISSING_BUNDLE_FILE)

    meta = json.loads(meta_path.read_text())
    if meta.get("schema_version") != CROP_SCHEMA_VERSION:
        raise CropIntegrityError(CropIntegrityErrorCode.SCHEMA_VERSION_MISMATCH)

    with np.load(arrays_path) as npz:
        pet_suvbw = np.asarray(npz["pet_suvbw"], dtype=np.float32)
        ct_hu = np.asarray(npz["ct_hu"], dtype=np.float32)
        valid_pet_mask = np.asarray(npz["valid_pet_mask"], dtype=np.uint8)
        iliac_label_mask = np.asarray(npz["iliac_label_mask"], dtype=np.uint8)
        reflection_affine = np.asarray(npz["reflection_affine"], dtype=np.float64)
        crop_to_pet_canonical_affine = np.asarray(
            npz["crop_to_pet_canonical_affine"], dtype=np.float64
        )
        original_voxel_from_pet_canonical_voxel = np.asarray(
            npz["original_voxel_from_pet_canonical_voxel"], dtype=np.float64
        )

    for arr in (pet_suvbw, ct_hu, valid_pet_mask, iliac_label_mask):
        if arr.shape != FIXED_CROP_SHAPE:
            raise CropIntegrityError(CropIntegrityErrorCode.SHAPE_MISMATCH)
    if not set(np.unique(iliac_label_mask).tolist()) <= _VALID_ILIAC_LABEL_MASK_VALUES:
        raise CropIntegrityError(CropIntegrityErrorCode.INVALID_ILIAC_LABEL_MASK)

    sidecar_dict = meta["geometry_sidecar"]
    geometry_sidecar = GeometrySidecar(
        canonical_shape=tuple(sidecar_dict["canonical_shape"]),
        original_shape=tuple(sidecar_dict["original_shape"]),
        canonical_affine_sha256=sidecar_dict["canonical_affine_sha256"],
        original_affine_sha256=sidecar_dict["original_affine_sha256"],
        original_voxel_from_canonical_voxel_sha256=sidecar_dict[
            "original_voxel_from_canonical_voxel_sha256"
        ],
    )

    bundle = CropBundle(
        schema_version=meta["schema_version"],
        subject=meta["subject"],
        session=meta["session"],
        pet_suvbw=pet_suvbw,
        ct_hu=ct_hu,
        valid_pet_mask=valid_pet_mask,
        iliac_label_mask=iliac_label_mask,
        reflection_affine=reflection_affine,
        crop_to_pet_canonical_affine=crop_to_pet_canonical_affine,
        crop_origin_voxel=tuple(meta["crop_origin_voxel"]),
        original_voxel_from_pet_canonical_voxel=original_voxel_from_pet_canonical_voxel,
        geometry_sidecar=geometry_sidecar,
        reflection_residual_mm=meta["reflection_residual_mm"],
        reflection_qc_flag=meta["reflection_qc_flag"],
        bbox_exceeds_fixed_crop=tuple(meta["bbox_exceeds_fixed_crop"]),
        crop_margin_mm=meta["crop_margin_mm"],
        pet_spacing_mm=tuple(meta["pet_spacing_mm"]),
        paired_point_count=meta["paired_point_count"],
        hashes=meta["hashes"],
    )

    if compute_bundle_hashes(bundle) != bundle.hashes:
        raise CropIntegrityError(CropIntegrityErrorCode.HASH_MISMATCH)
    return bundle


# ---------------------------------------------------------------------------
# Reflection-affine validation and the reflection helper
# ---------------------------------------------------------------------------


def validate_reflection_affine(reflection_affine: np.ndarray) -> np.ndarray:
    """Defense-in-depth re-verification that ``reflection_affine`` is a
    genuine physical mirror (Householder reflection): finite, a homogeneous
    4x4 affine, symmetric+involutive linear part, orientation-reversing.
    Mirrors ``src.vascutrace.quantification.measure._validate_reflection_
    affine``'s exact checks/tolerances (see module docstring) so a bundle
    produced by ``src.vascutrace.data.crops`` is guaranteed consumable by
    that downstream (P4) validator without surprises.
    """
    reflection = np.asarray(reflection_affine, dtype=np.float64)
    if reflection.shape != (4, 4):
        raise CropIntegrityError(CropIntegrityErrorCode.INVALID_REFLECTION_AFFINE)
    if not np.all(np.isfinite(reflection)):
        raise CropIntegrityError(CropIntegrityErrorCode.INVALID_REFLECTION_AFFINE)
    if not np.array_equal(reflection[3], np.array([0.0, 0.0, 0.0, 1.0])):
        raise CropIntegrityError(CropIntegrityErrorCode.INVALID_REFLECTION_AFFINE)
    linear = reflection[:3, :3]
    if not np.allclose(linear, linear.T, atol=_REFLECTION_SYMMETRY_ATOL):
        raise CropIntegrityError(CropIntegrityErrorCode.INVALID_REFLECTION_AFFINE)
    if not np.allclose(linear @ linear, np.eye(3), atol=_REFLECTION_INVOLUTION_ATOL):
        raise CropIntegrityError(CropIntegrityErrorCode.INVALID_REFLECTION_AFFINE)
    if np.linalg.det(linear) >= 0:
        raise CropIntegrityError(CropIntegrityErrorCode.INVALID_REFLECTION_AFFINE)
    return reflection


def _apply_affine(affine: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    homogeneous = np.concatenate([pts, ones], axis=1)
    return (np.asarray(affine, dtype=np.float64) @ homogeneous.T).T[:, :3]


def reflect_volume(
    volume: np.ndarray,
    crop_to_pet_canonical_affine: np.ndarray,
    reflection_affine: np.ndarray,
    *,
    order: int = 1,
) -> np.ndarray:
    """Physically reflect a crop-space array through ``reflection_affine``.

    Every output voxel's value comes from resampling ``volume`` at that
    voxel's true *reflected physical location* -- computed as
    crop-voxel -> world (via ``crop_to_pet_canonical_affine``) -> reflected
    world (via ``reflection_affine``) -> crop-voxel (via the inverse of
    ``crop_to_pet_canonical_affine``) -- never ``np.flip`` or any other
    array-index mirroring.

    This operates entirely within the crop's own RAS+ physical space (no
    RAS<->LPS bridge): ``reflection_affine`` is a same-space, same-patient
    mirror transform, not a cross-modality registration, matching the
    precedent already established for reflected-ROI sampling in
    ``src.vascutrace.quantification.measure.quantify_target`` (P4), which
    applies its ``reflection_affine`` argument directly in RAS as well.
    ``order=1`` (trilinear) is appropriate for continuous PET/CT intensity
    fields; pass ``order=0`` (nearest) for an integer/boolean mask.
    """
    reflection = validate_reflection_affine(reflection_affine)
    affine = np.asarray(crop_to_pet_canonical_affine, dtype=np.float64)
    if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
        raise CropIntegrityError(CropIntegrityErrorCode.INVALID_REFLECTION_AFFINE)
    inverse_affine = np.linalg.inv(affine)

    shape = volume.shape
    grids = np.meshgrid(*[np.arange(n, dtype=np.float64) for n in shape], indexing="ij")
    output_voxel = np.stack([g.reshape(-1) for g in grids], axis=1)

    world = _apply_affine(affine, output_voxel)
    reflected_world = _apply_affine(reflection, world)
    reflected_voxel = _apply_affine(inverse_affine, reflected_world)
    coords = reflected_voxel.T.reshape(3, *shape)

    resampled = map_coordinates(
        np.asarray(volume, dtype=np.float64),
        coords,
        order=order,
        mode="constant",
        cval=0.0,
    )
    if order == 0:
        return resampled.astype(volume.dtype)
    return resampled.astype(np.float32)
