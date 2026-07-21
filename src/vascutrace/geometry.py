"""Affine-aware PET/CT geometry contract (VascuTrace Phase 1).

RESEARCH_PROTOTYPE_WARNING
---------------------------------------------------------------------------
Research prototype. Trained and evaluated using simulated vascular-like
abnormalities, not confirmed human post-angioplasty lesions.
---------------------------------------------------------------------------

Implementation notes
============================================================================
This module is the sole gateway between raw NIfTI headers/affines and every
downstream PET/CT operation in VascuTrace. It answers three questions that a
paper reviewer will always ask about a multi-modal medical-imaging pipeline:
"is the affine you trusted actually well-posed?", "which physical coordinate
convention are you in at each step?", and "how did a value that started on
grid A end up being compared against a value on grid B?".

1. Grid validation (`validate_nifti_grid`)
    A NIfTI affine is only trustworthy if the header's two independent
    encodings of it (qform, an ortho-normal-basis quaternion + offset, and
    sform, a full 4x4 matrix) agree with each other and with the affine
    nibabel actually hands out (`img.affine`, nibabel's own qform/sform
    resolution). We require both to be *coded* (qform_code > 0 and
    sform_code > 0 -- an uncoded transform has no defined semantic meaning
    per the NIfTI-1 standard) and mutually consistent at `atol=1e-5` (NIfTI
    stores qform/sform as float32 in the header, so ~1e-7 disagreement is
    expected quantization noise; anything larger indicates the header was
    edited inconsistently and the geometry is ambiguous). We then check the
    affine is finite, non-singular and well-conditioned (a near-singular or
    high-condition-number linear part means voxel<->world is not reliably
    invertible), millimetre-scaled (the only unit this pipeline supports),
    has finite positive spacing, and is *not sheared*: after normalizing the
    three direction columns of the linear part to unit length, their
    pairwise dot products must be within 1e-4 of 0. A sheared voxel grid has
    no well-defined per-axis pixel spacing, which downstream code (Gaussian
    blur kernels, spacing-based measurements) implicitly assumes does not
    happen; we fail closed instead of silently mis-measuring.

2. RAS+ canonicalization (`canonicalize_nifti_to_ras`)
    NIfTI voxel axes can be stored in any of the 48 signed-permutation
    orientations relative to world space (radiological vs. neurological
    conventions, sagittal/axial/coronal-first storage, etc.). Every
    downstream algorithm (crop geometry, mirroring through the sagittal
    plane, spacing-aware blur) is written once, against one fixed axis
    convention: RAS+ ("as voxel index increases, world position increases
    toward Right, Anterior, Superior"). We determine the orientation with
    nibabel's `io_orientation`/`inv_ornt_aff` (read directly from
    `nibabel/orientations.py` in this environment's installed nibabel==5.4.2
    to confirm exact semantics rather than assuming them) and then move the
    voxel data ourselves via `_exact_voxel_remap`: because a NIfTI
    reorientation to RAS+ is *always* a signed permutation of the voxel
    lattice (transpose + flips only -- never a rotation, since rotation
    would not preserve "closest to an axis-aligned canonical form"), every
    output voxel maps to exactly one integer input voxel index. No
    interpolation is involved and the operation is exactly invertible,
    which is what makes an *exact* integer-label round trip possible later
    in `restore_mask_to_original_pet`. We independently re-derive nibabel's
    own `apply_orientation` output during development and confirmed
    bit-for-bit agreement (see the accompanying test suite), then dropped
    the dependency on `apply_orientation` itself so that the same
    `_exact_voxel_remap` primitive drives both the forward canonicalization
    and the later restoration, rather than trusting two independently
    written code paths to agree.

    We keep two named, explicit voxel-index mappings (never merely
    "transform"): `canonical_voxel_from_original_voxel` and
    `original_voxel_from_canonical_voxel`, matching the project's own
    imaging-physics contract for coordinate systems and grids. After
    canonicalizing we re-verify, at
    runtime, that (a) world coordinates computed via the canonical affine
    agree with world coordinates computed by mapping the same point back to
    original-voxel space and applying the original affine, to within 1e-6
    mm, and (b) the two named mappings are exact matrix inverses of each
    other to within 1e-10. These are defence-in-depth self-checks: for a
    validated input they are mathematically guaranteed to hold (a signed
    permutation composed with its own inverse is the identity in exact
    arithmetic), so they should never fire in practice, but they convert a
    silent latent bug into a typed, loud failure instead.

3. The RAS<->LPS bridge (`ras_points_to_lps`, `lps_points_to_ras`,
   `input_lps_from_output_lps`)
    NiBabel's world coordinates are RAS+ (neuroimaging convention).
    ITK/SimpleITK's physical coordinates are LPS (Left, Posterior, Superior
    -- the DICOM patient-coordinate convention). The two conventions differ
    by exactly a sign flip on the first two (x, y) axes:
    `point_lps = C_lps_from_ras @ point_ras`, where
    `C_lps_from_ras = diag(-1, -1, 1, 1)`. Because `C_lps_from_ras` squares
    to the identity (it is an involution), the RAS->LPS and LPS->RAS bridges
    are literally the same matrix multiplication, and
    `lps_points_to_ras(ras_points_to_lps(p)) == p` exactly (up to a possible
    signed-zero, since `-1 * -0.0 == 0.0` in IEEE-754 but does not always
    compare `array_equal` against a literal `0.0`; the test suite uses
    `np.array_equal` on inputs built to avoid exact zero components, plus a
    dedicated allclose check, to keep this distinction visible rather than
    hidden).

    For an *output*-physical-to-*input*-physical registration transform
    expressed in RAS (i.e. "given a point in the output grid's physical
    space, where is the corresponding point in the input grid's physical
    space"), the LPS-space equivalent is obtained by conjugating with the
    bridge on both sides:
    `T_input_lps_from_output_lps = C_lps_from_ras @ T_input_ras_from_output_ras @ C_lps_from_ras`.
    This is exactly the formula this project's imaging-physics contract
    specifies for handing a transform to an LPS-native physical-coordinate
    consumer. `resample_ct_to_pet`/`resample_mask_to_pet` route every output
    voxel through this exact bridge (with the RAS registration transform
    fixed at the identity -- see the "Registration assumption" note in
    `_output_voxel_grid_input_indices` below) so the bridge is exercised as
    load-bearing pipeline code, not a decorative standalone utility.

4. Resampling onto the PET SUVbw grid (`resample_ct_to_pet`,
   `resample_mask_to_pet`) and restoration (`restore_mask_to_original_pet`)
    The PET SUVbw grid is the sole reference/output grid for resampling in
    this project. We never resample by matching array shapes or by
    index-space zoom: every output voxel's *physical* coordinate is computed
    from the output (PET) affine, mapped through the RAS<->LPS bridge to the
    input grid's physical space, and only then converted to a fractional
    input-voxel coordinate via the input grid's own affine. CT is resampled
    with trilinear interpolation (`scipy.ndimage.map_coordinates`,
    `order=1`); an affine "HU ramp" phantom is interpolated *exactly* by a
    trilinear kernel (trilinear interpolation is exact for any function that
    is itself affine in the sampled coordinates), which is why the test
    suite uses such a ramp to isolate *coordinate-transform* correctness
    from interpolation-approximation error: a wrong axis order or a missed
    sign flip produces gross (not `1e-4`-scale) error, while a correct
    pipeline reproduces the analytic ramp to floating-point precision.
    Integer masks are resampled with nearest-neighbor interpolation
    (`order=0`), which can only ever copy a value that already exists in the
    source array, so "no new labels are invented" and "only source labels
    appear" hold by construction rather than by a separate filter step.
    Both resamplers zero-fill (`mode="constant", cval=0.0`) outside the
    source field of view.

    Internal processing happens on the RAS+-canonical grid. Every *public*
    spatial mask must be re-expressed in the original (pre-canonicalization)
    PET shape and affine before it leaves the pipeline
    (imaging-physics.md, "Emit every public spatial mask in original PET
    shape and affine. Internal arrays require a geometry sidecar.").
    `restore_mask_to_original_pet` performs that exact (again,
    interpolation-free) inverse remap and returns a small deterministic
    `GeometrySidecar` (shape + affine identity, fingerprinted by SHA-256)
    alongside the restored mask. If a caller later re-presents a sidecar
    that does not match the one freshly derived from the same
    `CanonicalVolume` (wrong shape, or an affine/mapping whose SHA-256
    digest disagrees), the mismatch is a `GeometryContractError` carrying
    `GeometryErrorCode.GEOMETRY_MISMATCH` and *only* the code string --
    never a raw path, header value, or voxel value -- as its message.

5. The immutable-array contract
    Every ndarray this module hands to a caller (coordinate arrays,
    `C_lps_from_ras`, grid affines/world bounds, RAS<->LPS bridge outputs,
    canonicalized volume data, and resampled arrays) is returned as an
    `ImmutableArray`: an owning (`OWNDATA=True`), non-writeable
    (`WRITEABLE=False`) `np.ndarray` subclass. A plain
    `a = result.copy(); a.setflags(write=False)` (the minimal recipe) gives
    you `OWNDATA=True`/`WRITEABLE=False`, but on a data-owning array (one
    with `base is None`) vanilla NumPy freely allows *undoing* that later
    (`a.setflags(write=True)` and `a.flags.writeable = True` both silently
    succeed, and `.resize()`/in-place `.shape = ...` reassignment bypass the
    writeable flag check entirely) -- verified empirically against this
    environment's installed NumPy before relying on it. `ImmutableArray`
    therefore overrides `setflags`, `resize`, the `shape`/`strides`/`dtype`
    property setters, and the `flags` object itself (via a thin
    `_FrozenFlags` wrapper) to reject any attempt to re-enable writeability
    or otherwise mutate structure, while still allowing genuine no-ops
    (re-asserting `write=False`) to pass silently, and while leaving
    ordinary read access, `.copy()` (an explicit, independently-owned,
    intentionally-mutable snapshot), and NumPy's native view-writeability
    propagation (e.g. `np.asarray(frozen)` returns a view that inherits
    `WRITEABLE=False` and cannot be flipped back, because its `base` is
    itself non-writeable) untouched.
============================================================================
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import nibabel as nib
import numpy as np
from nibabel.orientations import inv_ornt_aff, io_orientation
from scipy.ndimage import map_coordinates

__all__ = [
    "RESEARCH_PROTOTYPE_WARNING",
    "C_lps_from_ras",
    "GeometryErrorCode",
    "GeometryContractError",
    "GridGeometry",
    "CanonicalVolume",
    "SpatialArray",
    "GeometrySidecar",
    "validate_nifti_grid",
    "canonicalize_nifti_to_ras",
    "ras_points_to_lps",
    "lps_points_to_ras",
    "input_lps_from_output_lps",
    "resample_ct_to_pet",
    "resample_mask_to_pet",
    "restore_mask_to_original_pet",
]

RESEARCH_PROTOTYPE_WARNING = (
    "Research prototype. Trained and evaluated using simulated vascular-like "
    "abnormalities, not confirmed human post-angioplasty lesions."
)

# Numerical policy constants (see module docstring section 1 for rationale).
_AFFINE_CONSISTENCY_ATOL = 1e-5
_MIN_ABS_DETERMINANT = 1e-12
_MAX_CONDITION_NUMBER = 1e12
_SHEAR_DOT_TOLERANCE = 1e-4
_WORLD_AGREEMENT_TOLERANCE_MM = 1e-6
_INVERSE_RESIDUAL_TOLERANCE = 1e-10
_VOXEL_REMAP_INTEGER_TOLERANCE = 1e-9


# ---------------------------------------------------------------------------
# Immutable-array contract
# ---------------------------------------------------------------------------


class _FrozenFlags:
    """Read-mostly wrapper around a real ``numpy.flagsobj``.

    Delegates reads to the underlying flags object. Allows only no-op
    writes (assigning a flag its current value, or explicitly re-asserting
    ``writeable=False``); rejects any attempt to actually change a flag,
    most importantly ``writeable = True``.
    """

    __slots__ = ("_flags",)

    def __init__(self, flags: Any) -> None:
        object.__setattr__(self, "_flags", flags)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_flags"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        flags = object.__getattribute__(self, "_flags")
        if name == "writeable":
            if value:
                raise ValueError(
                    "cannot set WRITEABLE flag to True on an immutable array"
                )
            return  # no-op: already (and permanently) False
        current = getattr(flags, name, None)
        if value == current:
            return  # no-op: requested value already holds
        raise AttributeError(f"cannot modify flag {name!r} on an immutable array")

    def __repr__(self) -> str:
        return repr(object.__getattribute__(self, "_flags"))

    def __eq__(self, other: object) -> bool:
        return object.__getattribute__(self, "_flags") == other


class ImmutableArray(np.ndarray):
    """An owning, permanently read-only ``np.ndarray``.

    Construction copies the source into a freshly allocated, owned buffer
    (``OWNDATA=True``) and marks it non-writeable (``WRITEABLE=False``).
    Beyond that baseline (which vanilla ``copy(); setflags(write=False)``
    already provides), this subclass additionally rejects every attempt to
    resize, reshape/re-stride in place, change dtype in place, or re-enable
    writeability -- operations vanilla NumPy allows even on a
    ``WRITEABLE=False`` owning array. See the module docstring, section 5,
    for the empirical justification.
    """

    def __new__(cls, source: Any) -> "ImmutableArray":
        src = np.asarray(source)
        obj = super().__new__(cls, shape=src.shape, dtype=src.dtype)
        obj[...] = src
        np.ndarray.setflags(obj, write=False)
        return obj

    def setflags(
        self,
        write: bool | None = None,
        align: bool | None = None,
        uic: bool | None = None,
    ) -> None:
        if write:
            raise ValueError("cannot set WRITEABLE flag to True on an immutable array")
        # write=False/None and align/uic requests are inert no-ops here: this
        # array's alignment/WRITEBACKIFCOPY state never needs changing.
        return None

    def resize(self, *args: Any, **kwargs: Any) -> None:
        raise ValueError("cannot resize an immutable array")

    @property
    def shape(self) -> tuple[int, ...]:
        return np.ndarray.shape.__get__(self)

    @shape.setter
    def shape(self, value: Any) -> None:
        raise ValueError("cannot change the shape of an immutable array in place")

    @property
    def strides(self) -> tuple[int, ...]:
        return np.ndarray.strides.__get__(self)

    @strides.setter
    def strides(self, value: Any) -> None:
        raise ValueError("cannot change the strides of an immutable array in place")

    @property
    def dtype(self) -> np.dtype:
        return np.ndarray.dtype.__get__(self)

    @dtype.setter
    def dtype(self, value: Any) -> None:
        raise ValueError("cannot change the dtype of an immutable array in place")

    @property
    def flags(self) -> _FrozenFlags:
        return _FrozenFlags(np.ndarray.flags.__get__(self))


def _freeze(array: Any) -> ImmutableArray:
    """Return an owning, read-only ``ImmutableArray`` snapshot of ``array``."""
    return ImmutableArray(array)


# RAS<->LPS bridge matrix. Self-inverse (``C @ C == I``): the same matrix
# converts RAS->LPS and LPS->RAS. See module docstring, section 3, and
# the project contract for coordinate systems and grids.
C_lps_from_ras = _freeze(np.diag([-1.0, -1.0, 1.0, 1.0]))


# ---------------------------------------------------------------------------
# Error taxonomy
# ---------------------------------------------------------------------------


class GeometryErrorCode(StrEnum):
    """Typed, code-only failure reasons for the geometry contract."""

    INVALID_SHAPE = "INVALID_SHAPE"
    INVALID_AFFINE = "INVALID_AFFINE"
    NONFINITE_AFFINE = "NONFINITE_AFFINE"
    SINGULAR_AFFINE = "SINGULAR_AFFINE"
    AMBIGUOUS_AFFINE = "AMBIGUOUS_AFFINE"
    INVALID_SPATIAL_UNITS = "INVALID_SPATIAL_UNITS"
    INVALID_SPACING = "INVALID_SPACING"
    SHEARED_GRID = "SHEARED_GRID"
    GEOMETRY_MISMATCH = "GEOMETRY_MISMATCH"


class GeometryContractError(ValueError):
    """Raised for any geometry-contract violation.

    The exception message is exactly the failing :class:`GeometryErrorCode`
    value -- never a path, header field, or data value -- so it is always
    safe to log or surface verbatim.
    """

    def __init__(self, code: GeometryErrorCode) -> None:
        super().__init__(code.value)
        self.code = code


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class GridGeometry:
    """A validated voxel grid: shape, affine (voxel->world, RAS+ once
    canonicalized), derived spacing, units, and world-space bounding box
    computed from the eight voxel-center corners.

    ``eq``/``__hash__`` are left at Python's default identity-based
    behavior (not dataclass-generated) because tuple-wise ``==`` across
    ndarray fields raises ``ValueError`` (ambiguous truth value) rather
    than doing anything useful.
    """

    shape: tuple[int, int, int]
    affine: np.ndarray = field(repr=False)
    spacing: tuple[float, float, float]
    units: str
    world_bounds_min: np.ndarray = field(repr=False)
    world_bounds_max: np.ndarray = field(repr=False)


@dataclass(frozen=True, slots=True, eq=False)
class CanonicalVolume:
    """A volume canonicalized once to RAS+, with both named voxel-index
    mappings back to and from the pre-canonicalization ("original") grid.
    """

    data: np.ndarray = field(repr=False)
    geometry: GridGeometry
    original_geometry: GridGeometry
    canonical_voxel_from_original_voxel: np.ndarray = field(repr=False)
    original_voxel_from_canonical_voxel: np.ndarray = field(repr=False)


@dataclass(frozen=True, slots=True, eq=False)
class SpatialArray:
    """An owning, read-only ``(N, 3)`` array of physical coordinates in a
    named space (``"RAS"`` or ``"LPS"``).
    """

    points: np.ndarray = field(repr=False)
    space: str


@dataclass(frozen=True, slots=True)
class GeometrySidecar:
    """A small, deterministic fingerprint of a canonicalization, carried
    alongside a restored mask so a later caller can detect a corrupted or
    mismatched restoration. All fields are plain, hashable values (no
    ndarrays), so this record uses ordinary dataclass value equality.
    """

    canonical_shape: tuple[int, int, int]
    original_shape: tuple[int, int, int]
    canonical_affine_sha256: str
    original_affine_sha256: str
    original_voxel_from_canonical_voxel_sha256: str


# ---------------------------------------------------------------------------
# Shared numeric helpers
# ---------------------------------------------------------------------------


def _apply_affine(affine: Any, points: Any) -> np.ndarray:
    """Apply a 4x4 homogeneous affine to an ``(N, 3)`` array of points."""
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    homogeneous = np.concatenate([pts, ones], axis=1)
    return (np.asarray(affine, dtype=np.float64) @ homogeneous.T).T[:, :3]


def _affine_sha256(affine: Any) -> str:
    payload = np.ascontiguousarray(np.asarray(affine, dtype=np.float64)).tobytes()
    return hashlib.sha256(payload).hexdigest()


def _grid_geometry_from_affine(
    shape: tuple[int, int, int], affine: Any, units: str
) -> GridGeometry:
    """Validate a shape/affine/units triple (independent of any NIfTI
    header) and build the corresponding :class:`GridGeometry`.

    Factored out of :func:`validate_nifti_grid` so
    :func:`canonicalize_nifti_to_ras` can validate its *derived* canonical
    affine through the exact same numeric checks without re-deriving
    qform/sform header consistency (which has no meaning for a derived,
    header-less affine).
    """
    if len(shape) != 3 or any(int(d) <= 0 for d in shape):
        raise GeometryContractError(GeometryErrorCode.INVALID_SHAPE)

    affine_arr = np.asarray(affine, dtype=np.float64)
    if affine_arr.shape != (4, 4):
        raise GeometryContractError(GeometryErrorCode.INVALID_AFFINE)
    if not np.array_equal(affine_arr[3], np.array([0.0, 0.0, 0.0, 1.0])):
        raise GeometryContractError(GeometryErrorCode.INVALID_AFFINE)
    if not np.all(np.isfinite(affine_arr)):
        raise GeometryContractError(GeometryErrorCode.NONFINITE_AFFINE)

    linear = affine_arr[:3, :3]

    # Spacing (per-axis column norm) is checked *before* determinant/
    # condition below. A zero-norm column forces det == 0, so checking
    # determinant first would make INVALID_SPACING mathematically
    # unreachable (every degenerate-spacing case would already have been
    # caught as SINGULAR_AFFINE). Checking spacing first keeps the two
    # codes independently reachable: a zero/nonfinite column -> more
    # specific INVALID_SPACING; a matrix with three individually
    # positive-norm but linearly dependent columns (e.g. two parallel
    # direction columns of different lengths) passes this check yet is
    # still singular -> SINGULAR_AFFINE below.
    spacing = np.linalg.norm(linear, axis=0)
    if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise GeometryContractError(GeometryErrorCode.INVALID_SPACING)

    determinant = np.linalg.det(linear)
    if abs(determinant) <= _MIN_ABS_DETERMINANT:
        raise GeometryContractError(GeometryErrorCode.SINGULAR_AFFINE)
    # Condition number is evaluated on the 3x3 linear part (the
    # voxel->world Jacobian): translation does not affect invertibility.
    if np.linalg.cond(linear) > _MAX_CONDITION_NUMBER:
        raise GeometryContractError(GeometryErrorCode.SINGULAR_AFFINE)

    if units != "mm":
        raise GeometryContractError(GeometryErrorCode.INVALID_SPATIAL_UNITS)

    normalized = linear / spacing
    pairwise_dots = (
        abs(float(np.dot(normalized[:, 0], normalized[:, 1]))),
        abs(float(np.dot(normalized[:, 0], normalized[:, 2]))),
        abs(float(np.dot(normalized[:, 1], normalized[:, 2]))),
    )
    if any(d > _SHEAR_DOT_TOLERANCE for d in pairwise_dots):
        raise GeometryContractError(GeometryErrorCode.SHEARED_GRID)

    corner_indices = np.array(
        [
            [i, j, k]
            for i in (0, shape[0] - 1)
            for j in (0, shape[1] - 1)
            for k in (0, shape[2] - 1)
        ],
        dtype=np.float64,
    )
    world_corners = _apply_affine(affine_arr, corner_indices)

    return GridGeometry(
        shape=(int(shape[0]), int(shape[1]), int(shape[2])),
        affine=_freeze(affine_arr),
        spacing=(float(spacing[0]), float(spacing[1]), float(spacing[2])),
        units=units,
        world_bounds_min=_freeze(world_corners.min(axis=0)),
        world_bounds_max=_freeze(world_corners.max(axis=0)),
    )


def _permutation_axis_map(linear_3x3: np.ndarray) -> np.ndarray:
    """For an exact signed-permutation 3x3 matrix, return, for each output
    row, the index of the single input axis it maps from.
    """
    return np.argmax(np.abs(linear_3x3), axis=1)


def _exact_voxel_remap(
    source: np.ndarray,
    output_shape: tuple[int, int, int],
    input_voxel_from_output_voxel: np.ndarray,
) -> np.ndarray:
    """Move array data through an exact signed-permutation-plus-integer-shift
    voxel mapping (RAS+ canonicalization or its inverse restoration). No
    interpolation: every output voxel maps to exactly one integer input
    voxel index, so integer labels (and any other exact values) survive the
    round trip bit-for-bit. Cross-validated against nibabel's own
    ``apply_orientation`` during development (see module docstring,
    section 2).
    """
    grids = np.meshgrid(
        *[np.arange(n, dtype=np.float64) for n in output_shape], indexing="ij"
    )
    n_voxels = int(np.prod(output_shape))
    output_index_h = np.stack(
        [g.reshape(-1) for g in grids] + [np.ones(n_voxels, dtype=np.float64)], axis=0
    )
    input_index_h = (
        np.asarray(input_voxel_from_output_voxel, dtype=np.float64) @ output_index_h
    )
    input_index_float = input_index_h[:3]
    input_index_rounded = np.rint(input_index_float)
    if not np.allclose(
        input_index_float, input_index_rounded, atol=_VOXEL_REMAP_INTEGER_TOLERANCE
    ):
        # Defensive only: a validated RAS+ canonicalization mapping is an
        # exact signed permutation and can never produce a fractional
        # index. This guards against a corrupted/foreign mapping being
        # substituted for the real one.
        raise GeometryContractError(GeometryErrorCode.INVALID_AFFINE)
    input_index = input_index_rounded.astype(np.int64)
    remapped = source[input_index[0], input_index[1], input_index[2]]
    return remapped.reshape(output_shape)


# ---------------------------------------------------------------------------
# Public API: grid validation
# ---------------------------------------------------------------------------


def validate_nifti_grid(img: nib.Nifti1Image) -> GridGeometry:
    """Validate a NIfTI image's shape, qform/sform/image affine consistency,
    finiteness, invertibility/conditioning, spatial units, spacing, and
    shear, per the numerical and physical-coordinate policy in the module
    docstring. Raises :class:`GeometryContractError` with the specific
    :class:`GeometryErrorCode` on the first violation found, in the order:
    shape, affine coding/consistency, finiteness, singularity/conditioning,
    units, spacing, shear.
    """
    shape = tuple(img.shape)
    if len(shape) != 3 or any(int(d) <= 0 for d in shape):
        # Covers both non-3D shapes and the common "singleton 4D" NIfTI
        # case (trailing size-1 axis from multi-volume-capable headers).
        raise GeometryContractError(GeometryErrorCode.INVALID_SHAPE)

    header = img.header
    qform_affine, qform_code = header.get_qform(coded=True)
    sform_affine, sform_code = header.get_sform(coded=True)
    if not qform_code or not sform_code:
        # An uncoded qform/sform has no defined meaning per the NIfTI-1
        # standard: the orientation is ambiguous, not merely "unset".
        raise GeometryContractError(GeometryErrorCode.AMBIGUOUS_AFFINE)

    image_affine = np.asarray(img.affine, dtype=np.float64)
    for candidate in (
        np.asarray(qform_affine, dtype=np.float64),
        np.asarray(sform_affine, dtype=np.float64),
    ):
        if not np.allclose(
            candidate, image_affine, rtol=0.0, atol=_AFFINE_CONSISTENCY_ATOL
        ):
            raise GeometryContractError(GeometryErrorCode.AMBIGUOUS_AFFINE)

    spatial_units, _temporal_units = header.get_xyzt_units()

    return _grid_geometry_from_affine(
        (int(shape[0]), int(shape[1]), int(shape[2])), image_affine, spatial_units
    )


# ---------------------------------------------------------------------------
# Public API: RAS+ canonicalization
# ---------------------------------------------------------------------------


def _verify_canonicalization_invariants(
    *,
    original_affine: np.ndarray,
    canonical_affine: np.ndarray,
    original_voxel_from_canonical_voxel: np.ndarray,
    canonical_voxel_from_original_voxel: np.ndarray,
    canonical_shape: tuple[int, int, int],
) -> None:
    residual = np.asarray(
        original_voxel_from_canonical_voxel, dtype=np.float64
    ) @ np.asarray(canonical_voxel_from_original_voxel, dtype=np.float64) - np.eye(4)
    if np.max(np.abs(residual)) > _INVERSE_RESIDUAL_TOLERANCE:
        raise GeometryContractError(GeometryErrorCode.INVALID_AFFINE)

    corner_indices = np.array(
        [
            [i, j, k]
            for i in (0, canonical_shape[0] - 1)
            for j in (0, canonical_shape[1] - 1)
            for k in (0, canonical_shape[2] - 1)
        ],
        dtype=np.float64,
    )
    world_via_canonical = _apply_affine(canonical_affine, corner_indices)
    original_indices = _apply_affine(
        original_voxel_from_canonical_voxel, corner_indices
    )
    world_via_original = _apply_affine(original_affine, original_indices)
    if (
        np.max(np.abs(world_via_canonical - world_via_original))
        > _WORLD_AGREEMENT_TOLERANCE_MM
    ):
        raise GeometryContractError(GeometryErrorCode.INVALID_AFFINE)


def canonicalize_nifti_to_ras(img: nib.Nifti1Image) -> CanonicalVolume:
    """Validate ``img`` and canonicalize it to RAS+ exactly once.

    Returns a :class:`CanonicalVolume` carrying the canonical data array,
    the canonical :class:`GridGeometry`, the pre-canonicalization ("original")
    :class:`GridGeometry`, and both named voxel-index mappings between the
    two grids. See the module docstring, section 2, for the algorithm and
    its cross-validation against nibabel's own orientation machinery.
    """
    original_geometry = validate_nifti_grid(img)
    original_affine = np.asarray(img.affine, dtype=np.float64)
    original_shape = original_geometry.shape
    original_data = np.asarray(img.dataobj)  # preserves the on-disk/array dtype exactly

    orientation = io_orientation(original_affine)
    original_voxel_from_canonical_voxel = inv_ornt_aff(orientation, original_shape)
    canonical_voxel_from_original_voxel = np.linalg.inv(
        original_voxel_from_canonical_voxel
    )
    canonical_affine = original_affine @ original_voxel_from_canonical_voxel

    axis_map = _permutation_axis_map(canonical_voxel_from_original_voxel[:3, :3])
    canonical_shape = tuple(int(original_shape[axis_map[p]]) for p in range(3))

    canonical_data = _exact_voxel_remap(
        original_data, canonical_shape, original_voxel_from_canonical_voxel
    )

    canonical_geometry = _grid_geometry_from_affine(
        canonical_shape, canonical_affine, original_geometry.units
    )

    _verify_canonicalization_invariants(
        original_affine=original_affine,
        canonical_affine=np.asarray(canonical_geometry.affine, dtype=np.float64),
        original_voxel_from_canonical_voxel=original_voxel_from_canonical_voxel,
        canonical_voxel_from_original_voxel=canonical_voxel_from_original_voxel,
        canonical_shape=canonical_shape,
    )

    return CanonicalVolume(
        data=_freeze(canonical_data),
        geometry=canonical_geometry,
        original_geometry=original_geometry,
        canonical_voxel_from_original_voxel=_freeze(
            canonical_voxel_from_original_voxel
        ),
        original_voxel_from_canonical_voxel=_freeze(
            original_voxel_from_canonical_voxel
        ),
    )


# ---------------------------------------------------------------------------
# Public API: RAS<->LPS bridge
# ---------------------------------------------------------------------------


def _bridge_points(points: Any, target_space: str) -> SpatialArray:
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise GeometryContractError(GeometryErrorCode.INVALID_SHAPE)
    transformed = _apply_affine(C_lps_from_ras, pts)
    return SpatialArray(points=_freeze(transformed), space=target_space)


def ras_points_to_lps(points: Any) -> SpatialArray:
    """Convert ``(3,)`` or ``(N, 3)`` RAS+ physical points to LPS.

    Always returns ``(N, 3)`` points (a single input point becomes an
    ``(1, 3)`` result) for a uniform, unambiguous return shape.
    """
    return _bridge_points(points, "LPS")


def lps_points_to_ras(points: Any) -> SpatialArray:
    """Convert ``(3,)`` or ``(N, 3)`` LPS physical points to RAS+.

    ``C_lps_from_ras`` is self-inverse, so this uses the identical bridge
    matrix multiplication as :func:`ras_points_to_lps`; the two are
    literally the same transform applied in either direction.
    """
    return _bridge_points(points, "RAS")


def input_lps_from_output_lps(t_input_ras_from_output_ras: Any) -> np.ndarray:
    """Conjugate an output-RAS-to-input-RAS registration affine into the
    equivalent output-LPS-to-input-LPS affine:
    ``C @ T_input_ras_from_output_ras @ C``, where ``C = C_lps_from_ras``.

    Used to map PET-grid (output) physical coordinates back into a source
    (input, e.g. CT or a mask) grid's physical space for resampling -- see
    :func:`resample_ct_to_pet`/:func:`resample_mask_to_pet` and the module
    docstring, section 3.
    """
    t = np.asarray(t_input_ras_from_output_ras, dtype=np.float64)
    if t.shape != (4, 4):
        raise GeometryContractError(GeometryErrorCode.INVALID_AFFINE)
    if not np.all(np.isfinite(t)):
        raise GeometryContractError(GeometryErrorCode.NONFINITE_AFFINE)
    bridge = np.asarray(C_lps_from_ras, dtype=np.float64)
    return _freeze(bridge @ t @ bridge)


# ---------------------------------------------------------------------------
# Public API: resampling onto the PET SUVbw grid
# ---------------------------------------------------------------------------


def _output_voxel_grid_input_indices(
    output_geometry: GridGeometry, input_geometry: GridGeometry
) -> np.ndarray:
    """For every voxel index of ``output_geometry``'s grid, compute the
    corresponding fractional voxel index into ``input_geometry``'s grid, by
    walking output-voxel -> output-world(RAS) -> output-world(LPS) ->
    input-world(LPS) -> input-world(RAS) -> input-voxel.

    Registration assumption: within one VascuTrace case, PET and CT (and
    any mask defined on one of those grids) come from the same scan session
    and share one physical patient RAS space, so the output-to-input RAS
    registration transform used here is the identity. The LPS round trip is
    still performed explicitly (not skipped as an algebraic no-op) so that
    the RAS<->LPS bridge is genuinely exercised as the mechanism a future
    non-identity registration transform would also flow through.

    Returns an array shaped ``(3, *output_geometry.shape)``, the coordinate
    layout ``scipy.ndimage.map_coordinates`` expects.
    """
    shape = output_geometry.shape
    grids = np.meshgrid(*[np.arange(n, dtype=np.float64) for n in shape], indexing="ij")
    output_voxel = np.stack([g.reshape(-1) for g in grids], axis=1)  # (N, 3)

    output_world_ras = _apply_affine(output_geometry.affine, output_voxel)
    output_world_lps = ras_points_to_lps(output_world_ras).points

    t_input_lps_from_output_lps = input_lps_from_output_lps(np.eye(4))
    input_world_lps = _apply_affine(t_input_lps_from_output_lps, output_world_lps)
    input_world_ras = lps_points_to_ras(input_world_lps).points

    input_affine_inverse = np.linalg.inv(
        np.asarray(input_geometry.affine, dtype=np.float64)
    )
    input_voxel = _apply_affine(input_affine_inverse, input_world_ras)

    return input_voxel.T.reshape(3, *shape)


def resample_ct_to_pet(ct: CanonicalVolume, pet_geometry: GridGeometry) -> np.ndarray:
    """Trilinearly resample a canonicalized CT volume (HU) onto the PET
    SUVbw reference grid. Returns a ``float32`` array shaped
    ``pet_geometry.shape``, zero outside the CT field of view.

    Both ``ct`` and ``pet_geometry`` must already be RAS+-canonicalized
    (see :func:`canonicalize_nifti_to_ras`): this function assumes their
    affines encode a shared physical RAS+ patient space.
    """
    coordinates = _output_voxel_grid_input_indices(pet_geometry, ct.geometry)
    resampled = map_coordinates(
        np.asarray(ct.data, dtype=np.float64),
        coordinates,
        order=1,
        mode="constant",
        cval=0.0,
    )
    return _freeze(resampled.astype(np.float32))


def resample_mask_to_pet(
    mask: CanonicalVolume, pet_geometry: GridGeometry
) -> np.ndarray:
    """Nearest-neighbor resample a canonicalized integer-labeled mask onto
    the PET SUVbw reference grid. Returns an integer array (the source
    mask's own dtype) shaped ``pet_geometry.shape``, containing only labels
    present in the source mask, zero outside the source field of view.

    Both ``mask`` and ``pet_geometry`` must already be RAS+-canonicalized
    (see :func:`canonicalize_nifti_to_ras`).
    """
    source = np.asarray(mask.data)
    if not np.issubdtype(source.dtype, np.integer):
        raise TypeError("resample_mask_to_pet requires an integer-labeled mask array")
    coordinates = _output_voxel_grid_input_indices(pet_geometry, mask.geometry)
    resampled = map_coordinates(
        source.astype(np.float64),
        coordinates,
        order=0,
        mode="constant",
        cval=0.0,
    )
    return _freeze(np.rint(resampled).astype(source.dtype))


# ---------------------------------------------------------------------------
# Public API: restoration to the original (native) PET grid
# ---------------------------------------------------------------------------


def _sidecar_for(canonical_volume: CanonicalVolume) -> GeometrySidecar:
    return GeometrySidecar(
        canonical_shape=canonical_volume.geometry.shape,
        original_shape=canonical_volume.original_geometry.shape,
        canonical_affine_sha256=_affine_sha256(canonical_volume.geometry.affine),
        original_affine_sha256=_affine_sha256(
            canonical_volume.original_geometry.affine
        ),
        original_voxel_from_canonical_voxel_sha256=_affine_sha256(
            canonical_volume.original_voxel_from_canonical_voxel
        ),
    )


def restore_mask_to_original_pet(
    canonical_mask: np.ndarray,
    canonical_volume: CanonicalVolume,
    sidecar: GeometrySidecar | None = None,
) -> tuple[np.ndarray, GeometrySidecar]:
    """Restore a mask defined on ``canonical_volume``'s RAS+ grid back to
    its original (pre-canonicalization) PET shape and affine, via the exact
    inverse voxel mapping (no interpolation -- see
    :func:`_exact_voxel_remap`).

    Every public spatial mask must be re-expressed in the original PET
    shape/affine before leaving the pipeline (imaging-physics.md). This
    function is both how that happens and the sole place that produces a
    :class:`GeometrySidecar`: a small, deterministic fingerprint (shape +
    SHA-256 of the relevant affines/mapping) of the canonicalization that
    produced the restoration.

    If ``sidecar`` is omitted, a fresh sidecar derived from
    ``canonical_volume`` is returned alongside the restored mask (the
    common, first-use case). If ``sidecar`` is supplied, it is verified
    against the sidecar freshly derived from ``canonical_volume``; any
    mismatch (wrong shape, or an affine/mapping whose digest disagrees --
    i.e. a corrupted or foreign sidecar) raises
    ``GeometryContractError(GeometryErrorCode.GEOMETRY_MISMATCH)``, whose
    message is exactly the code string.
    """
    expected_sidecar = _sidecar_for(canonical_volume)

    mask = np.asarray(canonical_mask)
    if not np.issubdtype(mask.dtype, np.integer):
        raise TypeError(
            "restore_mask_to_original_pet requires an integer-labeled mask array"
        )
    if tuple(mask.shape) != expected_sidecar.canonical_shape:
        raise GeometryContractError(GeometryErrorCode.GEOMETRY_MISMATCH)
    if sidecar is not None and sidecar != expected_sidecar:
        raise GeometryContractError(GeometryErrorCode.GEOMETRY_MISMATCH)

    restored = _exact_voxel_remap(
        mask,
        expected_sidecar.original_shape,
        canonical_volume.original_voxel_from_canonical_voxel,
    )
    return _freeze(restored), expected_sidecar
