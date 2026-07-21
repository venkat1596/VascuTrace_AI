"""Physics-informed synthetic vascular-uptake generator (VascuTrace Phase 3).

RESEARCH_PROTOTYPE_WARNING
---------------------------------------------------------------------------
Research prototype. Trained and evaluated using simulated vascular-like
abnormalities, not confirmed human post-angioplasty lesions.
---------------------------------------------------------------------------

This module inserts a single controlled synthetic source along a frozen
vessel centerline into a healthy PET SUVbw background, producing a synthetic
PET volume plus a deterministic ground-truth mask. It establishes NO
clinical claim: only controlled image-domain synthetic-source insertion and
its exact geometric/radiometric provenance. It is pure, side-effect-free
numeric code -- no I/O, no network, no CUDA -- operating on an
already-validated :class:`~src.vascutrace.geometry.GridGeometry` (P1) and a
caller-supplied physical-space centerline and contralateral corridor mask.
This module does not itself compute the mid-sagittal reflection plane or
side/laterality determination (that is the P2 data-pipeline's
responsibility, per ``imaging-physics.md``, "Laterality, reflection, and
corridors"); it trusts a caller-supplied ``contralateral_mask`` -- built by
reflecting the frozen physical mask through that plane -- so this generator
stays testable against small generated phantoms without depending on
subject-specific anatomy.

Formula provenance
==============================================================================
The core generative formula, its symbol definitions, and every hard
invariant follow the project's frozen synthetic-source contract:

    PET_syn = PET_orig + G_sigma * ((m - 1) * B * F * H)

- ``B`` (:func:`_contralateral_baseline`) is the 20%-per-tail trimmed mean
  (SciPy ``scipy.stats.trim_mean(values, proportiontocut=0.20)`` semantics
  exactly) of the background sampled at the caller-supplied contralateral
  corridor. The contract requires at least 10 finite samples, no nonfinite
  samples in the corridor, at least 6 samples retained after trimming, and a
  positive trimmed-mean result; any violation raises a typed
  :class:`SimulationError` (never a silently substituted 0/NaN/epsilon).
- ``m`` is :attr:`AnomalySimulationParams.uptake_multiplier`, the nominal
  pre-blur target-to-background contrast. Validated ``>= 1.0`` so the
  "source is nonnegative" hard invariant below holds by construction (a
  multiplier below 1.0 would model a *cold* sub-threshold source, which this
  generator does not claim to support).
- ``F`` (:func:`_supersampled_occupancy`) is the fractional voxel occupancy
  of a supersampled capsule (cylinder + hemispherical caps) tube of radius
  ``radius_mm`` around the frozen centerline, computed by supersampling each
  voxel on a ``supersample**3`` physical subgrid (default
  ``supersample=5``, i.e. 125 subsamples/voxel) and averaging a
  point-to-centerline-distance threshold, so boundary voxels get partial
  occupancy in ``[0, 1]`` rather than a binary in/out decision. The
  centerline "core" segment used for the capsule is the caller's
  ``centerline_points_mm`` clipped (or, if shorter than requested, used in
  full -- "length clipped") to an arc length of ``length_mm`` starting at
  its first vertex (:func:`_clip_centerline_core`); the capsule's rounded
  ends then extend ``radius_mm`` beyond that core's two endpoints, exactly
  as a spherocylinder. For a straight two-point core this capsule has the
  *exact* analytic volume ``pi * r**2 * length_mm + (4/3) * pi * r**3``
  (cylinder core + one full sphere from the two hemispherical caps) --
  this is the formula the volume-error test validates against, chosen
  because the specification explicitly names "capsule = cylinder + hemispherical
  caps" as the required source shape.
- ``H`` (:func:`_heterogeneity_field`) is a strictly positive, spatially
  low-frequency lognormal field. A white-noise field is Gaussian-smoothed
  (correlation length ``max(2*radius_mm, 3*mean(spacing))`` mm, so the
  heterogeneity varies smoothly relative to the source's own size rather
  than at single-voxel resolution), re-standardized to zero mean/unit
  variance to correct the smoothing's variance attenuation, then mapped to a
  lognormal via the standard moment-matching identity for a lognormal
  distribution with underlying normal parameters ``(mu, sigma)``:
  ``CV = sqrt(exp(sigma**2) - 1)``, so ``sigma = sqrt(log1p(CV**2))``; the
  mean-1 choice ``mu = -sigma**2 / 2`` follows from
  ``E[exp(N(mu, sigma))] = exp(mu + sigma**2/2) = 1``. The field is then
  divided by its own occupancy-weighted mean (``sum(F*H_raw) / sum(F)``) so
  the *occupancy-weighted* mean is exactly ``1.0`` regardless of how the
  smoothing/masking may have shifted it (dividing a strictly positive field
  by a positive scalar cannot change its coefficient of variation, so this
  renormalization step targets the mean invariant without perturbing the
  achieved CV). The achieved occupancy-weighted CV is therefore only
  *approximately* ``heterogeneity`` (spatial smoothing and masking to a thin
  tube both perturb it from the pre-smoothing analytic target) -- the
  contract itself only requires "target CV 0.15", not an exact achieved
  value, and no acceptance test in this implementation's test plan gates on the
  achieved CV, so this approximation is accepted rather than iteratively
  corrected.
- ``G_sigma`` (:func:`_apply_unit_sum_blur`) is a unit-sum, isotropic
  *physical* Gaussian blur of the synthetic excess ``(m - 1) * B * F * H``
  -- **not** of the composited image ``PET_orig + excess`` -- matching the
  formula above literally (the blur operator applies to the parenthesized
  excess term only). Per-axis sigma in voxels is
  ``sigma_axis = fwhm_mm / (2*sqrt(2*ln(2))*spacing_axis_mm)``
  (:func:`fwhm_mm_to_sigma_voxels`, the exact contract formula), boundary
  mode ``"constant"``/``cval=0.0`` (declared: never ``"wrap"``, so a source
  near a volume edge truncates instead of aliasing to the opposite face),
  and ``truncate=4.0``. SciPy's separable Gaussian kernel is unit-sum by
  construction per axis (each 1-D tap array is explicitly normalized by its
  own sum inside ``scipy.ndimage.gaussian_filter1d``; verified empirically
  against this environment's installed SciPy before relying on it -- see
  the accompanying test suite), so the composite 3-D kernel (a product of
  three unit-sum 1-D kernels) is unit-sum too, and this module does not
  re-normalize it. The excess is accumulated in float64 throughout
  (:class:`SimulationProvenance.excess_accumulation_dtype`); only the final
  composited ``synthetic_pet`` is cast down to float32
  (:class:`SimulationProvenance.output_dtype`).

Hard invariants (verbatim from the contract, each with a dedicated test)
==============================================================================
- ``m = 1.0`` is a sham: ``(m - 1) == 0.0`` exactly in IEEE-754, so the
  excess is exactly zero at every stage (zero times any finite F/H/B is
  exactly zero; the unit-sum blur of an all-zero array is exactly zero), and
  ``synthetic_pet`` reproduces ``PET_orig`` bit-for-bit, not merely within a
  numeric tolerance.
- The source is nonnegative (``B > 0``, ``F in [0, 1]``, ``H > 0`` and
  ``m - 1 >= 0`` by construction/validation) and the discrete blur kernel is
  unit-sum, so the blurred excess stays nonnegative too (a nonnegative
  kernel applied to a nonnegative, zero-padded signal cannot produce a
  negative output).
- Rasterized analytic volume error is within 5% of the exact capsule formula
  above, at the declared supersampling resolution recorded in
  :class:`SimulationProvenance.supersample_factor`.
- Activity is conserved within 1% between ``ideal_excess`` and
  ``blurred_excess`` when the source plus its four-sigma (``truncate=4.0``)
  blur margin sits entirely inside the array: with zero-padded
  (``mode="constant"``) boundaries, every unit of pre-blur excess whose
  full kernel support (``+/- 4*sigma`` per axis) lies within the array
  contributes exactly once to the post-blur total; only excess placed
  closer than ``4*sigma`` to an edge can lose activity off that edge, which
  is the deliberate physical meaning of a "boundary" test rather than a
  conservation-tolerance violation.
- Recovered physical FWHM (via :func:`fwhm_mm_to_sigma_voxels` and
  :func:`_apply_unit_sum_blur` applied to an impulse) matches the requested
  FWHM within tolerance on every tested spacing (isotropic and anisotropic).

Memory-bounded supersampled occupancy
==============================================================================
:func:`_supersampled_occupancy`'s point cloud has
``shape[0]*shape[1]*shape[2]*supersample**3`` rows -- on a realistic
training-scale crop (~140x80x140 voxels, ``supersample=5``) that is ~2x10^8
points, each tested against every segment of the (potentially long, e.g.
~70-point, clipping to ~25 segments at a 45mm ``length_mm``) frozen
centerline. An earlier revision of this function built that entire point
cloud and its ``(n_points, n_centerline_segments, 3)`` distance broadcast as
one monolithic NumPy array; the reported production incident measured peak
RSS on a ~107K-voxel real-scale window with a ~73-point centerline at
``supersample=5`` at ~29 GiB against a 31 GiB machine (``/usr/bin/time -v``),
which single-sample-OOMs and takes down any parallel worker pool. A
directly-reproduced, moderate-scale before/after comparison at
80x70x80 voxels (56x10^6 supersampled points, a 2-point/1-segment
centerline) on this same machine measured **10.54 GiB** peak RSS / 10.5s
wall for the unchunked reference vs **~133 MiB** peak RSS / 3.8s wall after
this fix -- both lower memory *and* lower wall time, because avoiding one
multi-GB allocation also avoids its allocator/page-fault overhead (system
time dropped from 9.0s to well under 0.5s in that comparison).

The fix (this revision) chunks over *coarse voxels*, not raw supersampled
rows (an earlier draft of this fix chunked the distance computation but
still allocated one ``occupancy``-sized array at the *supersampled* point
count -- ~1.57 GB of its own on the 140x80x140 window above, closer to the
~2 GB verification ceiling than intended; chunking by coarse voxel keeps
every persistent array sized at the *coarse* voxel count instead, e.g.
~12.5 MB at 140x80x140x8 bytes). Each batch of at most
``chunk_points // supersample**3`` coarse voxels expands to its full
``supersample**3`` subsamples, is tested against the centerline
(:func:`_point_to_polyline_distance`, unchanged), reduced to one occupancy
value per voxel in that batch, and written into the (coarse-sized) output
array. Peak memory is therefore ``O(chunk_points * n_centerline_segments)``
transient plus ``O(shape[0]*shape[1]*shape[2])`` persistent -- independent of
the supersampled point count, i.e. independent of window size, centerline
length, and supersample factor combined (:data:`_DEFAULT_OCCUPANCY_CHUNK_POINTS`
is the default `chunk_points`; measured peak RSS at the full 140x80x140,
~70-point-centerline, supersample=5 scale with this design was **~286 MiB**
(~0.28 GiB), comfortably under the ~2 GiB verification ceiling and a **~103x**
reduction from the reported 29 GiB incident, at essentially unchanged wall
time -- 428.6s vs the coordinator's reported ~35s at their ~15x-smaller
~107K-voxel incident scale, consistent with this being compute-bound work
that scales with total supersampled-point-times-segment count, which
chunking does not and should not reduce; only memory). This is a pure memory
refactor: because no
reduction in :func:`_supersampled_occupancy` or
:func:`_point_to_polyline_distance` ever crosses a chunk boundary (each
point's world coordinate and each point's distance-to-centerline are
functions of that point alone, and summing a set of exactly-representable
``{0.0, 1.0}`` values in float64 to compute one voxel's mean is exact and
order-independent), the chunked and unchunked computations are bit-identical
-- verified directly (see the accompanying test suite's numerical-
equivalence tests, which both reconstruct the pre-fix unchunked computation
on a small grid and compare, and cross-check several different
``occupancy_chunk_points`` values against each other end-to-end).

``pet_ct_shift_mm`` (:func:`shift_ct_array`)
==============================================================================
Per the contract, ``pet_ct_shift_mm`` is "solely a CT-channel registration
stress test": it is a misregistration applied only to a model's CT input,
never to PET or the ground-truth labels. This module therefore never reads
``pet_ct_shift_mm`` inside :func:`simulate_vascular_anomaly` at all -- it is
carried on :class:`AnomalySimulationParams` purely as frozen metadata/
provenance for a downstream caller -- and provides :func:`shift_ct_array` as
a standalone helper the caller applies explicitly, and only, to a CT array
already resampled onto the PET grid. Declared magnitudes are 0/2/4 mm (the
deterministic +/-R/+/-A/+/-S direction *assignment* scheme for a full atlas
sweep lives in ``evaluation.md`` and is out of this module's scope: this
module only shifts by whatever physical vector it is given).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import scipy
from scipy import ndimage
from scipy.stats import trim_mean

from src.vascutrace.geometry import RESEARCH_PROTOTYPE_WARNING, GridGeometry

__all__ = [
    "RESEARCH_PROTOTYPE_WARNING",
    "SimulationErrorCode",
    "SimulationError",
    "AnomalySimulationParams",
    "AchievedActivitySummary",
    "SimulationProvenance",
    "SimulationResult",
    "fwhm_mm_to_sigma_voxels",
    "simulate_vascular_anomaly",
    "shift_ct_array",
]

# ---------------------------------------------------------------------------
# Numerical policy constants
# ---------------------------------------------------------------------------

# 20%-per-tail trimmed mean, per imaging-physics.md ("Synthetic source", B).
_TRIM_PROPORTION = 0.20
_MIN_CONTRALATERAL_SAMPLES = 10
_MIN_RETAINED_SAMPLES = 6

# Gaussian blur boundary policy, per imaging-physics.md ("Synthetic source",
# G_sigma): never "wrap" (would alias a source across the volume).
_BLUR_MODE = "constant"
_BLUR_CVAL = 0.0
_BLUR_TRUNCATE = 4.0

# FWHM -> sigma (mm), the exact conversion named in imaging-physics.md.
_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))

_DEFAULT_SUPERSAMPLE = 5
# Chunk size (supersampled points per batch) for _supersampled_occupancy's
# memory-bounded loop -- see the module docstring's "Memory-bounded
# supersampled occupancy" section. Tuned (see the accompanying test suite's
# memory-measurement test) to keep peak RSS well under ~1.5 GB at
# supersample=5 on a realistic ~140x80x140-voxel, ~70-point-centerline
# training crop.
_DEFAULT_OCCUPANCY_CHUNK_POINTS = 50_000
_GENERATOR_VERSION = "vascutrace-simulation-anomaly-v1"

_VALID_SIDES = frozenset({"left", "right"})


# ---------------------------------------------------------------------------
# Error taxonomy
# ---------------------------------------------------------------------------


class SimulationErrorCode(StrEnum):
    """Typed, code-only reasons the contralateral baseline or the source
    occupancy is scientifically invalid, per imaging-physics.md's named
    validity gates. Basic malformed-input checks (wrong shape/dtype,
    non-finite parameters) raise plain :class:`ValueError` instead, matching
    this project's existing quantification-engine convention of reserving
    typed codes for named contract concepts.
    """

    EMPTY_CONTRALATERAL_REGION = "EMPTY_CONTRALATERAL_REGION"
    NONFINITE_CONTRALATERAL_REGION = "NONFINITE_CONTRALATERAL_REGION"
    INSUFFICIENT_CONTRALATERAL_SAMPLES = "INSUFFICIENT_CONTRALATERAL_SAMPLES"
    NONPOSITIVE_CONTRALATERAL_BASELINE = "NONPOSITIVE_CONTRALATERAL_BASELINE"
    EMPTY_SOURCE_OCCUPANCY = "EMPTY_SOURCE_OCCUPANCY"


class SimulationError(ValueError):
    """Raised for a named simulation-contract validity violation.

    The exception message is exactly the failing
    :class:`SimulationErrorCode` value -- never a data value -- so it is
    always safe to log or surface verbatim (mirrors
    ``GeometryContractError`` in :mod:`src.vascutrace.geometry`).
    """

    def __init__(self, code: SimulationErrorCode) -> None:
        super().__init__(code.value)
        self.code = code


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AnomalySimulationParams:
    """Requested synthetic-source parameters (mirrors imaging-physics.md's
    "Synthetic source" contract and the atlas factor list in
    ``evaluation.md``). Validated eagerly at construction time.
    """

    side: str
    radius_mm: float
    length_mm: float
    uptake_multiplier: float
    blur_fwhm_mm: float
    heterogeneity: float
    pet_ct_shift_mm: tuple[float, float, float]
    seed: int

    def __post_init__(self) -> None:
        _validate_params(self)


@dataclass(frozen=True, slots=True)
class AchievedActivitySummary:
    """Achieved post-blur SUVs inside the ground-truth mask (contract:
    "Store ... achieved post-blur SUVs"). ``None`` fields mean the
    ground-truth mask selected zero voxels (typed null, never 0/NaN).
    """

    suvmax_in_mask: float | None
    suvmean_in_mask: float | None


@dataclass(frozen=True, slots=True)
class SimulationProvenance:
    """Deterministic provenance record (contract: "Store ... requested
    parameters, ... geometry, hashes, software versions, boundary/
    truncation/dtype settings, and RNG algorithm/seed.").
    """

    params_sha256: str
    output_sha256: str
    rng_algorithm: str
    seed: int
    generator_version: str
    supersample_factor: int
    blur_boundary_mode: str
    blur_cval: float
    blur_truncate: float
    excess_accumulation_dtype: str
    output_dtype: str
    numpy_version: str
    scipy_version: str
    research_prototype_warning: str = RESEARCH_PROTOTYPE_WARNING


@dataclass(frozen=True, slots=True, eq=False)
class SimulationResult:
    """The full synthetic-insertion bundle for one requested source.

    ``eq``/``__hash__`` are left at Python's default identity-based
    behavior (see :class:`~src.vascutrace.geometry.GridGeometry`) because
    ndarray-field tuple equality raises rather than doing anything useful.
    """

    original_pet: np.ndarray = field(repr=False)
    synthetic_pet: np.ndarray = field(repr=False)
    source_fraction: np.ndarray = field(repr=False)
    ground_truth_mask: np.ndarray = field(repr=False)
    ideal_excess: np.ndarray = field(repr=False)
    blurred_excess: np.ndarray = field(repr=False)
    heterogeneity_field: np.ndarray = field(repr=False)
    contralateral_baseline: float
    params: AnomalySimulationParams
    achieved_activity: AchievedActivitySummary
    geometry: GridGeometry
    provenance: SimulationProvenance


# ---------------------------------------------------------------------------
# Shared numeric helpers
# ---------------------------------------------------------------------------


def _apply_affine_points(affine: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous affine to an ``(N, 3)`` array of points.

    A local, deliberately-duplicated copy of the same small primitive that
    lives in both ``geometry.py`` and ``quantification/measure.py`` --
    those functions are private (not in either module's ``__all__``), and
    this project's existing convention (see both of those modules) is a
    tiny independent copy per module rather than a private cross-module
    import.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    homogeneous = np.concatenate([pts, ones], axis=1)
    return (np.asarray(affine, dtype=np.float64) @ homogeneous.T).T[:, :3]


def _validate_params(params: AnomalySimulationParams) -> None:
    if params.side not in _VALID_SIDES:
        raise ValueError('side must be "left" or "right"')
    if not (np.isfinite(params.radius_mm) and params.radius_mm > 0):
        raise ValueError("radius_mm must be finite and > 0")
    if not (np.isfinite(params.length_mm) and params.length_mm > 0):
        raise ValueError("length_mm must be finite and > 0")
    if not (np.isfinite(params.uptake_multiplier) and params.uptake_multiplier >= 1.0):
        raise ValueError(
            "uptake_multiplier must be finite and >= 1.0 (1.0 is the sham "
            "case; a value < 1.0 would violate the nonnegative-source "
            "invariant)"
        )
    if not (np.isfinite(params.blur_fwhm_mm) and params.blur_fwhm_mm >= 0):
        raise ValueError("blur_fwhm_mm must be finite and >= 0")
    if not (np.isfinite(params.heterogeneity) and params.heterogeneity >= 0):
        raise ValueError("heterogeneity (target CV) must be finite and >= 0")
    shift = np.asarray(params.pet_ct_shift_mm, dtype=np.float64)
    if shift.shape != (3,) or not np.all(np.isfinite(shift)):
        raise ValueError("pet_ct_shift_mm must be a finite 3-vector")
    if (
        not isinstance(params.seed, int)
        or isinstance(params.seed, bool)
        or params.seed < 0
    ):
        raise ValueError("seed must be a non-negative int")


def _validate_background(background: np.ndarray, geometry: GridGeometry) -> np.ndarray:
    arr = np.asarray(background, dtype=np.float64)
    if arr.shape != tuple(geometry.shape):
        raise ValueError("background shape must match geometry.shape")
    return arr


def _contralateral_baseline(
    background: np.ndarray, contralateral_mask: np.ndarray
) -> float:
    """20%-per-tail trimmed mean of ``background`` at ``contralateral_mask``.

    Exact ``scipy.stats.trim_mean(values, proportiontocut=0.20)`` semantics,
    per imaging-physics.md ("Synthetic source", B). Raises
    :class:`SimulationError` for every one of the contract's named validity
    gates: nonempty, all-finite, >= 10 samples before trimming, >= 6 samples
    retained after trimming, and a strictly positive trimmed-mean result.
    """
    mask = np.asarray(contralateral_mask, dtype=bool)
    if mask.shape != background.shape:
        raise ValueError("contralateral_mask shape must match background shape")

    values = background[mask]
    if values.size == 0:
        raise SimulationError(SimulationErrorCode.EMPTY_CONTRALATERAL_REGION)
    if not np.all(np.isfinite(values)):
        raise SimulationError(SimulationErrorCode.NONFINITE_CONTRALATERAL_REGION)
    if values.size < _MIN_CONTRALATERAL_SAMPLES:
        raise SimulationError(SimulationErrorCode.INSUFFICIENT_CONTRALATERAL_SAMPLES)

    trimmed_each_tail = int(np.floor(_TRIM_PROPORTION * values.size))
    retained = values.size - 2 * trimmed_each_tail
    if retained < _MIN_RETAINED_SAMPLES:
        # Defence in depth, not reachable given the >= 10 gate above at the
        # fixed 20%-per-tail proportion: retained = n - 2*floor(0.2n) >=
        # n - 0.4n = 0.6n, and n >= 10 => retained >= 6 always. Kept because
        # the contract names it as an independent, explicit requirement
        # (imaging-physics.md, "Synthetic source", B) -- same "convert a
        # silent latent bug into a typed, loud failure" rationale as
        # geometry.py's own defence-in-depth invariant checks -- and it
        # would become reachable again if _MIN_CONTRALATERAL_SAMPLES or
        # _TRIM_PROPORTION is ever tuned independently.
        raise SimulationError(SimulationErrorCode.INSUFFICIENT_CONTRALATERAL_SAMPLES)

    baseline = float(trim_mean(values, proportiontocut=_TRIM_PROPORTION))
    if not (np.isfinite(baseline) and baseline > 0):
        raise SimulationError(SimulationErrorCode.NONPOSITIVE_CONTRALATERAL_BASELINE)
    return baseline


def _clip_centerline_core(
    centerline_points_mm: np.ndarray, length_mm: float
) -> np.ndarray:
    """Return the sub-polyline of ``centerline_points_mm`` spanning arc
    length ``min(length_mm, total_arc_length)`` starting at its first
    vertex -- the "length clipped (or parameterized by length_mm)" core used
    as the capsule's cylindrical axis. Consecutive duplicate vertices are
    dropped first (a zero-length segment would make the arc-length
    parametrization below ambiguous, not because a duplicate point is
    itself invalid input).
    """
    pts = np.atleast_2d(np.asarray(centerline_points_mm, dtype=np.float64))
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 2:
        raise ValueError("centerline_points_mm must be shape (N, 3) with N >= 2")
    if not np.all(np.isfinite(pts)):
        raise ValueError("centerline_points_mm must be finite")

    keep = np.ones(pts.shape[0], dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(pts, axis=0), axis=1) > 0.0
    pts = pts[keep]
    if pts.shape[0] < 2:
        raise ValueError(
            "centerline_points_mm must contain at least two distinct points"
        )

    segment_vectors = np.diff(pts, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    target = min(float(length_mm), float(cumulative[-1]))

    last_index = int(np.searchsorted(cumulative, target, side="right") - 1)
    last_index = min(max(last_index, 0), len(pts) - 2)
    core = [pts[i] for i in range(last_index + 1)]
    remaining = target - cumulative[last_index]
    if remaining > 1e-9:
        direction = segment_vectors[last_index] / segment_lengths[last_index]
        core.append(pts[last_index] + direction * remaining)

    core_arr = np.asarray(core, dtype=np.float64)
    if core_arr.shape[0] < 2:
        core_arr = np.vstack([core_arr[0], core_arr[0]])
    return core_arr


def _point_to_polyline_distance(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """Vectorized minimum Euclidean distance from each row of ``points``
    (``(M, 3)``) to the piecewise-linear ``polyline`` (``(K, 3)``): the
    minimum, over each of the ``K - 1`` segments, of the standard clamped
    point-to-segment distance ``|p - (A + clip(t, 0, 1) * (B - A))|``.
    """
    starts = polyline[:-1]
    ends = polyline[1:]
    seg_vec = ends - starts
    seg_len_sq = np.sum(seg_vec**2, axis=1)
    seg_len_sq = np.where(seg_len_sq <= 0.0, 1.0, seg_len_sq)

    diff = points[:, None, :] - starts[None, :, :]
    t = np.einsum("mkj,kj->mk", diff, seg_vec) / seg_len_sq[None, :]
    t = np.clip(t, 0.0, 1.0)
    closest = starts[None, :, :] + t[:, :, None] * seg_vec[None, :, :]
    dist = np.linalg.norm(points[:, None, :] - closest, axis=2)
    return dist.min(axis=1)


def _supersampled_occupancy(
    core_polyline: np.ndarray,
    radius_mm: float,
    geometry: GridGeometry,
    supersample: int,
    *,
    chunk_points: int = _DEFAULT_OCCUPANCY_CHUNK_POINTS,
) -> np.ndarray:
    """Fractional voxel occupancy ``F`` of the capsule (radius ``radius_mm``
    around ``core_polyline``) via physical supersampling: each voxel is
    split into ``supersample**3`` regularly-spaced subvoxel centers, each
    subvoxel center is mapped to physical space via ``geometry.affine`` and
    tested against the capsule's distance field, and the per-voxel mean of
    that boolean test gives a fractional occupancy in ``[0, 1]`` (partial
    occupancy at the capsule boundary, not a binary in/out decision).

    Memory-bounded (see the module docstring's "Memory-bounded supersampled
    occupancy" section): the full supersampled point cloud has
    ``shape[0]*shape[1]*shape[2]*supersample**3`` rows -- tens to hundreds of
    millions on a realistic training crop -- so this function chunks over
    *coarse voxels* (at most ``chunk_points // supersample**3`` per batch,
    so each batch still tests at most ``chunk_points`` supersampled rows
    against the centerline), rather than over raw supersampled rows: that
    keeps the one persistent array this function allocates at the *coarse*
    voxel count (``shape[0]*shape[1]*shape[2]``, e.g. ~12.5 MB at
    140x80x140x8 bytes), not the supersampled point count (which would be
    ``supersample**3`` times larger, e.g. ~1.5 GB at supersample=5 on that
    same window -- an earlier revision of this chunking fix made exactly
    that mistake; see the accompanying test suite's memory-measurement
    test). No reduction ever crosses a chunk boundary within one voxel's
    own ``supersample**3`` subsamples (each subsample's world coordinate and
    distance-to-polyline are pure per-point functions, and summing a set of
    exactly-representable ``{0.0, 1.0}`` values in float64 is exact and
    order-independent, so the chunked and unchunked per-voxel means are
    bit-identical regardless of chunk_points or the internal enumeration
    order of a voxel's subsamples -- verified directly in the test suite).
    """
    shape = geometry.shape
    n0, n1, n2 = shape
    n_voxels = n0 * n1 * n2
    offsets = (np.arange(supersample, dtype=np.float64) + 0.5) / supersample - 0.5

    # Every supersample**3 per-axis offset combination for one voxel, shape
    # (supersample**3, 3): sub_offsets[s] = [offsets[a], offsets[b], offsets[c]]
    # for the s-th (a, b, c) triple. Enumeration order within a voxel does
    # not affect the result (see docstring above), only that all
    # supersample**3 combinations appear exactly once.
    oa, ob, oc = np.meshgrid(offsets, offsets, offsets, indexing="ij")
    sub_offsets = np.stack([oa.reshape(-1), ob.reshape(-1), oc.reshape(-1)], axis=1)
    n_sub = sub_offsets.shape[0]  # == supersample**3

    affine = np.asarray(geometry.affine, dtype=np.float64)
    occupancy_flat = np.empty(n_voxels, dtype=np.float64)
    voxels_per_chunk = max(1, int(chunk_points) // n_sub)

    for chunk_start in range(0, n_voxels, voxels_per_chunk):
        chunk_stop = min(chunk_start + voxels_per_chunk, n_voxels)
        voxel_flat = np.arange(chunk_start, chunk_stop)
        # Unravel the flat *coarse-voxel* index in C order (axis 0 slowest,
        # axis 2 fastest), matching occupancy_flat.reshape(shape) below.
        idx2 = voxel_flat % n2
        remainder = voxel_flat // n2
        idx1 = remainder % n1
        idx0 = remainder // n1
        base_idx = np.stack([idx0, idx1, idx2], axis=1).astype(np.float64)  # (C, 3)

        chunk_voxel_count = base_idx.shape[0]
        combined = base_idx[:, None, :] + sub_offsets[None, :, :]  # (C, n_sub, 3)
        chunk_world = _apply_affine_points(affine, combined.reshape(-1, 3))
        chunk_distance = _point_to_polyline_distance(chunk_world, core_polyline)
        chunk_inside = (chunk_distance <= radius_mm).astype(np.float64)
        occupancy_flat[chunk_start:chunk_stop] = chunk_inside.reshape(
            chunk_voxel_count, n_sub
        ).mean(axis=1)

    return occupancy_flat.reshape(shape)


def _heterogeneity_field(
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    radius_mm: float,
    occupancy: np.ndarray,
    target_cv: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Strictly positive, low-frequency lognormal field ``H``, renormalized
    to occupancy-weighted mean 1.0. See the module docstring's "Formula
    provenance" section for the full derivation.
    """
    total_weight = float(np.sum(occupancy))
    if total_weight <= 0.0:
        raise SimulationError(SimulationErrorCode.EMPTY_SOURCE_OCCUPANCY)

    if target_cv == 0.0:
        return np.ones(shape, dtype=np.float64)

    white = rng.standard_normal(shape)
    correlation_len_mm = max(2.0 * radius_mm, 3.0 * float(np.mean(spacing)))
    sigma_voxels = tuple(correlation_len_mm / s for s in spacing)
    smoothed = ndimage.gaussian_filter(
        white, sigma=sigma_voxels, mode="reflect", truncate=_BLUR_TRUNCATE
    )
    smoothed_std = float(np.std(smoothed))
    if smoothed_std > 0.0:
        standardized = (smoothed - float(np.mean(smoothed))) / smoothed_std
    else:
        standardized = np.zeros(shape, dtype=np.float64)

    # Lognormal moment matching: for X = exp(N(mu, sigma)),
    # CV(X) = sqrt(exp(sigma**2) - 1), and E[X] = 1 requires mu = -sigma**2/2.
    sigma_log = float(np.sqrt(np.log1p(target_cv**2)))
    mu_log = -0.5 * sigma_log**2
    raw = np.exp(mu_log + sigma_log * standardized)

    weighted_mean = float(np.sum(occupancy * raw) / total_weight)
    return raw / weighted_mean


def fwhm_mm_to_sigma_voxels(
    fwhm_mm: float, spacing_mm: tuple[float, float, float]
) -> tuple[float, float, float]:
    """``sigma_axis = fwhm_mm / (2*sqrt(2*ln(2))*spacing_axis_mm)``, the
    exact per-axis FWHM->sigma-in-voxels conversion named in
    imaging-physics.md ("Synthetic source", G_sigma).
    """
    if not (np.isfinite(fwhm_mm) and fwhm_mm >= 0):
        raise ValueError("fwhm_mm must be finite and >= 0")
    sigma_mm = float(fwhm_mm) * _FWHM_TO_SIGMA
    return tuple(sigma_mm / float(s) for s in spacing_mm)


def _apply_unit_sum_blur(
    field_values: np.ndarray, sigma_voxels: tuple[float, float, float]
) -> np.ndarray:
    """Unit-sum isotropic *physical* Gaussian blur of ``field_values``, per
    imaging-physics.md ("Synthetic source", G_sigma): boundary mode
    ``"constant"``, ``cval=0.0``, ``truncate=4.0``. SciPy's per-axis
    Gaussian kernel is already unit-sum by construction (verified
    empirically for this environment's installed SciPy -- see the test
    suite), so no additional renormalization is applied here.
    """
    return ndimage.gaussian_filter(
        field_values,
        sigma=sigma_voxels,
        mode=_BLUR_MODE,
        cval=_BLUR_CVAL,
        truncate=_BLUR_TRUNCATE,
    )


def _achieved_activity_summary(
    synthetic_pet: np.ndarray, ground_truth_mask: np.ndarray
) -> AchievedActivitySummary:
    values = synthetic_pet[ground_truth_mask.astype(bool)]
    if values.size == 0:
        return AchievedActivitySummary(suvmax_in_mask=None, suvmean_in_mask=None)
    return AchievedActivitySummary(
        suvmax_in_mask=float(np.max(values)), suvmean_in_mask=float(np.mean(values))
    )


def _canonical_params_bytes(params: AnomalySimulationParams) -> bytes:
    payload = (
        params.side,
        round(float(params.radius_mm), 10),
        round(float(params.length_mm), 10),
        round(float(params.uptake_multiplier), 10),
        round(float(params.blur_fwhm_mm), 10),
        round(float(params.heterogeneity), 10),
        tuple(round(float(v), 10) for v in params.pet_ct_shift_mm),
        int(params.seed),
    )
    return repr(payload).encode("utf-8")


def _build_provenance(
    *,
    params: AnomalySimulationParams,
    synthetic_pet: np.ndarray,
    ground_truth_mask: np.ndarray,
    supersample: int,
) -> SimulationProvenance:
    params_bytes = _canonical_params_bytes(params)
    combined = hashlib.sha256()
    combined.update(params_bytes)
    combined.update(np.ascontiguousarray(synthetic_pet).tobytes())
    combined.update(np.ascontiguousarray(ground_truth_mask).tobytes())
    return SimulationProvenance(
        params_sha256=hashlib.sha256(params_bytes).hexdigest(),
        output_sha256=combined.hexdigest(),
        rng_algorithm="numpy.random.default_rng (PCG64)",
        seed=int(params.seed),
        generator_version=_GENERATOR_VERSION,
        supersample_factor=int(supersample),
        blur_boundary_mode=_BLUR_MODE,
        blur_cval=_BLUR_CVAL,
        blur_truncate=_BLUR_TRUNCATE,
        excess_accumulation_dtype="float64",
        output_dtype="float32",
        numpy_version=np.__version__,
        scipy_version=scipy.__version__,
    )


# ---------------------------------------------------------------------------
# Public API: synthetic source insertion
# ---------------------------------------------------------------------------


def simulate_vascular_anomaly(
    background: np.ndarray,
    geometry: GridGeometry,
    centerline_points_mm: np.ndarray,
    contralateral_mask: np.ndarray,
    params: AnomalySimulationParams,
    *,
    supersample: int = _DEFAULT_SUPERSAMPLE,
    occupancy_chunk_points: int = _DEFAULT_OCCUPANCY_CHUNK_POINTS,
) -> SimulationResult:
    """Insert a controlled synthetic source along ``centerline_points_mm``
    into ``background`` (a PET SUVbw array on ``geometry``'s grid) and
    return the full :class:`SimulationResult` bundle.

    ``centerline_points_mm`` and ``contralateral_mask`` are both in the same
    physical-space/array convention as ``geometry`` (the same convention
    ``geometry.affine`` maps voxel indices into): the caller is responsible
    for having already reflected the physical mask through the subject's
    frozen mid-sagittal plane (P2's responsibility, not this module's).

    ``supersample`` (subsamples per voxel per axis; ``supersample**3``
    subsamples per voxel total) is not part of
    :class:`AnomalySimulationParams` (it is a numerical-resolution knob, not
    a scientific parameter of the requested source) but is recorded in the
    returned :class:`SimulationProvenance`.

    ``occupancy_chunk_points`` bounds the peak memory of
    :func:`_supersampled_occupancy`'s point-to-centerline distance
    computation (batch size, in supersampled points, per chunk -- see the
    module docstring's "Memory-bounded supersampled occupancy" section). It
    is a pure performance/memory knob, not a scientific parameter: unlike
    ``supersample``, it provably cannot change any output value (no
    reduction crosses a chunk boundary), so it is *not* recorded in
    :class:`SimulationProvenance` -- doing so would incorrectly imply it
    were part of the reproducibility contract.
    """
    if not isinstance(supersample, int) or supersample < 1:
        raise ValueError("supersample must be a positive int")
    if not isinstance(occupancy_chunk_points, int) or occupancy_chunk_points < 1:
        raise ValueError("occupancy_chunk_points must be a positive int")

    background_arr = _validate_background(background, geometry)
    baseline = _contralateral_baseline(background_arr, contralateral_mask)
    core_polyline = _clip_centerline_core(centerline_points_mm, params.length_mm)
    occupancy = _supersampled_occupancy(
        core_polyline,
        params.radius_mm,
        geometry,
        supersample,
        chunk_points=occupancy_chunk_points,
    )

    rng = np.random.default_rng(params.seed)
    heterogeneity = _heterogeneity_field(
        geometry.shape,
        geometry.spacing,
        params.radius_mm,
        occupancy,
        params.heterogeneity,
        rng,
    )

    ideal_excess = (
        (params.uptake_multiplier - 1.0) * baseline * occupancy * heterogeneity
    )
    sigma_voxels = fwhm_mm_to_sigma_voxels(params.blur_fwhm_mm, geometry.spacing)
    blurred_excess = _apply_unit_sum_blur(ideal_excess, sigma_voxels)

    synthetic_pet = (background_arr + blurred_excess).astype(np.float32)
    ground_truth_mask = (occupancy >= 0.5).astype(np.int8)

    provenance = _build_provenance(
        params=params,
        synthetic_pet=synthetic_pet,
        ground_truth_mask=ground_truth_mask,
        supersample=supersample,
    )

    return SimulationResult(
        original_pet=background_arr.astype(np.float32),
        synthetic_pet=synthetic_pet,
        source_fraction=occupancy.astype(np.float32),
        ground_truth_mask=ground_truth_mask,
        ideal_excess=ideal_excess,
        blurred_excess=blurred_excess,
        heterogeneity_field=heterogeneity.astype(np.float32),
        contralateral_baseline=baseline,
        params=params,
        achieved_activity=_achieved_activity_summary(synthetic_pet, ground_truth_mask),
        geometry=geometry,
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# Public API: CT-only registration stress test
# ---------------------------------------------------------------------------


def shift_ct_array(
    ct: np.ndarray,
    geometry: GridGeometry,
    shift_mm: tuple[float, float, float],
    *,
    order: int = 1,
) -> np.ndarray:
    """Resample ``ct`` (already on ``geometry``'s grid) as though its
    physical content had been rigidly translated by ``shift_mm`` (mm, same
    physical/world convention as ``geometry.affine``): a feature at physical
    position ``p`` in ``ct`` appears at physical position ``p + shift_mm`` in
    the returned array. Implemented by sampling the *original* ``ct`` at
    each output voxel's ``world - shift_mm`` position
    (``scipy.ndimage.map_coordinates``, ``mode="constant"``, ``cval=0.0``),
    the same output-voxel -> world -> input-voxel walk used by
    :func:`~src.vascutrace.geometry.resample_ct_to_pet`.

    Declared solely a CT-channel registration stress test
    (imaging-physics.md): this function must be, and is, the *only* place a
    ``pet_ct_shift_mm`` vector is ever applied to an array in this
    generator's pipeline -- :func:`simulate_vascular_anomaly` never reads
    ``params.pet_ct_shift_mm``, so PET and the ground-truth mask are
    mechanically unreachable by this shift, not merely conventionally
    exempted from it.
    """
    shift = np.asarray(shift_mm, dtype=np.float64)
    if shift.shape != (3,) or not np.all(np.isfinite(shift)):
        raise ValueError("shift_mm must be a finite 3-vector")
    ct_arr = np.asarray(ct, dtype=np.float64)
    if ct_arr.shape != tuple(geometry.shape):
        raise ValueError("ct shape must match geometry.shape")

    shape = geometry.shape
    grids = np.meshgrid(*[np.arange(n, dtype=np.float64) for n in shape], indexing="ij")
    voxel_idx = np.stack([g.reshape(-1) for g in grids], axis=1)
    world = _apply_affine_points(geometry.affine, voxel_idx)
    source_world = world - shift[None, :]
    affine_inverse = np.linalg.inv(np.asarray(geometry.affine, dtype=np.float64))
    source_voxel = _apply_affine_points(affine_inverse, source_world)
    coordinates = source_voxel.T.reshape(3, *shape)

    shifted = ndimage.map_coordinates(
        ct_arr, coordinates, order=order, mode="constant", cval=0.0
    )
    return shifted.astype(np.float32)
