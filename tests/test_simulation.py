"""Tests for the physics-informed synthetic vascular-uptake generator
(VascuTrace Phase 3, ``src/vascutrace/simulation/anomaly.py``).

Every fixture in this file is a small, generated, non-patient phantom built
directly with ``nibabel.Nifti1Image`` (routed through
:func:`src.vascutrace.geometry.validate_nifti_grid`, mirroring
``tests/test_geometry.py``'s own fixture convention) and plain NumPy arrays
-- nothing here reads ``Data/``, the network, CUDA, or any external service.

A few tests import and exercise the module's private numeric cores
(``_apply_unit_sum_blur``, ``_point_to_polyline_distance``,
``_clip_centerline_core``) directly, the same deliberate, documented choice
``test_geometry.py`` makes for ``_grid_geometry_from_affine``: these
functions are the exact code the public API runs, and testing them directly
makes it possible to isolate one physical property (e.g. FWHM recovery) from
the rest of the pipeline (contralateral baseline, occupancy, heterogeneity)
without those unrelated stages adding noise to the assertion.
"""

from __future__ import annotations

import resource
import time

import numpy as np
import nibabel as nib
import pytest

from src.vascutrace.geometry import GridGeometry, validate_nifti_grid
from src.vascutrace.simulation.anomaly import (
    RESEARCH_PROTOTYPE_WARNING,
    AnomalySimulationParams,
    SimulationError,
    SimulationErrorCode,
    fwhm_mm_to_sigma_voxels,
    shift_ct_array,
    simulate_vascular_anomaly,
    _apply_affine_points,
    _apply_unit_sum_blur,
    _clip_centerline_core,
    _point_to_polyline_distance,
    _supersampled_occupancy,
)

# ---------------------------------------------------------------------------
# Fixture helpers (generated phantoms only)
# ---------------------------------------------------------------------------


def _make_geometry(
    shape: tuple[int, int, int] = (30, 20, 20),
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> GridGeometry:
    affine = np.eye(4)
    affine[0, 0], affine[1, 1], affine[2, 2] = spacing
    affine[:3, 3] = origin
    data = np.zeros(shape, dtype=np.float32)
    img = nib.Nifti1Image(data, affine)
    img.header.set_qform(affine, code=1)
    img.header.set_sform(affine, code=1)
    img.header.set_xyzt_units(xyz="mm", t="sec")
    return validate_nifti_grid(img)


def _flat_background(
    shape: tuple[int, int, int], value: float = 1.0, noise: float = 0.05, seed: int = 0
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (value + noise * rng.standard_normal(shape)).astype(np.float32)


def _contralateral_mask(
    shape: tuple[int, int, int], x_slice: slice = slice(20, 28)
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[x_slice, :, :] = True
    return mask


def _params(**overrides: object) -> AnomalySimulationParams:
    defaults: dict[str, object] = dict(
        side="left",
        radius_mm=3.0,
        length_mm=15.0,
        uptake_multiplier=1.6,
        blur_fwhm_mm=4.0,
        heterogeneity=0.15,
        pet_ct_shift_mm=(0.0, 0.0, 0.0),
        seed=42,
    )
    defaults.update(overrides)
    return AnomalySimulationParams(**defaults)  # type: ignore[arg-type]


def _straight_centerline(
    start: tuple[float, float, float] = (5.0, 10.0, 10.0),
    end: tuple[float, float, float] = (25.0, 10.0, 10.0),
) -> np.ndarray:
    return np.array([start, end], dtype=np.float64)


def _curved_centerline(
    center: tuple[float, float, float] = (10.0, 10.0, 10.0),
    radius: float = 8.0,
    n_points: int = 6,
) -> np.ndarray:
    """A quarter-circle arc in the x/y plane, at fixed z -- a simple,
    exactly-reproducible curved polyline for the "curved centerline" phantom.
    """
    angles = np.linspace(0.0, np.pi / 2, n_points)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    z = np.full(n_points, center[2])
    return np.stack([x, y, z], axis=1)


# ---------------------------------------------------------------------------
# RESEARCH_PROTOTYPE_WARNING / error taxonomy
# ---------------------------------------------------------------------------


def test_research_prototype_warning_exact_text() -> None:
    assert RESEARCH_PROTOTYPE_WARNING == (
        "Research prototype. Trained and evaluated using simulated vascular-like "
        "abnormalities, not confirmed human post-angioplasty lesions."
    )


def test_simulation_error_message_is_code_only() -> None:
    err = SimulationError(SimulationErrorCode.EMPTY_CONTRALATERAL_REGION)
    assert err.code is SimulationErrorCode.EMPTY_CONTRALATERAL_REGION
    assert str(err) == "EMPTY_CONTRALATERAL_REGION"
    assert isinstance(err, ValueError)


def test_simulation_error_code_members() -> None:
    expected = {
        "EMPTY_CONTRALATERAL_REGION",
        "NONFINITE_CONTRALATERAL_REGION",
        "INSUFFICIENT_CONTRALATERAL_SAMPLES",
        "NONPOSITIVE_CONTRALATERAL_BASELINE",
        "EMPTY_SOURCE_OCCUPANCY",
    }
    assert {member.value for member in SimulationErrorCode} == expected


# ---------------------------------------------------------------------------
# Generated analytic phantoms: straight and curved centerlines
# ---------------------------------------------------------------------------


def test_straight_centerline_generated_phantom_produces_plausible_source() -> None:
    geometry = _make_geometry()
    background = _flat_background(geometry.shape)
    contralateral = _contralateral_mask(geometry.shape)
    result = simulate_vascular_anomaly(
        background, geometry, _straight_centerline(), contralateral, _params()
    )

    assert result.synthetic_pet.shape == geometry.shape
    assert result.synthetic_pet.dtype == np.float32
    assert result.ground_truth_mask.shape == geometry.shape
    assert result.ground_truth_mask.dtype == np.int8
    assert result.ground_truth_mask.sum() > 0
    assert set(np.unique(result.ground_truth_mask)) <= {0, 1}
    assert result.source_fraction.min() >= 0.0
    assert result.source_fraction.max() <= 1.0
    # Boundary voxels of the capsule get partial (non-binary) occupancy.
    assert np.any((result.source_fraction > 0.0) & (result.source_fraction < 1.0))


def test_curved_centerline_generated_phantom_produces_plausible_source() -> None:
    geometry = _make_geometry(shape=(30, 30, 20))
    background = _flat_background(geometry.shape)
    contralateral = _contralateral_mask(geometry.shape, x_slice=slice(0, 8))
    centerline = _curved_centerline(center=(15.0, 15.0, 10.0), radius=8.0)
    result = simulate_vascular_anomaly(
        background, geometry, centerline, contralateral, _params(length_mm=10.0)
    )

    assert result.synthetic_pet.shape == geometry.shape
    assert result.ground_truth_mask.sum() > 0
    assert np.all(np.isfinite(result.synthetic_pet))


# ---------------------------------------------------------------------------
# Seed determinism
# ---------------------------------------------------------------------------


def test_same_seed_is_bitwise_identical() -> None:
    geometry = _make_geometry()
    background = _flat_background(geometry.shape)
    contralateral = _contralateral_mask(geometry.shape)
    centerline = _straight_centerline()
    params = _params(seed=7)

    first = simulate_vascular_anomaly(
        background, geometry, centerline, contralateral, params
    )
    second = simulate_vascular_anomaly(
        background, geometry, centerline, contralateral, params
    )

    assert np.array_equal(first.synthetic_pet, second.synthetic_pet)
    assert np.array_equal(first.ground_truth_mask, second.ground_truth_mask)
    assert np.array_equal(first.heterogeneity_field, second.heterogeneity_field)
    assert first.provenance.output_sha256 == second.provenance.output_sha256


def test_different_seed_changes_heterogeneity_and_output() -> None:
    geometry = _make_geometry()
    background = _flat_background(geometry.shape)
    contralateral = _contralateral_mask(geometry.shape)
    centerline = _straight_centerline()

    result_a = simulate_vascular_anomaly(
        background, geometry, centerline, contralateral, _params(seed=1)
    )
    result_b = simulate_vascular_anomaly(
        background, geometry, centerline, contralateral, _params(seed=2)
    )

    assert not np.array_equal(
        result_a.heterogeneity_field, result_b.heterogeneity_field
    )
    assert not np.array_equal(result_a.synthetic_pet, result_b.synthetic_pet)
    assert result_a.provenance.output_sha256 != result_b.provenance.output_sha256
    # The geometric occupancy itself does not depend on the RNG seed.
    assert np.array_equal(result_a.source_fraction, result_b.source_fraction)


# ---------------------------------------------------------------------------
# Degenerate valid input: heterogeneity == 0.0 (the target_cv == 0.0 early
# return in _heterogeneity_field) -- every other test hardcodes 0.15.
# ---------------------------------------------------------------------------


def test_heterogeneity_zero_is_uniform_field_and_invariants_still_hold() -> None:
    """``heterogeneity=0.0`` is a valid (not erroneous) request: it must
    short-circuit to a uniform field of *exactly* 1.0 (not merely close to
    1.0), and the sham and activity-conservation hard invariants -- already
    proven at ``heterogeneity=0.15`` elsewhere in this file -- must still
    hold when the lognormal field degenerates to a constant.
    """
    spacing = (1.0, 1.0, 1.0)
    fwhm_mm = 4.0
    sigma_mm = fwhm_mm * (1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0))))
    margin_mm = 4.0 * sigma_mm  # matches the activity-conservation fixture

    geometry = _make_geometry(shape=(40, 30, 30), spacing=spacing)
    background = _flat_background(geometry.shape)
    contralateral = _contralateral_mask(geometry.shape, x_slice=slice(0, 8))
    radius_mm = 3.0
    centerline = _straight_centerline(start=(15.0, 15.0, 15.0), end=(25.0, 15.0, 15.0))
    assert (
        radius_mm + margin_mm < 15.0
    )  # sanity: fixture actually has the claimed margin

    params = _params(
        radius_mm=radius_mm, length_mm=10.0, blur_fwhm_mm=fwhm_mm, heterogeneity=0.0
    )
    result = simulate_vascular_anomaly(
        background, geometry, centerline, contralateral, params
    )

    assert np.array_equal(
        result.heterogeneity_field, np.ones(geometry.shape, dtype=np.float32)
    )

    # Sham identity still holds with heterogeneity == 0.0.
    sham_result = simulate_vascular_anomaly(
        background,
        geometry,
        centerline,
        contralateral,
        _params(
            radius_mm=radius_mm,
            length_mm=10.0,
            blur_fwhm_mm=fwhm_mm,
            heterogeneity=0.0,
            uptake_multiplier=1.0,
        ),
    )
    assert np.array_equal(sham_result.synthetic_pet, background)

    # Activity conservation still holds with heterogeneity == 0.0 (ideal_excess
    # is now exactly (m - 1) * B * F, with no lognormal field multiplying it).
    total_before = float(result.ideal_excess.sum())
    total_after = float(result.blurred_excess.sum())
    relative_error = abs(total_after - total_before) / total_before
    assert relative_error <= 0.01, (total_before, total_after, relative_error)


# ---------------------------------------------------------------------------
# Sham identity (m == 1.0)
# ---------------------------------------------------------------------------


def test_sham_uptake_multiplier_one_reproduces_background_exactly() -> None:
    geometry = _make_geometry()
    background = _flat_background(geometry.shape)
    contralateral = _contralateral_mask(geometry.shape)
    centerline = _straight_centerline()
    params = _params(uptake_multiplier=1.0)

    result = simulate_vascular_anomaly(
        background, geometry, centerline, contralateral, params
    )

    assert np.array_equal(result.synthetic_pet, background)
    assert np.array_equal(result.ideal_excess, np.zeros(geometry.shape))
    assert np.array_equal(result.blurred_excess, np.zeros(geometry.shape))


# ---------------------------------------------------------------------------
# Source nonnegativity
# ---------------------------------------------------------------------------


def test_source_is_nonnegative_at_every_stage() -> None:
    geometry = _make_geometry()
    background = _flat_background(geometry.shape)
    contralateral = _contralateral_mask(geometry.shape)
    centerline = _straight_centerline()
    result = simulate_vascular_anomaly(
        background, geometry, centerline, contralateral, _params(uptake_multiplier=2.0)
    )

    assert np.all(result.ideal_excess >= 0.0)
    assert np.all(result.blurred_excess >= 0.0)
    assert np.all(result.heterogeneity_field > 0.0)


def test_uptake_multiplier_below_one_is_rejected() -> None:
    with pytest.raises(ValueError):
        _params(uptake_multiplier=0.9)


# ---------------------------------------------------------------------------
# Degenerate valid input: sub-voxel radius -> nonzero occupancy but an empty
# ground-truth mask, and a typed-null (never 0.0/NaN) achieved activity.
# ---------------------------------------------------------------------------


def test_subvoxel_radius_yields_empty_mask_and_typed_null_achieved_activity() -> None:
    """A radius far smaller than the voxel spacing produces nonzero partial
    occupancy (the capsule genuinely intersects the grid) that never reaches
    the ``F >= 0.5`` binarization threshold anywhere, so the ground-truth
    mask is legitimately empty. ``AchievedActivitySummary`` must then be the
    typed null ``(None, None)`` -- never ``0.0``/``NaN`` standing in for
    "no voxels selected" (the "typed null, never 0/NaN" contract also used
    by the sibling P4 quantification engine's ``Measurement``).
    """
    geometry = _make_geometry()
    background = _flat_background(geometry.shape)
    contralateral = _contralateral_mask(geometry.shape)
    centerline = _straight_centerline()
    params = _params(radius_mm=0.05)

    result = simulate_vascular_anomaly(
        background, geometry, centerline, contralateral, params
    )

    assert result.source_fraction.max() > 0.0  # the capsule does intersect the grid
    assert (
        result.ground_truth_mask.sum() == 0
    )  # but never crosses the F >= 0.5 threshold
    assert result.achieved_activity.suvmax_in_mask is None
    assert result.achieved_activity.suvmean_in_mask is None


# ---------------------------------------------------------------------------
# Inserted volume error <= 5% vs. the analytic capsule volume
# ---------------------------------------------------------------------------


def test_inserted_volume_error_within_5_percent_of_analytic_capsule() -> None:
    """Rasterized analytic volume vs. the exact spherocylinder ("capsule")
    formula: cylinder core ``pi*r**2*L`` plus one full sphere
    ``(4/3)*pi*r**3`` from the two hemispherical end caps -- the formula
    named in the module docstring's "Formula provenance" section, chosen
    (over a bare cylinder ``pi*r**2*L``) because the specification requires
    hemispherical capsule end caps.
    """
    geometry = _make_geometry(shape=(30, 20, 20), spacing=(1.0, 1.0, 1.0))
    background = _flat_background(geometry.shape)
    contralateral = _contralateral_mask(geometry.shape)
    radius_mm, length_mm = 3.0, 15.0
    # Straight centerline exactly as long as length_mm: no ambiguity from
    # arc-length clipping landing mid-segment.
    centerline = _straight_centerline(start=(5.0, 10.0, 10.0), end=(20.0, 10.0, 10.0))
    params = _params(radius_mm=radius_mm, length_mm=length_mm)

    result = simulate_vascular_anomaly(
        background, geometry, centerline, contralateral, params
    )

    voxel_volume_mm3 = float(np.prod(geometry.spacing))
    rasterized_volume = float(result.source_fraction.sum()) * voxel_volume_mm3
    analytic_volume = (
        np.pi * radius_mm**2 * length_mm + (4.0 / 3.0) * np.pi * radius_mm**3
    )

    relative_error = abs(rasterized_volume - analytic_volume) / analytic_volume
    assert relative_error <= 0.05, (rasterized_volume, analytic_volume, relative_error)


# ---------------------------------------------------------------------------
# Activity conservation <= 1% across the unit-sum Gaussian blur
# ---------------------------------------------------------------------------


def test_activity_conserved_within_1_percent_when_margin_is_sufficient() -> None:
    """With the source's occupancy support at least ``4*sigma`` (the
    declared blur truncation radius) from every array edge, zero-padded
    (``mode="constant"``) boundaries cannot truncate any of the unit-sum
    kernel's support, so total pre-/post-blur excess must match closely.
    """
    spacing = (1.0, 1.0, 1.0)
    fwhm_mm = 4.0
    sigma_mm = fwhm_mm * (1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0))))
    margin_mm = 4.0 * sigma_mm  # ~6.8 mm; keep >= this from every edge

    geometry = _make_geometry(shape=(40, 30, 30), spacing=spacing)
    background = _flat_background(geometry.shape)
    contralateral = _contralateral_mask(geometry.shape, x_slice=slice(0, 8))
    radius_mm = 3.0
    centerline = _straight_centerline(start=(15.0, 15.0, 15.0), end=(25.0, 15.0, 15.0))
    assert (
        radius_mm + margin_mm < 15.0
    )  # sanity: fixture actually has the claimed margin

    params = _params(radius_mm=radius_mm, length_mm=10.0, blur_fwhm_mm=fwhm_mm)
    result = simulate_vascular_anomaly(
        background, geometry, centerline, contralateral, params
    )

    total_before = float(result.ideal_excess.sum())
    total_after = float(result.blurred_excess.sum())
    relative_error = abs(total_after - total_before) / total_before
    assert relative_error <= 0.01, (total_before, total_after, relative_error)


# ---------------------------------------------------------------------------
# Supported FWHM recovery on an impulse (every supported spacing)
# ---------------------------------------------------------------------------


def _measure_fwhm_mm(profile: np.ndarray, spacing_mm: float) -> float:
    """Linear-interpolation half-max crossing distance along a 1-D profile."""
    peak = profile.max()
    half = peak / 2.0
    peak_index = int(np.argmax(profile))

    i = peak_index
    while profile[i] > half:
        i -= 1
    left = i + (half - profile[i]) / (profile[i + 1] - profile[i])

    i = peak_index
    while profile[i] > half:
        i += 1
    right = (i - 1) + (profile[i - 1] - half) / (profile[i - 1] - profile[i])

    return (right - left) * spacing_mm


@pytest.mark.parametrize(
    "spacing_mm",
    [(1.0, 1.0, 1.0), (1.0, 2.0, 0.5)],
    ids=["isotropic", "anisotropic"],
)
def test_fwhm_recovery_on_impulse_within_tolerance(
    spacing_mm: tuple[float, float, float],
) -> None:
    shape = (41, 41, 41)
    requested_fwhm_mm = 6.0
    impulse = np.zeros(shape, dtype=np.float64)
    impulse[20, 20, 20] = 1.0

    sigma_voxels = fwhm_mm_to_sigma_voxels(requested_fwhm_mm, spacing_mm)
    blurred = _apply_unit_sum_blur(impulse, sigma_voxels)

    for axis, spacing in enumerate(spacing_mm):
        index = [20, 20, 20]
        index[axis] = slice(None)  # type: ignore[call-overload]
        profile = blurred[tuple(index)]
        recovered_fwhm = _measure_fwhm_mm(profile, spacing)
        assert recovered_fwhm == pytest.approx(requested_fwhm_mm, rel=0.1)


def test_unit_sum_blur_kernel_conserves_total_for_centered_impulse() -> None:
    shape = (21, 21, 21)
    impulse = np.zeros(shape, dtype=np.float64)
    impulse[10, 10, 10] = 1.0
    sigma_voxels = fwhm_mm_to_sigma_voxels(4.0, (1.0, 1.0, 1.0))
    blurred = _apply_unit_sum_blur(impulse, sigma_voxels)
    assert blurred.sum() == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Boundary behavior: truncates safely, no wraparound
# ---------------------------------------------------------------------------


def test_source_near_edge_truncates_without_wraparound() -> None:
    geometry = _make_geometry(shape=(20, 20, 20), spacing=(1.0, 1.0, 1.0))
    background = _flat_background(geometry.shape)
    contralateral = _contralateral_mask(geometry.shape, x_slice=slice(0, 8))
    # Centerline hugging the x=0 face: the radius-3 capsule pokes outside
    # the array's physical extent on that side.
    centerline = _straight_centerline(start=(1.0, 10.0, 10.0), end=(1.0, 18.0, 10.0))
    params = _params(
        radius_mm=3.0, length_mm=8.0, uptake_multiplier=1.8, blur_fwhm_mm=4.0
    )

    result = simulate_vascular_anomaly(
        background, geometry, centerline, contralateral, params
    )

    assert result.synthetic_pet.shape == geometry.shape
    assert np.all(np.isfinite(result.synthetic_pet))
    # Occupancy exists right at the near edge (voxel index 0)...
    assert result.source_fraction[0, 10, 10] > 0.0
    # ...but the opposite face (far from the source, well beyond any blur
    # margin) is completely unaffected: no periodic wraparound leaked
    # activity to voxel index 19 (would happen under mode="wrap").
    assert np.array_equal(result.synthetic_pet[19, :, :], background[19, :, :])
    assert np.array_equal(result.source_fraction[19, :, :], np.zeros((20, 20)))


# ---------------------------------------------------------------------------
# Invalid background -> typed SimulationError
# ---------------------------------------------------------------------------


def test_empty_contralateral_region_raises_typed_error() -> None:
    geometry = _make_geometry()
    background = _flat_background(geometry.shape)
    empty_mask = np.zeros(geometry.shape, dtype=bool)
    with pytest.raises(SimulationError) as excinfo:
        simulate_vascular_anomaly(
            background, geometry, _straight_centerline(), empty_mask, _params()
        )
    assert excinfo.value.code is SimulationErrorCode.EMPTY_CONTRALATERAL_REGION


def test_nonfinite_contralateral_region_raises_typed_error() -> None:
    geometry = _make_geometry()
    background = _flat_background(geometry.shape)
    contralateral = _contralateral_mask(geometry.shape)
    background[contralateral] = np.nan
    with pytest.raises(SimulationError) as excinfo:
        simulate_vascular_anomaly(
            background, geometry, _straight_centerline(), contralateral, _params()
        )
    assert excinfo.value.code is SimulationErrorCode.NONFINITE_CONTRALATERAL_REGION


def test_insufficient_contralateral_samples_raises_typed_error() -> None:
    geometry = _make_geometry()
    background = _flat_background(geometry.shape)
    sparse_mask = np.zeros(geometry.shape, dtype=bool)
    sparse_mask[20, 0, :5] = True  # 5 voxels < the required 10
    with pytest.raises(SimulationError) as excinfo:
        simulate_vascular_anomaly(
            background, geometry, _straight_centerline(), sparse_mask, _params()
        )
    assert excinfo.value.code is SimulationErrorCode.INSUFFICIENT_CONTRALATERAL_SAMPLES


def test_nonpositive_contralateral_baseline_raises_typed_error() -> None:
    geometry = _make_geometry()
    background = _flat_background(geometry.shape)
    contralateral = _contralateral_mask(geometry.shape)
    background[contralateral] = -1.0  # trimmed mean will be exactly -1.0
    with pytest.raises(SimulationError) as excinfo:
        simulate_vascular_anomaly(
            background, geometry, _straight_centerline(), contralateral, _params()
        )
    assert excinfo.value.code is SimulationErrorCode.NONPOSITIVE_CONTRALATERAL_BASELINE


def test_source_entirely_outside_grid_raises_empty_source_occupancy() -> None:
    """The fifth named :class:`SimulationErrorCode` -- ``EMPTY_SOURCE_OCCUPANCY``
    -- is otherwise only referenced by the enum-membership test; this
    exercises it behaviorally. A centerline placed far outside the grid's
    physical extent, with a small radius, produces all-zero occupancy
    (``sum(F) == 0``), which :func:`_heterogeneity_field` must reject before
    ever attempting to renormalize by the (zero) occupancy-weighted mean.
    """
    geometry = _make_geometry(shape=(10, 10, 10), spacing=(1.0, 1.0, 1.0))
    background = _flat_background(geometry.shape)
    contralateral = _contralateral_mask(geometry.shape, x_slice=slice(0, 8))
    far_centerline = np.array([[500.0, 500.0, 500.0], [520.0, 500.0, 500.0]])
    params = _params(radius_mm=0.5, length_mm=15.0)

    with pytest.raises(SimulationError) as excinfo:
        simulate_vascular_anomaly(
            background, geometry, far_centerline, contralateral, params
        )
    assert excinfo.value.code is SimulationErrorCode.EMPTY_SOURCE_OCCUPANCY


# ---------------------------------------------------------------------------
# Provenance / hash: present and stable
# ---------------------------------------------------------------------------


def test_provenance_hash_present_and_stable_for_identical_inputs() -> None:
    geometry = _make_geometry()
    background = _flat_background(geometry.shape)
    contralateral = _contralateral_mask(geometry.shape)
    centerline = _straight_centerline()
    params = _params(seed=99)

    result = simulate_vascular_anomaly(
        background, geometry, centerline, contralateral, params
    )
    repeat = simulate_vascular_anomaly(
        background, geometry, centerline, contralateral, params
    )

    assert isinstance(result.provenance.output_sha256, str)
    assert len(result.provenance.output_sha256) == 64  # SHA-256 hex digest
    assert isinstance(result.provenance.params_sha256, str)
    assert len(result.provenance.params_sha256) == 64
    assert result.provenance.output_sha256 == repeat.provenance.output_sha256
    assert result.provenance.params_sha256 == repeat.provenance.params_sha256
    assert result.provenance.seed == 99
    assert result.provenance.research_prototype_warning == RESEARCH_PROTOTYPE_WARNING
    assert result.provenance.supersample_factor == 5
    assert result.provenance.blur_boundary_mode == "constant"
    assert result.provenance.blur_truncate == 4.0
    assert result.provenance.excess_accumulation_dtype == "float64"
    assert result.provenance.output_dtype == "float32"


def test_provenance_hash_changes_with_params() -> None:
    geometry = _make_geometry()
    background = _flat_background(geometry.shape)
    contralateral = _contralateral_mask(geometry.shape)
    centerline = _straight_centerline()

    result_a = simulate_vascular_anomaly(
        background, geometry, centerline, contralateral, _params(radius_mm=3.0)
    )
    result_b = simulate_vascular_anomaly(
        background, geometry, centerline, contralateral, _params(radius_mm=4.0)
    )
    assert result_a.provenance.params_sha256 != result_b.provenance.params_sha256
    assert result_a.provenance.output_sha256 != result_b.provenance.output_sha256


# ---------------------------------------------------------------------------
# CT-registration stress test: pet_ct_shift_mm never moves PET or labels
# ---------------------------------------------------------------------------


_CT_SHIFT_MAGNITUDES_MM = [
    (0.0, 0.0, 0.0),
    (2.0, 0.0, 0.0),
    (0.0, -2.0, 0.0),
    (0.0, 0.0, 4.0),
    (-4.0, 0.0, 0.0),
]


def test_ct_shift_never_moves_pet_or_ground_truth_labels() -> None:
    """The real stress test: run the *same* background/geometry/centerline
    with every declared 0/2/4 mm ``pet_ct_shift_mm`` value and require
    ``synthetic_pet``/``ground_truth_mask`` to be bit-identical across all
    of them, then separately confirm the shifted CT array actually differs
    from the unshifted one (so this isn't vacuously true because nothing
    moved). Comparing *across* different shift values -- not just re-running
    the same value twice -- is what actually proves shift-invariance rather
    than mere determinism.
    """
    geometry = _make_geometry(shape=(24, 24, 24))
    background = _flat_background(geometry.shape)
    contralateral = _contralateral_mask(geometry.shape, x_slice=slice(0, 8))
    centerline = _straight_centerline(start=(10.0, 12.0, 12.0), end=(18.0, 12.0, 12.0))

    results = [
        simulate_vascular_anomaly(
            background,
            geometry,
            centerline,
            contralateral,
            _params(pet_ct_shift_mm=shift_mm),
        )
        for shift_mm in _CT_SHIFT_MAGNITUDES_MM
    ]

    reference = results[0]
    for other in results[1:]:
        assert np.array_equal(other.synthetic_pet, reference.synthetic_pet)
        assert np.array_equal(other.ground_truth_mask, reference.ground_truth_mask)
        assert np.array_equal(other.source_fraction, reference.source_fraction)

    # A distinct, deterministic CT phantom -- never touched by the simulator
    # (params.pet_ct_shift_mm is not even read inside simulate_vascular_anomaly;
    # only shift_ct_array reads it, and only on a caller-supplied CT array).
    ct = np.arange(np.prod(geometry.shape), dtype=np.float32).reshape(geometry.shape)
    ct_baseline = ct.copy()

    for shift_mm in _CT_SHIFT_MAGNITUDES_MM:
        shifted_ct = shift_ct_array(ct, geometry, shift_mm)
        # The caller's own CT array is never mutated in place.
        assert np.array_equal(ct, ct_baseline)
        if shift_mm == (0.0, 0.0, 0.0):
            assert np.array_equal(shifted_ct, ct)
        else:
            assert not np.array_equal(shifted_ct, ct)


def test_shift_ct_array_translates_a_point_source_by_the_requested_vector() -> None:
    geometry = _make_geometry(shape=(20, 20, 20), spacing=(1.0, 1.0, 1.0))
    ct = np.zeros(geometry.shape, dtype=np.float32)
    ct[10, 10, 10] = 1000.0

    shifted = shift_ct_array(ct, geometry, (2.0, 0.0, 0.0))
    assert shifted[12, 10, 10] == pytest.approx(1000.0)
    assert shifted[10, 10, 10] == pytest.approx(0.0)

    shifted_neg = shift_ct_array(ct, geometry, (0.0, -3.0, 0.0))
    assert shifted_neg[10, 7, 10] == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# Private numeric core: point-to-polyline distance / centerline clipping
# ---------------------------------------------------------------------------


def test_point_to_polyline_distance_matches_point_to_segment_for_two_points() -> None:
    polyline = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    points = np.array(
        [
            [5.0, 3.0, 0.0],  # perpendicular to the middle of the segment
            [-2.0, 0.0, 0.0],  # beyond the start endpoint
            [12.0, 0.0, 0.0],  # beyond the end endpoint
        ]
    )
    distances = _point_to_polyline_distance(points, polyline)
    assert distances == pytest.approx([3.0, 2.0, 2.0])


def test_clip_centerline_core_clips_to_requested_length() -> None:
    centerline = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    core = _clip_centerline_core(centerline, length_mm=5.0)
    assert core[0] == pytest.approx([0.0, 0.0, 0.0])
    assert core[-1] == pytest.approx([5.0, 0.0, 0.0])


def test_clip_centerline_core_uses_full_centerline_when_shorter_than_requested() -> (
    None
):
    centerline = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    core = _clip_centerline_core(centerline, length_mm=100.0)
    assert core[0] == pytest.approx([0.0, 0.0, 0.0])
    assert core[-1] == pytest.approx([10.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Basic parameter/shape validation
# ---------------------------------------------------------------------------


def test_invalid_side_is_rejected() -> None:
    with pytest.raises(ValueError):
        _params(side="both")


def test_background_shape_mismatch_is_rejected() -> None:
    geometry = _make_geometry()
    wrong_shape_background = np.ones((geometry.shape[0] + 1, *geometry.shape[1:]))
    with pytest.raises(ValueError):
        simulate_vascular_anomaly(
            wrong_shape_background,
            geometry,
            _straight_centerline(),
            _contralateral_mask(geometry.shape),
            _params(),
        )


# ---------------------------------------------------------------------------
# Memory-bounded occupancy chunking (production OOM incident fix)
#
# Root cause: on a real-scale training window (~140x80x140 voxels, a ~70-
# point centerline, supersample=5), the pre-fix _supersampled_occupancy
# built one monolithic (n_supersampled_points, n_centerline_segments, 3)
# NumPy array -- ~2x10^8 points x ~25 segments x 3 -- and separately
# allocated one persistent array at the full supersampled point count.
# Reported production measurement (/usr/bin/time -v): ~29 GiB peak RSS
# against a 31 GiB machine, ~35s wall, on a ~107K-voxel window. This
# single-sample-OOMs and takes down any parallel worker pool.
#
# These tests cover exactly what the incident asked for: (1) the chunked
# result is bit-identical to the unchunked pre-fix computation, at every
# chunk size, on both a straight and a curved centerline; and (2) peak
# process memory for a realistic-scale call is bounded to a small, fixed
# constant, not proportional to the supersampled point count.
# ---------------------------------------------------------------------------


def _reference_unchunked_occupancy(
    core_polyline: np.ndarray,
    radius_mm: float,
    geometry: GridGeometry,
    supersample: int,
) -> np.ndarray:
    """A verbatim reconstruction of the pre-memory-fix implementation of
    ``_supersampled_occupancy``: one monolithic
    ``(n_points, n_centerline_segments, 3)`` broadcast, no chunking. Used
    only to prove the chunked, production implementation is bit-identical
    to it on grids small enough for the monolithic path to be feasible at
    all (the whole point of the fix is that it is *not* feasible at
    training scale -- see the module docstring's "Memory-bounded
    supersampled occupancy" section).
    """
    shape = geometry.shape
    offsets = (np.arange(supersample, dtype=np.float64) + 0.5) / supersample - 0.5
    axis_indices = [
        (np.arange(n, dtype=np.float64)[:, None] + offsets[None, :]).reshape(-1)
        for n in shape
    ]
    gi, gj, gk = np.meshgrid(*axis_indices, indexing="ij")
    voxel_idx = np.stack([gi.reshape(-1), gj.reshape(-1), gk.reshape(-1)], axis=1)
    world = _apply_affine_points(geometry.affine, voxel_idx)
    distance = _point_to_polyline_distance(world, core_polyline)
    inside = (distance <= radius_mm).astype(np.float64)
    inside = inside.reshape(
        shape[0], supersample, shape[1], supersample, shape[2], supersample
    )
    return inside.mean(axis=(1, 3, 5))


@pytest.mark.parametrize("chunk_points", [1, 7, 13, 50, 400, 10_000, 5_000_000])
def test_chunked_occupancy_matches_unchunked_reference_straight(
    chunk_points: int,
) -> None:
    """Numerical equivalence, straight centerline: every ``chunk_points``
    value -- including ones that force a chunk boundary in the middle of a
    voxel's own supersample**3 subsamples (1, 7, 13) and one so large the
    whole grid is one chunk (5,000,000) -- must reproduce the pre-fix
    unchunked reference exactly (0 difference, not merely within 1e-6).
    """
    geometry = _make_geometry(shape=(17, 13, 11))
    centerline = np.array([[3.0, 5.0, 4.0], [13.0, 7.0, 6.0]])
    core = _clip_centerline_core(centerline, length_mm=8.0)

    reference = _reference_unchunked_occupancy(core, 2.5, geometry, supersample=4)
    chunked = _supersampled_occupancy(
        core, 2.5, geometry, supersample=4, chunk_points=chunk_points
    )

    assert np.array_equal(reference, chunked)
    assert np.max(np.abs(reference - chunked)) <= 1e-6


@pytest.mark.parametrize("chunk_points", [1, 3, 17, 1_000])
def test_chunked_occupancy_matches_unchunked_reference_curved(
    chunk_points: int,
) -> None:
    """Numerical equivalence, curved (multi-segment) centerline."""
    geometry = _make_geometry(shape=(21, 19, 15))
    centerline = _curved_centerline(center=(6.0, 6.0, 6.0), radius=4.0, n_points=7)
    core = _clip_centerline_core(centerline, length_mm=6.0)

    reference = _reference_unchunked_occupancy(core, 1.7, geometry, supersample=5)
    chunked = _supersampled_occupancy(
        core, 1.7, geometry, supersample=5, chunk_points=chunk_points
    )

    assert np.array_equal(reference, chunked)


def test_full_pipeline_bit_identical_across_occupancy_chunk_points() -> None:
    """End-to-end equivalence: the coordinator's exact ask -- same
    ``synthetic_pet``, ``ground_truth_mask``, and achieved fields --
    compared across chunk sizes spanning from "one voxel's worth of
    subsamples per chunk" up to "the entire grid in one chunk" (the latter
    exercises the same code path the unchunked pre-fix implementation did,
    since chunk_points >> total supersampled point count collapses the loop
    to a single iteration).
    """
    geometry = _make_geometry()
    background = _flat_background(geometry.shape)
    contralateral = _contralateral_mask(geometry.shape)
    centerline = _straight_centerline()
    params = _params(seed=11)

    one_chunk = simulate_vascular_anomaly(
        background,
        geometry,
        centerline,
        contralateral,
        params,
        occupancy_chunk_points=10_000_000,
    )
    many_chunks = simulate_vascular_anomaly(
        background,
        geometry,
        centerline,
        contralateral,
        params,
        occupancy_chunk_points=17,
    )

    assert np.array_equal(one_chunk.synthetic_pet, many_chunks.synthetic_pet)
    assert np.array_equal(one_chunk.ground_truth_mask, many_chunks.ground_truth_mask)
    assert np.array_equal(one_chunk.source_fraction, many_chunks.source_fraction)
    assert np.array_equal(one_chunk.ideal_excess, many_chunks.ideal_excess)
    assert np.array_equal(one_chunk.blurred_excess, many_chunks.blurred_excess)
    assert one_chunk.achieved_activity == many_chunks.achieved_activity
    assert one_chunk.contralateral_baseline == many_chunks.contralateral_baseline
    assert one_chunk.provenance.output_sha256 == many_chunks.provenance.output_sha256


def test_occupancy_chunk_points_must_be_a_positive_int() -> None:
    geometry = _make_geometry()
    background = _flat_background(geometry.shape)
    contralateral = _contralateral_mask(geometry.shape)
    with pytest.raises(ValueError):
        simulate_vascular_anomaly(
            background,
            geometry,
            _straight_centerline(),
            contralateral,
            _params(),
            occupancy_chunk_points=0,
        )


def test_peak_memory_bounded_on_realistic_training_scale_window() -> None:
    """Reproduces the reported incident's scale class in-process: a
    realistic training window (80x70x80 = 448,000 voxels; supersample=5
    gives 56,000,000 supersampled points -- the same order of magnitude as
    the ~2x10^8-point production incident) with a source spanning most of
    the window. The pre-fix unchunked implementation, run standalone via
    ``/usr/bin/time -v`` on this exact configuration on this same machine,
    measured **10.54 GiB** peak RSS / 10.5s wall; this fix measured **~133
    MiB** peak RSS / 3.8s wall via the same subprocess methodology (a ~81x
    memory reduction, and faster wall time too, since avoiding a multi-GB
    allocation also avoids its allocator overhead). This test asserts the
    *in-process* bound -- ``resource.getrusage(RUSAGE_SELF).ru_maxrss`` is a
    high-water mark, so the delta before/after this call isolates how much
    higher this specific call pushes the process's peak, not the process's
    total footprint (which already includes pytest/numpy/etc. baseline
    usage) -- comfortably under a generous 500 MiB ceiling (observed: low
    tens of MiB), clear of the coordinator's explicit "< ~2 GiB" bound with
    wide margin for CI variance.
    """
    shape = (80, 70, 80)
    geometry = _make_geometry(shape=shape, spacing=(1.0, 1.0, 1.0))
    background = _flat_background(shape)
    contralateral = _contralateral_mask(shape, x_slice=slice(0, 8))
    centerline = np.array([[5.0, 35.0, 40.0], [45.0, 35.0, 40.0]])
    params = _params(radius_mm=3.0, length_mm=40.0, blur_fwhm_mm=4.0)

    before_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start = time.time()
    result = simulate_vascular_anomaly(
        background, geometry, centerline, contralateral, params, supersample=5
    )
    elapsed_s = time.time() - start
    after_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    delta_mib = (after_kb - before_kb) / 1024.0  # ru_maxrss is in KiB on Linux
    assert result.ground_truth_mask.shape == shape
    assert result.ground_truth_mask.sum() > 0
    assert delta_mib < 500.0, (
        f"additional peak RSS for this call: {delta_mib:.1f} MiB "
        f"(elapsed {elapsed_s:.1f}s)"
    )
