"""Tests for the deterministic PET quantification engine (VascuTrace Phase 4).

Every fixture in this file is a small, hand-computable, generated array --
nothing here reads ``Data/``, the network, CUDA, or any external service.
:class:`~src.vascutrace.geometry.GridGeometry` is a plain dataclass with no
constructor-time validation, so fixtures build it directly from a known
shape/affine pair rather than routing through a NIfTI header.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.vascutrace.geometry import GridGeometry
from src.vascutrace.quantification import (
    RESEARCH_PROTOTYPE_WARNING,
    Measurement,
    QuantificationErrorCode,
    QuantificationResult,
    quantify_target,
)

# ---------------------------------------------------------------------------
# Fixture helpers (generated arrays only)
# ---------------------------------------------------------------------------


def _make_geometry(shape: tuple[int, int, int], affine: np.ndarray) -> GridGeometry:
    """Build a :class:`GridGeometry` directly from a known shape/affine pair
    (no NIfTI header, no validation pass -- ``GridGeometry`` itself performs
    none at construction time; that lives in ``geometry.py``'s builder
    functions, which this test suite deliberately does not need).
    """
    affine = np.asarray(affine, dtype=np.float64)
    corners = np.array(
        [
            [i, j, k]
            for i in (0, shape[0] - 1)
            for j in (0, shape[1] - 1)
            for k in (0, shape[2] - 1)
        ],
        dtype=np.float64,
    )
    homogeneous = np.concatenate([corners, np.ones((8, 1))], axis=1)
    world_corners = (affine @ homogeneous.T).T[:, :3]
    spacing = tuple(float(x) for x in np.linalg.norm(affine[:3, :3], axis=0))
    return GridGeometry(
        shape=tuple(int(s) for s in shape),
        affine=affine,
        spacing=spacing,
        units="mm",
        world_bounds_min=world_corners.min(axis=0),
        world_bounds_max=world_corners.max(axis=0),
    )


def _identity_affine(
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    affine = np.eye(4)
    affine[:3, 3] = offset
    return affine


def _spacing_affine(spacing: tuple[float, float, float]) -> np.ndarray:
    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
    return affine


def _mirror_affine_through_x(offset: float) -> np.ndarray:
    """A physical mirror reflecting world x through the plane ``x == offset``:
    ``T(x, y, z) = (-x + 2*offset, y, z)``.
    """
    m = np.eye(4)
    m[0, 0] = -1.0
    m[0, 3] = 2.0 * offset
    return m


# A 6x4x4 grid whose x-voxel-index i maps to world x = i - 2.5 (so the grid
# is symmetric about world x == 0 and the mirror below pairs voxel i with
# voxel (5 - i) *exactly*, with no rounding error).
_SHAPE = (6, 4, 4)
_GEOMETRY = _make_geometry(_SHAPE, _identity_affine(offset=(-2.5, 0.0, 0.0)))
_REFLECTION_X0 = _mirror_affine_through_x(offset=0.0)


def _suvbw(shape: tuple[int, int, int]) -> np.ndarray:
    return np.zeros(shape, dtype=np.float64)


def _bool_mask(shape: tuple[int, int, int]) -> np.ndarray:
    return np.zeros(shape, dtype=bool)


# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------


def test_research_prototype_warning_exact_text() -> None:
    assert RESEARCH_PROTOTYPE_WARNING == (
        "Research prototype. Trained and evaluated using simulated vascular-like "
        "abnormalities, not confirmed human post-angioplasty lesions."
    )


def test_quantification_error_code_members() -> None:
    expected = {
        "EMPTY_MASK",
        "NONFINITE_TARGET",
        "CONTRALATERAL_OUT_OF_DOMAIN",
        "NONFINITE_CONTRALATERAL",
        "INVALID_CONTRALATERAL_DENOMINATOR",
    }
    assert {member.value for member in QuantificationErrorCode} == expected


def test_measurement_is_valid_property() -> None:
    assert Measurement(3.0).is_valid is True
    assert Measurement(None, "EMPTY_MASK").is_valid is False
    assert Measurement(0.0).is_valid is True  # a legitimate zero is valid


# ---------------------------------------------------------------------------
# Known-array happy path: SUVmax/mean, volume, extent, asymmetry, TBR,
# laterality -- all hand-computed from the exact frozen formulas.
# ---------------------------------------------------------------------------


def _left_side_scenario() -> tuple[np.ndarray, np.ndarray]:
    """Target = 3 voxels at x-index {0,1,2} (world x < 0, i.e. "left" of the
    x==0 mirror plane); contralateral = their exact mirror images at
    x-index {5,4,3}. All at fixed j=1, k=1.
    """
    suv = _suvbw(_SHAPE)
    mask = _bool_mask(_SHAPE)
    mask[0, 1, 1] = mask[1, 1, 1] = mask[2, 1, 1] = True
    suv[0, 1, 1] = 2.0
    suv[1, 1, 1] = 4.0
    suv[2, 1, 1] = 6.0
    suv[3, 1, 1] = 1.0
    suv[4, 1, 1] = 1.0
    suv[5, 1, 1] = 1.0
    return suv, mask


def test_known_array_target_suvmax_and_suvmean() -> None:
    suv, mask = _left_side_scenario()
    result = quantify_target(suv, mask, _GEOMETRY, _REFLECTION_X0)
    assert result.target_suvmax == Measurement(6.0)
    assert result.target_suvmean == Measurement(pytest.approx(4.0))


def test_known_array_metabolic_volume_ml_from_affine_determinant() -> None:
    # voxel_count=3, |det(diag(1,1,1))| = 1 mm^3/voxel -> 3 * 1 / 1000 mL.
    suv, mask = _left_side_scenario()
    result = quantify_target(suv, mask, _GEOMETRY, _REFLECTION_X0)
    assert result.metabolic_volume_ml == Measurement(pytest.approx(0.003))


def test_known_array_metabolic_volume_ml_nontrivial_spacing() -> None:
    shape = (4, 4, 4)
    geometry = _make_geometry(shape, _spacing_affine((2.0, 3.0, 4.0)))
    suv = _suvbw(shape)
    mask = _bool_mask(shape)
    mask[0, 0, 0] = mask[1, 0, 0] = mask[1, 1, 0] = mask[1, 1, 1] = True  # 4 voxels
    reflection = _mirror_affine_through_x(offset=0.0)
    result = quantify_target(suv, mask, geometry, reflection)
    # |det(diag(2,3,4))| = 24 mm^3/voxel; 4 voxels -> 96 mm^3 -> 0.096 mL.
    assert result.metabolic_volume_ml == Measurement(pytest.approx(0.096))


def test_known_array_longitudinal_extent_linear_mask() -> None:
    suv, mask = _left_side_scenario()
    result = quantify_target(suv, mask, _GEOMETRY, _REFLECTION_X0)
    # 3 collinear points 1 mm apart along x -> span = 2 mm.
    assert result.longitudinal_extent_mm == Measurement(pytest.approx(2.0))


def test_known_array_asymmetry_index_and_tbr() -> None:
    suv, mask = _left_side_scenario()
    result = quantify_target(suv, mask, _GEOMETRY, _REFLECTION_X0)
    target_mean, contra_mean = 4.0, 1.0
    expected_asymmetry = (target_mean - contra_mean) / (contra_mean + 1e-6)
    # TBR numerator is target_suvmean (not target_suvmax), per the frozen
    # contract (imaging-physics.md:127: "target_to_contralateral_ratio =
    # target_mean / (contra_mean + 1e-6)").
    expected_tbr = target_mean / (contra_mean + 1e-6)
    assert result.contralateral_suvmax == Measurement(pytest.approx(1.0))
    assert result.contralateral_suvmean == Measurement(pytest.approx(1.0))
    assert result.asymmetry_index == Measurement(pytest.approx(expected_asymmetry))
    assert result.target_to_background_ratio == Measurement(pytest.approx(expected_tbr))


def test_known_array_laterality_left() -> None:
    suv, mask = _left_side_scenario()
    result = quantify_target(suv, mask, _GEOMETRY, _REFLECTION_X0)
    assert result.laterality == "left"


def test_known_array_laterality_right() -> None:
    suv = _suvbw(_SHAPE)
    mask = _bool_mask(_SHAPE)
    mask[3, 1, 1] = mask[4, 1, 1] = mask[5, 1, 1] = True
    result = quantify_target(suv, mask, _GEOMETRY, _REFLECTION_X0)
    assert result.laterality == "right"


def test_known_array_laterality_bilateral_straddles_plane() -> None:
    suv = _suvbw(_SHAPE)
    mask = _bool_mask(_SHAPE)
    mask[2, 1, 1] = True  # world x = -0.5
    mask[3, 1, 1] = True  # world x = 0.5
    result = quantify_target(suv, mask, _GEOMETRY, _REFLECTION_X0)
    assert result.laterality == "bilateral"


def test_laterality_none_for_empty_mask() -> None:
    suv = _suvbw(_SHAPE)
    mask = _bool_mask(_SHAPE)
    result = quantify_target(suv, mask, _GEOMETRY, _REFLECTION_X0)
    assert result.laterality == "none"


# ---------------------------------------------------------------------------
# Explicit centerline_direction vs. PCA fallback (both hand-computable)
# ---------------------------------------------------------------------------


def test_longitudinal_extent_pca_vs_explicit_direction_differ_as_expected() -> None:
    # Two occupied voxels at world (0,0,0) and (3,4,0): a 3-4-5 triangle.
    shape = (6, 6, 2)
    geometry = _make_geometry(shape, _identity_affine())
    suv = _suvbw(shape)
    mask = _bool_mask(shape)
    mask[0, 0, 0] = True
    mask[3, 4, 0] = True
    reflection = _mirror_affine_through_x(offset=10.0)  # keep out of scope here

    pca_result = quantify_target(suv, mask, geometry, reflection)
    # PCA of exactly 2 points is the line through them: span == their distance.
    assert pca_result.longitudinal_extent_mm == Measurement(pytest.approx(5.0))

    explicit_result = quantify_target(
        suv, mask, geometry, reflection, centerline_direction=np.array([1.0, 0.0, 0.0])
    )
    # Projected onto the x-axis only: span == |3 - 0| == 3.
    assert explicit_result.longitudinal_extent_mm == Measurement(pytest.approx(3.0))


# ---------------------------------------------------------------------------
# Invalid/empty branches: every one returns a typed structured null.
# ---------------------------------------------------------------------------


def test_empty_mask_is_structured_null_everywhere() -> None:
    suv = _suvbw(_SHAPE)
    mask = _bool_mask(_SHAPE)
    result = quantify_target(suv, mask, _GEOMETRY, _REFLECTION_X0)

    empty_null = Measurement(None, QuantificationErrorCode.EMPTY_MASK.value)
    assert result.target_suvmax == empty_null
    assert result.target_suvmean == empty_null
    assert result.metabolic_volume_ml == empty_null
    assert result.longitudinal_extent_mm == empty_null
    assert result.contralateral_suvmax == empty_null
    assert result.contralateral_suvmean == empty_null
    assert result.asymmetry_index == empty_null
    assert result.target_to_background_ratio == empty_null
    assert result.laterality == "none"

    assert result.qc.empty_mask is True
    assert result.qc.invalid_region is True
    assert result.qc.boundary_touching is False
    assert result.qc.fragmented is False
    assert result.qc.component_count == 0
    assert result.qc.partial_volume_risk is False
    assert result.qc.misregistration_risk_placeholder is False


def test_nonfinite_target_voxel_nulls_only_target_suv_measurements() -> None:
    suv, mask = _left_side_scenario()
    suv[1, 1, 1] = np.nan  # one of three target voxels: partial contamination
    result = quantify_target(suv, mask, _GEOMETRY, _REFLECTION_X0)

    reason = QuantificationErrorCode.NONFINITE_TARGET.value
    assert result.target_suvmax == Measurement(None, reason)
    assert result.target_suvmean == Measurement(None, reason)
    # Volume and extent are purely geometric -- unaffected by SUV finiteness.
    assert result.metabolic_volume_ml == Measurement(pytest.approx(0.003))
    assert result.longitudinal_extent_mm == Measurement(pytest.approx(2.0))
    # Downstream ratios propagate the target's nonfinite reason.
    assert result.asymmetry_index == Measurement(None, reason)
    assert result.target_to_background_ratio == Measurement(None, reason)
    # invalid_region is a *distinct*, stronger signal than NONFINITE_TARGET:
    # only one of three voxels is bad here, so the region is not "entirely"
    # invalid even though its SUV measurements are still null.
    assert result.qc.empty_mask is False
    assert result.qc.invalid_region is False


def test_invalid_region_true_when_all_target_values_nonfinite() -> None:
    suv, mask = _left_side_scenario()
    suv[0, 1, 1] = np.nan
    suv[1, 1, 1] = np.inf
    suv[2, 1, 1] = -np.inf  # all three target voxels nonfinite
    result = quantify_target(suv, mask, _GEOMETRY, _REFLECTION_X0)

    reason = QuantificationErrorCode.NONFINITE_TARGET.value
    assert result.target_suvmax == Measurement(None, reason)
    assert result.target_suvmean == Measurement(None, reason)
    # Nonempty (three voxels selected), so this is not EMPTY_MASK -- but the
    # region is entirely unusable, so invalid_region is a distinct True.
    assert result.qc.empty_mask is False
    assert result.qc.invalid_region is True


def test_invalid_region_false_for_a_fully_valid_mask() -> None:
    suv, mask = _left_side_scenario()
    result = quantify_target(suv, mask, _GEOMETRY, _REFLECTION_X0)
    assert result.qc.empty_mask is False
    assert result.qc.invalid_region is False


def test_nonfinite_contralateral_voxel_nulls_contralateral_and_ratios() -> None:
    suv, mask = _left_side_scenario()
    suv[4, 1, 1] = np.inf
    result = quantify_target(suv, mask, _GEOMETRY, _REFLECTION_X0)

    reason = QuantificationErrorCode.NONFINITE_CONTRALATERAL.value
    assert result.contralateral_suvmax == Measurement(None, reason)
    assert result.contralateral_suvmean == Measurement(None, reason)
    assert result.asymmetry_index == Measurement(None, reason)
    assert result.target_to_background_ratio == Measurement(None, reason)
    # Target measurements are unaffected.
    assert result.target_suvmax == Measurement(pytest.approx(6.0))


def test_contralateral_out_of_domain_is_structured_null() -> None:
    shape = (4, 4, 4)
    geometry = _make_geometry(shape, _identity_affine())  # voxel i -> world x = i
    reflection = _mirror_affine_through_x(offset=0.0)  # world x -> -x
    suv = _suvbw(shape)
    mask = _bool_mask(shape)
    mask[1, 1, 1] = (
        True  # world x=1 -> reflected x=-1 -> voxel index -1 (out of bounds)
    )
    suv[1, 1, 1] = 5.0

    result = quantify_target(suv, mask, geometry, reflection)

    reason = QuantificationErrorCode.CONTRALATERAL_OUT_OF_DOMAIN.value
    assert result.contralateral_suvmax == Measurement(None, reason)
    assert result.contralateral_suvmean == Measurement(None, reason)
    assert result.asymmetry_index == Measurement(None, reason)
    assert result.target_to_background_ratio == Measurement(None, reason)
    # Target measurements remain valid.
    assert result.target_suvmax == Measurement(pytest.approx(5.0))


def test_contralateral_denominator_floor_nulls_only_ratios() -> None:
    suv, mask = _left_side_scenario()
    suv[3, 1, 1] = suv[4, 1, 1] = suv[5, 1, 1] = 0.0  # contra mean == 0.0 <= 1e-6
    result = quantify_target(suv, mask, _GEOMETRY, _REFLECTION_X0)

    # The contralateral mean itself is a *valid* (legitimately zero) result --
    # only the ratios that would divide by it become null.
    assert result.contralateral_suvmean == Measurement(0.0)
    assert result.contralateral_suvmean.is_valid is True

    reason = QuantificationErrorCode.INVALID_CONTRALATERAL_DENOMINATOR.value
    assert result.asymmetry_index == Measurement(None, reason)
    assert result.target_to_background_ratio == Measurement(None, reason)
    # Target measurements remain valid and unaffected.
    assert result.target_suvmean == Measurement(pytest.approx(4.0))


def test_contralateral_voxel_collision_is_deduplicated() -> None:
    # An oblique reflection plane combined with anisotropic voxel spacing can
    # legitimately map two *distinct* target voxels onto the same rounded
    # contralateral voxel index. This exact collision was found by direct
    # numeric search (not assumed): with spacing (2.0, 0.5, 1.0), offset
    # (-3, -3, 0), and a Householder mirror through the plane with unit
    # normal (0.6, 0.8, 0.0), target voxels (2,1,0) and (2,2,0) both round to
    # contralateral voxel (3,5,0), while a third target voxel (2,0,0) maps
    # singly to a *different* contralateral voxel (3,6,0).
    #
    # This 2-collide-1-distinct shape makes dedup genuinely observable: with
    # dedup (``np.unique``, the implementation), the ROI is the 2-voxel set
    # {(3,5,0)=9.0, (3,6,0)=2.0} -> mean == 5.5. Without dedup, the ROI would
    # be the 3-entry multiset [9.0, 9.0, 2.0] (the colliding voxel read
    # twice) -> mean == 20/3 ~= 6.667. The two hypotheses are numerically
    # distinguishable, so this test proves dedup is actually happening
    # rather than merely being consistent with it.
    shape = (4, 7, 1)
    spacing_affine = np.diag([2.0, 0.5, 1.0, 1.0])
    spacing_affine[:3, 3] = [-3.0, -3.0, 0.0]
    geometry = _make_geometry(shape, spacing_affine)

    normal = np.array([0.6, 0.8, 0.0])
    reflection = np.eye(4)
    reflection[:3, :3] = np.eye(3) - 2.0 * np.outer(normal, normal)

    suv = _suvbw(shape)
    mask = _bool_mask(shape)
    mask[2, 1, 0] = True  # -> contralateral (3, 5, 0): collision voxel A
    mask[2, 2, 0] = True  # -> contralateral (3, 5, 0): collision voxel B
    mask[2, 0, 0] = True  # -> contralateral (3, 6, 0): distinct, non-colliding
    suv[2, 1, 0] = 3.0
    suv[2, 2, 0] = 7.0
    suv[2, 0, 0] = 1.0
    suv[3, 5, 0] = 9.0  # the single contralateral voxel two targets collide onto
    suv[3, 6, 0] = 2.0  # the distinct contralateral voxel the third target hits

    result = quantify_target(suv, mask, geometry, reflection)

    assert result.contralateral_suvmax == Measurement(pytest.approx(9.0))
    assert result.contralateral_suvmean == Measurement(pytest.approx(5.5))


# ---------------------------------------------------------------------------
# QC families: fragmentation (26-connectivity), boundary-touching,
# partial-volume risk, misregistration placeholder.
# ---------------------------------------------------------------------------


def test_fragmentation_two_separated_components_flagged() -> None:
    shape = (8, 8, 8)
    geometry = _make_geometry(shape, _identity_affine())
    reflection = _mirror_affine_through_x(offset=20.0)  # keep contralateral irrelevant
    suv = _suvbw(shape)
    mask = _bool_mask(shape)
    mask[0, 0, 0] = True
    mask[5, 5, 5] = True  # far apart: two genuinely separate components

    result = quantify_target(suv, mask, geometry, reflection)
    assert result.qc.component_count == 2
    assert result.qc.fragmented is True


def test_fragmentation_diagonal_adjacency_counts_as_one_component_26conn() -> None:
    # Two voxels sharing only a corner are connected under 26-connectivity
    # (would be 2 separate components under 6-connectivity) -- this proves
    # the engine uses 26-connectivity, not face-only adjacency.
    shape = (8, 8, 8)
    geometry = _make_geometry(shape, _identity_affine())
    reflection = _mirror_affine_through_x(offset=20.0)
    suv = _suvbw(shape)
    mask = _bool_mask(shape)
    mask[2, 2, 2] = True
    mask[3, 3, 3] = True  # corner-adjacent to (2,2,2)

    result = quantify_target(suv, mask, geometry, reflection)
    assert result.qc.component_count == 1
    assert result.qc.fragmented is False


def test_boundary_touching_flag() -> None:
    shape = (6, 6, 6)
    geometry = _make_geometry(shape, _identity_affine())
    reflection = _mirror_affine_through_x(offset=20.0)
    suv = _suvbw(shape)

    touching_mask = _bool_mask(shape)
    touching_mask[0, 3, 3] = True  # index 0 on the first axis: touches the edge
    touching = quantify_target(suv, touching_mask, geometry, reflection)
    assert touching.qc.boundary_touching is True

    interior_mask = _bool_mask(shape)
    interior_mask[3, 3, 3] = True  # fully interior
    interior = quantify_target(suv, interior_mask, geometry, reflection)
    assert interior.qc.boundary_touching is False


def test_partial_volume_risk_voxel_count_threshold() -> None:
    shape = (10, 10, 10)
    geometry = _make_geometry(shape, _identity_affine())
    reflection = _mirror_affine_through_x(offset=20.0)
    suv = _suvbw(shape)

    small_mask = _bool_mask(shape)
    small_mask[0:2, 0:2, 0:2] = True  # 8 voxels, well under the 27-voxel rule
    small = quantify_target(suv, small_mask, geometry, reflection)
    assert small.qc.partial_volume_risk is True

    exactly_27_mask = _bool_mask(shape)
    exactly_27_mask[0:3, 0:3, 0:3] = True  # a full 3x3x3 cube: exactly 27 voxels
    at_threshold = quantify_target(suv, exactly_27_mask, geometry, reflection)
    assert at_threshold.qc.component_count == 1
    assert at_threshold.qc.partial_volume_risk is False


def test_partial_volume_risk_uses_blur_fwhm_when_supplied() -> None:
    shape = (10, 10, 10)
    geometry = _make_geometry(shape, _identity_affine())
    reflection = _mirror_affine_through_x(offset=20.0)
    suv = _suvbw(shape)
    mask = _bool_mask(shape)
    mask[0:5, 0, 0] = True  # 5 voxels, 4 mm extent along x, well over 27-voxel rule N/A

    small_fwhm = quantify_target(
        suv, mask, geometry, reflection, blur_fwhm_mm=2.0
    )  # extent (4 mm) >= fwhm (2 mm)
    assert small_fwhm.qc.partial_volume_risk is False

    large_fwhm = quantify_target(
        suv, mask, geometry, reflection, blur_fwhm_mm=10.0
    )  # extent (4 mm) < fwhm (10 mm)
    assert large_fwhm.qc.partial_volume_risk is True


def test_misregistration_risk_placeholder_is_deterministic_passthrough() -> None:
    suv, mask = _left_side_scenario()
    no_shift = quantify_target(suv, mask, _GEOMETRY, _REFLECTION_X0)
    assert no_shift.qc.misregistration_risk_placeholder is False

    with_shift = quantify_target(
        suv, mask, _GEOMETRY, _REFLECTION_X0, registration_shift_mm=2.0
    )
    assert with_shift.qc.misregistration_risk_placeholder is True


# ---------------------------------------------------------------------------
# Raw-vs-normalized separation: the engine measures exactly what it is
# given, with no internal clip/scale step.
# ---------------------------------------------------------------------------


def test_raw_array_measured_as_is_no_silent_normalization() -> None:
    suv = _suvbw(_SHAPE)
    mask = _bool_mask(_SHAPE)
    mask[0, 1, 1] = True
    mask[1, 1, 1] = True
    # One value inside the [0, 10] network-clip range, one value well above
    # it. If the engine silently applied the project's network-normalization
    # (clip to [0, 10] then divide by 10 -- see imaging-physics.md, "Network
    # inputs"), suvmax would come back as 1.0, not 15.0.
    suv[0, 1, 1] = 0.05
    suv[1, 1, 1] = 15.0

    result = quantify_target(suv, mask, _GEOMETRY, _REFLECTION_X0)
    assert result.target_suvmax == Measurement(pytest.approx(15.0))
    assert result.target_suvmean == Measurement(pytest.approx((0.05 + 15.0) / 2))


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_determinism_identical_inputs_identical_output() -> None:
    suv, mask = _left_side_scenario()
    result_1 = quantify_target(suv, mask, _GEOMETRY, _REFLECTION_X0)
    result_2 = quantify_target(suv.copy(), mask.copy(), _GEOMETRY, _REFLECTION_X0)
    assert result_1 == result_2
    assert isinstance(result_1, QuantificationResult)


# ---------------------------------------------------------------------------
# Input-contract validation (ValueError / TypeError, not a structured null:
# these are caller-contract violations, not data-derived invalid results).
# ---------------------------------------------------------------------------


def test_suvbw_shape_mismatch_raises_value_error() -> None:
    wrong_shape_suv = np.zeros((3, 3, 3))
    mask = _bool_mask(_SHAPE)
    with pytest.raises(ValueError):
        quantify_target(wrong_shape_suv, mask, _GEOMETRY, _REFLECTION_X0)


def test_mask_shape_mismatch_raises_value_error() -> None:
    suv = _suvbw(_SHAPE)
    wrong_shape_mask = np.zeros((3, 3, 3), dtype=bool)
    with pytest.raises(ValueError):
        quantify_target(suv, wrong_shape_mask, _GEOMETRY, _REFLECTION_X0)


def test_mask_non_integer_dtype_raises_type_error() -> None:
    suv = _suvbw(_SHAPE)
    float_mask = np.zeros(_SHAPE, dtype=np.float32)
    with pytest.raises(TypeError):
        quantify_target(suv, float_mask, _GEOMETRY, _REFLECTION_X0)


@pytest.mark.parametrize(
    "bad_reflection",
    [
        np.eye(3),  # wrong shape
        np.array(  # non-finite entry
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, np.nan, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        np.array(  # not symmetric
            [
                [0.0, 1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        np.diag([-2.0, 1.0, 1.0, 1.0]),  # symmetric but not an involution
        np.eye(4),  # orientation-preserving (det > 0), not a mirror
        # Full 3-axis point inversion (-I): symmetric, involutive, and
        # det < 0 (det((-I)) == -1) -- satisfies every check *except* the
        # eigenvalue-multiplicity one (all three eigenvalues are -1, not
        # "one -1, two +1"), so this is exactly the case that check exists
        # to reject: -I is not a single-plane mirror.
        np.diag([-1.0, -1.0, -1.0, 1.0]),
        # A different "improper but involutive" transform: an oblique skew
        # involution (L @ L == I, det(L) == -1) that is NOT symmetric, i.e.
        # not an orthogonal reflection about any plane at all.
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [2.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    ],
)
def test_malformed_reflection_affine_raises_value_error(
    bad_reflection: np.ndarray,
) -> None:
    suv, mask = _left_side_scenario()
    with pytest.raises(ValueError):
        quantify_target(suv, mask, _GEOMETRY, bad_reflection)


def test_full_point_inversion_rejected_eigenvalue_multiplicity_check() -> None:
    # Dedicated, non-parametrized repro for the specific bug this check
    # fixes: -I passes symmetric + involutive + det<0 but is not a mirror.
    full_inversion = np.eye(4)
    full_inversion[:3, :3] = -np.eye(3)
    suv, mask = _left_side_scenario()
    with pytest.raises(ValueError, match="single-plane mirror"):
        quantify_target(suv, mask, _GEOMETRY, full_inversion)


def test_true_mirrors_are_accepted_axis_aligned_and_oblique() -> None:
    # A true single-plane Householder mirror -- axis-aligned (the fixture
    # used throughout this file) and a rotated/oblique one -- must both be
    # accepted (no raise), proving the new eigenvalue check does not
    # over-reject genuine mid-sagittal-style mirrors.
    suv, mask = _left_side_scenario()
    axis_aligned_result = quantify_target(suv, mask, _GEOMETRY, _REFLECTION_X0)
    assert axis_aligned_result.laterality == "left"

    oblique_normal = np.array([0.6, 0.8, 0.0])  # unit vector, not axis-aligned
    oblique_mirror = np.eye(4)
    oblique_mirror[:3, :3] = np.eye(3) - 2.0 * np.outer(oblique_normal, oblique_normal)
    oblique_result = quantify_target(suv, mask, _GEOMETRY, oblique_mirror)
    assert oblique_result.target_suvmax.is_valid


def test_centerline_direction_wrong_shape_raises_value_error() -> None:
    suv, mask = _left_side_scenario()
    with pytest.raises(ValueError):
        quantify_target(
            suv,
            mask,
            _GEOMETRY,
            _REFLECTION_X0,
            centerline_direction=np.array([1.0, 0.0]),
        )


def test_centerline_direction_zero_vector_raises_value_error() -> None:
    suv, mask = _left_side_scenario()
    with pytest.raises(ValueError):
        quantify_target(
            suv,
            mask,
            _GEOMETRY,
            _REFLECTION_X0,
            centerline_direction=np.array([0.0, 0.0, 0.0]),
        )


def test_centerline_direction_nonfinite_raises_value_error() -> None:
    suv, mask = _left_side_scenario()
    with pytest.raises(ValueError):
        quantify_target(
            suv,
            mask,
            _GEOMETRY,
            _REFLECTION_X0,
            centerline_direction=np.array([np.nan, 0.0, 0.0]),
        )
