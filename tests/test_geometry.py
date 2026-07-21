"""Tests for the affine-aware PET/CT geometry contract (VascuTrace Phase 1).

Every fixture in this file is a small, generated, non-patient NIfTI phantom
built directly with ``nibabel.Nifti1Image`` and explicit qform/sform/units
headers -- nothing here reads ``Data/``, the network, or any external
service.

Two error codes (``SINGULAR_AFFINE``, ``INVALID_SPACING``) and one
(``NONFINITE_AFFINE``) are exercised against the module's private numeric
core, ``_grid_geometry_from_affine``, rather than through a full
``nib.Nifti1Image``: nibabel's own ``Nifti1Image`` constructor independently
rejects non-finite and singular/degenerate-column affines at construction
time (it cannot decompose them into a qform quaternion), so a legitimately
"coded" NIfTI header carrying one of those pathological affines cannot exist
in the first place. This was confirmed empirically (see the implementation documentation)
before committing to this test strategy, rather than assumed. Testing the
shared private core directly is a deliberate, documented choice, not a
shortcut around the public contract: ``validate_nifti_grid`` delegates to
this exact function for every one of these checks after resolving the NIfTI
header, so covering the function directly still verifies each branch of the
same code the public API runs.
"""

from __future__ import annotations

import dataclasses

import nibabel as nib
import numpy as np
import pytest

from src.vascutrace.geometry import (
    RESEARCH_PROTOTYPE_WARNING,
    C_lps_from_ras,
    CanonicalVolume,
    GeometryContractError,
    GeometryErrorCode,
    GeometrySidecar,
    GridGeometry,
    SpatialArray,
    _exact_voxel_remap,
    _grid_geometry_from_affine,
    canonicalize_nifti_to_ras,
    input_lps_from_output_lps,
    lps_points_to_ras,
    ras_points_to_lps,
    resample_ct_to_pet,
    resample_mask_to_pet,
    restore_mask_to_original_pet,
    validate_nifti_grid,
)

# ---------------------------------------------------------------------------
# Fixture helpers (generated phantoms only)
# ---------------------------------------------------------------------------


def _rotation_matrix(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """A standard Rodrigues rotation matrix, used only to build non-trivial
    (rotated, anisotropic) affines for test fixtures.
    """
    axis = axis / np.linalg.norm(axis)
    angle = np.radians(angle_deg)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    one_minus_c = 1 - c
    return np.array(
        [
            [
                x * x * one_minus_c + c,
                x * y * one_minus_c - z * s,
                x * z * one_minus_c + y * s,
            ],
            [
                y * x * one_minus_c + z * s,
                y * y * one_minus_c + c,
                y * z * one_minus_c - x * s,
            ],
            [
                z * x * one_minus_c - y * s,
                z * y * one_minus_c + x * s,
                z * z * one_minus_c + c,
            ],
        ]
    )


def _rotated_anisotropic_affine(
    axis: tuple[float, float, float],
    angle_deg: float,
    spacing: tuple[float, float, float],
    offset: tuple[float, float, float],
) -> np.ndarray:
    linear = _rotation_matrix(np.array(axis), angle_deg) @ np.diag(spacing)
    affine = np.eye(4)
    affine[:3, :3] = linear
    affine[:3, 3] = offset
    return affine


def _make_nifti_image(
    shape: tuple[int, int, int],
    affine: np.ndarray,
    data: np.ndarray | None = None,
    *,
    dtype: np.dtype = np.float32,
    qform_code: int = 1,
    sform_code: int = 1,
    xyz_units: str = "mm",
) -> nib.Nifti1Image:
    """Build a small, generated, non-patient NIfTI phantom with explicit,
    independently-set qform/sform codes and spatial units (nibabel's
    defaults are *not* validation-passing: fresh images have an uncoded
    qform and 'unknown' units, which is exactly why several invalid-header
    tests below simply omit one of these calls).
    """
    if data is None:
        rng = np.random.default_rng(0)
        if np.issubdtype(dtype, np.integer):
            data = rng.integers(0, 4, size=shape).astype(dtype)
        else:
            data = rng.random(shape).astype(dtype)
    img = nib.Nifti1Image(data, affine)
    if qform_code:
        img.header.set_qform(affine, code=qform_code)
    if sform_code:
        img.header.set_sform(affine, code=sform_code)
    if xyz_units:
        img.header.set_xyzt_units(xyz=xyz_units, t="sec")
    return img


VALID_SHAPE = (6, 7, 8)
VALID_AFFINE = _rotated_anisotropic_affine(
    axis=(0.2, 0.6, 0.3),
    angle_deg=17.0,
    spacing=(1.5, 2.0, 2.5),
    offset=(5.0, -3.0, 2.0),
)


def _valid_label_image() -> nib.Nifti1Image:
    labels = np.zeros(VALID_SHAPE, dtype=np.int16)
    labels[1:3, 2:4, 3:5] = 1
    labels[4, 5, 6] = 2
    labels[0, 0, 0] = 3
    return _make_nifti_image(VALID_SHAPE, VALID_AFFINE, data=labels, dtype=np.int16)


# ---------------------------------------------------------------------------
# RESEARCH_PROTOTYPE_WARNING / C_lps_from_ras / GeometryContractError
# ---------------------------------------------------------------------------


def test_research_prototype_warning_exact_text() -> None:
    assert RESEARCH_PROTOTYPE_WARNING == (
        "Research prototype. Trained and evaluated using simulated vascular-like "
        "abnormalities, not confirmed human post-angioplasty lesions."
    )


def test_c_lps_from_ras_value() -> None:
    assert np.array_equal(np.asarray(C_lps_from_ras), np.diag([-1.0, -1.0, 1.0, 1.0]))


def test_geometry_contract_error_message_is_code_only() -> None:
    err = GeometryContractError(GeometryErrorCode.SHEARED_GRID)
    assert err.code is GeometryErrorCode.SHEARED_GRID
    assert str(err) == "SHEARED_GRID"
    assert isinstance(err, ValueError)


def test_geometry_error_code_members() -> None:
    expected = {
        "INVALID_SHAPE",
        "INVALID_AFFINE",
        "NONFINITE_AFFINE",
        "SINGULAR_AFFINE",
        "AMBIGUOUS_AFFINE",
        "INVALID_SPATIAL_UNITS",
        "INVALID_SPACING",
        "SHEARED_GRID",
        "GEOMETRY_MISMATCH",
    }
    assert {member.value for member in GeometryErrorCode} == expected


# ---------------------------------------------------------------------------
# validate_nifti_grid: valid case + every invalid branch (exact error code)
# ---------------------------------------------------------------------------


def test_validate_nifti_grid_valid_rotated_anisotropic() -> None:
    img = _make_nifti_image(VALID_SHAPE, VALID_AFFINE)
    geometry = validate_nifti_grid(img)
    assert isinstance(geometry, GridGeometry)
    assert geometry.shape == VALID_SHAPE
    assert geometry.units == "mm"
    expected_spacing = tuple(float(x) for x in (1.5, 2.0, 2.5))
    assert geometry.spacing == pytest.approx(expected_spacing, abs=1e-6)
    assert np.array_equal(np.asarray(geometry.affine), VALID_AFFINE)


@pytest.mark.parametrize(
    "shape",
    [
        (6, 7, 8, 1),  # singleton 4D
        (6, 7),  # 2D
        (0, 7, 8),  # zero-length axis
    ],
)
def test_validate_nifti_grid_invalid_shape(shape: tuple[int, ...]) -> None:
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    data = np.zeros(shape, dtype=np.int16)
    img = _make_nifti_image(
        shape[:3] if len(shape) >= 3 else (1, 1, 1), affine, data=data
    )
    with pytest.raises(GeometryContractError) as exc_info:
        validate_nifti_grid(img)
    assert exc_info.value.code is GeometryErrorCode.INVALID_SHAPE


def test_validate_nifti_grid_uncoded_qform_is_ambiguous() -> None:
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    img = _make_nifti_image((3, 4, 5), affine, qform_code=0)
    with pytest.raises(GeometryContractError) as exc_info:
        validate_nifti_grid(img)
    assert exc_info.value.code is GeometryErrorCode.AMBIGUOUS_AFFINE


def test_validate_nifti_grid_mismatched_qform_sform_is_ambiguous() -> None:
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    other_affine = np.diag([3.0, 3.0, 3.0, 1.0])
    data = np.zeros((3, 4, 5), dtype=np.int16)
    img = nib.Nifti1Image(data, affine)
    img.header.set_qform(affine, code=1)
    img.header.set_sform(other_affine, code=1)
    img.header.set_xyzt_units(xyz="mm", t="sec")
    with pytest.raises(GeometryContractError) as exc_info:
        validate_nifti_grid(img)
    assert exc_info.value.code is GeometryErrorCode.AMBIGUOUS_AFFINE


def test_validate_nifti_grid_unknown_units_is_invalid() -> None:
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    img = _make_nifti_image(
        (3, 4, 5), affine, xyz_units=""
    )  # nibabel default: 'unknown'
    assert img.header.get_xyzt_units()[0] == "unknown"
    with pytest.raises(GeometryContractError) as exc_info:
        validate_nifti_grid(img)
    assert exc_info.value.code is GeometryErrorCode.INVALID_SPATIAL_UNITS


def test_grid_geometry_from_affine_nonfinite() -> None:
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    affine[0, 0] = np.nan
    with pytest.raises(GeometryContractError) as exc_info:
        _grid_geometry_from_affine((3, 4, 5), affine, "mm")
    assert exc_info.value.code is GeometryErrorCode.NONFINITE_AFFINE

    affine_inf = np.diag([2.0, 2.0, 2.0, 1.0])
    affine_inf[1, 2] = np.inf
    with pytest.raises(GeometryContractError) as exc_info:
        _grid_geometry_from_affine((3, 4, 5), affine_inf, "mm")
    assert exc_info.value.code is GeometryErrorCode.NONFINITE_AFFINE


def test_grid_geometry_from_affine_zero_spacing_column_is_invalid_spacing() -> None:
    # Column 1 (the "j" voxel axis) has zero length: a genuinely degenerate
    # axis. This is also singular (det == 0), but INVALID_SPACING is the
    # more specific, independently-reachable code for exactly this case
    # (see the module's check-ordering comment); SINGULAR_AFFINE is
    # covered separately below by a matrix with no zero-norm column.
    affine = np.eye(4)
    affine[:3, :3] = np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
    with pytest.raises(GeometryContractError) as exc_info:
        _grid_geometry_from_affine((3, 4, 5), affine, "mm")
    assert exc_info.value.code is GeometryErrorCode.INVALID_SPACING


def test_grid_geometry_from_affine_singular_with_positive_spacing() -> None:
    # Every column has positive, finite norm (2, 1, 2) but columns 0 and 1
    # are parallel, so the matrix is singular without any zero-norm column
    # -- this is only reachable as SINGULAR_AFFINE, not INVALID_SPACING.
    affine = np.eye(4)
    affine[:3, :3] = np.array([[2.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
    assert np.all(np.linalg.norm(affine[:3, :3], axis=0) > 0)
    with pytest.raises(GeometryContractError) as exc_info:
        _grid_geometry_from_affine((3, 4, 5), affine, "mm")
    assert exc_info.value.code is GeometryErrorCode.SINGULAR_AFFINE


def test_grid_geometry_from_affine_ill_conditioned_is_singular() -> None:
    # Determinant (0.1) is comfortably above the 1e-12 threshold -- this
    # fixture is chosen specifically to fail only the *condition-number*
    # sub-check (cond == 1e13 > 1e12), not the determinant sub-check, so it
    # exercises that branch distinctly rather than incidentally overlapping
    # with the (separately tested) exact-zero-determinant case.
    affine = np.eye(4)
    affine[:3, :3] = np.diag([1e6, 1.0, 1e-7])
    assert abs(np.linalg.det(affine[:3, :3])) > 1e-12
    assert np.linalg.cond(affine[:3, :3]) > 1e12
    with pytest.raises(GeometryContractError) as exc_info:
        _grid_geometry_from_affine((3, 4, 5), affine, "mm")
    assert exc_info.value.code is GeometryErrorCode.SINGULAR_AFFINE


def test_grid_geometry_from_affine_sheared() -> None:
    affine = np.eye(4)
    affine[:3, :3] = np.array([[2.0, 0.3, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
    with pytest.raises(GeometryContractError) as exc_info:
        _grid_geometry_from_affine((3, 4, 5), affine, "mm")
    assert exc_info.value.code is GeometryErrorCode.SHEARED_GRID


def test_grid_geometry_from_affine_valid_passes_through_core() -> None:
    geometry = _grid_geometry_from_affine(
        (3, 4, 5), np.diag([2.0, 2.0, 2.0, 1.0]), "mm"
    )
    assert geometry.shape == (3, 4, 5)
    assert geometry.spacing == (2.0, 2.0, 2.0)


# ---------------------------------------------------------------------------
# canonicalize_nifti_to_ras: labels, fiducials, inverse maps, RAS+ sign
# ---------------------------------------------------------------------------


def test_canonicalize_exact_integer_label_round_trip() -> None:
    img = _valid_label_image()
    original_labels = np.asarray(img.dataobj).copy()

    canonical = canonicalize_nifti_to_ras(img)
    assert isinstance(canonical, CanonicalVolume)
    assert np.asarray(canonical.data).dtype == np.int16
    # RAS+ canonicalization is a pure signed permutation: the *set* of
    # labels present must be unchanged, and every original voxel's label
    # must survive somewhere in the canonical array.
    assert set(np.asarray(canonical.data).ravel().tolist()) == set(
        original_labels.ravel().tolist()
    )

    restored, _sidecar = restore_mask_to_original_pet(
        np.asarray(canonical.data), canonical
    )
    assert np.asarray(restored).dtype == np.int16
    assert np.array_equal(np.asarray(restored), original_labels)


def test_canonicalize_result_is_ras_plus_oriented() -> None:
    # RAS+ means: as each canonical voxel index increases, world position
    # increases toward R, A, S respectively -- i.e. the canonical affine's
    # linear part has strictly positive entries on its "dominant" diagonal
    # once its (signed-permutation) column order is fixed to (R, A, S) rows.
    img = _valid_label_image()
    canonical = canonicalize_nifti_to_ras(img)
    linear = np.asarray(canonical.geometry.affine)[:3, :3]
    dominant = linear[np.arange(3), np.argmax(np.abs(linear), axis=1)]
    assert np.all(dominant > 0)


def test_canonicalize_fiducial_world_coordinate_agreement() -> None:
    """World-coordinate agreement within 1e-6 mm (contract requirement):
    for explicit fiducial voxels, world position computed via the canonical
    affine must agree with world position computed by mapping the same
    point back to the original grid and applying the original affine.
    """
    img = _valid_label_image()
    canonical = canonicalize_nifti_to_ras(img)

    original_affine = VALID_AFFINE
    canonical_affine = np.asarray(canonical.geometry.affine)
    original_from_canonical = np.asarray(canonical.original_voxel_from_canonical_voxel)

    fiducials = np.array(
        [
            [0, 0, 0],
            [canonical.geometry.shape[0] - 1, 0, 0],
            [0, canonical.geometry.shape[1] - 1, 0],
            [0, 0, canonical.geometry.shape[2] - 1],
            [2, 3, 4],
        ],
        dtype=np.float64,
    )
    fiducials_h = np.concatenate([fiducials, np.ones((fiducials.shape[0], 1))], axis=1)

    world_via_canonical = (canonical_affine @ fiducials_h.T).T[:, :3]
    original_voxels_h = (original_from_canonical @ fiducials_h.T).T
    world_via_original = (original_affine @ original_voxels_h.T).T[:, :3]

    assert np.max(np.abs(world_via_canonical - world_via_original)) < 1e-6


def test_canonicalize_inverse_voxel_maps_are_exact_inverses() -> None:
    """Mapping inverse residual within 1e-10 (contract requirement)."""
    img = _valid_label_image()
    canonical = canonicalize_nifti_to_ras(img)
    forward = np.asarray(canonical.canonical_voxel_from_original_voxel)
    backward = np.asarray(canonical.original_voxel_from_canonical_voxel)
    residual = forward @ backward - np.eye(4)
    assert np.max(np.abs(residual)) < 1e-10


def test_canonicalize_original_geometry_matches_validate_nifti_grid() -> None:
    img = _valid_label_image()
    expected = validate_nifti_grid(img)
    canonical = canonicalize_nifti_to_ras(img)
    assert canonical.original_geometry.shape == expected.shape
    assert np.array_equal(
        np.asarray(canonical.original_geometry.affine), np.asarray(expected.affine)
    )


def test_canonicalize_is_idempotent_on_an_already_ras_plus_grid() -> None:
    # An axis-aligned, already-RAS+ affine should canonicalize to itself
    # (identity voxel mapping), since it is already the closest-canonical
    # orientation.
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    img = _make_nifti_image((4, 5, 6), affine)
    canonical = canonicalize_nifti_to_ras(img)
    assert canonical.geometry.shape == (4, 5, 6)
    assert np.array_equal(
        np.asarray(canonical.canonical_voxel_from_original_voxel), np.eye(4)
    )
    assert np.array_equal(np.asarray(canonical.data), np.asarray(img.dataobj))


# ---------------------------------------------------------------------------
# _exact_voxel_remap: memory-efficient transpose/flip == former dense gather
# ---------------------------------------------------------------------------


def _dense_gather_remap_reference(
    source: np.ndarray,
    output_shape: tuple[int, int, int],
    input_voxel_from_output_voxel: np.ndarray,
) -> np.ndarray:
    """The former ``_exact_voxel_remap`` implementation, kept here verbatim as
    the reference the memory-efficient transpose/flip version must match
    bit-for-bit. Builds the full float64 per-voxel index grid and gathers.
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
    input_index = np.rint(input_index_h[:3]).astype(np.int64)
    remapped = source[input_index[0], input_index[1], input_index[2]]
    return remapped.reshape(output_shape)


def test_exact_voxel_remap_matches_dense_gather() -> None:
    """The transpose/flip implementation is byte-identical to the former
    dense-index gather across all 48 signed axis permutations, for a source
    with three distinct dimensions (so any wrong axis/order/flip is caught).
    """
    import itertools

    rng = np.random.default_rng(20260721)
    src_shape = (4, 5, 6)
    source = rng.integers(-7, 100, size=src_shape).astype(np.int16)

    for perm in itertools.permutations(
        range(3)
    ):  # perm[j] = output axis of source axis j
        for signs in itertools.product((1.0, -1.0), repeat=3):
            linear = np.zeros((3, 3), dtype=np.float64)
            output_shape = [0, 0, 0]
            translation = np.zeros(3, dtype=np.float64)
            for j in range(3):
                linear[j, perm[j]] = signs[j]
                output_shape[perm[j]] = src_shape[j]
                translation[j] = (src_shape[j] - 1) if signs[j] < 0 else 0
            mapping = np.eye(4, dtype=np.float64)
            mapping[:3, :3] = linear
            mapping[:3, 3] = translation
            output_shape_t = (output_shape[0], output_shape[1], output_shape[2])

            reference = _dense_gather_remap_reference(source, output_shape_t, mapping)
            fast = _exact_voxel_remap(source, output_shape_t, mapping)

            assert fast.shape == reference.shape == output_shape_t
            assert fast.dtype == reference.dtype == source.dtype
            assert np.array_equal(fast, reference), (perm, signs)


def test_exact_voxel_remap_rejects_non_permutation_mapping() -> None:
    source = np.zeros((4, 5, 6), dtype=np.int16)
    fractional = np.eye(4, dtype=np.float64)
    fractional[0, 0] = 0.5  # not a signed permutation entry
    with pytest.raises(GeometryContractError) as excinfo:
        _exact_voxel_remap(source, (4, 5, 6), fractional)
    assert excinfo.value.code == GeometryErrorCode.INVALID_AFFINE


def test_exact_voxel_remap_rejects_noninteger_translation() -> None:
    source = np.zeros((4, 5, 6), dtype=np.int16)
    shifted = np.eye(4, dtype=np.float64)
    shifted[0, 3] = 0.5  # fractional translation
    with pytest.raises(GeometryContractError) as excinfo:
        _exact_voxel_remap(source, (4, 5, 6), shifted)
    assert excinfo.value.code == GeometryErrorCode.INVALID_AFFINE


# ---------------------------------------------------------------------------
# RAS <-> LPS bridge: sign convention, involution, transform direction
# ---------------------------------------------------------------------------


def test_ras_to_lps_sign_convention_asymmetric_fiducial() -> None:
    # Asymmetric R/A/S components so a sign-flip bug on the wrong axis
    # cannot hide behind a symmetric input (per imaging-physics.md:
    # "Unit-test this bridge with asymmetric R/A/S fiducials").
    point_ras = np.array([[11.0, -7.0, 3.0]])
    result = ras_points_to_lps(point_ras)
    assert isinstance(result, SpatialArray)
    assert result.space == "LPS"
    assert np.array_equal(np.asarray(result.points), np.array([[-11.0, 7.0, 3.0]]))


def test_lps_to_ras_sign_convention_asymmetric_fiducial() -> None:
    point_lps = np.array([[-11.0, 7.0, 3.0]])
    result = lps_points_to_ras(point_lps)
    assert result.space == "RAS"
    assert np.array_equal(np.asarray(result.points), np.array([[11.0, -7.0, 3.0]]))


def test_ras_lps_reflection_is_an_exact_involution() -> None:
    points = np.array([[1.0, 2.0, 3.0], [-4.0, 5.5, -6.25], [9.0, -9.0, 0.5]])
    round_tripped = lps_points_to_ras(ras_points_to_lps(points).points)
    assert np.array_equal(np.asarray(round_tripped.points), points)


def test_ras_points_to_lps_accepts_single_point_returns_n_by_3() -> None:
    single = np.array([1.0, 2.0, 3.0])
    result = ras_points_to_lps(single)
    assert np.asarray(result.points).shape == (1, 3)
    assert np.array_equal(np.asarray(result.points), np.array([[-1.0, -2.0, 3.0]]))


def test_input_lps_from_output_lps_identity() -> None:
    bridged = input_lps_from_output_lps(np.eye(4))
    assert np.array_equal(np.asarray(bridged), np.eye(4))


def test_input_lps_from_output_lps_conjugation_formula() -> None:
    t_ras = np.eye(4)
    t_ras[:3, :3] = _rotation_matrix(np.array([0.0, 0.0, 1.0]), 90.0)
    t_ras[:3, 3] = [1.0, 2.0, 3.0]
    bridged = input_lps_from_output_lps(t_ras)
    expected = np.asarray(C_lps_from_ras) @ t_ras @ np.asarray(C_lps_from_ras)
    assert np.allclose(np.asarray(bridged), expected, atol=0, rtol=0)


def test_input_lps_from_output_lps_physical_transform_direction() -> None:
    """The bridged transform must map an output-LPS point to the *same*
    physical location as applying the original RAS transform in RAS space
    and then converting to LPS -- i.e. the two paths (convert-then-
    transform vs. transform-then-convert) must agree exactly.
    """
    t_ras_input_from_output = np.eye(4)
    t_ras_input_from_output[:3, 3] = [4.0, -5.0, 6.0]

    output_point_ras = np.array([[10.0, 20.0, 30.0]])
    expected_input_point_ras = output_point_ras + t_ras_input_from_output[:3, 3]

    output_point_lps = ras_points_to_lps(output_point_ras).points
    t_lps = input_lps_from_output_lps(t_ras_input_from_output)
    output_point_lps_h = np.concatenate(
        [np.asarray(output_point_lps), np.ones((1, 1))], axis=1
    )
    input_point_lps = (np.asarray(t_lps) @ output_point_lps_h.T).T[:, :3]
    input_point_ras_via_bridge = lps_points_to_ras(input_point_lps).points

    assert np.allclose(
        np.asarray(input_point_ras_via_bridge), expected_input_point_ras, atol=1e-9
    )


def test_geometry_contract_error_on_malformed_bridge_input() -> None:
    with pytest.raises(GeometryContractError) as exc_info:
        input_lps_from_output_lps(np.eye(3))
    assert exc_info.value.code is GeometryErrorCode.INVALID_AFFINE

    bad = np.eye(4)
    bad[0, 0] = np.nan
    with pytest.raises(GeometryContractError) as exc_info:
        input_lps_from_output_lps(bad)
    assert exc_info.value.code is GeometryErrorCode.NONFINITE_AFFINE


# ---------------------------------------------------------------------------
# resample_ct_to_pet: analytic rotated-anisotropic HU ramp, <= 1e-4 HU
# ---------------------------------------------------------------------------


def _analytic_ramp_hu(world_points: np.ndarray) -> np.ndarray:
    a, b, c, d = 2.0, -1.5, 0.7, 100.0
    return a * world_points[:, 0] + b * world_points[:, 1] + c * world_points[:, 2] + d


def test_resample_ct_to_pet_analytic_ramp_accuracy() -> None:
    ct_shape = (10, 11, 12)
    ct_affine = _rotated_anisotropic_affine(
        axis=(0.4, 0.1, 0.9),
        angle_deg=31.0,
        spacing=(1.1, 0.9, 1.3),
        offset=(0.0, 0.0, 0.0),
    )
    idx = np.indices(ct_shape, dtype=np.float64)
    idx_h = np.stack(
        [
            idx[0].ravel(),
            idx[1].ravel(),
            idx[2].ravel(),
            np.ones(int(np.prod(ct_shape))),
        ]
    )
    ct_world = (ct_affine @ idx_h).T[:, :3]
    ct_hu = _analytic_ramp_hu(ct_world).reshape(ct_shape).astype(np.float32)
    ct_img = _make_nifti_image(ct_shape, ct_affine, data=ct_hu, dtype=np.float32)
    ct_canonical = canonicalize_nifti_to_ras(ct_img)

    pet_shape = (8, 9, 10)
    pet_affine = _rotated_anisotropic_affine(
        axis=(0.1, 0.3, 0.2),
        angle_deg=9.0,
        spacing=(2.0, 2.0, 2.0),
        offset=(1.0, 1.0, 1.0),
    )
    pet_img = _make_nifti_image(
        pet_shape,
        pet_affine,
        data=np.zeros(pet_shape, dtype=np.float32),
        dtype=np.float32,
    )
    pet_canonical = canonicalize_nifti_to_ras(pet_img)

    resampled = resample_ct_to_pet(ct_canonical, pet_canonical.geometry)
    resampled_arr = np.asarray(resampled)
    assert resampled_arr.dtype == np.float32
    assert resampled_arr.shape == pet_canonical.geometry.shape

    pet_out_shape = pet_canonical.geometry.shape
    idx_p = np.indices(pet_out_shape, dtype=np.float64)
    idx_p_h = np.stack(
        [
            idx_p[0].ravel(),
            idx_p[1].ravel(),
            idx_p[2].ravel(),
            np.ones(int(np.prod(pet_out_shape))),
        ]
    )
    world_p = (np.asarray(pet_canonical.geometry.affine) @ idx_p_h).T[:, :3]
    expected = _analytic_ramp_hu(world_p).reshape(pet_out_shape)

    # "Inside the CT field of view" must be judged in the CT grid's own
    # fractional voxel-index space (the CT footprint is a rotated box, not
    # an axis-aligned one), with a full-voxel margin from every boundary so
    # linear interpolation's boundary blending against the constant
    # zero-fill cannot contaminate the comparison. A world-space axis-
    # aligned bounding-box test would over-include points outside the
    # actual rotated footprint and was empirically confirmed, during
    # development, to produce spurious ~100 HU "errors" that were entirely
    # a test-methodology artifact, not a resampler defect.
    ct_affine_canonical = np.asarray(ct_canonical.geometry.affine)
    ct_shape_canonical = np.array(ct_canonical.geometry.shape, dtype=np.float64)
    world_p_h = np.concatenate([world_p, np.ones((world_p.shape[0], 1))], axis=1)
    ct_voxel_p = (np.linalg.inv(ct_affine_canonical) @ world_p_h.T).T[:, :3]
    margin = 1.0
    inside = np.all(
        (ct_voxel_p >= margin) & (ct_voxel_p <= (ct_shape_canonical - 1 - margin)),
        axis=1,
    ).reshape(pet_out_shape)
    assert inside.sum() >= 20, (
        "fixture must exercise a nontrivial interior overlap region"
    )

    max_error = np.max(
        np.abs(resampled_arr.astype(np.float64)[inside] - expected[inside])
    )
    assert max_error <= 1e-4


def test_resample_ct_to_pet_zero_outside_field_of_view() -> None:
    ct_shape = (3, 3, 3)
    ct_affine = np.diag([1.0, 1.0, 1.0, 1.0])
    ct_img = _make_nifti_image(
        ct_shape,
        ct_affine,
        data=np.full(ct_shape, 500.0, dtype=np.float32),
        dtype=np.float32,
    )
    ct_canonical = canonicalize_nifti_to_ras(ct_img)

    # A PET grid far away in physical space: entirely outside the CT FOV.
    pet_affine = np.diag([1.0, 1.0, 1.0, 1.0])
    pet_affine[:3, 3] = [1000.0, 1000.0, 1000.0]
    pet_img = _make_nifti_image(
        (2, 2, 2),
        pet_affine,
        data=np.zeros((2, 2, 2), dtype=np.float32),
        dtype=np.float32,
    )
    pet_canonical = canonicalize_nifti_to_ras(pet_img)

    resampled = resample_ct_to_pet(ct_canonical, pet_canonical.geometry)
    assert np.all(np.asarray(resampled) == 0.0)


# ---------------------------------------------------------------------------
# resample_mask_to_pet: nearest-neighbor, source labels only, zero outside
# ---------------------------------------------------------------------------


def test_resample_mask_to_pet_preserves_only_source_labels() -> None:
    ct_shape = (10, 11, 12)
    ct_affine = _rotated_anisotropic_affine(
        axis=(0.4, 0.1, 0.9),
        angle_deg=31.0,
        spacing=(1.1, 0.9, 1.3),
        offset=(0.0, 0.0, 0.0),
    )
    mask_labels = np.zeros(ct_shape, dtype=np.int16)
    mask_labels[2:4, 3:5, 4:6] = 7
    mask_labels[6, 6, 6] = 3
    mask_img = _make_nifti_image(ct_shape, ct_affine, data=mask_labels, dtype=np.int16)
    mask_canonical = canonicalize_nifti_to_ras(mask_img)

    pet_shape = (8, 9, 10)
    pet_affine = _rotated_anisotropic_affine(
        axis=(0.1, 0.3, 0.2),
        angle_deg=9.0,
        spacing=(2.0, 2.0, 2.0),
        offset=(1.0, 1.0, 1.0),
    )
    pet_img = _make_nifti_image(
        pet_shape,
        pet_affine,
        data=np.zeros(pet_shape, dtype=np.float32),
        dtype=np.float32,
    )
    pet_canonical = canonicalize_nifti_to_ras(pet_img)

    resampled = resample_mask_to_pet(mask_canonical, pet_canonical.geometry)
    resampled_arr = np.asarray(resampled)
    assert resampled_arr.dtype == np.int16
    assert resampled_arr.shape == pet_canonical.geometry.shape
    present_labels = set(resampled_arr.ravel().tolist())
    source_labels = set(mask_labels.ravel().tolist())
    assert present_labels <= source_labels
    # Nearest-neighbor never blends: every resampled value must be a value
    # that was literally present in the source array.
    assert present_labels.issubset(set(np.unique(mask_labels).tolist()))


def test_resample_mask_to_pet_zero_outside_field_of_view() -> None:
    mask_shape = (3, 3, 3)
    mask_affine = np.diag([1.0, 1.0, 1.0, 1.0])
    mask_img = _make_nifti_image(
        mask_shape,
        mask_affine,
        data=np.full(mask_shape, 9, dtype=np.int16),
        dtype=np.int16,
    )
    mask_canonical = canonicalize_nifti_to_ras(mask_img)

    pet_affine = np.diag([1.0, 1.0, 1.0, 1.0])
    pet_affine[:3, 3] = [1000.0, 1000.0, 1000.0]
    pet_img = _make_nifti_image(
        (2, 2, 2),
        pet_affine,
        data=np.zeros((2, 2, 2), dtype=np.float32),
        dtype=np.float32,
    )
    pet_canonical = canonicalize_nifti_to_ras(pet_img)

    resampled = resample_mask_to_pet(mask_canonical, pet_canonical.geometry)
    assert np.all(np.asarray(resampled) == 0)


def test_resample_mask_to_pet_requires_integer_dtype() -> None:
    ct_shape = (3, 3, 3)
    ct_affine = np.diag([1.0, 1.0, 1.0, 1.0])
    float_mask_img = _make_nifti_image(
        ct_shape, ct_affine, data=np.zeros(ct_shape, dtype=np.float32), dtype=np.float32
    )
    float_mask_canonical = canonicalize_nifti_to_ras(float_mask_img)
    pet_img = _make_nifti_image(
        ct_shape, ct_affine, data=np.zeros(ct_shape, dtype=np.float32), dtype=np.float32
    )
    pet_canonical = canonicalize_nifti_to_ras(pet_img)
    with pytest.raises(TypeError):
        resample_mask_to_pet(float_mask_canonical, pet_canonical.geometry)


# ---------------------------------------------------------------------------
# restore_mask_to_original_pet: round trip, deterministic sidecar, mismatch
# ---------------------------------------------------------------------------


def test_restore_mask_round_trip_and_deterministic_sidecar() -> None:
    img = _valid_label_image()
    original_labels = np.asarray(img.dataobj).copy()
    canonical = canonicalize_nifti_to_ras(img)

    restored_1, sidecar_1 = restore_mask_to_original_pet(
        np.asarray(canonical.data), canonical
    )
    restored_2, sidecar_2 = restore_mask_to_original_pet(
        np.asarray(canonical.data), canonical
    )

    assert np.array_equal(np.asarray(restored_1), original_labels)
    assert np.array_equal(np.asarray(restored_2), original_labels)
    assert sidecar_1 == sidecar_2
    assert isinstance(sidecar_1, GeometrySidecar)
    assert sidecar_1.canonical_shape == canonical.geometry.shape
    assert sidecar_1.original_shape == canonical.original_geometry.shape


def test_restore_mask_accepts_and_reverifies_a_matching_sidecar() -> None:
    img = _valid_label_image()
    canonical = canonicalize_nifti_to_ras(img)
    _restored, sidecar = restore_mask_to_original_pet(
        np.asarray(canonical.data), canonical
    )

    restored_again, sidecar_again = restore_mask_to_original_pet(
        np.asarray(canonical.data), canonical, sidecar=sidecar
    )
    assert sidecar_again == sidecar
    assert np.array_equal(np.asarray(restored_again), np.asarray(_restored))


def test_restore_mask_corrupted_sidecar_raises_geometry_mismatch() -> None:
    img = _valid_label_image()
    canonical = canonicalize_nifti_to_ras(img)
    _restored, sidecar = restore_mask_to_original_pet(
        np.asarray(canonical.data), canonical
    )

    corrupted = dataclasses.replace(sidecar, original_shape=(1, 1, 1))
    with pytest.raises(GeometryContractError) as exc_info:
        restore_mask_to_original_pet(
            np.asarray(canonical.data), canonical, sidecar=corrupted
        )
    assert exc_info.value.code is GeometryErrorCode.GEOMETRY_MISMATCH
    # Non-sensitive message: exactly the code, nothing else (no path, no
    # shape/affine value, no voxel data).
    assert str(exc_info.value) == "GEOMETRY_MISMATCH"

    corrupted_hash = dataclasses.replace(sidecar, original_affine_sha256="0" * 64)
    with pytest.raises(GeometryContractError) as exc_info:
        restore_mask_to_original_pet(
            np.asarray(canonical.data), canonical, sidecar=corrupted_hash
        )
    assert exc_info.value.code is GeometryErrorCode.GEOMETRY_MISMATCH


def test_restore_mask_shape_mismatch_raises_geometry_mismatch() -> None:
    img = _valid_label_image()
    canonical = canonicalize_nifti_to_ras(img)
    wrong_shape_mask = np.zeros((1, 1, 1), dtype=np.int16)
    with pytest.raises(GeometryContractError) as exc_info:
        restore_mask_to_original_pet(wrong_shape_mask, canonical)
    assert exc_info.value.code is GeometryErrorCode.GEOMETRY_MISMATCH


def test_restore_mask_requires_integer_dtype() -> None:
    img = _valid_label_image()
    canonical = canonicalize_nifti_to_ras(img)
    float_mask = np.asarray(canonical.data).astype(np.float32)
    with pytest.raises(TypeError):
        restore_mask_to_original_pet(float_mask, canonical)


# ---------------------------------------------------------------------------
# Immutable-array contract: every exported ndarray, comprehensively
# ---------------------------------------------------------------------------


def _immutable_array_samples() -> dict[str, np.ndarray]:
    img = _valid_label_image()
    geometry = validate_nifti_grid(img)
    canonical = canonicalize_nifti_to_ras(img)
    ct_shape = (4, 4, 4)
    ct_affine = np.diag([1.0, 1.0, 1.0, 1.0])
    ct_img = _make_nifti_image(
        ct_shape, ct_affine, data=np.ones(ct_shape, dtype=np.float32), dtype=np.float32
    )
    ct_canonical = canonicalize_nifti_to_ras(ct_img)
    mask_img = _make_nifti_image(
        ct_shape, ct_affine, data=np.ones(ct_shape, dtype=np.int16), dtype=np.int16
    )
    mask_canonical = canonicalize_nifti_to_ras(mask_img)
    pet_img = _make_nifti_image(
        (3, 3, 3),
        ct_affine,
        data=np.zeros((3, 3, 3), dtype=np.float32),
        dtype=np.float32,
    )
    pet_canonical = canonicalize_nifti_to_ras(pet_img)

    return {
        "C_lps_from_ras": C_lps_from_ras,
        "GridGeometry.affine": geometry.affine,
        "GridGeometry.world_bounds_min": geometry.world_bounds_min,
        "GridGeometry.world_bounds_max": geometry.world_bounds_max,
        "CanonicalVolume.data": canonical.data,
        "CanonicalVolume.canonical_voxel_from_original_voxel": (
            canonical.canonical_voxel_from_original_voxel
        ),
        "CanonicalVolume.original_voxel_from_canonical_voxel": (
            canonical.original_voxel_from_canonical_voxel
        ),
        "SpatialArray.points (ras_points_to_lps)": ras_points_to_lps(
            np.array([1.0, 2.0, 3.0])
        ).points,
        "input_lps_from_output_lps output": input_lps_from_output_lps(np.eye(4)),
        "resample_ct_to_pet output": resample_ct_to_pet(
            ct_canonical, pet_canonical.geometry
        ),
        "resample_mask_to_pet output": resample_mask_to_pet(
            mask_canonical, pet_canonical.geometry
        ),
        "restore_mask_to_original_pet output": restore_mask_to_original_pet(
            np.asarray(canonical.data), canonical
        )[0],
    }


@pytest.fixture(params=list(_immutable_array_samples().keys()))
def immutable_sample(request: pytest.FixtureRequest) -> tuple[str, np.ndarray]:
    name = request.param
    return name, _immutable_array_samples()[name]


def test_immutable_array_is_owning_and_read_only(
    immutable_sample: tuple[str, np.ndarray],
) -> None:
    name, arr = immutable_sample
    assert arr.flags.owndata is True, name
    assert arr.flags.writeable is False, name
    assert arr.base is None, name


def test_immutable_array_rejects_item_assignment(
    immutable_sample: tuple[str, np.ndarray],
) -> None:
    _name, arr = immutable_sample
    with pytest.raises(ValueError):
        arr[(0,) * arr.ndim] = 12345.0


def test_immutable_array_rejects_setflags_write_true(
    immutable_sample: tuple[str, np.ndarray],
) -> None:
    _name, arr = immutable_sample
    with pytest.raises(ValueError):
        arr.setflags(write=True)


def test_immutable_array_rejects_flags_writeable_true(
    immutable_sample: tuple[str, np.ndarray],
) -> None:
    _name, arr = immutable_sample
    with pytest.raises(ValueError):
        arr.flags.writeable = True


def test_immutable_array_rejects_resize(
    immutable_sample: tuple[str, np.ndarray],
) -> None:
    _name, arr = immutable_sample
    with pytest.raises(ValueError):
        arr.resize(arr.shape, refcheck=False)


def test_immutable_array_rejects_in_place_shape_change(
    immutable_sample: tuple[str, np.ndarray],
) -> None:
    _name, arr = immutable_sample
    with pytest.raises(ValueError):
        arr.shape = (int(arr.size),)


def test_immutable_array_rejects_in_place_stride_change(
    immutable_sample: tuple[str, np.ndarray],
) -> None:
    _name, arr = immutable_sample
    with pytest.raises(ValueError):
        arr.strides = arr.strides


def test_immutable_array_no_op_writeable_false_is_allowed(
    immutable_sample: tuple[str, np.ndarray],
) -> None:
    _name, arr = immutable_sample
    arr.setflags(write=False)  # must not raise
    arr.flags.writeable = False  # must not raise
    assert arr.flags.writeable is False


def test_immutable_array_snapshot_unchanged_after_rejected_mutations(
    immutable_sample: tuple[str, np.ndarray],
) -> None:
    _name, arr = immutable_sample
    shape_before = arr.shape
    dtype_before = arr.dtype
    strides_before = arr.strides
    bytes_before = arr.tobytes()
    owndata_before = arr.flags.owndata
    writeable_before = arr.flags.writeable

    for mutate in (
        lambda: arr.__setitem__((0,) * arr.ndim, 999),
        lambda: arr.setflags(write=True),
        lambda: setattr(arr.flags, "writeable", True),
        lambda: arr.resize(arr.shape, refcheck=False),
        lambda: setattr(arr, "shape", (int(arr.size),)),
        lambda: setattr(arr, "strides", arr.strides),
    ):
        try:
            mutate()
        except Exception:
            pass

    assert arr.shape == shape_before
    assert arr.dtype == dtype_before
    assert arr.strides == strides_before
    assert arr.tobytes() == bytes_before
    assert arr.flags.owndata == owndata_before
    assert arr.flags.writeable == writeable_before


def test_immutable_array_copy_is_independent_and_mutable() -> None:
    # An explicit .copy() is a deliberate, independently-owned snapshot: it
    # is legitimate for it to be a fresh, ordinarily-mutable array (the
    # immutability contract protects the *exported* array from being
    # mutated in place, not the caller's ability to ever obtain a mutable
    # copy of its values).
    copy = np.asarray(C_lps_from_ras).copy()
    copy[0, 0] = 42.0
    assert copy[0, 0] == 42.0
    assert np.asarray(C_lps_from_ras)[0, 0] == -1.0


def test_immutable_array_asarray_view_stays_read_only() -> None:
    view = np.asarray(C_lps_from_ras)
    assert view.flags.writeable is False
    with pytest.raises(ValueError):
        view.flags.writeable = True
    with pytest.raises(ValueError):
        view[0, 0] = 42.0
