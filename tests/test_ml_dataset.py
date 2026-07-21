"""Tests for the P6 dataset builder (``src.vascutrace.ml.dataset``).

Generated-fixture tests (no ``Data/`` access, no ``local_data`` marker) build
a full-``FIXED_CROP_SHAPE`` synthetic :class:`CropBundle` via
``make_crop_bundle`` (matching the fixture pattern already established in
``tests/test_data_pipeline.py``'s ``_synthetic_bundle_fields``) with a known,
small, compact iliac-label mask so tests stay fast. Most tests use
``_FAST_CONFIG`` (``supersample=1``) -- deliberately cheaper than
``DatasetConfig``'s production default -- because they exercise PLUMBING
(shapes, target presence/absence, determinism, reflection identity, caching
equivalence), not the P3 volume-accuracy claim itself; only
``TestVolumeErrorAtRealSpacing`` tests that claim, at the real production
supersample, matching this project's practice elsewhere of using the
cheapest config that still exercises the code path under test.

One ``@pytest.mark.local_data`` class exercises one-or-more real,
already-computed P2 crop bundles from the gitignored
``data/processed/p2/crops/p2-crop-v2/`` tree (never ``Data/`` directly, and
never written to -- read-only). Aggregate-only: shapes, counts, and boolean
structural checks -- never per-voxel values, subject identifiers in
assertions text, or printed raw data.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.vascutrace.data.contract import (
    FIXED_CROP_SHAPE,
    ILIAC_LABEL_LEFT,
    ILIAC_LABEL_RIGHT,
    CropBundle,
    load_crop_bundle,
    make_crop_bundle,
    reflect_volume,
    save_crop_bundle,
)
from src.vascutrace.data.crops import build_reflection_affine
from src.vascutrace.geometry import GeometrySidecar, GridGeometry
from src.vascutrace.ml.dataset import (
    DatasetConfig,
    SiameseCropDataset,
    build_bilateral_views,
    build_sample,
    frozen_validation_set,
    iliac_centerlines,
)
from src.vascutrace.ml.dataset import (
    _build_bundle_precompute as build_bundle_precompute,
)
from src.vascutrace.ml.dataset import (
    _positive_center_z_candidates as positive_center_z_candidates,
)
from src.vascutrace.ml.dataset import (
    _true_core_last_index as true_core_last_index,
)
from src.vascutrace.ml.tensor_schema import (
    CHANNELS_PER_VIEW,
    FIRST_CENTER_Z,
    HALF_K,
    IN_PLANE_HW,
    K,
    LAST_CENTER_Z,
    PET_DIFF_SHAPE,
    TARGET_SHAPE,
)
from src.vascutrace.simulation.anomaly import (
    AnomalySimulationParams,
    simulate_vascular_anomaly,
)

# ---------------------------------------------------------------------------
# Shared synthetic-fixture builders
# ---------------------------------------------------------------------------

# Isotropic, rotation-free affine: world_mm = voxel_index * 2.0. Chosen so
# geometric reasoning in these tests (e.g. the physical-reflection test) is
# easy to verify by hand.
_STANDARD_AFFINE = np.diag([2.0, 2.0, 2.0, 1.0])
# Mirror plane at world x = 144 mm == voxel x = 72 (the crop's own X
# mid-index, NOT (144-1)/2 = 71.5 -- deliberately off from the array-flip
# center by half a voxel, see TestReflectionIsPhysicalNotFlip).
_STANDARD_REFLECTION = build_reflection_affine(
    np.array([1.0, 0.0, 0.0]), np.array([144.0, 0.0, 0.0])
)

# Deliberately cheaper than DatasetConfig()'s production default
# (supersample=5, chosen for P3's <5% volume-error invariant -- see
# TestVolumeErrorAtRealSpacing and DatasetConfig.supersample's own
# docstring comment). Whether a "positive" sample's target is empty or
# nonzero, and every other plumbing property these tests check, does not
# depend on supersample -- a voxel at the true vessel centerline
# (distance ~0 from the centerline) reads occupancy ~1.0 at any
# supersample >= 1. Using supersample=1 here keeps the CPU suite fast.
_FAST_CONFIG = DatasetConfig(supersample=1)


def _geometry_sidecar() -> GeometrySidecar:
    return GeometrySidecar(
        canonical_shape=(440, 440, 531),
        original_shape=(440, 440, 531),
        canonical_affine_sha256="a" * 64,
        original_affine_sha256="b" * 64,
        original_voxel_from_canonical_voxel_sha256="c" * 64,
    )


def _make_bundle(
    *,
    pet_suvbw: np.ndarray,
    ct_hu: np.ndarray,
    valid_pet_mask: np.ndarray,
    iliac_label_mask: np.ndarray,
    crop_to_pet_canonical_affine: np.ndarray = _STANDARD_AFFINE,
    reflection_affine: np.ndarray = _STANDARD_REFLECTION,
    subject: str = "SYNTH_SUBJECT_001",
    session: str = "Test",
) -> CropBundle:
    spacing = tuple(
        float(v) for v in np.linalg.norm(crop_to_pet_canonical_affine[:3, :3], axis=0)
    )
    return make_crop_bundle(
        subject=subject,
        session=session,
        pet_suvbw=pet_suvbw,
        ct_hu=ct_hu,
        valid_pet_mask=valid_pet_mask,
        iliac_label_mask=iliac_label_mask,
        reflection_affine=reflection_affine,
        crop_to_pet_canonical_affine=crop_to_pet_canonical_affine,
        crop_origin_voxel=(0, 0, 0),
        original_voxel_from_pet_canonical_voxel=np.eye(4),
        geometry_sidecar=_geometry_sidecar(),
        reflection_residual_mm=1.0,
        reflection_qc_flag=False,
        bbox_exceeds_fixed_crop=(False, False, False),
        crop_margin_mm=15.0,
        pet_spacing_mm=spacing,
        paired_point_count=1,
    )


def _standard_iliac_label_mask() -> np.ndarray:
    """Left block at voxel X [60, 64), right block at voxel X [80, 84) --
    an exact mirror pair about voxel x=72 under ``_STANDARD_REFLECTION``
    (72 - 60 == 84 - 72 == 12). Both spans Y [35, 45), Z [50, 90) (40
    axial slices, one left+right centroid pair per slice).
    """
    mask = np.zeros(FIXED_CROP_SHAPE, dtype=np.uint8)
    mask[60:64, 35:45, 50:90] = ILIAC_LABEL_LEFT
    mask[80:84, 35:45, 50:90] = ILIAC_LABEL_RIGHT
    return mask


def _standard_bundle(seed: int = 42) -> CropBundle:
    rng = np.random.default_rng(seed)
    pet = (rng.random(FIXED_CROP_SHAPE, dtype=np.float32) * 8.0 + 1.0).astype(
        np.float32
    )
    ct = (rng.random(FIXED_CROP_SHAPE, dtype=np.float32) * 200.0 - 100.0).astype(
        np.float32
    )
    valid = np.ones(FIXED_CROP_SHAPE, dtype=np.uint8)
    return _make_bundle(
        pet_suvbw=pet,
        ct_hu=ct,
        valid_pet_mask=valid,
        iliac_label_mask=_standard_iliac_label_mask(),
    )


# A center_z within the standard fixture's labeled Z-range [50, 90) and,
# given length_mm=45 / spacing_z=2.0mm starting at the lowest labeled slice
# (z=50), inside the lesion's own core span too.
_STANDARD_CENTER_Z_IN_LESION = 60
# Still inside the labeled Z-range but well past the 45 mm lesion core
# (core ends near z=50+23=73), so the target here must be all-zero even on
# a "positive" sample.
_STANDARD_CENTER_Z_PAST_LESION = 88


# ---------------------------------------------------------------------------
# 1. Sample tensor shapes/dtypes match tensor_schema exactly
# ---------------------------------------------------------------------------


class TestSampleShapesAndDtypes:
    def test_healthy_sample_shapes_and_dtypes(self) -> None:
        bundle = _standard_bundle()
        sample = build_sample(
            bundle, _STANDARD_CENTER_Z_IN_LESION, seed=1, positive=False
        )

        expected_view_shape = (CHANNELS_PER_VIEW, *IN_PLANE_HW)
        assert tuple(sample.left_view.shape) == expected_view_shape
        assert tuple(sample.right_view.shape) == expected_view_shape
        assert tuple(sample.pet_diff.shape) == PET_DIFF_SHAPE
        assert tuple(sample.target_mask.shape) == TARGET_SHAPE
        assert tuple(sample.valid_mask.shape) == TARGET_SHAPE
        assert tuple(sample.raw_pet.shape) == TARGET_SHAPE

        for tensor in (
            sample.left_view,
            sample.right_view,
            sample.pet_diff,
            sample.target_mask,
            sample.valid_mask,
            sample.raw_pet,
        ):
            assert isinstance(tensor, torch.Tensor)
            assert tensor.dtype == torch.float32

    def test_positive_sample_shapes_and_dtypes(self) -> None:
        bundle = _standard_bundle()
        sample = build_sample(
            bundle,
            _STANDARD_CENTER_Z_IN_LESION,
            seed=7,
            positive=True,
            side="left",
            config=_FAST_CONFIG,
        )
        assert tuple(sample.left_view.shape) == (CHANNELS_PER_VIEW, *IN_PLANE_HW)
        assert tuple(sample.pet_diff.shape) == (K, *IN_PLANE_HW)
        assert sample.left_view.dtype == torch.float32
        assert set(sample.meta) == {
            "subject",
            "session",
            "center_z",
            "positive",
            "side",
            "sim_params",
            "tensor_schema_version",
            "dataset_builder_version",
        }


# ---------------------------------------------------------------------------
# 2. Positive nonzero target / healthy all-zero target
# ---------------------------------------------------------------------------


class TestPositiveAndHealthyTargets:
    def test_positive_sample_target_nonzero_where_lesion_is(self) -> None:
        bundle = _standard_bundle()
        sample = build_sample(
            bundle,
            _STANDARD_CENTER_Z_IN_LESION,
            seed=123,
            positive=True,
            side="left",
            config=_FAST_CONFIG,
        )
        assert float(sample.target_mask.sum()) > 0.0

    def test_healthy_sample_target_is_all_zero(self) -> None:
        bundle = _standard_bundle()
        sample = build_sample(
            bundle, _STANDARD_CENTER_Z_IN_LESION, seed=123, positive=False
        )
        assert torch.count_nonzero(sample.target_mask).item() == 0

    def test_positive_sample_target_is_zero_far_from_lesion_core(self) -> None:
        """Spatial precision, not just "positive metadata": a center slice
        well past the fixed 45 mm lesion core has an all-zero target even
        though ``positive=True`` and the lesion truly was inserted
        elsewhere in the volume.
        """
        bundle = _standard_bundle()
        sample = build_sample(
            bundle,
            _STANDARD_CENTER_Z_PAST_LESION,
            seed=123,
            positive=True,
            side="left",
            config=_FAST_CONFIG,
        )
        assert torch.count_nonzero(sample.target_mask).item() == 0


# ---------------------------------------------------------------------------
# 3. Normalization ranges
# ---------------------------------------------------------------------------


class TestNormalizationRanges:
    def test_pet_in_zero_one_and_ct_in_minus_one_one(self) -> None:
        bundle = _standard_bundle()
        sample = build_sample(
            bundle, _STANDARD_CENTER_Z_IN_LESION, seed=5, positive=False
        )
        for view in (sample.left_view, sample.right_view):
            pet_channels = view[:K]
            ct_channels = view[K:]
            assert torch.all(pet_channels >= 0.0)
            assert torch.all(pet_channels <= 1.0)
            assert torch.all(ct_channels >= -1.0)
            assert torch.all(ct_channels <= 1.0)
        assert torch.all(sample.pet_diff >= -1.0)
        assert torch.all(sample.pet_diff <= 1.0)

    def test_clipping_actually_triggers_on_out_of_range_values(self) -> None:
        """Bounds alone could pass trivially on already-in-range data; this
        proves ``build_bilateral_views`` actually clips rather than
        passing values through unchanged.
        """
        pet_crop = np.full(FIXED_CROP_SHAPE, 50.0, dtype=np.float32)  # > PET_CLIP[1]
        ct_crop = np.full(FIXED_CROP_SHAPE, -5000.0, dtype=np.float32)  # < CT_CLIP[0]
        valid = np.ones(FIXED_CROP_SHAPE, dtype=np.uint8)

        views = build_bilateral_views(
            pet_crop=pet_crop,
            ct_crop=ct_crop,
            valid_pet_mask=valid,
            crop_to_pet_canonical_affine=_STANDARD_AFFINE,
            reflection_affine=_STANDARD_REFLECTION,
            center_z=70,
        )
        # Interior point (well away from the reflected boundary column) --
        # both left and right views must read exactly the clipped value.
        h_interior, w_interior = 72, 40
        assert np.allclose(views.left_view[:K, h_interior, w_interior], 1.0)
        assert np.allclose(views.right_view[:K, h_interior, w_interior], 1.0)
        assert np.allclose(views.left_view[K:, h_interior, w_interior], -1.0)
        assert np.allclose(views.right_view[K:, h_interior, w_interior], -1.0)
        # Global bounds must hold everywhere too (incl. any zero-padded
        # boundary voxel from the reflection resample).
        assert views.left_view[:K].min() >= 0.0 and views.left_view[:K].max() <= 1.0
        assert views.left_view[K:].min() >= -1.0 and views.left_view[K:].max() <= 1.0


# ---------------------------------------------------------------------------
# 4. Reflection uses the physical affine, not an array flip
# ---------------------------------------------------------------------------


class TestReflectionIsPhysicalNotFlip:
    def test_right_view_matches_reflect_volume_not_np_flip(self) -> None:
        # Single asymmetric marker voxel: x=50 (off-center; not equidistant
        # from any trivial symmetry), y=40, z=70. Background otherwise 0.
        pet_crop = np.zeros(FIXED_CROP_SHAPE, dtype=np.float32)
        pet_crop[50, 40, 70] = 8.0
        ct_crop = np.zeros(FIXED_CROP_SHAPE, dtype=np.float32)
        valid = np.ones(FIXED_CROP_SHAPE, dtype=np.uint8)
        center_z = 70

        views = build_bilateral_views(
            pet_crop=pet_crop,
            ct_crop=ct_crop,
            valid_pet_mask=valid,
            crop_to_pet_canonical_affine=_STANDARD_AFFINE,
            reflection_affine=_STANDARD_REFLECTION,
            center_z=center_z,
        )

        # Ground truth: the frozen, already-tested physical-mirror
        # primitive this module reuses (never reimplemented here).
        expected_reflected = reflect_volume(
            pet_crop, _STANDARD_AFFINE, _STANDARD_REFLECTION, order=1
        )
        z0, z1 = center_z - HALF_K, center_z + HALF_K + 1
        expected_slab = np.moveaxis(expected_reflected[:, :, z0:z1], 2, 0)
        expected_norm = np.clip(expected_slab, 0.0, 10.0) / 10.0
        np.testing.assert_allclose(views.right_view[:K], expected_norm, atol=1e-6)

        # The naive bug this test guards against: np.flip along the R/L
        # (X) axis instead of a true physical reflection.
        flipped = np.flip(pet_crop, axis=0)
        flipped_slab = np.moveaxis(flipped[:, :, z0:z1], 2, 0)
        flipped_norm = np.clip(flipped_slab, 0.0, 10.0) / 10.0
        assert not np.allclose(views.right_view[:K], flipped_norm, atol=1e-6)

        # Concretely: physical reflection maps voxel x=50 -> x=94
        # (world 2*144 - 100 = 188 -> voxel 94), NOT the array-flip target
        # x=93 (143 - 50) -- these differ by exactly one voxel for this
        # fixture's affine/reflection-plane offset.
        marker_channel = int(HALF_K)  # center slice within the K slab
        assert float(views.right_view[marker_channel, 94, 40]) > 0.0
        assert float(views.right_view[marker_channel, 93, 40]) == 0.0


# ---------------------------------------------------------------------------
# 5. Center-slice alignment
# ---------------------------------------------------------------------------


class TestCenterSliceAlignment:
    def test_valid_mask_is_exactly_the_center_slice(self) -> None:
        pet_crop = np.ones(FIXED_CROP_SHAPE, dtype=np.float32)
        ct_crop = np.zeros(FIXED_CROP_SHAPE, dtype=np.float32)
        valid = np.ones(FIXED_CROP_SHAPE, dtype=np.uint8)
        valid[:, :, 70] = 0  # only z=70 is invalid

        views_on = build_bilateral_views(
            pet_crop=pet_crop,
            ct_crop=ct_crop,
            valid_pet_mask=valid,
            crop_to_pet_canonical_affine=_STANDARD_AFFINE,
            reflection_affine=_STANDARD_REFLECTION,
            center_z=69,
        )
        views_off = build_bilateral_views(
            pet_crop=pet_crop,
            ct_crop=ct_crop,
            valid_pet_mask=valid,
            crop_to_pet_canonical_affine=_STANDARD_AFFINE,
            reflection_affine=_STANDARD_REFLECTION,
            center_z=70,
        )
        assert np.all(views_on.valid_mask == 1.0)
        assert np.all(views_off.valid_mask == 0.0)

    def test_target_mask_alignment_via_build_sample(self) -> None:
        """Same underlying ``_center_slice`` primitive as valid_mask; this
        confirms it also applies correctly to the simulated
        ``ground_truth_mask`` used for ``target_mask``.
        """
        bundle = _standard_bundle()
        in_lesion = build_sample(
            bundle,
            _STANDARD_CENTER_Z_IN_LESION,
            seed=123,
            positive=True,
            side="left",
            config=_FAST_CONFIG,
        )
        past_lesion = build_sample(
            bundle,
            _STANDARD_CENTER_Z_PAST_LESION,
            seed=123,
            positive=True,
            side="left",
            config=_FAST_CONFIG,
        )
        assert float(in_lesion.target_mask.sum()) > 0.0
        assert torch.count_nonzero(past_lesion.target_mask).item() == 0


# ---------------------------------------------------------------------------
# 6. Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_build_sample_same_seed_is_identical(self) -> None:
        bundle = _standard_bundle()
        first = build_sample(
            bundle,
            _STANDARD_CENTER_Z_IN_LESION,
            seed=777,
            positive=True,
            side="left",
            config=_FAST_CONFIG,
        )
        second = build_sample(
            bundle,
            _STANDARD_CENTER_Z_IN_LESION,
            seed=777,
            positive=True,
            side="left",
            config=_FAST_CONFIG,
        )
        assert torch.equal(first.left_view, second.left_view)
        assert torch.equal(first.right_view, second.right_view)
        assert torch.equal(first.pet_diff, second.pet_diff)
        assert torch.equal(first.target_mask, second.target_mask)
        assert torch.equal(first.valid_mask, second.valid_mask)
        assert torch.equal(first.raw_pet, second.raw_pet)
        assert first.meta == second.meta

    def test_build_sample_different_seed_changes_sim_params(self) -> None:
        bundle = _standard_bundle()
        first = build_sample(
            bundle,
            _STANDARD_CENTER_Z_IN_LESION,
            seed=1,
            positive=True,
            side="left",
            config=_FAST_CONFIG,
        )
        second = build_sample(
            bundle,
            _STANDARD_CENTER_Z_IN_LESION,
            seed=2,
            positive=True,
            side="left",
            config=_FAST_CONFIG,
        )
        assert first.meta["sim_params"] != second.meta["sim_params"]

    def test_dataset_same_seed_is_identical_across_instances(
        self, tmp_path: Path
    ) -> None:
        bundle = _standard_bundle()
        bundle_dir = save_crop_bundle(bundle, tmp_path)

        config = DatasetConfig(samples_per_bundle=3, supersample=1)
        dataset_a = SiameseCropDataset(
            [bundle_dir], seed=2026, positive_fraction=0.5, config=config
        )
        dataset_b = SiameseCropDataset(
            [bundle_dir], seed=2026, positive_fraction=0.5, config=config
        )
        assert len(dataset_a) == len(dataset_b) == 3

        for i in range(len(dataset_a)):
            sample_a = dataset_a[i]
            sample_b = dataset_b[i]
            assert torch.equal(sample_a.left_view, sample_b.left_view)
            assert torch.equal(sample_a.target_mask, sample_b.target_mask)
            assert sample_a.meta == sample_b.meta


# ---------------------------------------------------------------------------
# 7. Centerline derivation correctness
# ---------------------------------------------------------------------------


class TestIliacCenterlines:
    def test_per_slice_centroid_ordering_on_a_known_label(self) -> None:
        mask = np.zeros(FIXED_CROP_SHAPE, dtype=np.uint8)
        # z=10: single left voxel at (5, 5, 10) -> centroid (5, 5, 10).
        mask[5, 5, 10] = ILIAC_LABEL_LEFT
        # z=11: deliberately no left label -> must be skipped, not
        # interpolated.
        # z=12: two left voxels -> centroid (6, 5, 12).
        mask[5, 5, 12] = ILIAC_LABEL_LEFT
        mask[7, 5, 12] = ILIAC_LABEL_LEFT
        # z=9: one left voxel, so ascending z order is exercised
        # (construction order z=10,12,9 above; expected output order
        # 9, 10, 12).
        mask[3, 5, 9] = ILIAC_LABEL_LEFT

        bundle = _make_bundle(
            pet_suvbw=np.ones(FIXED_CROP_SHAPE, dtype=np.float32),
            ct_hu=np.zeros(FIXED_CROP_SHAPE, dtype=np.float32),
            valid_pet_mask=np.ones(FIXED_CROP_SHAPE, dtype=np.uint8),
            iliac_label_mask=mask,
            crop_to_pet_canonical_affine=np.eye(4),  # world mm == voxel index
        )

        centerlines = iliac_centerlines(bundle)
        left = centerlines["left"]
        right = centerlines["right"]

        assert right.shape == (0, 3)  # no right label anywhere in this mask
        np.testing.assert_allclose(
            left, np.array([[3.0, 5.0, 9.0], [5.0, 5.0, 10.0], [6.0, 5.0, 12.0]])
        )

    def test_both_sides_present_on_standard_fixture(self) -> None:
        bundle = _standard_bundle()
        centerlines = iliac_centerlines(bundle)
        assert centerlines["left"].shape == (40, 3)
        assert centerlines["right"].shape == (40, 3)
        # Ascending-Z ordering.
        assert np.all(np.diff(centerlines["left"][:, 2]) > 0)
        assert np.all(np.diff(centerlines["right"][:, 2]) > 0)


# ---------------------------------------------------------------------------
# 8. Train/inference preprocessing identity
# ---------------------------------------------------------------------------


class TestTrainInferencePreprocessingIdentity:
    def test_build_sample_and_direct_build_bilateral_views_agree(self) -> None:
        bundle = _standard_bundle()
        sample = build_sample(
            bundle, _STANDARD_CENTER_Z_IN_LESION, seed=9, positive=False
        )

        # The exact same reusable function a future inference script would
        # call directly on the raw, unmodified crop arrays.
        direct = build_bilateral_views(
            pet_crop=bundle.pet_suvbw,
            ct_crop=bundle.ct_hu,
            valid_pet_mask=bundle.valid_pet_mask,
            crop_to_pet_canonical_affine=bundle.crop_to_pet_canonical_affine,
            reflection_affine=bundle.reflection_affine,
            center_z=_STANDARD_CENTER_Z_IN_LESION,
        )

        assert torch.equal(sample.left_view, torch.from_numpy(direct.left_view))
        assert torch.equal(sample.right_view, torch.from_numpy(direct.right_view))
        assert torch.equal(sample.pet_diff, torch.from_numpy(direct.pet_diff))
        assert torch.equal(sample.valid_mask, torch.from_numpy(direct.valid_mask))
        assert torch.equal(sample.raw_pet, torch.from_numpy(direct.raw_pet))

    def test_build_bilateral_views_is_deterministic_pure_function(self) -> None:
        bundle = _standard_bundle()
        first = build_bilateral_views(
            pet_crop=bundle.pet_suvbw,
            ct_crop=bundle.ct_hu,
            valid_pet_mask=bundle.valid_pet_mask,
            crop_to_pet_canonical_affine=bundle.crop_to_pet_canonical_affine,
            reflection_affine=bundle.reflection_affine,
            center_z=_STANDARD_CENTER_Z_IN_LESION,
        )
        second = build_bilateral_views(
            pet_crop=bundle.pet_suvbw,
            ct_crop=bundle.ct_hu,
            valid_pet_mask=bundle.valid_pet_mask,
            crop_to_pet_canonical_affine=bundle.crop_to_pet_canonical_affine,
            reflection_affine=bundle.reflection_affine,
            center_z=_STANDARD_CENTER_Z_IN_LESION,
        )
        np.testing.assert_array_equal(first.left_view, second.left_view)
        np.testing.assert_array_equal(first.right_view, second.right_view)


# ---------------------------------------------------------------------------
# Raw PET retention (unnormalized, separate from the normalized views)
# ---------------------------------------------------------------------------


class TestRawPetRetention:
    def test_raw_pet_is_unnormalized_center_slice(self) -> None:
        bundle = _standard_bundle()
        sample = build_sample(
            bundle, _STANDARD_CENTER_Z_IN_LESION, seed=3, positive=False
        )
        expected_raw = bundle.pet_suvbw[:, :, _STANDARD_CENTER_Z_IN_LESION]
        np.testing.assert_allclose(sample.raw_pet[0].numpy(), expected_raw, atol=1e-6)
        # Structural relationship to the network-normalized center PET
        # channel (index HALF_K within the K slab): the normalized channel
        # is exactly clip(raw_pet, 0, 10) / 10 -- proving raw_pet feeds the
        # network channel WITHOUT itself being clipped/scaled.
        center_channel = sample.left_view[HALF_K]
        reconstructed = torch.clip(sample.raw_pet[0], 0.0, 10.0) / 10.0
        assert torch.allclose(center_channel, reconstructed, atol=1e-6)


# ---------------------------------------------------------------------------
# SiameseCropDataset / frozen_validation_set
# ---------------------------------------------------------------------------


class TestSiameseCropDataset:
    def test_length_matches_bundles_times_samples_per_bundle(
        self, tmp_path: Path
    ) -> None:
        bundle_dir = save_crop_bundle(_standard_bundle(), tmp_path)
        dataset = SiameseCropDataset(
            [bundle_dir],
            seed=1,
            config=DatasetConfig(samples_per_bundle=4, supersample=1),
        )
        assert len(dataset) == 4

    def test_getitem_returns_well_formed_sample(self, tmp_path: Path) -> None:
        bundle_dir = save_crop_bundle(_standard_bundle(), tmp_path)
        dataset = SiameseCropDataset(
            [bundle_dir],
            seed=1,
            config=DatasetConfig(samples_per_bundle=2, supersample=1),
        )
        sample = dataset[0]
        assert tuple(sample.left_view.shape) == (CHANNELS_PER_VIEW, *IN_PLANE_HW)
        assert FIRST_CENTER_Z <= sample.meta["center_z"] <= LAST_CENTER_Z

    def test_only_passed_bundles_are_ever_touched(self, tmp_path: Path) -> None:
        bundle_a = _standard_bundle(seed=1)
        bundle_b = _make_bundle(
            pet_suvbw=_standard_bundle(seed=2).pet_suvbw,
            ct_hu=_standard_bundle(seed=2).ct_hu,
            valid_pet_mask=np.ones(FIXED_CROP_SHAPE, dtype=np.uint8),
            iliac_label_mask=_standard_iliac_label_mask(),
            subject="SYNTH_SUBJECT_002",
        )
        dir_a = save_crop_bundle(bundle_a, tmp_path)
        save_crop_bundle(bundle_b, tmp_path)  # present on disk, never passed in

        dataset = SiameseCropDataset(
            [dir_a], seed=1, config=DatasetConfig(samples_per_bundle=3, supersample=1)
        )
        assert dataset.bundle_dirs == (dir_a,)
        for i in range(len(dataset)):
            assert dataset[i].meta["subject"] == "SYNTH_SUBJECT_001"

    def test_negative_index_and_out_of_range(self, tmp_path: Path) -> None:
        bundle_dir = save_crop_bundle(_standard_bundle(), tmp_path)
        dataset = SiameseCropDataset(
            [bundle_dir],
            seed=1,
            config=DatasetConfig(samples_per_bundle=2, supersample=1),
        )
        assert dataset[-1].meta["center_z"] == dataset[1].meta["center_z"]
        with pytest.raises(IndexError):
            dataset[2]


class TestFrozenValidationSet:
    def test_matches_a_fresh_dataset_of_the_same_arguments(
        self, tmp_path: Path
    ) -> None:
        bundle_dir = save_crop_bundle(_standard_bundle(), tmp_path)
        config = DatasetConfig(samples_per_bundle=3, supersample=1)

        materialized = frozen_validation_set(
            [bundle_dir], seed=55, positive_fraction=0.5, config=config
        )
        fresh_dataset = SiameseCropDataset(
            [bundle_dir], seed=55, positive_fraction=0.5, config=config
        )

        assert len(materialized) == len(fresh_dataset) == 3
        for materialized_sample, fresh_sample in zip(
            materialized,
            (fresh_dataset[i] for i in range(len(fresh_dataset))),
            strict=True,
        ):
            assert torch.equal(materialized_sample.left_view, fresh_sample.left_view)
            assert materialized_sample.meta == fresh_sample.meta

    def test_stable_across_repeated_calls(self, tmp_path: Path) -> None:
        bundle_dir = save_crop_bundle(_standard_bundle(), tmp_path)
        config = DatasetConfig(samples_per_bundle=2, supersample=1)
        first = frozen_validation_set([bundle_dir], seed=7, config=config)
        second = frozen_validation_set([bundle_dir], seed=7, config=config)
        for a, b in zip(first, second, strict=True):
            assert torch.equal(a.target_mask, b.target_mask)


# ---------------------------------------------------------------------------
# Adversarial-review fix 1: DatasetConfig.supersample's <5% volume-error
# invariant, measured at the REAL P2 crop spacing.
# ---------------------------------------------------------------------------


class TestVolumeErrorAtRealSpacing:
    """Validates :class:`DatasetConfig`'s chosen ``supersample`` default
    against P3's own <5% rasterized-analytic-capsule-volume-error
    invariant (``imaging-physics.md``, "Synthetic source"), at this
    implementation's real P2 crop spacing (1.65, 1.65, 2.0 mm) and
    ``length_mm=45``, swept across the full sampleable ``radius_mm``
    range ``[2, 6]`` -- reproducing (at a smaller, fast-to-run generated
    grid, matching ``tests/test_simulation.py``'s own volume-error test
    pattern) the exact measurement that justifies
    ``DatasetConfig.supersample``'s default value (see that field's own
    docstring comment for the full sweep this test's single-supersample
    check is drawn from).
    """

    _SPACING_MM = (1.65, 1.65, 2.0)

    def _measure_relative_error(self, radius_mm: float, supersample: int) -> float:
        length_mm = DatasetConfig().length_mm
        spacing = self._SPACING_MM
        margin = radius_mm + 20.0
        length_vox = (
            int(np.ceil(length_mm / spacing[0])) + int(np.ceil(margin / spacing[0])) * 2
        )
        y_vox = int(np.ceil(margin / spacing[1])) * 2 + 4
        z_vox = int(np.ceil(margin / spacing[2])) * 2 + 4
        shape = (length_vox, y_vox, z_vox)

        affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
        corners = np.array(
            [
                [i, j, k]
                for i in (0, shape[0] - 1)
                for j in (0, shape[1] - 1)
                for k in (0, shape[2] - 1)
            ],
            dtype=np.float64,
        )
        world = (affine @ np.concatenate([corners, np.ones((8, 1))], axis=1).T).T[:, :3]
        geometry = GridGeometry(
            shape=shape,
            affine=affine,
            spacing=spacing,
            units="mm",
            world_bounds_min=world.min(axis=0),
            world_bounds_max=world.max(axis=0),
        )

        background = np.ones(shape, dtype=np.float32)
        contralateral = np.zeros(shape, dtype=bool)
        contralateral[0:3, :, :] = True

        start = (margin, y_vox * spacing[1] / 2.0, z_vox * spacing[2] / 2.0)
        end = (margin + length_mm, y_vox * spacing[1] / 2.0, z_vox * spacing[2] / 2.0)
        centerline = np.array([start, end], dtype=np.float64)

        params = AnomalySimulationParams(
            side="left",
            radius_mm=radius_mm,
            length_mm=length_mm,
            uptake_multiplier=1.6,
            blur_fwhm_mm=4.0,
            heterogeneity=0.15,
            pet_ct_shift_mm=(0.0, 0.0, 0.0),
            seed=42,
        )
        result = simulate_vascular_anomaly(
            background,
            geometry,
            centerline,
            contralateral,
            params,
            supersample=supersample,
        )

        voxel_volume_mm3 = float(np.prod(spacing))
        rasterized_volume = float(result.source_fraction.sum()) * voxel_volume_mm3
        analytic_volume = (
            np.pi * radius_mm**2 * length_mm + (4.0 / 3.0) * np.pi * radius_mm**3
        )
        return abs(rasterized_volume - analytic_volume) / analytic_volume

    @pytest.mark.parametrize("radius_mm", [2.0, 3.0, 4.0, 5.0, 6.0])
    def test_volume_error_under_five_percent_at_chosen_supersample(
        self, radius_mm: float
    ) -> None:
        supersample = DatasetConfig().supersample
        relative_error = self._measure_relative_error(radius_mm, supersample)
        assert relative_error < 0.05, (radius_mm, supersample, relative_error)

    def test_supersample_one_is_not_sufficient(self) -> None:
        """Negative control: proves this test actually discriminates --
        the previously-shipped ``supersample=1`` default fails this same
        check (this is exactly the adversarial-review finding this test
        suite now guards against regressing on).
        """
        relative_error = self._measure_relative_error(radius_mm=2.0, supersample=1)
        assert relative_error >= 0.05, relative_error


# ---------------------------------------------------------------------------
# Adversarial-review fix 2: positive center_z candidates must be restricted
# to the TRUE arc-length-clipped lesion core, not an over-estimate.
# ---------------------------------------------------------------------------


class TestPositiveCenterZCandidatesTrueCore:
    def test_true_core_last_index_matches_hand_computed_arc_length(self) -> None:
        # Straight centerline, one point per mm along Z (spacing_z == 1
        # here for easy hand-verification): length_mm=10 -> core spans
        # points at cumulative arc length <= 10, i.e. indices 0..10
        # (11 points, 0-indexed), so the last INDEX is 10.
        points = np.array([[0.0, 0.0, float(z)] for z in range(21)])
        assert true_core_last_index(points, length_mm=10.0) == 10
        # A length_mm longer than the whole centerline clips to the last
        # point.
        assert true_core_last_index(points, length_mm=1000.0) == 20

    def test_candidates_never_exceed_true_core_even_with_overshoot_padding(
        self,
    ) -> None:
        """The specific adversarial-review finding: the OLD implementation
        used ``_core_z_span_slices`` (a deliberate over-estimate for the
        simulation window's own safety margin) as the candidate cutoff,
        which could include slices past the true arc-length-clipped core.
        This test builds a bundle whose lesion core provably ends at a
        known z-index and asserts every returned candidate is at or
        before it.
        """
        bundle = _standard_bundle()
        cfg = _FAST_CONFIG  # length_mm=45 (default), spacing_z=2.0mm here
        centerline = iliac_centerlines(bundle)["left"]
        # True core: cumulative arc length <= 45mm from z=50 (first vertex).
        last_index = true_core_last_index(centerline, cfg.length_mm)
        true_core_upper_z = int(
            centerline[last_index, 2] / 2.0
        )  # world mm -> voxel z (spacing 2.0)

        candidates = positive_center_z_candidates(bundle, "left", cfg)
        assert candidates.max() <= true_core_upper_z
        assert candidates.min() >= FIRST_CENTER_Z

        # And every returned candidate genuinely produces a nonzero target
        # (the whole point of restricting to the true core).
        for z in (int(candidates.min()), int(candidates.max())):
            sample = build_sample(
                bundle, z, seed=999, positive=True, side="left", config=cfg
            )
            assert float(sample.target_mask.sum()) > 0.0, z


# ---------------------------------------------------------------------------
# Adversarial-review fix 3: per-bundle caching correctness + throughput.
# ---------------------------------------------------------------------------


class TestPerBundleCaching:
    def test_precompute_produces_bit_identical_samples(self) -> None:
        bundle = _standard_bundle()
        cfg = _FAST_CONFIG
        precompute = build_bundle_precompute(bundle, cfg)

        cases = [
            (False, None, 111),
            (True, "left", 222),
            (True, "right", 333),
        ]
        for positive, side, seed in cases:
            direct = build_sample(
                bundle,
                _STANDARD_CENTER_Z_IN_LESION,
                seed=seed,
                positive=positive,
                side=side,
                config=cfg,
            )
            cached = build_sample(
                bundle,
                _STANDARD_CENTER_Z_IN_LESION,
                seed=seed,
                positive=positive,
                side=side,
                config=cfg,
                _precompute=precompute,
            )
            assert torch.equal(direct.left_view, cached.left_view), (positive, side)
            assert torch.equal(direct.right_view, cached.right_view), (positive, side)
            assert torch.equal(direct.pet_diff, cached.pet_diff), (positive, side)
            assert torch.equal(direct.target_mask, cached.target_mask), (positive, side)
            assert torch.equal(direct.valid_mask, cached.valid_mask), (positive, side)
            assert torch.equal(direct.raw_pet, cached.raw_pet), (positive, side)
            assert direct.meta == cached.meta, (positive, side)

    def test_dataset_uses_precompute_and_agrees_with_direct_build_sample(
        self, tmp_path: Path
    ) -> None:
        bundle = _standard_bundle()
        bundle_dir = save_crop_bundle(bundle, tmp_path)
        # positive_fraction=1.0 -- deliberately forces every sample
        # positive so this test exercises the cached positive-sample code
        # path unconditionally, not at the mercy of a small sample count's
        # luck of the draw.
        config = DatasetConfig(samples_per_bundle=3, supersample=1)
        dataset = SiameseCropDataset(
            [bundle_dir], seed=42, positive_fraction=1.0, config=config
        )
        # Force the dataset's own (cached) code path.
        samples = [dataset[i] for i in range(len(dataset))]
        assert len(dataset._precompute_cache) == 1
        assert all(s.meta["positive"] for s in samples)
        for sample in samples:
            assert torch.all(torch.isfinite(sample.left_view))


# ---------------------------------------------------------------------------
# Adversarial-review fix 3: picklability + DataLoader(num_workers>0) safety.
# ---------------------------------------------------------------------------


class TestPicklingAndDataLoader:
    def test_dataset_is_picklable_before_and_after_use(self, tmp_path: Path) -> None:
        bundle_dir = save_crop_bundle(_standard_bundle(), tmp_path)
        config = DatasetConfig(samples_per_bundle=2, supersample=1)
        dataset = SiameseCropDataset([bundle_dir], seed=1, config=config)

        reloaded_before_use = pickle.loads(pickle.dumps(dataset))
        assert len(reloaded_before_use) == len(dataset)

        _ = dataset[0]  # populate the bundle/precompute caches
        reloaded_after_use = pickle.loads(pickle.dumps(dataset))
        sample = reloaded_after_use[0]
        assert tuple(sample.left_view.shape) == (CHANNELS_PER_VIEW, *IN_PLANE_HW)

    def test_dataloader_with_multiple_workers(self, tmp_path: Path) -> None:
        """``batch_size=None`` (per-item, no default_collate) is this
        Dataset's demonstrated ``DataLoader`` usage pattern: ``Sample`` is
        a plain dataclass of tensors, not a dict/namedtuple, so PyTorch's
        ``default_collate`` cannot batch it automatically (verified
        directly during this implementation's implementation) -- a training loop
        wanting batched tensors supplies its own small ``collate_fn``
        (trivial, given every ``Sample`` field is already a tensor), which
        is that loop's responsibility, not this dataset module's.
        """
        bundle_dir = save_crop_bundle(_standard_bundle(), tmp_path)
        config = DatasetConfig(samples_per_bundle=4, supersample=1)
        dataset = SiameseCropDataset(
            [bundle_dir], seed=1, positive_fraction=0.5, config=config
        )
        loader = DataLoader(dataset, batch_size=None, num_workers=2)
        items = list(loader)
        assert len(items) == 4
        for item in items:
            assert tuple(item.left_view.shape) == (CHANNELS_PER_VIEW, *IN_PLANE_HW)
            assert torch.all(torch.isfinite(item.left_view))


# ---------------------------------------------------------------------------
# @pytest.mark.local_data -- one-or-more real, already-computed P2 crop
# bundles.
# ---------------------------------------------------------------------------

_LOCAL_CROPS_ROOT = Path("data/processed/p2/crops/p2-crop-v2")


def _all_available_real_bundle_dirs() -> list[Path]:
    if not _LOCAL_CROPS_ROOT.is_dir():
        return []
    return sorted(
        candidate
        for candidate in _LOCAL_CROPS_ROOT.glob("*/*")
        if (candidate / "bundle.json").is_file()
        and (candidate / "bundle.npz").is_file()
    )


def _first_available_real_bundle_dir() -> Path | None:
    dirs = _all_available_real_bundle_dirs()
    return dirs[0] if dirs else None


@pytest.mark.local_data
class TestRealBundle:
    def test_centerlines_and_samples_on_one_real_bundle(self) -> None:
        bundle_dir = _first_available_real_bundle_dir()
        if bundle_dir is None:
            pytest.skip("no local p2-crop-v2 bundle available")
        bundle = load_crop_bundle(bundle_dir)

        centerlines = iliac_centerlines(bundle)
        assert centerlines["left"].shape[0] > 0
        assert centerlines["right"].shape[0] > 0
        assert centerlines["left"].shape[1] == 3
        assert centerlines["right"].shape[1] == 3

        left_z_indices = np.unique(
            np.argwhere(bundle.iliac_label_mask == ILIAC_LABEL_LEFT)[:, 2]
        )
        # A few slices into the left corridor (not the very first, to give
        # the capsule's rounded cap room to reach full cross-section), but
        # comfortably inside the fixed 45 mm lesion core.
        center_index = min(5, left_z_indices.size - 1)
        center_z = int(
            np.clip(left_z_indices[center_index], FIRST_CENTER_Z, LAST_CENTER_Z)
        )

        positive_sample = build_sample(
            bundle, center_z, seed=2026, positive=True, side="left"
        )
        healthy_sample = build_sample(bundle, center_z, seed=2026, positive=False)

        for sample in (positive_sample, healthy_sample):
            assert tuple(sample.left_view.shape) == (CHANNELS_PER_VIEW, *IN_PLANE_HW)
            assert tuple(sample.right_view.shape) == (CHANNELS_PER_VIEW, *IN_PLANE_HW)
            assert tuple(sample.pet_diff.shape) == PET_DIFF_SHAPE
            assert tuple(sample.target_mask.shape) == TARGET_SHAPE
            assert tuple(sample.valid_mask.shape) == TARGET_SHAPE
            assert tuple(sample.raw_pet.shape) == TARGET_SHAPE
            assert sample.left_view.dtype == torch.float32
            assert sample.raw_pet.dtype == torch.float32

        # Lesion present in the positive target (aggregate count only).
        assert float(positive_sample.target_mask.sum()) > 0.0
        # Healthy sample's target is exactly zero.
        assert torch.count_nonzero(healthy_sample.target_mask).item() == 0

        # raw_pet retained and unnormalized: a structural (not per-voxel
        # value) check -- the normalized network PET channel is exactly
        # clip(raw_pet, 0, 10) / 10, proving raw_pet is not itself clipped.
        for sample in (positive_sample, healthy_sample):
            center_channel = sample.left_view[HALF_K]
            reconstructed = torch.clip(sample.raw_pet[0], 0.0, 10.0) / 10.0
            assert torch.allclose(center_channel, reconstructed, atol=1e-6)


@pytest.mark.local_data
class TestEmptyPositiveTargetRateOnRealBundles:
    def test_random_candidate_path_rarely_produces_an_empty_target(self) -> None:
        """Exercises ``SiameseCropDataset.__getitem__``'s own random
        ``center_z``-candidate path (NOT a caller-pinned ``center_z``) on
        every available real bundle and asserts the empty-positive-target
        rate stays low -- the adversarial-review-fix-2 acceptance check.
        """
        bundle_dirs = _all_available_real_bundle_dirs()[:6]
        if not bundle_dirs:
            pytest.skip("no local p2-crop-v2 bundles available")

        # supersample is deliberately cheap here (1, not the production
        # default of 5) -- see _FAST_CONFIG's docstring comment: whether a
        # slice is empty or not does not depend on supersample, and this
        # keeps dozens of samples across every available real bundle
        # tractable within a routine local_data run.
        config = DatasetConfig(samples_per_bundle=20, supersample=1)
        dataset = SiameseCropDataset(
            bundle_dirs, seed=2026, positive_fraction=1.0, config=config
        )

        total = len(dataset)
        empty_count = sum(
            1
            for i in range(total)
            if torch.count_nonzero(dataset[i].target_mask).item() == 0
        )
        empty_rate = empty_count / total
        assert empty_rate < 0.05, (empty_count, total, empty_rate)
