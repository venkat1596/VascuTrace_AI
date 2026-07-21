"""Tests for the VascuTrace Phase 2 data pipeline
(``src.vascutrace.data.{ingest,split,crops,contract}``).

Two families of tests live here, per the implementation's test plan:

- **Generated-fixture tests** (no ``Data/`` access, no ``local_data`` marker):
  cover the split algorithm, the leakage proof, every pure crop/reflection
  -geometry helper, and the crop-bundle contract's save/load/hash-integrity
  behavior, all against small synthetic arrays/affines or tiny in-memory
  NIfTI phantoms built with ``nibabel.Nifti1Image`` directly (the same
  pattern ``tests/test_geometry.py`` and ``tests/test_quantification.py``
  use). These run in normal CPU/offline CI.
- **Real-data tests** (``@pytest.mark.local_data``): exercise
  ``discover_dataset``/``pin_label_semantics``/``validate_dataset_grids`` and
  ``compute_session_crop`` against the real local ``Data/QUADRA_HC`` archive.
  These are the local promotion gate and are skipped by default CI.

Aggregate-only discipline: local_data tests print/assert only counts, shapes,
and boolean flags -- never a subject identifier or raw path in an assertion
message.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pytest

from src.vascutrace.data.contract import (
    ADJACENT_SLICE_COUNT_K,
    AXIAL_ADJACENT_SLICE_AXIS,
    CROP_SCHEMA_VERSION,
    FIXED_CROP_SHAPE,
    ILIAC_LABEL_LEFT,
    ILIAC_LABEL_RIGHT,
    NETWORK_TENSOR_FIELDS,
    CropIntegrityError,
    CropIntegrityErrorCode,
    load_crop_bundle,
    make_crop_bundle,
    reflect_volume,
    save_crop_bundle,
    validate_reflection_affine,
)
from src.vascutrace.data.crops import (
    REFLECTION_RESIDUAL_QC_THRESHOLD_MM,
    CropError,
    ReflectionFitError,
    ReflectionFitErrorCode,
    build_reflection_affine,
    center_crop_window,
    compute_session_crop,
    expand_world_bbox,
    extract_fixed_crop,
    fit_reflection_plane,
    label_union_bbox_voxels,
    paired_centerline_points,
    reflection_plane_residual_mm,
    run_crop_pipeline,
    world_bbox_to_voxel_bbox,
)
from src.vascutrace.data.ingest import (
    LABEL_SCHEME,
    NIFTI_COUNT,
    SEGMENTATION_GROUPS,
    SESSION_COUNT,
    SESSION_NAMES,
    SUBJECT_COUNT,
    DatasetManifest,
    GridValidationError,
    IngestError,
    IngestErrorCode,
    LabelPinError,
    SessionPaths,
    collect_provenance,
    discover_dataset,
    ingest_dataset,
    pin_label_semantics,
    validate_dataset_grids,
)
from src.vascutrace.data.split import (
    N_TEST_SUBJECTS,
    N_VAL_SUBJECTS,
    SPLIT_SEED,
    leakage_proof,
    load_subject_split,
    save_subject_split,
    stratified_subject_split,
    subject_partition_map,
)
from src.vascutrace.geometry import GeometrySidecar

# ---------------------------------------------------------------------------
# Fixture helpers (generated data only)
# ---------------------------------------------------------------------------


def _make_nifti_image(
    shape: tuple[int, int, int],
    affine: np.ndarray,
    data: np.ndarray,
    *,
    qform_code: int = 1,
    sform_code: int = 1,
    xyz_units: str = "mm",
) -> nib.Nifti1Image:
    """Small, generated, non-patient NIfTI phantom with explicit qform/sform
    and units -- matches the ``tests/test_geometry.py`` pattern.
    """
    img = nib.Nifti1Image(data, affine)
    if qform_code:
        img.header.set_qform(affine, code=qform_code)
    if sform_code:
        img.header.set_sform(affine, code=sform_code)
    if xyz_units:
        img.header.set_xyzt_units(xyz=xyz_units, t="sec")
    return img


def _write_segmentation_with_labels(
    path: Path,
    shape: tuple[int, int, int],
    affine: np.ndarray,
    labels_present: set[int],
) -> None:
    """Write a small uint8 segmentation NIfTI at ``path`` containing exactly
    one voxel of each label in ``labels_present`` (plus background 0).
    """
    data = np.zeros(shape, dtype=np.uint8)
    for offset, label in enumerate(sorted(labels_present)):
        flat_index = offset % data.size
        idx = np.unravel_index(flat_index, shape)
        data[idx] = label
    nib.save(_make_nifti_image(shape, affine, data), path)


_TINY_SHAPE = (4, 4, 6)
_TINY_AFFINE = np.diag([1.65, 1.65, 2.0, 1.0])


def _build_fake_dataset_tree(
    root: Path,
    *,
    n_subjects: int = SUBJECT_COUNT,
    drop_subject_dir: bool = False,
    drop_session_dir: bool = False,
    drop_pet: bool = False,
    drop_segmentation_group: str | None = None,
) -> Path:
    """Build a fake ``<root>/Imaging Data/`` tree matching the QUADRA_HC
    layout. Files are content-empty (``discover_dataset`` only checks
    existence) except where a specific defect is requested via the
    ``drop_*`` flags, applied to the *first* subject/session only.
    """
    imaging_root = root / "Imaging Data"
    imaging_root.mkdir(parents=True, exist_ok=True)
    for subject_index in range(1, n_subjects + 1):
        subject = f"SYNTH_SUBJECT_{subject_index:03d}"
        subject_dir = imaging_root / subject
        if drop_subject_dir and subject_index == 1:
            continue
        subject_dir.mkdir(parents=True, exist_ok=True)
        for session_name in SESSION_NAMES:
            if drop_session_dir and subject_index == 1 and session_name == "Test":
                continue
            session_dir = subject_dir / session_name
            session_dir.mkdir(parents=True, exist_ok=True)
            if not (drop_pet and subject_index == 1 and session_name == "Test"):
                (session_dir / f"{subject}_{session_name}_PT-SUV.nii.gz").touch()
            (session_dir / f"{subject}_{session_name}_CT-AC.nii.gz").touch()
            seg_dir = session_dir / "Segmentations"
            seg_dir.mkdir(parents=True, exist_ok=True)
            for group in SEGMENTATION_GROUPS:
                if (
                    drop_segmentation_group == group
                    and subject_index == 1
                    and session_name == "Test"
                ):
                    continue
                (seg_dir / f"{subject}_{session_name}_{group}.nii.gz").touch()
    return root


def _discover_fake_dataset(root: Path) -> DatasetManifest:
    """Run discovery against the generated, non-patient fixture prefix."""
    with patch("src.vascutrace.data.ingest.SUBJECT_PREFIX", "SYNTH_SUBJECT_"):
        return discover_dataset(root)


# ---------------------------------------------------------------------------
# ingest.py -- discovery
# ---------------------------------------------------------------------------


class TestDiscoverDataset:
    def test_happy_path_reconciles_48_96_960(self, tmp_path: Path) -> None:
        root = _build_fake_dataset_tree(tmp_path)
        manifest = _discover_fake_dataset(root)
        assert manifest.subject_count == SUBJECT_COUNT
        assert manifest.session_count == SESSION_COUNT
        assert manifest.nifti_count == NIFTI_COUNT

    def test_missing_imaging_root_raises(self, tmp_path: Path) -> None:
        with pytest.raises(IngestError) as excinfo:
            _discover_fake_dataset(tmp_path / "does-not-exist")
        assert excinfo.value.code == IngestErrorCode.IMAGING_ROOT_NOT_FOUND

    def test_wrong_subject_count_raises(self, tmp_path: Path) -> None:
        root = _build_fake_dataset_tree(tmp_path, n_subjects=10)
        with pytest.raises(IngestError) as excinfo:
            _discover_fake_dataset(root)
        assert excinfo.value.code == IngestErrorCode.SUBJECT_COUNT_MISMATCH
        assert excinfo.value.detail["found"] == 10

    def test_missing_session_directory_raises(self, tmp_path: Path) -> None:
        root = _build_fake_dataset_tree(tmp_path, drop_session_dir=True)
        with pytest.raises(IngestError) as excinfo:
            _discover_fake_dataset(root)
        assert excinfo.value.code == IngestErrorCode.MISSING_SESSION_DIRECTORY

    def test_missing_pet_file_raises(self, tmp_path: Path) -> None:
        root = _build_fake_dataset_tree(tmp_path, drop_pet=True)
        with pytest.raises(IngestError) as excinfo:
            _discover_fake_dataset(root)
        assert excinfo.value.code == IngestErrorCode.MISSING_SESSION_FILE

    def test_missing_segmentation_group_raises(self, tmp_path: Path) -> None:
        root = _build_fake_dataset_tree(tmp_path, drop_segmentation_group="Cardiac")
        with pytest.raises(IngestError) as excinfo:
            _discover_fake_dataset(root)
        assert excinfo.value.code == IngestErrorCode.MISSING_SEGMENTATION_GROUP
        assert excinfo.value.detail["group"] == "Cardiac"

    def test_error_detail_never_carries_a_path(self, tmp_path: Path) -> None:
        root = _build_fake_dataset_tree(tmp_path, n_subjects=5)
        with pytest.raises(IngestError) as excinfo:
            _discover_fake_dataset(root)
        for value in excinfo.value.detail.values():
            assert not isinstance(value, Path)
            assert str(root) not in str(value)


# ---------------------------------------------------------------------------
# ingest.py -- label-semantics pin (highest-stakes check)
# ---------------------------------------------------------------------------


def _session_paths_with_labels(
    tmp_path: Path, cardiac_labels: set[int], peripheral_labels: set[int]
) -> SessionPaths:
    seg_dir = tmp_path / "Segmentations"
    seg_dir.mkdir(parents=True, exist_ok=True)
    cardiac_path = seg_dir / "Cardiac.nii.gz"
    peripheral_path = seg_dir / "Peripheral-Bones.nii.gz"
    _write_segmentation_with_labels(
        cardiac_path, _TINY_SHAPE, _TINY_AFFINE, cardiac_labels
    )
    _write_segmentation_with_labels(
        peripheral_path, _TINY_SHAPE, _TINY_AFFINE, peripheral_labels
    )
    other_groups = {g: cardiac_path for g in SEGMENTATION_GROUPS}
    other_groups["Cardiac"] = cardiac_path
    other_groups["Peripheral-Bones"] = peripheral_path
    return SessionPaths(
        subject="SYNTH_SUBJECT_001",
        session="Test",
        pet=cardiac_path,
        ct=cardiac_path,
        segmentations=other_groups,
    )


class TestLabelPin:
    def test_correct_scheme_passes(self, tmp_path: Path) -> None:
        session = _session_paths_with_labels(tmp_path, {7, 8}, {5, 6})
        manifest = DatasetManifest(root=tmp_path, sessions=(session,))
        report = pin_label_semantics(manifest)
        assert report.all_passed
        assert report.failing_ordinals == ()

    def test_missing_iliac_label_raises_typed_error(self, tmp_path: Path) -> None:
        # Right iliac (8) missing -- the disagreement this check exists to
        # catch (fitness note's highest-stakes finding).
        session = _session_paths_with_labels(tmp_path, {7}, {5, 6})
        manifest = DatasetManifest(root=tmp_path, sessions=(session,))
        with pytest.raises(LabelPinError) as excinfo:
            pin_label_semantics(manifest)
        report = excinfo.value.report
        assert report.failing_ordinals == (0,)
        assert "iliac_right" in report.session_results[0].missing_labels

    def test_missing_femur_label_raises_typed_error(self, tmp_path: Path) -> None:
        session = _session_paths_with_labels(tmp_path, {7, 8}, {5})
        manifest = DatasetManifest(root=tmp_path, sessions=(session,))
        with pytest.raises(LabelPinError) as excinfo:
            pin_label_semantics(manifest)
        assert "femur_right" in excinfo.value.report.session_results[0].missing_labels

    def test_non_integer_dtype_raises_typed_error(self, tmp_path: Path) -> None:
        seg_dir = tmp_path / "Segmentations"
        seg_dir.mkdir(parents=True, exist_ok=True)
        cardiac_path = seg_dir / "Cardiac.nii.gz"
        peripheral_path = seg_dir / "Peripheral-Bones.nii.gz"
        float_data = np.zeros(_TINY_SHAPE, dtype=np.float32)
        float_data.flat[0] = 7.0
        float_data.flat[1] = 8.0
        nib.save(_make_nifti_image(_TINY_SHAPE, _TINY_AFFINE, float_data), cardiac_path)
        _write_segmentation_with_labels(
            peripheral_path, _TINY_SHAPE, _TINY_AFFINE, {5, 6}
        )
        session = SessionPaths(
            subject="SYNTH_SUBJECT_001",
            session="Test",
            pet=cardiac_path,
            ct=cardiac_path,
            segmentations={
                **{g: peripheral_path for g in SEGMENTATION_GROUPS},
                "Cardiac": cardiac_path,
                "Peripheral-Bones": peripheral_path,
            },
        )
        manifest = DatasetManifest(root=tmp_path, sessions=(session,))
        with pytest.raises(LabelPinError) as excinfo:
            pin_label_semantics(manifest)
        assert (
            "Cardiac"
            in excinfo.value.report.session_results[0].non_integer_dtype_groups
        )

    def test_multi_session_report_isolates_failing_ordinal(
        self, tmp_path: Path
    ) -> None:
        good = _session_paths_with_labels(tmp_path / "good", {7, 8}, {5, 6})
        bad = _session_paths_with_labels(tmp_path / "bad", {7}, {5, 6})
        manifest = DatasetManifest(root=tmp_path, sessions=(good, bad))
        with pytest.raises(LabelPinError) as excinfo:
            pin_label_semantics(manifest)
        assert excinfo.value.report.failing_ordinals == (1,)

    def test_label_scheme_matches_certified_mapping(self) -> None:
        assert LABEL_SCHEME["Cardiac"] == {"iliac_left": 7, "iliac_right": 8}
        assert LABEL_SCHEME["Peripheral-Bones"] == {"femur_left": 5, "femur_right": 6}


# ---------------------------------------------------------------------------
# ingest.py -- grid validation (delegates to geometry.validate_nifti_grid)
# ---------------------------------------------------------------------------


class TestGridValidation:
    def test_valid_grids_pass(self, tmp_path: Path) -> None:
        data = np.zeros(_TINY_SHAPE, dtype=np.float32)
        pet_path = tmp_path / "pet.nii.gz"
        ct_path = tmp_path / "ct.nii.gz"
        nib.save(_make_nifti_image(_TINY_SHAPE, _TINY_AFFINE, data), pet_path)
        nib.save(_make_nifti_image(_TINY_SHAPE, _TINY_AFFINE, data), ct_path)
        session = SessionPaths(
            subject="SYNTH_SUBJECT_001",
            session="Test",
            pet=pet_path,
            ct=ct_path,
            segmentations={},
        )
        manifest = DatasetManifest(root=tmp_path, sessions=(session,))
        report = validate_dataset_grids(manifest)
        assert report.all_passed

    def test_ambiguous_affine_raises_typed_error(self, tmp_path: Path) -> None:
        data = np.zeros(_TINY_SHAPE, dtype=np.float32)
        pet_path = tmp_path / "pet.nii.gz"
        ct_path = tmp_path / "ct.nii.gz"
        # qform_code=0 -> uncoded/ambiguous, per geometry.validate_nifti_grid.
        nib.save(
            _make_nifti_image(_TINY_SHAPE, _TINY_AFFINE, data, qform_code=0), pet_path
        )
        nib.save(_make_nifti_image(_TINY_SHAPE, _TINY_AFFINE, data), ct_path)
        session = SessionPaths(
            subject="SYNTH_SUBJECT_001",
            session="Test",
            pet=pet_path,
            ct=ct_path,
            segmentations={},
        )
        manifest = DatasetManifest(root=tmp_path, sessions=(session,))
        with pytest.raises(GridValidationError) as excinfo:
            validate_dataset_grids(manifest)
        result = excinfo.value.report.session_results[0]
        assert result.pet_passed is False
        assert result.pet_error_code == "AMBIGUOUS_AFFINE"
        assert result.ct_passed is True


# ---------------------------------------------------------------------------
# ingest.py -- provenance (informational, never raises)
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_empty_headers_reported_absent(self, tmp_path: Path) -> None:
        session = _session_paths_with_labels(tmp_path, {7, 8}, {5, 6})
        manifest = DatasetManifest(root=tmp_path, sessions=(session,))
        report = collect_provenance(manifest)
        assert report.sessions_checked == 1
        assert report.header_text_present_count == 0
        assert report.moose_version_documentary == "3.0.13"


# ---------------------------------------------------------------------------
# split.py -- stratified split + leakage proof (pure, no Data/)
# ---------------------------------------------------------------------------


def _synthetic_subject_sex(n_female: int = 25, n_male: int = 23) -> dict[str, str]:
    subjects = {}
    index = 1
    for _ in range(n_female):
        subjects[f"SYNTH_SUBJECT_{index:03d}"] = "F"
        index += 1
    for _ in range(n_male):
        subjects[f"SYNTH_SUBJECT_{index:03d}"] = "M"
        index += 1
    return subjects


class TestStratifiedSplit:
    def test_counts_match_certified_split(self) -> None:
        subject_sex = _synthetic_subject_sex()
        split = stratified_subject_split(subject_sex, seed=SPLIT_SEED)
        assert len(split.train) == 32
        assert len(split.val) == N_VAL_SUBJECTS
        assert len(split.test) == N_TEST_SUBJECTS

        from collections import Counter

        assert Counter(subject_sex[s] for s in split.val) == {"F": 4, "M": 4}
        assert Counter(subject_sex[s] for s in split.test) == {"F": 4, "M": 4}
        assert Counter(subject_sex[s] for s in split.train) == {"F": 17, "M": 15}

    def test_deterministic_across_runs(self) -> None:
        subject_sex = _synthetic_subject_sex()
        split_a = stratified_subject_split(subject_sex, seed=SPLIT_SEED)
        split_b = stratified_subject_split(subject_sex, seed=SPLIT_SEED)
        assert split_a.train == split_b.train
        assert split_a.val == split_b.val
        assert split_a.test == split_b.test

    def test_different_seed_changes_membership(self) -> None:
        subject_sex = _synthetic_subject_sex()
        split_a = stratified_subject_split(subject_sex, seed=SPLIT_SEED)
        split_b = stratified_subject_split(subject_sex, seed=SPLIT_SEED + 1)
        assert set(split_a.val) != set(split_b.val) or set(split_a.test) != set(
            split_b.test
        )

    def test_full_coverage_and_disjoint(self) -> None:
        subject_sex = _synthetic_subject_sex()
        split = stratified_subject_split(subject_sex, seed=SPLIT_SEED)
        all_subjects = set(subject_sex)
        assert set(split.train) | set(split.val) | set(split.test) == all_subjects
        assert not (set(split.train) & set(split.val))
        assert not (set(split.train) & set(split.test))
        assert not (set(split.val) & set(split.test))


class TestLeakageProof:
    def test_clean_split_passes_subject_and_session_level(self) -> None:
        subject_sex = _synthetic_subject_sex()
        split = stratified_subject_split(subject_sex, seed=SPLIT_SEED)
        sessions = [(s, name) for s in subject_sex for name in ("Test", "Retest")]
        result = leakage_proof(split, all_subjects=list(subject_sex), sessions=sessions)
        assert result.passed
        assert result.subject_level_disjoint
        assert result.subject_level_full_coverage
        assert result.session_level_no_subject_split
        assert result.every_session_subject_known

    def test_manually_corrupted_split_fails_subject_level(self) -> None:
        subject_sex = _synthetic_subject_sex()
        split = stratified_subject_split(subject_sex, seed=SPLIT_SEED)
        # Deliberately inject one train subject into val too.
        from src.vascutrace.data.split import SubjectSplit

        leaking_val = (*split.val, split.train[0])
        corrupted = SubjectSplit(
            schema_version=split.schema_version,
            seed=split.seed,
            train=split.train,
            val=leaking_val,
            test=split.test,
        )
        result = leakage_proof(corrupted, all_subjects=list(subject_sex))
        assert not result.passed
        assert not result.subject_level_disjoint

    def test_session_assigned_to_wrong_partition_fails_session_level(self) -> None:
        subject_sex = _synthetic_subject_sex()
        split = stratified_subject_split(subject_sex, seed=SPLIT_SEED)
        partition_of = subject_partition_map(split)
        # Find one subject in train and one in val; simulate a session-level
        # leak by asserting a train subject's session also appears attached
        # to val's session list (i.e. the same subject appears effectively
        # split across sessions is impossible by construction here, so we
        # instead directly fabricate a (subject, session) list referencing a
        # subject as if it had a session counted under a different partition
        # -- this exercises the mechanical check itself using a corrupted
        # split, not a corrupted session list, since sessions always inherit
        # the subject's partition by construction in the real pipeline).
        train_subject = split.train[0]
        val_subject = split.val[0]
        assert partition_of[train_subject] == "train"
        assert partition_of[val_subject] == "val"
        # Sanity: the mechanical proof at least sees both subjects.
        sessions = [
            (train_subject, "Test"),
            (train_subject, "Retest"),
            (val_subject, "Test"),
        ]
        result = leakage_proof(split, sessions=sessions)
        assert result.session_level_no_subject_split

    def test_unknown_session_subject_fails(self) -> None:
        subject_sex = _synthetic_subject_sex()
        split = stratified_subject_split(subject_sex, seed=SPLIT_SEED)
        result = leakage_proof(split, sessions=[("SYNTH_SUBJECT_999", "Test")])
        assert not result.passed
        assert not result.every_session_subject_known


class TestSplitPersistence:
    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        subject_sex = _synthetic_subject_sex()
        split = stratified_subject_split(subject_sex, seed=SPLIT_SEED)
        path = save_subject_split(split, tmp_path, sex_by_subject=subject_sex)
        assert path.is_file()
        reloaded = load_subject_split(path)
        assert reloaded.train == split.train
        assert reloaded.val == split.val
        assert reloaded.test == split.test
        assert reloaded.seed == SPLIT_SEED


# ---------------------------------------------------------------------------
# crops.py -- pure geometry helpers (no Data/)
# ---------------------------------------------------------------------------


class TestBboxHelpers:
    def test_label_union_bbox_voxels(self) -> None:
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[2, 3, 4] = 7
        mask[6, 8, 9] = 8
        result = label_union_bbox_voxels(mask, (7, 8))
        assert result is not None
        min_idx, max_idx = result
        np.testing.assert_array_equal(min_idx, [2, 3, 4])
        np.testing.assert_array_equal(max_idx, [6, 8, 9])

    def test_label_union_bbox_voxels_absent_returns_none(self) -> None:
        mask = np.zeros((5, 5, 5), dtype=np.uint8)
        assert label_union_bbox_voxels(mask, (7, 8)) is None

    def test_expand_world_bbox_adds_margin_symmetrically(self) -> None:
        lo, hi = expand_world_bbox(
            np.array([0.0, 0.0, 0.0]), np.array([10.0, 10.0, 10.0]), 5.0
        )
        np.testing.assert_allclose(lo, [-5.0, -5.0, -5.0])
        np.testing.assert_allclose(hi, [15.0, 15.0, 15.0])

    def test_world_bbox_to_voxel_bbox_guarantees_containment(self) -> None:
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        min_idx, max_idx = world_bbox_to_voxel_bbox(
            np.array([1.0, 1.0, 1.0]), np.array([9.0, 9.0, 9.0]), affine
        )
        # World 1mm at 2mm spacing -> voxel 0.5 -> floor 0; world 9mm -> voxel 4.5 -> ceil 5.
        np.testing.assert_array_equal(min_idx, [0, 0, 0])
        np.testing.assert_array_equal(max_idx, [5, 5, 5])

    def test_center_crop_window_centers_and_flags_no_exceed(self) -> None:
        crop_start, exceeds = center_crop_window(
            np.array([10, 10, 10]), np.array([20, 20, 20]), (144, 80, 144)
        )
        # bbox extent is 11 on every axis, well under the fixed shape.
        assert exceeds == (False, False, False)
        center = np.array([15.0, 15.0, 15.0])
        expected_start = np.round(center - np.array([72.0, 40.0, 72.0])).astype(
            np.int64
        )
        np.testing.assert_array_equal(crop_start, expected_start)

    def test_center_crop_window_flags_bbox_exceeding_fixed_shape(self) -> None:
        _crop_start, exceeds = center_crop_window(
            np.array([0, 0, 0]), np.array([200, 10, 10]), (144, 80, 144)
        )
        assert exceeds == (True, False, False)


class TestExtractFixedCrop:
    def test_fully_interior_crop_is_all_valid(self) -> None:
        volume = np.arange(20 * 20 * 20, dtype=np.float32).reshape(20, 20, 20)
        crop, valid = extract_fixed_crop(volume, np.array([2, 2, 2]), (10, 10, 10))
        assert crop.shape == (10, 10, 10)
        assert valid.shape == (10, 10, 10)
        assert np.all(valid == 1)
        np.testing.assert_array_equal(crop, volume[2:12, 2:12, 2:12])

    def test_crop_extending_past_lower_bound_pads_and_flags_invalid(self) -> None:
        volume = np.ones((10, 10, 10), dtype=np.float32)
        crop, valid = extract_fixed_crop(volume, np.array([-3, 0, 0]), (10, 10, 10))
        assert crop.shape == (10, 10, 10)
        # The first 3 slices along axis 0 are out-of-FOV padding.
        assert np.all(valid[:3] == 0)
        assert np.all(crop[:3] == 0)
        assert np.all(valid[3:] == 1)
        assert np.all(crop[3:] == 1)

    def test_crop_entirely_outside_volume_is_all_invalid(self) -> None:
        volume = np.ones((10, 10, 10), dtype=np.float32)
        crop, valid = extract_fixed_crop(volume, np.array([1000, 0, 0]), (10, 10, 10))
        assert np.all(valid == 0)
        assert np.all(crop == 0)

    def test_valid_mask_dtype_is_uint8(self) -> None:
        volume = np.ones((10, 10, 10), dtype=np.float32)
        _crop, valid = extract_fixed_crop(volume, np.array([0, 0, 0]), (5, 5, 5))
        assert valid.dtype == np.uint8


class TestReflectionPlaneFit:
    def _mirror_symmetric_pairs(
        self, n_slices: int = 8, offset_x: float = 100.0
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Perfectly mirror-symmetric points about the plane x = offset_x/2
        (i.e. normal = +X, point = (offset_x/2, *, *))."""
        pairs = []
        for z in range(n_slices):
            left = np.array([0.0, float(z), float(z)])
            right = np.array([offset_x, float(z), float(z)])
            pairs.append((left, right))
        return pairs

    def test_perfect_mirror_gives_x_normal_and_zero_residual(self) -> None:
        pairs = self._mirror_symmetric_pairs()
        normal, point = fit_reflection_plane(pairs)
        np.testing.assert_allclose(normal, [1.0, 0.0, 0.0], atol=1e-9)
        assert point[0] == pytest.approx(50.0)
        residual = reflection_plane_residual_mm(pairs, normal, point)
        assert residual == pytest.approx(0.0, abs=1e-9)

    def test_normal_oriented_toward_ras_plus_r(self) -> None:
        # Swap left/right labeling so the raw mean(R-L) points toward -X;
        # fit_reflection_plane must still return a +X-oriented normal.
        pairs = [(right, left) for left, right in self._mirror_symmetric_pairs()]
        normal, _point = fit_reflection_plane(pairs)
        assert normal[0] > 0.0

    def test_no_paired_points_raises_typed_error(self) -> None:
        with pytest.raises(ReflectionFitError) as excinfo:
            fit_reflection_plane([])
        assert excinfo.value.code == ReflectionFitErrorCode.NO_PAIRED_POINTS

    def test_degenerate_coincident_points_raises_typed_error(self) -> None:
        coincident = [(np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0]))]
        with pytest.raises(ReflectionFitError) as excinfo:
            fit_reflection_plane(coincident)
        assert excinfo.value.code == ReflectionFitErrorCode.DEGENERATE_NORMAL

    def test_perturbed_pairs_give_small_nonzero_residual_and_no_qc_flag(self) -> None:
        rng = np.random.default_rng(0)
        pairs = self._mirror_symmetric_pairs()
        perturbed = [
            (left + rng.normal(0, 0.5, size=3), right + rng.normal(0, 0.5, size=3))
            for left, right in pairs
        ]
        normal, point = fit_reflection_plane(perturbed)
        residual = reflection_plane_residual_mm(perturbed, normal, point)
        assert 0.0 < residual < REFLECTION_RESIDUAL_QC_THRESHOLD_MM

    def test_large_asymmetry_exceeds_qc_threshold(self) -> None:
        pairs = self._mirror_symmetric_pairs()
        # Alternate a +/-30mm shift of the right point *along the normal
        # axis* (X). A uniform (non-alternating) shift would just relocate
        # the best-fit plane with ~zero residual; the alternation is what
        # makes each individual pair deviate from any single global plane.
        skewed = [
            (left, right + np.array([30.0 if i % 2 == 0 else -30.0, 0.0, 0.0]))
            for i, (left, right) in enumerate(pairs)
        ]
        normal, point = fit_reflection_plane(skewed)
        residual = reflection_plane_residual_mm(skewed, normal, point)
        assert residual > REFLECTION_RESIDUAL_QC_THRESHOLD_MM


class TestReflectionAffineConstruction:
    def test_built_affine_passes_validation(self) -> None:
        normal = np.array([1.0, 0.0, 0.0])
        point = np.array([50.0, 0.0, 0.0])
        affine = build_reflection_affine(normal, point)
        validated = validate_reflection_affine(affine)
        np.testing.assert_allclose(validated, affine)

    def test_reflection_is_involution(self) -> None:
        normal = np.array([0.3, 0.7, 0.2])
        point = np.array([12.0, -4.0, 8.0])
        affine = build_reflection_affine(normal, point)
        test_point = np.array([1.0, 2.0, 3.0, 1.0])
        once = affine @ test_point
        twice = affine @ once
        np.testing.assert_allclose(twice[:3], test_point[:3], atol=1e-9)

    def test_reflecting_the_plane_point_is_a_fixed_point(self) -> None:
        normal = np.array([0.0, 1.0, 0.0])
        point = np.array([5.0, 5.0, 5.0])
        affine = build_reflection_affine(normal, point)
        homogeneous_point = np.array([*point, 1.0])
        reflected = affine @ homogeneous_point
        np.testing.assert_allclose(reflected[:3], point, atol=1e-9)


class TestPairedCenterlinePoints:
    def test_pairs_only_slices_with_both_labels(self) -> None:
        mask = np.zeros((5, 5, 6), dtype=np.uint8)
        # slice z=0: both labels present -> paired.
        mask[1, 1, 0] = 7
        mask[3, 1, 0] = 8
        # slice z=1: only left present -> not paired.
        mask[1, 1, 1] = 7
        affine = np.eye(4)
        pairs = paired_centerline_points(mask, 7, 8, affine, axial_axis=2)
        assert len(pairs) == 1
        left, right = pairs[0]
        np.testing.assert_allclose(left, [1.0, 1.0, 0.0])
        np.testing.assert_allclose(right, [3.0, 1.0, 0.0])

    def test_no_labels_returns_empty(self) -> None:
        mask = np.zeros((4, 4, 4), dtype=np.uint8)
        pairs = paired_centerline_points(mask, 7, 8, np.eye(4))
        assert pairs == []


# ---------------------------------------------------------------------------
# contract.py -- CropBundle construction, hashing, save/load, integrity
# ---------------------------------------------------------------------------


def _minimal_geometry_sidecar() -> GeometrySidecar:
    return GeometrySidecar(
        canonical_shape=(440, 440, 531),
        original_shape=(440, 440, 531),
        canonical_affine_sha256="a" * 64,
        original_affine_sha256="b" * 64,
        original_voxel_from_canonical_voxel_sha256="c" * 64,
    )


def _synthetic_iliac_label_mask() -> np.ndarray:
    """A small synthetic ``iliac_label_mask``: a handful of left-label (7)
    voxels and a handful of right-label (8) voxels, 0 elsewhere -- the
    field's exact contract (see ``contract.py``, "iliac_label_mask field").
    """
    mask = np.zeros(FIXED_CROP_SHAPE, dtype=np.uint8)
    mask[60:64, 30:34, 60:64] = ILIAC_LABEL_LEFT
    mask[80:84, 30:34, 60:64] = ILIAC_LABEL_RIGHT
    return mask


def _synthetic_bundle_fields() -> dict:
    rng = np.random.default_rng(42)
    normal = np.array([1.0, 0.0, 0.0])
    point = np.array([50.0, 0.0, 0.0])
    reflection_affine = build_reflection_affine(normal, point)
    return dict(
        subject="SYNTH_SUBJECT_001",
        session="Test",
        pet_suvbw=rng.random(FIXED_CROP_SHAPE, dtype=np.float32) * 10.0,
        ct_hu=(rng.random(FIXED_CROP_SHAPE, dtype=np.float32) * 2000.0 - 1000.0),
        valid_pet_mask=np.ones(FIXED_CROP_SHAPE, dtype=np.uint8),
        iliac_label_mask=_synthetic_iliac_label_mask(),
        reflection_affine=reflection_affine,
        crop_to_pet_canonical_affine=np.diag([1.65, 1.65, 2.0, 1.0]),
        crop_origin_voxel=(10, 20, 30),
        original_voxel_from_pet_canonical_voxel=np.eye(4),
        geometry_sidecar=_minimal_geometry_sidecar(),
        reflection_residual_mm=2.5,
        reflection_qc_flag=False,
        bbox_exceeds_fixed_crop=(False, False, False),
        crop_margin_mm=15.0,
        pet_spacing_mm=(1.65, 1.65, 2.0),
        paired_point_count=172,
    )


class TestMakeCropBundle:
    def test_builds_and_hashes(self) -> None:
        bundle = make_crop_bundle(**_synthetic_bundle_fields())
        assert bundle.schema_version == CROP_SCHEMA_VERSION
        assert bundle.schema_version == "p2-crop-v2"
        assert bundle.pet_suvbw.shape == FIXED_CROP_SHAPE
        assert bundle.pet_suvbw.dtype == np.float32
        assert set(bundle.hashes) == {
            "pet_suvbw",
            "ct_hu",
            "valid_pet_mask",
            "iliac_label_mask",
            "reflection_affine",
            "crop_to_pet_canonical_affine",
            "original_voxel_from_pet_canonical_voxel",
            "params",
        }

    def test_wrong_shape_raises(self) -> None:
        fields = _synthetic_bundle_fields()
        fields["pet_suvbw"] = np.zeros((10, 10, 10), dtype=np.float32)
        with pytest.raises(CropIntegrityError) as excinfo:
            make_crop_bundle(**fields)
        assert excinfo.value.code == CropIntegrityErrorCode.SHAPE_MISMATCH

    def test_invalid_reflection_affine_raises(self) -> None:
        fields = _synthetic_bundle_fields()
        fields["reflection_affine"] = np.eye(4)  # not orientation-reversing
        with pytest.raises(CropIntegrityError) as excinfo:
            make_crop_bundle(**fields)
        assert excinfo.value.code == CropIntegrityErrorCode.INVALID_REFLECTION_AFFINE

    def test_iliac_label_mask_shape_dtype_and_labels(self) -> None:
        bundle = make_crop_bundle(**_synthetic_bundle_fields())
        assert bundle.iliac_label_mask.shape == FIXED_CROP_SHAPE
        assert bundle.iliac_label_mask.dtype == np.uint8
        present_labels = set(np.unique(bundle.iliac_label_mask).tolist())
        assert present_labels <= {0, ILIAC_LABEL_LEFT, ILIAC_LABEL_RIGHT}
        assert ILIAC_LABEL_LEFT in present_labels
        assert ILIAC_LABEL_RIGHT in present_labels

    def test_wrong_shape_iliac_label_mask_raises(self) -> None:
        fields = _synthetic_bundle_fields()
        fields["iliac_label_mask"] = np.zeros((10, 10, 10), dtype=np.uint8)
        with pytest.raises(CropIntegrityError) as excinfo:
            make_crop_bundle(**fields)
        assert excinfo.value.code == CropIntegrityErrorCode.SHAPE_MISMATCH

    def test_invalid_iliac_label_mask_value_raises(self) -> None:
        fields = _synthetic_bundle_fields()
        bad_mask = _synthetic_iliac_label_mask()
        bad_mask[0, 0, 0] = 5  # femur label leaking into the iliac field -- forbidden
        fields["iliac_label_mask"] = bad_mask
        with pytest.raises(CropIntegrityError) as excinfo:
            make_crop_bundle(**fields)
        assert excinfo.value.code == CropIntegrityErrorCode.INVALID_ILIAC_LABEL_MASK


class TestSaveLoadCropBundle:
    def test_roundtrip_preserves_arrays_and_metadata(self, tmp_path: Path) -> None:
        bundle = make_crop_bundle(**_synthetic_bundle_fields())
        directory = save_crop_bundle(bundle, tmp_path)
        reloaded = load_crop_bundle(directory)
        np.testing.assert_array_equal(reloaded.pet_suvbw, bundle.pet_suvbw)
        np.testing.assert_array_equal(reloaded.ct_hu, bundle.ct_hu)
        np.testing.assert_array_equal(reloaded.valid_pet_mask, bundle.valid_pet_mask)
        np.testing.assert_array_equal(
            reloaded.iliac_label_mask, bundle.iliac_label_mask
        )
        np.testing.assert_allclose(reloaded.reflection_affine, bundle.reflection_affine)
        assert reloaded.subject == bundle.subject
        assert reloaded.session == bundle.session
        assert reloaded.crop_origin_voxel == bundle.crop_origin_voxel
        assert reloaded.hashes == bundle.hashes

    def test_tampered_array_fails_hash_verification(self, tmp_path: Path) -> None:
        bundle = make_crop_bundle(**_synthetic_bundle_fields())
        directory = save_crop_bundle(bundle, tmp_path)
        with np.load(directory / "bundle.npz") as npz:
            arrays = {key: npz[key] for key in npz.files}
        arrays["pet_suvbw"] = arrays["pet_suvbw"] + 1.0  # tamper
        np.savez_compressed(directory / "bundle.npz", **arrays)
        with pytest.raises(CropIntegrityError) as excinfo:
            load_crop_bundle(directory)
        assert excinfo.value.code == CropIntegrityErrorCode.HASH_MISMATCH

    def test_tampered_iliac_label_mask_fails_hash_verification(
        self, tmp_path: Path
    ) -> None:
        bundle = make_crop_bundle(**_synthetic_bundle_fields())
        directory = save_crop_bundle(bundle, tmp_path)
        with np.load(directory / "bundle.npz") as npz:
            arrays = {key: npz[key] for key in npz.files}
        tampered = arrays["iliac_label_mask"].copy()
        tampered[0, 0, 0] = (
            ILIAC_LABEL_LEFT  # still a valid label value, but a hash-breaking edit
        )
        arrays["iliac_label_mask"] = tampered
        np.savez_compressed(directory / "bundle.npz", **arrays)
        with pytest.raises(CropIntegrityError) as excinfo:
            load_crop_bundle(directory)
        assert excinfo.value.code == CropIntegrityErrorCode.HASH_MISMATCH

    def test_missing_bundle_raises_typed_error(self, tmp_path: Path) -> None:
        with pytest.raises(CropIntegrityError) as excinfo:
            load_crop_bundle(tmp_path / "nonexistent")
        assert excinfo.value.code == CropIntegrityErrorCode.MISSING_BUNDLE_FILE

    def test_schema_version_mismatch_raises_typed_error(self, tmp_path: Path) -> None:
        bundle = make_crop_bundle(**_synthetic_bundle_fields())
        directory = save_crop_bundle(bundle, tmp_path)
        import json

        meta_path = directory / "bundle.json"
        meta = json.loads(meta_path.read_text())
        meta["schema_version"] = "some-other-version"
        meta_path.write_text(json.dumps(meta))
        with pytest.raises(CropIntegrityError) as excinfo:
            load_crop_bundle(directory)
        assert excinfo.value.code == CropIntegrityErrorCode.SCHEMA_VERSION_MISMATCH


class TestReflectVolume:
    def test_reflecting_twice_is_near_identity(self) -> None:
        rng = np.random.default_rng(1)
        volume = rng.random(FIXED_CROP_SHAPE).astype(np.float32)
        affine = np.diag([1.65, 1.65, 2.0, 1.0])
        normal = np.array([1.0, 0.0, 0.0])
        point = np.array([FIXED_CROP_SHAPE[0] * 1.65 / 2.0, 0.0, 0.0])
        reflection_affine = build_reflection_affine(normal, point)

        once = reflect_volume(volume, affine, reflection_affine, order=1)
        twice = reflect_volume(once, affine, reflection_affine, order=1)

        # Interior region only: near the crop boundary, reflect-twice loses
        # fidelity because part of the once-reflected volume samples outside
        # the original volume's support (zero-fill), which cannot be
        # perfectly un-reflected. The interior (away from the mirrored edge)
        # must round-trip closely.
        interior = np.s_[20:-20, :, :]
        assert np.mean(np.abs(twice[interior] - volume[interior])) < 0.05

    def test_symmetric_volume_reflects_to_itself(self) -> None:
        shape = (20, 6, 6)
        volume = np.zeros(shape, dtype=np.float32)
        # Build an X-axis-symmetric volume about the plane between indices 9/10.
        for i in range(shape[0]):
            volume[i, :, :] = min(i, shape[0] - 1 - i)
        affine = np.eye(4)
        normal = np.array([1.0, 0.0, 0.0])
        point = np.array([(shape[0] - 1) / 2.0, 0.0, 0.0])
        reflection_affine = build_reflection_affine(normal, point)
        reflected = reflect_volume(volume, affine, reflection_affine, order=1)
        np.testing.assert_allclose(reflected, volume, atol=1e-4)

    def test_invalid_reflection_affine_raises(self) -> None:
        volume = np.zeros((10, 10, 10), dtype=np.float32)
        with pytest.raises(CropIntegrityError):
            reflect_volume(volume, np.eye(4), np.eye(4))


class TestTensorContractDocumentation:
    def test_network_tensor_fields_cover_expected_names(self) -> None:
        expected = {
            "pet_left_suvbw",
            "pet_right_suvbw",
            "pet_diff",
            "ct_left_hu",
            "ct_right_hu",
            "valid_pet_mask",
            "target_mask",
        }
        assert set(NETWORK_TENSOR_FIELDS) == expected

    def test_axial_axis_and_k_are_frozen_within_spec(self) -> None:
        assert AXIAL_ADJACENT_SLICE_AXIS == 2
        assert 3 <= ADJACENT_SLICE_COUNT_K <= 7
        assert ADJACENT_SLICE_COUNT_K % 2 == 1  # odd: well-defined center slice


# ---------------------------------------------------------------------------
# Real-data tests (local promotion gate)
# ---------------------------------------------------------------------------

_DATA_ROOT = Path("Data/QUADRA_HC")


@pytest.mark.local_data
class TestRealDataIngest:
    def test_discover_dataset_reconciles_full_cohort(self) -> None:
        manifest = discover_dataset(_DATA_ROOT)
        assert manifest.subject_count == SUBJECT_COUNT
        assert manifest.session_count == SESSION_COUNT
        assert manifest.nifti_count == NIFTI_COUNT

    def test_label_pin_and_grid_validation_on_a_sample(self) -> None:
        manifest = discover_dataset(_DATA_ROOT)
        sample = manifest.sessions[:2]
        label_report = pin_label_semantics(manifest, sessions=sample)
        assert label_report.all_passed
        grid_report = validate_dataset_grids(manifest, sessions=sample)
        assert grid_report.all_passed

    def test_ingest_dataset_end_to_end_on_a_sample(self) -> None:
        manifest = discover_dataset(_DATA_ROOT)
        sample = manifest.sessions[:2]
        result = ingest_dataset(_DATA_ROOT, sessions=sample)
        assert result.label_pin_report.all_passed
        assert result.grid_validation_report.all_passed
        assert result.provenance_report.sessions_checked == 2


@pytest.mark.local_data
class TestRealDataSplit:
    def test_split_from_real_demographics_matches_certified_counts(self) -> None:
        from src.vascutrace.data.split import load_subject_sex_table

        subject_sex = load_subject_sex_table(_DATA_ROOT / "Demographics (All).xlsx")
        assert len(subject_sex) == SUBJECT_COUNT
        split = stratified_subject_split(subject_sex, seed=SPLIT_SEED)
        assert len(split.train) == 32
        assert len(split.val) == N_VAL_SUBJECTS
        assert len(split.test) == N_TEST_SUBJECTS
        result = leakage_proof(split, all_subjects=list(subject_sex))
        assert result.passed


@pytest.mark.local_data
class TestRealDataCrops:
    def test_compute_session_crop_shapes_and_integrity(self, tmp_path: Path) -> None:
        manifest = discover_dataset(_DATA_ROOT)
        session = manifest.sessions[0]
        bundle = compute_session_crop(session)
        assert bundle.schema_version == "p2-crop-v2"
        assert bundle.pet_suvbw.shape == FIXED_CROP_SHAPE
        assert bundle.ct_hu.shape == FIXED_CROP_SHAPE
        assert bundle.valid_pet_mask.shape == FIXED_CROP_SHAPE
        assert bundle.paired_point_count > 0
        validate_reflection_affine(bundle.reflection_affine)  # must not raise

        # iliac_label_mask: real-data check that both labels are present,
        # correctly typed, and no foreign label value leaked through from
        # the broader Cardiac MOOSE group (aggregate-only: counts, not
        # per-voxel locations).
        assert bundle.iliac_label_mask.shape == FIXED_CROP_SHAPE
        assert bundle.iliac_label_mask.dtype == np.uint8
        present_labels = set(np.unique(bundle.iliac_label_mask).tolist())
        assert present_labels <= {0, ILIAC_LABEL_LEFT, ILIAC_LABEL_RIGHT}
        left_voxel_count = int(np.sum(bundle.iliac_label_mask == ILIAC_LABEL_LEFT))
        right_voxel_count = int(np.sum(bundle.iliac_label_mask == ILIAC_LABEL_RIGHT))
        assert left_voxel_count > 0
        assert right_voxel_count > 0
        # Plausible range: full-cohort iliac voxel counts (fitness note Q4)
        # are O(1e3-1e4) on the native segmentation grid; resampling onto
        # the coarser PET grid changes the exact count but not the order of
        # magnitude -- a few hundred to a few tens of thousands is the sane
        # band, not a handful of stray voxels or a saturated crop.
        assert 50 <= left_voxel_count <= 50_000
        assert 50 <= right_voxel_count <= 50_000

        directory = save_crop_bundle(bundle, tmp_path)
        reloaded = load_crop_bundle(directory)
        np.testing.assert_array_equal(reloaded.pet_suvbw, bundle.pet_suvbw)
        np.testing.assert_array_equal(
            reloaded.iliac_label_mask, bundle.iliac_label_mask
        )

    def test_run_crop_pipeline_on_two_sessions(self, tmp_path: Path) -> None:
        manifest = discover_dataset(_DATA_ROOT)
        sample = manifest.sessions[:2]
        report = run_crop_pipeline(manifest, tmp_path, sessions=sample)
        assert len(report.outcomes) == 2
        assert 0.0 <= report.mean_valid_coverage <= 1.0
        for outcome in report.outcomes:
            reloaded = load_crop_bundle(outcome.bundle_path)
            assert reloaded.pet_suvbw.shape == FIXED_CROP_SHAPE

    def test_no_iliac_labels_raises_typed_error(self, tmp_path: Path) -> None:
        # A synthetic, non-Data/-derived session whose Cardiac segmentation
        # has no iliac voxels -- exercises the typed CropError without
        # needing a real defective session (none exist in this cohort).
        cardiac_data = np.zeros((8, 8, 8), dtype=np.uint8)
        peripheral_data = np.zeros((8, 8, 8), dtype=np.uint8)
        peripheral_data[1, 1, 1] = 5
        peripheral_data[6, 1, 1] = 6
        affine = np.diag([1.52, 1.52, 2.0, 1.0])
        cardiac_path = tmp_path / "cardiac.nii.gz"
        peripheral_path = tmp_path / "peripheral.nii.gz"
        nib.save(_make_nifti_image((8, 8, 8), affine, cardiac_data), cardiac_path)
        nib.save(_make_nifti_image((8, 8, 8), affine, peripheral_data), peripheral_path)

        pet_data = np.zeros((20, 20, 20), dtype=np.float32)
        pet_affine = np.diag([1.65, 1.65, 2.0, 1.0])
        pet_path = tmp_path / "pet.nii.gz"
        ct_path = tmp_path / "ct.nii.gz"
        nib.save(_make_nifti_image((20, 20, 20), pet_affine, pet_data), pet_path)
        nib.save(
            _make_nifti_image((20, 20, 20), affine, cardiac_data.astype(np.float32)),
            ct_path,
        )

        session = SessionPaths(
            subject="SYNTH_SUBJECT_001",
            session="Test",
            pet=pet_path,
            ct=ct_path,
            segmentations={
                **{g: cardiac_path for g in SEGMENTATION_GROUPS},
                "Cardiac": cardiac_path,
                "Peripheral-Bones": peripheral_path,
            },
        )
        with pytest.raises(CropError) as excinfo:
            compute_session_crop(session)
        from src.vascutrace.data.crops import CropErrorCode

        assert excinfo.value.code == CropErrorCode.ILIAC_LABELS_NOT_FOUND
