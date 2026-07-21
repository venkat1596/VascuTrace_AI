"""Tests for the P6 training loop / checkpointing / CLI
(``src.vascutrace.ml.checkpoint``, ``src.vascutrace.ml.train``,
``src.vascutrace.ml.cli``).

CPU-only, generated-fixture tests -- no ``Data/`` access anywhere in this
file. Bundles are built via ``make_crop_bundle``/``save_crop_bundle``
directly (matching the fixture pattern already established in
``tests/test_ml_dataset.py``'s ``_standard_bundle``), never read from the
real ``data/processed/p2/crops/...`` tree. Models use ``ModelConfig
(base_channels=4)`` and datasets use a small ``samples_per_bundle`` +
``limit_train_batches``/``limit_val_batches`` so the whole file stays well
under the ~60s CPU budget. Any test that requires a real CUDA device is
marked ``@pytest.mark.gpu`` and skipped by the CPU/offline gate
(``pytest -m "not gpu"``).
"""

from __future__ import annotations

import json
import math
import random
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

import src.vascutrace.ml.train as train_module
from src.vascutrace.data.contract import (
    FIXED_CROP_SHAPE,
    ILIAC_LABEL_LEFT,
    ILIAC_LABEL_RIGHT,
    CropBundle,
    make_crop_bundle,
    save_crop_bundle,
)
from src.vascutrace.data.crops import build_reflection_affine
from src.vascutrace.geometry import RESEARCH_PROTOTYPE_WARNING, GeometrySidecar
from src.vascutrace.ml.cache import (
    CachePrepError,
    CacheSchemaError,
    CachedSampleDataset,
    precompute_synthetic_cache,
)
from src.vascutrace.ml.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointError,
    CheckpointPayload,
    capture_rng_state,
    load_checkpoint,
    restore_rng,
    save_checkpoint,
)
from src.vascutrace.ml.dataset import DatasetConfig, SiameseCropDataset
from src.vascutrace.ml.losses import (
    combo_loss,
    focal_tversky_loss,
    soft_bce_loss,
    soft_combo_loss,
    soft_dice_semimetric_loss,
)
from src.vascutrace.ml.model import ModelConfig, build_model, model_signature
from src.vascutrace.ml.tensor_schema import CROP_SCHEMA_VERSION as _CROP_SCHEMA_VERSION
from src.vascutrace.ml.tensor_schema import TENSOR_SCHEMA_VERSION
from src.vascutrace.ml.train import (
    CheckpointCompatibilityError,
    CudaUnavailableError,
    TrainConfig,
    TrainConfigError,
    ValidationMetrics,
    compute_split_hash,
    resume,
    train,
)
from src.vascutrace.ml.train import _augment_batch  # noqa: PLC2701 -- direct unit test
from src.vascutrace.ml.train import _lr_at_step  # noqa: PLC2701 -- direct unit test
from src.vascutrace.ml.train import (  # noqa: PLC2701 -- direct unit test
    _per_sample_clipped_negative_score,
)
from src.vascutrace.ml.train import _select_metric_value  # noqa: PLC2701 -- direct unit test
from src.vascutrace.ml.train import _update_ema  # noqa: PLC2701 -- direct unit test

# ---------------------------------------------------------------------------
# Shared synthetic-fixture builders (deliberately duplicated from
# tests/test_ml_dataset.py's own helpers, matching this project's
# small-primitive-duplication convention -- see e.g.
# src.vascutrace.simulation.anomaly._apply_affine_points's own docstring).
# ---------------------------------------------------------------------------

_STANDARD_AFFINE = np.diag([2.0, 2.0, 2.0, 1.0])
_STANDARD_REFLECTION = build_reflection_affine(
    np.array([1.0, 0.0, 0.0]), np.array([144.0, 0.0, 0.0])
)


def _geometry_sidecar() -> GeometrySidecar:
    return GeometrySidecar(
        canonical_shape=(440, 440, 531),
        original_shape=(440, 440, 531),
        canonical_affine_sha256="a" * 64,
        original_affine_sha256="b" * 64,
        original_voxel_from_canonical_voxel_sha256="c" * 64,
    )


def _standard_iliac_label_mask() -> np.ndarray:
    mask = np.zeros(FIXED_CROP_SHAPE, dtype=np.uint8)
    mask[60:64, 35:45, 50:90] = ILIAC_LABEL_LEFT
    mask[80:84, 35:45, 50:90] = ILIAC_LABEL_RIGHT
    return mask


def _make_bundle(*, subject: str, seed: int, session: str = "Test") -> CropBundle:
    rng = np.random.default_rng(seed)
    pet = (rng.random(FIXED_CROP_SHAPE, dtype=np.float32) * 8.0 + 1.0).astype(
        np.float32
    )
    ct = (rng.random(FIXED_CROP_SHAPE, dtype=np.float32) * 200.0 - 100.0).astype(
        np.float32
    )
    valid = np.ones(FIXED_CROP_SHAPE, dtype=np.uint8)
    spacing = tuple(float(v) for v in np.linalg.norm(_STANDARD_AFFINE[:3, :3], axis=0))
    return make_crop_bundle(
        subject=subject,
        session=session,
        pet_suvbw=pet,
        ct_hu=ct,
        valid_pet_mask=valid,
        iliac_label_mask=_standard_iliac_label_mask(),
        reflection_affine=_STANDARD_REFLECTION,
        crop_to_pet_canonical_affine=_STANDARD_AFFINE,
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


def _two_bundle_dirs(root: Path) -> tuple[Path, Path]:
    """Two DISTINCT bundle directories (different subjects) -- required
    because ``TrainConfig`` rejects any overlap between
    ``train_bundle_dirs`` and ``val_bundle_dirs`` (leakage-safety).
    """
    train_dir = save_crop_bundle(
        _make_bundle(subject="SYNTH_SUBJECT_101", seed=1), root
    )
    val_dir = save_crop_bundle(_make_bundle(subject="SYNTH_SUBJECT_102", seed=2), root)
    return train_dir, val_dir


@pytest.fixture(scope="module")
def shared_bundle_dirs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Two read-only synthetic bundle directories, built ONCE and reused by
    every test in this module. Safe to share: no test in this file ever
    writes into a bundle's ``bundle.npz``/``bundle.json`` -- only
    ``SiameseCropDataset`` reads them (via ``dataset.py``'s own per
    -instance load cache), and dataset sample construction
    (``build_bilateral_views`` -> ``reflect_volume``, a full-volume
    trilinear resample) dominates a ``train()`` call's wall-clock time far
    more than model forward/backward on this tiny ``base_channels=4``
    network (measured directly: ~0.6s/sample for a healthy sample's
    reflect/slice/normalize vs. ~0.1s for 1 forward+backward step at
    batch=2) -- rebuilding two fresh bundles per test, as every other P6
    dataset/model test file does, would blow this file's ~60s CPU budget
    many times over once several ``train()`` calls per test are involved.
    """
    root = tmp_path_factory.mktemp("shared_bundles")
    return _two_bundle_dirs(root)


def _tiny_train_config(
    *, train_dir: Path, val_dir: Path, run_root: Path, **overrides
) -> TrainConfig:
    """A small, fast ``TrainConfig``: ``base_channels=4``, 2 samples per
    bundle (dataset len 2 -> 2 single-sample steps/epoch), all-healthy
    (``positive_fraction=0.0``) samples by default so no test pays the
    synthetic-lesion simulation cost unless a test explicitly asks for it
    -- see ``dataset.py``'s module docstring, item 5: only a ``positive``
    sample invokes P3's simulator. ``samples_per_bundle``/``batch_size``
    are deliberately minimal: per-sample cost is dominated by
    ``build_bilateral_views``'s ``reflect_volume`` (a full-volume
    trilinear resample, ~0.6s/sample measured directly), not model
    forward/backward, so keeping the sample count as small as possible
    while still exercising >= 2 training steps + 1 validation batch is
    what keeps this file inside its CPU wall-clock budget.
    """
    seed = overrides.pop("seed", 0)
    defaults: dict = dict(
        train_bundle_dirs=(train_dir,),
        val_bundle_dirs=(val_dir,),
        run_root=run_root,
        seed=seed,
        val_seed=seed,
        batch_size=1,
        max_epochs=1,
        lr=1e-3,
        weight_decay=0.0,
        early_stop_patience=100,
        device="cpu",
        amp=False,
        num_workers=0,
        model_config=ModelConfig(base_channels=4, seed=seed),
        dataset_config=DatasetConfig(samples_per_bundle=2),
        train_positive_fraction=0.0,
        val_positive_fraction=0.0,
        limit_train_batches=2,
        limit_val_batches=1,
        log_every_n_steps=1,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


def _tiny_payload(
    *, epoch: int, global_step: int, take_optimizer_step: bool = False
) -> CheckpointPayload:
    """A minimal, structurally-valid :class:`CheckpointPayload` for tests
    that exercise ``checkpoint.py`` directly (no ``TrainConfig``/dataset
    involved).
    """
    model = build_model(ModelConfig(base_channels=4, seed=0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler(device="cpu", enabled=False)

    if take_optimizer_step:
        from src.vascutrace.ml.tensor_schema import (
            LEFT_VIEW_SHAPE,
            PET_DIFF_SHAPE,
            RIGHT_VIEW_SHAPE,
            TARGET_SHAPE,
        )
        from src.vascutrace.ml.model import dice_bce_loss

        left = torch.randn(1, *LEFT_VIEW_SHAPE)
        right = torch.randn(1, *RIGHT_VIEW_SHAPE)
        diff = torch.randn(1, *PET_DIFF_SHAPE)
        target = (torch.rand(1, *TARGET_SHAPE) > 0.5).float()
        logits = model(left, right, diff)
        loss = dice_bce_loss(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    generator = torch.Generator()
    generator.manual_seed(0)

    return CheckpointPayload(
        checkpoint_schema_version=CHECKPOINT_SCHEMA_VERSION,
        tensor_schema_version=TENSOR_SCHEMA_VERSION,
        crop_schema_version=_CROP_SCHEMA_VERSION,
        model_signature=model_signature(model.config),
        model_config=model.config,
        dataset_config=DatasetConfig(),
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        scaler_state_dict=scaler.state_dict(),
        rng_state=capture_rng_state(generator),
        epoch=epoch,
        global_step=global_step,
        best_val_metric=None,
        best_val_metric_name="val_dice",
    )


def _metrics_records(metrics_path: Path) -> list[dict]:
    return [json.loads(line) for line in metrics_path.read_text().splitlines() if line]


def _first_train_loss(metrics_path: Path) -> float:
    for record in _metrics_records(metrics_path):
        if record["event"] == "train_step":
            return record["train_loss"]
    raise AssertionError(f"no train_step record found in {metrics_path}")


# ---------------------------------------------------------------------------
# 1. Config validation rejects bad configs BEFORE allocation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_rejects_non_positive_batch_size_before_allocation(
        self, tmp_path: Path, shared_bundle_dirs: tuple[Path, Path]
    ) -> None:
        train_dir, val_dir = shared_bundle_dirs
        run_root = tmp_path / "run"
        with pytest.raises(TrainConfigError, match="batch_size"):
            _tiny_train_config(
                train_dir=train_dir, val_dir=val_dir, run_root=run_root, batch_size=0
            )
        # __post_init__ raised before any construction succeeded -- no
        # model/dataset/CUDA allocation and no run_root directory.
        assert not run_root.exists()

    def test_rejects_non_positive_max_epochs(
        self, tmp_path: Path, shared_bundle_dirs: tuple[Path, Path]
    ) -> None:
        train_dir, val_dir = shared_bundle_dirs
        with pytest.raises(TrainConfigError, match="max_epochs"):
            _tiny_train_config(
                train_dir=train_dir,
                val_dir=val_dir,
                run_root=tmp_path / "run",
                max_epochs=0,
            )

    def test_rejects_non_positive_lr(
        self, tmp_path: Path, shared_bundle_dirs: tuple[Path, Path]
    ) -> None:
        train_dir, val_dir = shared_bundle_dirs
        with pytest.raises(TrainConfigError, match="lr"):
            _tiny_train_config(
                train_dir=train_dir, val_dir=val_dir, run_root=tmp_path / "run", lr=-1.0
            )

    def test_rejects_negative_weight_decay(
        self, tmp_path: Path, shared_bundle_dirs: tuple[Path, Path]
    ) -> None:
        train_dir, val_dir = shared_bundle_dirs
        with pytest.raises(TrainConfigError, match="weight_decay"):
            _tiny_train_config(
                train_dir=train_dir,
                val_dir=val_dir,
                run_root=tmp_path / "run",
                weight_decay=-0.1,
            )

    def test_rejects_unknown_device(
        self, tmp_path: Path, shared_bundle_dirs: tuple[Path, Path]
    ) -> None:
        train_dir, val_dir = shared_bundle_dirs
        with pytest.raises(TrainConfigError, match="device"):
            _tiny_train_config(
                train_dir=train_dir,
                val_dir=val_dir,
                run_root=tmp_path / "run",
                device="tpu",
            )

    def test_rejects_empty_train_bundle_dirs(
        self, tmp_path: Path, shared_bundle_dirs: tuple[Path, Path]
    ) -> None:
        _, val_dir = shared_bundle_dirs
        with pytest.raises(TrainConfigError, match="train_bundle_dirs"):
            TrainConfig(
                train_bundle_dirs=(),
                val_bundle_dirs=(val_dir,),
                run_root=tmp_path / "run",
                model_config=ModelConfig(base_channels=4),
            )

    def test_rejects_overlapping_train_and_val_dirs(
        self, tmp_path: Path, shared_bundle_dirs: tuple[Path, Path]
    ) -> None:
        train_dir, _ = shared_bundle_dirs
        with pytest.raises(TrainConfigError, match="leakage"):
            _tiny_train_config(
                train_dir=train_dir, val_dir=train_dir, run_root=tmp_path / "run"
            )

    def test_rejects_non_positive_early_stop_patience(
        self, tmp_path: Path, shared_bundle_dirs: tuple[Path, Path]
    ) -> None:
        train_dir, val_dir = shared_bundle_dirs
        with pytest.raises(TrainConfigError, match="early_stop_patience"):
            _tiny_train_config(
                train_dir=train_dir,
                val_dir=val_dir,
                run_root=tmp_path / "run",
                early_stop_patience=0,
            )

    @pytest.mark.parametrize(
        "field_name",
        [
            "constrained_iou_min_precision",
            "constrained_iou_min_f1",
            "constrained_iou_min_clean",
        ],
    )
    def test_rejects_out_of_range_constrained_iou_floor(
        self,
        tmp_path: Path,
        shared_bundle_dirs: tuple[Path, Path],
        field_name: str,
    ) -> None:
        train_dir, val_dir = shared_bundle_dirs
        with pytest.raises(TrainConfigError, match=field_name):
            _tiny_train_config(
                train_dir=train_dir,
                val_dir=val_dir,
                run_root=tmp_path / "run",
                **{field_name: 1.5},
            )
        with pytest.raises(TrainConfigError, match=field_name):
            _tiny_train_config(
                train_dir=train_dir,
                val_dir=val_dir,
                run_root=tmp_path / "run",
                **{field_name: -0.1},
            )


# ---------------------------------------------------------------------------
# 2. A tiny CPU run runs a few steps + a val pass with finite losses and
#    produces a checkpoint.
# ---------------------------------------------------------------------------


class TestTinyCpuTrainingRun:
    def test_runs_steps_and_validation_with_finite_losses_and_checkpoint(
        self, tmp_path: Path, shared_bundle_dirs: tuple[Path, Path]
    ) -> None:
        train_dir, val_dir = shared_bundle_dirs
        run_root = tmp_path / "run"
        config = _tiny_train_config(
            train_dir=train_dir, val_dir=val_dir, run_root=run_root
        )

        result = train(config)

        assert result.final_epoch == 0
        assert result.global_step == 2  # limit_train_batches=2, batch_size=2
        assert result.last_checkpoint_path.is_file()
        assert result.best_checkpoint_path is not None
        assert result.best_checkpoint_path.is_file()
        assert result.best_val_metric is not None
        assert np.isfinite(result.best_val_metric)

        records = _metrics_records(result.metrics_log_path)
        train_records = [r for r in records if r["event"] == "train_step"]
        val_records = [r for r in records if r["event"] == "validation"]
        assert len(train_records) == 2
        assert len(val_records) == 1
        for record in train_records:
            assert np.isfinite(record["train_loss"])
            assert np.isfinite(record["lr"])
        for record in val_records:
            # See train.py's module docstring, item 11: the blended Dice
            # is kept/logged for continuity but is no longer the field
            # named "val_dice" -- every one of the new positive-focused
            # aggregates is logged too, and every one must be finite (this
            # module's own _finite_or NaN-sanitization).
            for key in (
                "blended_dice",
                "mean_positive_dice",
                "mean_positive_iou",
                "detection_precision",
                "detection_recall",
                "detection_f1",
                "negative_clean_rate",
                "dice_x_clean",
                "det_f1_gated_dice",
                "selection_metric_value",
            ):
                assert np.isfinite(record[key]), key
            assert record["selection_metric_name"] == config.selection_metric

        assert result.manifest_path.is_file()
        manifest = json.loads(result.manifest_path.read_text())
        assert manifest["research_prototype_warning"] == RESEARCH_PROTOTYPE_WARNING
        assert manifest["calibration_status"] == "uncalibrated"
        assert manifest["tensor_schema_version"] == TENSOR_SCHEMA_VERSION

        last_payload = load_checkpoint(result.last_checkpoint_path)
        assert last_payload.calibration_status == "uncalibrated"
        assert last_payload.research_prototype_warning == RESEARCH_PROTOTYPE_WARNING
        assert last_payload.epoch == 0
        assert last_payload.global_step == 2


# ---------------------------------------------------------------------------
# 3. Checkpoint round-trip: save -> load restores model/optimizer/scaler/
#    RNG/epoch/step exactly.
# ---------------------------------------------------------------------------


class TestCheckpointRoundTrip:
    def test_save_load_restores_state_exactly(self, tmp_path: Path) -> None:
        payload = _tiny_payload(epoch=3, global_step=17, take_optimizer_step=True)
        path = tmp_path / "ckpt.pt"

        save_checkpoint(path, payload)
        loaded = load_checkpoint(path)

        assert loaded.epoch == 3
        assert loaded.global_step == 17
        assert loaded.calibration_status == "uncalibrated"
        assert loaded.research_prototype_warning == RESEARCH_PROTOTYPE_WARNING
        assert loaded.checkpoint_schema_version == CHECKPOINT_SCHEMA_VERSION

        # Model state_dict: exact tensor equality, every key.
        assert set(loaded.model_state_dict) == set(payload.model_state_dict)
        for key, value in payload.model_state_dict.items():
            assert torch.equal(loaded.model_state_dict[key], value)

        # Optimizer state_dict: param_groups (plain values) + per-param
        # Adam moment tensors ("state"), exact equality.
        assert (
            loaded.optimizer_state_dict["param_groups"]
            == payload.optimizer_state_dict["param_groups"]
        )
        assert set(loaded.optimizer_state_dict["state"]) == set(
            payload.optimizer_state_dict["state"]
        )
        for param_id, field_map in payload.optimizer_state_dict["state"].items():
            loaded_field_map = loaded.optimizer_state_dict["state"][param_id]
            for field_name, value in field_map.items():
                loaded_value = loaded_field_map[field_name]
                if isinstance(value, torch.Tensor):
                    assert torch.equal(loaded_value, value)
                else:
                    assert loaded_value == value

        assert loaded.scaler_state_dict == payload.scaler_state_dict

        # RNG state restoration.
        random.seed(999)
        np.random.seed(999)
        torch.manual_seed(999)
        restored_generator = restore_rng(loaded)

        assert random.getstate() == payload.rng_state.python_random
        assert torch.equal(torch.get_rng_state(), payload.rng_state.torch_cpu)
        assert torch.equal(
            restored_generator.get_state(), payload.rng_state.dataloader_generator
        )
        restored_np_state = np.random.get_state()
        assert restored_np_state[0] == payload.rng_state.numpy_random[0]
        np.testing.assert_array_equal(
            restored_np_state[1], payload.rng_state.numpy_random[1]
        )

    def test_load_missing_file_raises_checkpoint_error(self, tmp_path: Path) -> None:
        with pytest.raises(CheckpointError):
            load_checkpoint(tmp_path / "does_not_exist.pt")


# ---------------------------------------------------------------------------
# 4. Atomic-write safety: an interrupted temp write never corrupts the
#    previous good checkpoint.
# ---------------------------------------------------------------------------


class TestAtomicCheckpointWrites:
    def test_interrupted_write_never_corrupts_previous_checkpoint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        good_payload = _tiny_payload(epoch=1, global_step=10)
        path = tmp_path / "last.pt"
        save_checkpoint(path, good_payload)
        assert path.is_file()
        original_bytes = path.read_bytes()

        bad_payload = _tiny_payload(epoch=99, global_step=999)

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated interrupted write")

        monkeypatch.setattr("src.vascutrace.ml.checkpoint.torch.save", _boom)

        with pytest.raises(RuntimeError, match="simulated interrupted write"):
            save_checkpoint(path, bad_payload)

        # last.pt is byte-for-byte untouched -- os.replace was never reached.
        assert path.read_bytes() == original_bytes
        reloaded = load_checkpoint(path)
        assert reloaded.epoch == 1
        assert reloaded.global_step == 10

        # No orphaned temp file left behind in the directory.
        leftovers = [p for p in tmp_path.iterdir() if p != path]
        assert leftovers == []

    def test_successful_save_overwrites_cleanly(self, tmp_path: Path) -> None:
        path = tmp_path / "last.pt"
        save_checkpoint(path, _tiny_payload(epoch=0, global_step=1))
        save_checkpoint(path, _tiny_payload(epoch=5, global_step=50))

        reloaded = load_checkpoint(path)
        assert reloaded.epoch == 5
        assert reloaded.global_step == 50
        leftovers = [p for p in tmp_path.iterdir() if p != path]
        assert leftovers == []


# ---------------------------------------------------------------------------
# 5. Resume equivalence: N steps -> checkpoint -> resume -> continue vs 2N
#    steps in one go => same final weights and same RNG stream.
#
# Checkpointing happens once per epoch (the implementation's own "save last.pt
# every epoch"), so this is exercised at epoch granularity: one continuous
# 2-epoch run vs a 1-epoch run checkpointed then resumed for 1 more epoch.
# ---------------------------------------------------------------------------


class TestResumeEquivalence:
    def test_resume_matches_a_single_continuous_run(
        self, tmp_path: Path, shared_bundle_dirs: tuple[Path, Path]
    ) -> None:
        train_dir, val_dir = shared_bundle_dirs

        # lr_schedule="none" deliberately: this test's own point is GENERAL
        # resume mechanics (weights, RNG stream), exercised across a
        # max_epochs-CHANGING resume (1 -> 2) as the simplest way to
        # simulate "partial run, then resumed" without a separate
        # interruption mechanism. A "cosine" schedule's auto-computed
        # total_steps is NOT stable across a max_epochs-changing resume by
        # design (see train.py's module docstring, item 12, and
        # TrainConfig.lr_schedule_total_steps's own docstring) -- that is
        # exercised deliberately, and correctly, by
        # TestSchedulerResumeEquivalence below instead.
        baseline_root = tmp_path / "baseline"
        baseline_result = train(
            _tiny_train_config(
                train_dir=train_dir,
                val_dir=val_dir,
                run_root=baseline_root,
                max_epochs=2,
                seed=2026,
                lr_schedule="none",
            )
        )
        baseline_payload = load_checkpoint(baseline_result.last_checkpoint_path)

        split_root = tmp_path / "split"
        first_result = train(
            _tiny_train_config(
                train_dir=train_dir,
                val_dir=val_dir,
                run_root=split_root,
                max_epochs=1,
                seed=2026,
                lr_schedule="none",
            )
        )
        assert first_result.final_epoch == 0

        resumed_result = resume(
            split_root,
            _tiny_train_config(
                train_dir=train_dir,
                val_dir=val_dir,
                run_root=split_root,
                max_epochs=2,
                seed=2026,
                lr_schedule="none",
            ),
        )
        resumed_payload = load_checkpoint(resumed_result.last_checkpoint_path)

        assert resumed_result.final_epoch == baseline_result.final_epoch == 1
        assert resumed_result.global_step == baseline_result.global_step

        # Same best-checkpoint-selection state too (a real bug, found and
        # fixed during this implementation's own verification: last.pt's payload
        # used to be built with the PRE-this-epoch's-update
        # best_val_metric, corrupting best_epoch/best_val_metric across a
        # resume boundary landing exactly on the best epoch -- see
        # train.py's _execute, "Update best-tracking state BEFORE building
        # the checkpoint payload").
        assert resumed_result.best_epoch == baseline_result.best_epoch
        assert resumed_result.best_val_metric == baseline_result.best_val_metric

        # Same final weights.
        assert set(resumed_payload.model_state_dict) == set(
            baseline_payload.model_state_dict
        )
        for key in baseline_payload.model_state_dict:
            torch.testing.assert_close(
                resumed_payload.model_state_dict[key],
                baseline_payload.model_state_dict[key],
                atol=1e-6,
                rtol=1e-5,
            )

        # Same RNG stream (the DataLoader-shuffle generator's state after
        # the last epoch is bit-identical between the two paths).
        assert torch.equal(
            resumed_payload.rng_state.dataloader_generator,
            baseline_payload.rng_state.dataloader_generator,
        )


# ---------------------------------------------------------------------------
# 5b. Resume equivalence WITH the cosine+warmup scheduler enabled -- the
#     common, "resume with the SAME max_epochs" pattern, where total_steps
#     is auto-computed and automatically stable across the resume boundary.
# ---------------------------------------------------------------------------


class TestSchedulerResumeEquivalence:
    def test_resume_matches_continuous_run_with_cosine_schedule(
        self, tmp_path: Path, shared_bundle_dirs: tuple[Path, Path]
    ) -> None:
        train_dir, val_dir = shared_bundle_dirs
        # Both the baseline and the split/resumed path use the SAME
        # max_epochs=2 throughout (the realistic "resume an interrupted
        # run to its original target" pattern) -- so the cosine schedule's
        # auto-computed total_steps is identical in every call, with no
        # need for the lr_schedule_total_steps override.
        common = {
            "max_epochs": 2,
            "seed": 4242,
            "lr_schedule": "cosine",
            "warmup_steps": 1,
        }

        baseline_root = tmp_path / "sched_baseline"
        baseline_result = train(
            _tiny_train_config(
                train_dir=train_dir, val_dir=val_dir, run_root=baseline_root, **common
            )
        )

        split_root = tmp_path / "sched_split"
        first_result = train(
            _tiny_train_config(
                train_dir=train_dir,
                val_dir=val_dir,
                run_root=split_root,
                max_epochs=1,
                seed=common["seed"],
                lr_schedule=common["lr_schedule"],
                warmup_steps=common["warmup_steps"],
                # Pin the SAME total_steps a max_epochs=2 run would compute,
                # since this call's own max_epochs (1) differs from the
                # eventual target (2) -- see TrainConfig.
                # lr_schedule_total_steps's own docstring.
                lr_schedule_total_steps=4,
            )
        )
        assert first_result.final_epoch == 0
        resumed_result = resume(
            split_root,
            _tiny_train_config(
                train_dir=train_dir,
                val_dir=val_dir,
                run_root=split_root,
                lr_schedule_total_steps=4,
                **common,
            ),
        )

        baseline_lrs = [
            r["lr"]
            for r in _metrics_records(baseline_result.metrics_log_path)
            if r["event"] == "train_step"
        ]
        # first_result and resumed_result share the SAME run_root, so
        # resumed_result.metrics_log_path is the identical (append-mode)
        # metrics.jsonl file already containing both calls' records, in
        # chronological order -- read it once, not both.
        assert first_result.metrics_log_path == resumed_result.metrics_log_path
        resumed_lrs = [
            r["lr"]
            for r in _metrics_records(resumed_result.metrics_log_path)
            if r["event"] == "train_step"
        ]

        assert resumed_lrs == baseline_lrs

        baseline_payload = load_checkpoint(baseline_result.last_checkpoint_path)
        resumed_payload = load_checkpoint(resumed_result.last_checkpoint_path)
        for key in baseline_payload.model_state_dict:
            torch.testing.assert_close(
                resumed_payload.model_state_dict[key],
                baseline_payload.model_state_dict[key],
                atol=1e-6,
                rtol=1e-5,
            )


class TestResumeCompatibilityGuards:
    def test_resume_rejects_changed_training_configuration(
        self, tmp_path: Path, shared_bundle_dirs: tuple[Path, Path]
    ) -> None:
        train_dir, val_dir = shared_bundle_dirs
        run_root = tmp_path / "changed_config"
        train(
            _tiny_train_config(
                train_dir=train_dir,
                val_dir=val_dir,
                run_root=run_root,
                max_epochs=1,
                lr_schedule="none",
            )
        )

        incompatible = replace(
            _tiny_train_config(
                train_dir=train_dir,
                val_dir=val_dir,
                run_root=run_root,
                max_epochs=2,
                lr_schedule="none",
            ),
            loss="dice_bce",
        )
        with pytest.raises(
            CheckpointCompatibilityError, match="config_hash|hyperparams.loss"
        ):
            resume(run_root, incompatible)

    @pytest.mark.parametrize(
        "stateful_override",
        [
            {"ema_decay": 0.9},
            {"secondary_selection_metric": "mean_positive_iou"},
            {"constrained_iou_selection": True},
        ],
        ids=["ema", "secondary-selector", "constrained-iou-selector"],
    )
    def test_resume_rejects_checkpoint_external_feature_state(
        self,
        tmp_path: Path,
        shared_bundle_dirs: tuple[Path, Path],
        stateful_override: dict[str, object],
    ) -> None:
        train_dir, val_dir = shared_bundle_dirs
        run_root = tmp_path / next(iter(stateful_override))
        first_config = replace(
            _tiny_train_config(
                train_dir=train_dir,
                val_dir=val_dir,
                run_root=run_root,
                max_epochs=1,
                lr_schedule="none",
            ),
            **stateful_override,
        )
        train(first_config)

        resume_config = replace(first_config, max_epochs=2)
        with pytest.raises(
            CheckpointCompatibilityError,
            match="checkpoint-external state",
        ):
            resume(run_root, resume_config)


# ---------------------------------------------------------------------------
# 6. Missing-CUDA: device="cuda" with CUDA unavailable raises a clear
#    error, NO silent CPU fallback.
# ---------------------------------------------------------------------------


class TestNoCudaFallback:
    def test_cuda_device_without_cuda_raises_not_falls_back(
        self,
        tmp_path: Path,
        shared_bundle_dirs: tuple[Path, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        train_dir, val_dir = shared_bundle_dirs
        run_root = tmp_path / "run"
        config = _tiny_train_config(
            train_dir=train_dir, val_dir=val_dir, run_root=run_root, device="cuda"
        )

        with pytest.raises(CudaUnavailableError):
            train(config)

        # No silent CPU fallback happened: the run never got far enough to
        # create run_root or write a checkpoint.
        assert not run_root.exists()


# ---------------------------------------------------------------------------
# 7. Seed determinism: two runs, same seed => identical first-step loss.
# ---------------------------------------------------------------------------


class TestSeedDeterminism:
    def test_same_seed_gives_identical_first_step_loss(
        self, tmp_path: Path, shared_bundle_dirs: tuple[Path, Path]
    ) -> None:
        train_dir, val_dir = shared_bundle_dirs

        config_a = _tiny_train_config(
            train_dir=train_dir, val_dir=val_dir, run_root=tmp_path / "run_a", seed=4242
        )
        config_b = _tiny_train_config(
            train_dir=train_dir, val_dir=val_dir, run_root=tmp_path / "run_b", seed=4242
        )
        train(config_a)
        train(config_b)

        loss_a = _first_train_loss(config_a.run_root / "metrics.jsonl")
        loss_b = _first_train_loss(config_b.run_root / "metrics.jsonl")
        assert loss_a == loss_b

    def test_different_seed_changes_first_step_loss(
        self, tmp_path: Path, shared_bundle_dirs: tuple[Path, Path]
    ) -> None:
        train_dir, val_dir = shared_bundle_dirs

        config_a = _tiny_train_config(
            train_dir=train_dir, val_dir=val_dir, run_root=tmp_path / "run_a", seed=1
        )
        config_b = _tiny_train_config(
            train_dir=train_dir, val_dir=val_dir, run_root=tmp_path / "run_b", seed=2
        )
        train(config_a)
        train(config_b)

        loss_a = _first_train_loss(config_a.run_root / "metrics.jsonl")
        loss_b = _first_train_loss(config_b.run_root / "metrics.jsonl")
        assert loss_a != loss_b


# ---------------------------------------------------------------------------
# GPU-marked: real CUDA + AMP tiny run. Skipped by `-m "not gpu"`.
# ---------------------------------------------------------------------------


@pytest.mark.gpu
class TestGpuAmpTrainingRun:
    def test_tiny_cuda_amp_run_completes_with_finite_losses(
        self, tmp_path: Path, shared_bundle_dirs: tuple[Path, Path]
    ) -> None:
        if not torch.cuda.is_available():
            pytest.skip("no CUDA device available on this machine")

        train_dir, val_dir = shared_bundle_dirs
        config = _tiny_train_config(
            train_dir=train_dir,
            val_dir=val_dir,
            run_root=tmp_path / "gpu_run",
            device="cuda",
            amp=True,
        )
        result = train(config)
        assert result.best_val_metric is not None
        assert np.isfinite(result.best_val_metric)

        payload = load_checkpoint(result.last_checkpoint_path)
        # GradScaler is enabled on this run -- its state_dict is non-empty
        # (unlike the always-disabled CPU-path scaler tested elsewhere).
        assert payload.scaler_state_dict


# ---------------------------------------------------------------------------
# Precomputed synthetic-sample cache (src.vascutrace.ml.cache)
#
# ``supersample=1`` everywhere below is a deliberate CPU-test-speed choice
# (matching this project's established convention -- e.g.
# tests/test_ml_dataset.py's own fixtures), NOT a claim about P3's
# accuracy-mandated real-training default (5, see dataset.py's
# DatasetConfig.supersample docstring comment) -- these tests exercise
# cache.py's own mechanics (index resolution, parallel dispatch, npz
# round-trip, manifest/leakage checks), not P3's simulation accuracy.
# ---------------------------------------------------------------------------

_CACHE_TEST_CONFIG = DatasetConfig(supersample=1)


def _rewrite_cached_sample(
    path: Path,
    *,
    meta_updates: dict[str, object] | None = None,
    clear_target: bool = False,
    mutate_left_view: bool = False,
) -> None:
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: np.array(archive[name]) for name in archive.files}
    if meta_updates:
        meta = json.loads(str(payload["meta_json"]))
        meta.update(meta_updates)
        payload["meta_json"] = np.array(json.dumps(meta))
    if clear_target:
        payload["target_mask"] = np.zeros_like(payload["target_mask"])
        if "source_fraction" in payload:
            payload["source_fraction"] = np.zeros_like(payload["source_fraction"])
    if mutate_left_view:
        payload["left_view"] = payload["left_view"].copy()
        payload["left_view"].flat[0] += np.float32(1e-3)
    np.savez(path, **payload)


def _precompute_cache_pair(
    root: Path,
    bundle_dirs: tuple[Path, Path],
    *,
    train_seed: int = 31,
    val_seed: int = 32,
    n_positive: int = 1,
    n_negative: int = 1,
) -> tuple[Path, Path]:
    train_bundle, val_bundle = bundle_dirs
    train_cache = root / "train_cache"
    val_cache = root / "val_cache"
    precompute_synthetic_cache(
        [train_bundle],
        train_cache,
        n_positive_per_bundle=n_positive,
        n_negative_per_bundle=n_negative,
        seed=train_seed,
        config=_CACHE_TEST_CONFIG,
        num_workers=1,
    )
    precompute_synthetic_cache(
        [val_bundle],
        val_cache,
        n_positive_per_bundle=n_positive,
        n_negative_per_bundle=n_negative,
        seed=val_seed,
        config=_CACHE_TEST_CONFIG,
        num_workers=1,
    )
    return train_cache, val_cache


def _tiny_cache_train_config(
    *,
    train_cache: Path,
    val_cache: Path,
    run_root: Path,
    **overrides: object,
) -> TrainConfig:
    seed = int(overrides.pop("seed", 77))
    defaults: dict[str, object] = {
        "train_bundle_dirs": (),
        "val_bundle_dirs": (),
        "train_cache_dir": train_cache,
        "val_cache_dir": val_cache,
        "run_root": run_root,
        "seed": seed,
        "val_seed": seed,
        "batch_size": 2,
        "max_epochs": 1,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "early_stop_patience": 100,
        "device": "cpu",
        "amp": False,
        "num_workers": 0,
        "model_config": ModelConfig(base_channels=4, seed=seed),
        "limit_train_batches": 1,
        "limit_val_batches": 1,
        "log_every_n_steps": 1,
        "lr_schedule": "none",
    }
    defaults.update(overrides)
    return TrainConfig(**defaults)


def _scripted_validation_metrics(
    primary: float, *, secondary: float | None = None
) -> ValidationMetrics:
    iou = primary if secondary is None else secondary
    return ValidationMetrics(
        blended_dice=primary,
        mean_positive_dice=primary,
        mean_positive_iou=iou,
        detection_precision=1.0,
        detection_recall=1.0,
        detection_f1=1.0,
        negative_clean_rate=1.0,
        dice_x_clean=primary,
        det_f1_gated_dice=primary,
        n_positive=1,
        n_negative=1,
    )


@pytest.fixture(scope="module")
def shared_cache_dirs(
    tmp_path_factory: pytest.TempPathFactory,
    shared_bundle_dirs: tuple[Path, Path],
) -> tuple[Path, Path]:
    return _precompute_cache_pair(
        tmp_path_factory.mktemp("shared_feature_caches"),
        shared_bundle_dirs,
    )


class TestCacheRoundTrip:
    def test_cached_samples_are_schema_correct_and_match_direct_build(
        self, tmp_path: Path, shared_bundle_dirs: tuple[Path, Path]
    ) -> None:
        train_dir, _val_dir = shared_bundle_dirs
        cache_dir = tmp_path / "cache"

        manifest = precompute_synthetic_cache(
            [train_dir],
            cache_dir,
            n_positive_per_bundle=1,
            n_negative_per_bundle=1,
            seed=99,
            config=_CACHE_TEST_CONFIG,
            num_workers=2,
        )
        assert manifest.total_samples == 2
        assert manifest.total_positive == 1
        assert manifest.total_negative == 1
        assert manifest.research_prototype_warning == RESEARCH_PROTOTYPE_WARNING

        cached = CachedSampleDataset(cache_dir)
        assert len(cached) == 2
        for i in range(len(cached)):
            sample = cached[i]
            assert tuple(sample.left_view.shape) == (10, 144, 80)
            assert tuple(sample.right_view.shape) == (10, 144, 80)
            assert tuple(sample.pet_diff.shape) == (5, 144, 80)
            assert tuple(sample.target_mask.shape) == (1, 144, 80)
            assert tuple(sample.source_fraction.shape) == (1, 144, 80)
            assert tuple(sample.valid_mask.shape) == (1, 144, 80)
            assert tuple(sample.raw_pet.shape) == (1, 144, 80)
            for tensor in (
                sample.left_view,
                sample.right_view,
                sample.pet_diff,
                sample.target_mask,
                sample.source_fraction,
                sample.valid_mask,
                sample.raw_pet,
            ):
                assert tensor.dtype == torch.float32

        # Index 0 is the sole positive sample (precompute_synthetic_cache
        # appends the positive stream first); index 1 is the sole negative
        # sample -- see cache.py's module docstring, item 2, for exactly
        # which "virtual" SiameseCropDataset each stream matches.
        pos_dataset = SiameseCropDataset(
            [train_dir],
            seed=99,
            positive_fraction=1.0,
            config=replace(_CACHE_TEST_CONFIG, samples_per_bundle=1),
        )
        direct_positive = pos_dataset[0]
        cached_positive = cached[0]
        assert torch.equal(direct_positive.left_view, cached_positive.left_view)
        assert torch.equal(direct_positive.right_view, cached_positive.right_view)
        assert torch.equal(direct_positive.target_mask, cached_positive.target_mask)
        assert torch.equal(
            direct_positive.source_fraction, cached_positive.source_fraction
        )
        assert cached_positive.source_fraction.dtype == torch.float32
        assert bool((cached_positive.source_fraction >= 0.0).all())
        assert bool((cached_positive.source_fraction <= 1.0).all())
        assert torch.equal(
            cached_positive.source_fraction >= 0.5,
            cached_positive.target_mask >= 0.5,
        )
        assert cached_positive.meta["positive"] is True

        neg_dataset = SiameseCropDataset(
            [train_dir],
            seed=99,
            positive_fraction=0.0,
            config=replace(_CACHE_TEST_CONFIG, samples_per_bundle=1),
        )
        direct_negative = neg_dataset[0]
        cached_negative = cached[1]
        assert torch.equal(direct_negative.left_view, cached_negative.left_view)
        assert torch.count_nonzero(cached_negative.source_fraction).item() == 0
        assert cached_negative.meta["positive"] is False

    def test_bad_precompute_args_raise_typed_error(
        self, tmp_path: Path, shared_bundle_dirs: tuple[Path, Path]
    ) -> None:
        train_dir, _ = shared_bundle_dirs
        with pytest.raises(CachePrepError):
            precompute_synthetic_cache(
                [train_dir],
                tmp_path / "cache",
                n_positive_per_bundle=0,
                n_negative_per_bundle=0,
                seed=1,
                config=_CACHE_TEST_CONFIG,
                num_workers=1,
            )
        with pytest.raises(CachePrepError):
            precompute_synthetic_cache(
                [],
                tmp_path / "cache2",
                n_positive_per_bundle=1,
                n_negative_per_bundle=0,
                seed=1,
                config=_CACHE_TEST_CONFIG,
                num_workers=1,
            )

    def test_soft_target_manifest_fails_closed_on_corrupt_samples(
        self, tmp_path: Path, shared_bundle_dirs: tuple[Path, Path]
    ) -> None:
        train_dir, _ = shared_bundle_dirs
        cache_dir = tmp_path / "soft_cache"
        precompute_synthetic_cache(
            [train_dir],
            cache_dir,
            n_positive_per_bundle=1,
            n_negative_per_bundle=0,
            seed=13,
            config=_CACHE_TEST_CONFIG,
            num_workers=1,
        )
        sample_path = cache_dir / "sample_0000.npz"
        with np.load(sample_path, allow_pickle=False) as npz:
            original = {name: np.array(npz[name], copy=True) for name in npz.files}

        corruptions: dict[str, dict[str, np.ndarray] | None] = {
            "missing": None,
            "wrong shape": {"source_fraction": np.zeros((1, 1, 1), dtype=np.float32)},
            "wrong dtype": {
                "source_fraction": original["source_fraction"].astype(np.float64)
            },
            "non-finite": {
                "source_fraction": np.full_like(original["source_fraction"], np.nan)
            },
            "out of range": {
                "source_fraction": np.full_like(original["source_fraction"], 1.5)
            },
            "hard-threshold invariant": {
                "source_fraction": np.zeros_like(original["source_fraction"])
            },
        }

        for label, replacement in corruptions.items():
            payload = {
                name: np.array(value, copy=True) for name, value in original.items()
            }
            if replacement is None:
                payload.pop("source_fraction")
            else:
                payload.update(replacement)
            np.savez(sample_path, **payload)
            with pytest.raises(CacheSchemaError) as exc_info:
                CachedSampleDataset(cache_dir)[0]
            assert str(exc_info.value), label

        missing_tensor_payload = {
            name: np.array(value, copy=True) for name, value in original.items()
        }
        missing_tensor_payload.pop("left_view")
        np.savez(sample_path, **missing_tensor_payload)
        with pytest.raises(CacheSchemaError, match="left_view"):
            CachedSampleDataset(cache_dir)

        # A legacy manifest that does not claim soft-target support may
        # still load an old sample with no source_fraction field; the
        # compatibility value is explicitly all-zero.
        legacy_payload = {
            name: np.array(value, copy=True)
            for name, value in original.items()
            if name != "source_fraction"
        }
        np.savez(sample_path, **legacy_payload)
        manifest_path = cache_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["has_source_fraction"] = False
        manifest_path.write_text(json.dumps(manifest))
        legacy_sample = CachedSampleDataset(cache_dir)[0]
        assert torch.count_nonzero(legacy_sample.source_fraction).item() == 0


class TestCacheManifestIntegrity:
    def test_inventory_and_manifest_corruption_fail_at_dataset_construction(
        self, tmp_path: Path, shared_bundle_dirs: tuple[Path, Path]
    ) -> None:
        train_dir, _ = shared_bundle_dirs
        pristine = tmp_path / "pristine"
        precompute_synthetic_cache(
            [train_dir],
            pristine,
            n_positive_per_bundle=1,
            n_negative_per_bundle=1,
            seed=73,
            config=_CACHE_TEST_CONFIG,
            num_workers=1,
        )

        cases = (
            "missing_sample",
            "extra_sample",
            "non_contiguous",
            "inconsistent_totals",
            "invalid_total_positive",
            "missing_manifest_key",
        )
        for case in cases:
            candidate = tmp_path / case
            shutil.copytree(pristine, candidate)
            manifest_path = candidate / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            if case == "missing_sample":
                (candidate / "sample_0001.npz").unlink()
            elif case == "extra_sample":
                shutil.copy2(
                    candidate / "sample_0001.npz",
                    candidate / "sample_0002.npz",
                )
            elif case == "non_contiguous":
                (candidate / "sample_0001.npz").rename(candidate / "sample_0003.npz")
            elif case == "inconsistent_totals":
                manifest["total_samples"] = 3
                manifest_path.write_text(json.dumps(manifest))
            elif case == "invalid_total_positive":
                manifest["total_positive"] = -1
                manifest_path.write_text(json.dumps(manifest))
            else:
                manifest.pop("config_hash")
                manifest_path.write_text(json.dumps(manifest))

            with pytest.raises(CacheSchemaError) as exc_info:
                CachedSampleDataset(candidate)
            assert str(exc_info.value), case

    @pytest.mark.parametrize(
        ("case", "meta_updates", "clear_target"),
        (
            ("foreign_bundle", {"subject": "Other", "session": "Session"}, False),
            ("wrong_class", {"positive": False}, False),
            ("wrong_content", None, True),
        ),
    )
    def test_sample_metadata_and_content_must_bind_to_manifest(
        self,
        tmp_path: Path,
        shared_bundle_dirs: tuple[Path, Path],
        case: str,
        meta_updates: dict[str, object] | None,
        clear_target: bool,
    ) -> None:
        train_dir, _ = shared_bundle_dirs
        cache_dir = tmp_path / case
        precompute_synthetic_cache(
            [train_dir],
            cache_dir,
            n_positive_per_bundle=1,
            n_negative_per_bundle=1,
            seed=74,
            config=_CACHE_TEST_CONFIG,
            num_workers=1,
        )
        _rewrite_cached_sample(
            cache_dir / "sample_0000.npz",
            meta_updates=meta_updates,
            clear_target=clear_target,
        )

        with pytest.raises(CacheSchemaError):
            CachedSampleDataset(cache_dir)
        if clear_target:
            audit_dataset = CachedSampleDataset(
                cache_dir, enforce_manifest_class_content=False
            )
            assert audit_dataset[0].meta["positive"] is True
            assert not bool((audit_dataset[0].target_mask >= 0.5).any())
        else:
            with pytest.raises(CacheSchemaError):
                CachedSampleDataset(cache_dir, enforce_manifest_class_content=False)

    def test_per_bundle_sample_counts_must_match_manifest(
        self, tmp_path: Path, shared_bundle_dirs: tuple[Path, Path]
    ) -> None:
        cache_dir = tmp_path / "count_mismatch"
        precompute_synthetic_cache(
            list(shared_bundle_dirs),
            cache_dir,
            n_positive_per_bundle=1,
            n_negative_per_bundle=1,
            seed=75,
            config=_CACHE_TEST_CONFIG,
            num_workers=1,
        )
        manifest = json.loads((cache_dir / "manifest.json").read_text())
        subject, session = manifest["bundle_identities"][1].split("/", maxsplit=1)
        _rewrite_cached_sample(
            cache_dir / "sample_0000.npz",
            meta_updates={"subject": subject, "session": session},
        )

        with pytest.raises(CacheSchemaError, match="counts"):
            CachedSampleDataset(cache_dir)
        with pytest.raises(CacheSchemaError, match="counts"):
            CachedSampleDataset(cache_dir, enforce_manifest_class_content=False)


class TestCacheIdentityBinding:
    def test_cache_generation_identity_changes_split_hash_and_blocks_resume(
        self, tmp_path: Path, shared_bundle_dirs: tuple[Path, Path]
    ) -> None:
        train_bundle, _ = shared_bundle_dirs
        train_cache, val_cache = _precompute_cache_pair(
            tmp_path / "base",
            shared_bundle_dirs,
            train_seed=101,
            val_seed=102,
        )
        substitute_cache = tmp_path / "substitute_train"
        precompute_synthetic_cache(
            [train_bundle],
            substitute_cache,
            n_positive_per_bundle=1,
            n_negative_per_bundle=1,
            seed=999,
            config=_CACHE_TEST_CONFIG,
            num_workers=1,
        )

        run_root = tmp_path / "run"
        original = _tiny_cache_train_config(
            train_cache=train_cache,
            val_cache=val_cache,
            run_root=run_root,
            max_epochs=1,
        )
        substituted = replace(
            original,
            train_cache_dir=substitute_cache,
            max_epochs=2,
        )

        original_hash = compute_split_hash(original)
        assert original_hash != compute_split_hash(substituted)
        train(original)
        with pytest.raises(CheckpointCompatibilityError, match="split_hash"):
            resume(run_root, substituted)

        _rewrite_cached_sample(train_cache / "sample_0000.npz", mutate_left_view=True)
        assert compute_split_hash(original) != original_hash
        with pytest.raises(CheckpointCompatibilityError, match="split_hash"):
            resume(run_root, replace(original, max_epochs=2))


# ---------------------------------------------------------------------------
# Determinism: same args -> identical cache
# ---------------------------------------------------------------------------


class TestCacheDeterminism:
    def test_same_args_produce_identical_cached_tensors(
        self, tmp_path: Path, shared_bundle_dirs: tuple[Path, Path]
    ) -> None:
        train_dir, _ = shared_bundle_dirs
        cache_a = tmp_path / "cache_a"
        cache_b = tmp_path / "cache_b"

        for cache_dir in (cache_a, cache_b):
            precompute_synthetic_cache(
                [train_dir],
                cache_dir,
                n_positive_per_bundle=1,
                n_negative_per_bundle=1,
                seed=5,
                config=_CACHE_TEST_CONFIG,
                num_workers=2,
            )

        ds_a = CachedSampleDataset(cache_a)
        ds_b = CachedSampleDataset(cache_b)
        assert len(ds_a) == len(ds_b)
        for i in range(len(ds_a)):
            sample_a, sample_b = ds_a[i], ds_b[i]
            assert torch.equal(sample_a.left_view, sample_b.left_view)
            assert torch.equal(sample_a.right_view, sample_b.right_view)
            assert torch.equal(sample_a.target_mask, sample_b.target_mask)
            assert sample_a.meta == sample_b.meta


# ---------------------------------------------------------------------------
# train() on a tiny cache
# ---------------------------------------------------------------------------


class TestTrainWithCache:
    def test_train_runs_a_few_steps_on_cached_datasets_and_checkpoints(
        self, tmp_path: Path, shared_bundle_dirs: tuple[Path, Path]
    ) -> None:
        train_dir, val_dir = shared_bundle_dirs
        train_cache = tmp_path / "train_cache"
        val_cache = tmp_path / "val_cache"

        precompute_synthetic_cache(
            [train_dir],
            train_cache,
            n_positive_per_bundle=1,
            n_negative_per_bundle=1,
            seed=11,
            config=_CACHE_TEST_CONFIG,
            num_workers=1,
        )
        precompute_synthetic_cache(
            [val_dir],
            val_cache,
            n_positive_per_bundle=1,
            n_negative_per_bundle=1,
            seed=12,
            config=_CACHE_TEST_CONFIG,
            num_workers=1,
        )

        config = TrainConfig(
            train_bundle_dirs=(),
            val_bundle_dirs=(),
            train_cache_dir=train_cache,
            val_cache_dir=val_cache,
            run_root=tmp_path / "run",
            seed=1,
            batch_size=1,
            max_epochs=1,
            lr=1e-3,
            weight_decay=0.0,
            early_stop_patience=5,
            device="cpu",
            amp=False,
            num_workers=0,
            model_config=ModelConfig(base_channels=4, seed=1),
            limit_train_batches=2,
            limit_val_batches=2,
            log_every_n_steps=1,
        )
        result = train(config)

        assert result.global_step == 2
        assert result.last_checkpoint_path.is_file()
        assert result.best_val_metric is not None
        assert np.isfinite(result.best_val_metric)

        records = _metrics_records(result.metrics_log_path)
        train_records = [r for r in records if r["event"] == "train_step"]
        assert len(train_records) == 2
        for record in train_records:
            assert np.isfinite(record["train_loss"])


# ---------------------------------------------------------------------------
# Leakage guard: overlapping train/val cache -> typed error
# ---------------------------------------------------------------------------


class TestCacheLeakageGuard:
    def test_overlapping_train_and_val_cache_raises(
        self, tmp_path: Path, shared_bundle_dirs: tuple[Path, Path]
    ) -> None:
        train_dir, _ = shared_bundle_dirs
        cache_a = tmp_path / "cache_a"
        cache_b = tmp_path / "cache_b"

        # Both caches built from the SAME bundle -> overlapping
        # bundle_identities in their respective manifests.
        precompute_synthetic_cache(
            [train_dir],
            cache_a,
            n_positive_per_bundle=1,
            n_negative_per_bundle=0,
            seed=1,
            config=_CACHE_TEST_CONFIG,
            num_workers=1,
        )
        precompute_synthetic_cache(
            [train_dir],
            cache_b,
            n_positive_per_bundle=0,
            n_negative_per_bundle=1,
            seed=2,
            config=_CACHE_TEST_CONFIG,
            num_workers=1,
        )

        with pytest.raises(TrainConfigError, match="leakage"):
            TrainConfig(
                train_bundle_dirs=(),
                val_bundle_dirs=(),
                train_cache_dir=cache_a,
                val_cache_dir=cache_b,
                run_root=tmp_path / "run",
                model_config=ModelConfig(base_channels=4),
            )

    def test_only_one_cache_dir_set_raises(
        self, tmp_path: Path, shared_bundle_dirs: tuple[Path, Path]
    ) -> None:
        train_dir, _ = shared_bundle_dirs
        cache_a = tmp_path / "cache_only_one"
        precompute_synthetic_cache(
            [train_dir],
            cache_a,
            n_positive_per_bundle=0,
            n_negative_per_bundle=1,
            seed=1,
            config=_CACHE_TEST_CONFIG,
            num_workers=1,
        )
        with pytest.raises(TrainConfigError):
            TrainConfig(
                train_bundle_dirs=(),
                val_bundle_dirs=shared_bundle_dirs[1:2],
                train_cache_dir=cache_a,
                val_cache_dir=None,
                run_root=tmp_path / "run",
                model_config=ModelConfig(base_channels=4),
            )


# ---------------------------------------------------------------------------
# Manifest schema/version mismatch -> typed error
# ---------------------------------------------------------------------------


class TestCacheManifestSchemaMismatch:
    def test_cache_schema_version_mismatch_raises_typed_error(
        self, tmp_path: Path, shared_bundle_dirs: tuple[Path, Path]
    ) -> None:
        train_dir, _ = shared_bundle_dirs
        cache_dir = tmp_path / "cache_bad_version"
        precompute_synthetic_cache(
            [train_dir],
            cache_dir,
            n_positive_per_bundle=0,
            n_negative_per_bundle=1,
            seed=3,
            config=_CACHE_TEST_CONFIG,
            num_workers=1,
        )

        manifest_path = cache_dir / "manifest.json"
        payload = json.loads(manifest_path.read_text())
        payload["cache_schema_version"] = "not-a-real-cache-schema-version"
        manifest_path.write_text(json.dumps(payload))

        with pytest.raises(CacheSchemaError):
            CachedSampleDataset(cache_dir)

    def test_missing_manifest_raises_typed_error(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "no_manifest_here"
        empty_dir.mkdir()
        with pytest.raises(CacheSchemaError):
            CachedSampleDataset(empty_dir)


# ---------------------------------------------------------------------------
# Focal Tversky / Combo loss (src.vascutrace.ml.losses) -- finite,
# differentiable, and the alpha=FN-weight=0.7 > beta=FP-weight=0.3 recall
# -emphasis property this implementation is built on.
# ---------------------------------------------------------------------------


class TestFocalTverskyAndComboLoss:
    def test_focal_tversky_loss_is_finite_and_gradient_is_finite(self) -> None:
        torch.manual_seed(0)
        logits = torch.randn(2, 1, 8, 8, requires_grad=True)
        target = (torch.rand(2, 1, 8, 8) > 0.8).float()
        valid = torch.ones(2, 1, 8, 8)

        loss = focal_tversky_loss(logits, target, valid)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(logits.grad).all()

    def test_combo_loss_is_finite_and_gradient_is_finite(self) -> None:
        torch.manual_seed(0)
        logits = torch.randn(2, 1, 8, 8, requires_grad=True)
        target = (torch.rand(2, 1, 8, 8) > 0.8).float()
        valid = torch.ones(2, 1, 8, 8)

        loss = combo_loss(logits, target, valid)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(logits.grad).all()

    def test_empty_vs_empty_is_finite_not_nan(self) -> None:
        logits = torch.full((1, 1, 4, 4), -10.0)
        target = torch.zeros(1, 1, 4, 4)
        assert torch.isfinite(focal_tversky_loss(logits, target))
        assert torch.isfinite(combo_loss(logits, target))

    def test_valid_mask_excludes_invalid_voxels(self) -> None:
        target = torch.zeros(1, 1, 4, 4)
        target[0, 0, 0, 0] = 1.0
        valid = torch.ones(1, 1, 4, 4)
        valid[0, 0, 3, 3] = 0.0  # this one voxel is invalid

        logits_a = torch.zeros(1, 1, 4, 4)
        logits_b = logits_a.clone()
        # A wildly "confident positive" prediction, but ONLY at the
        # invalid voxel -- must not move the loss at all.
        logits_b[0, 0, 3, 3] = 20.0

        loss_a = focal_tversky_loss(logits_a, target, valid)
        loss_b = focal_tversky_loss(logits_b, target, valid)
        assert torch.allclose(loss_a, loss_b)

    def test_shape_mismatch_raises(self) -> None:
        logits = torch.zeros(1, 1, 4, 4)
        target = torch.zeros(1, 1, 4, 5)
        with pytest.raises(ValueError, match="shape"):
            focal_tversky_loss(logits, target)
        with pytest.raises(ValueError, match="shape"):
            combo_loss(logits, target)

    def test_increasing_false_negatives_raises_loss_more_than_false_positives(
        self,
    ) -> None:
        """The key recall-emphasis property this implementation is built on:
        alpha (FN weight) = 0.7 > beta (FP weight) = 0.3 (Abraham & Khan
        2019's own reference implementation -- see losses.py's module
        docstring, item 1), so for an EQUAL number of erroneous voxels,
        missing part of the true lesion (FN) must raise the loss MORE
        than an equivalently-sized spurious activation (FP) of the same
        size, starting from the same perfect baseline.
        """
        h, w = 10, 10
        target = torch.zeros(1, 1, h, w)
        target[0, 0, 2:6, 2:6] = 1.0  # a 4x4 = 16-voxel lesion

        baseline_logits = torch.where(target > 0, 10.0, -10.0)
        baseline_loss = focal_tversky_loss(baseline_logits, target)

        # FN case: flip 4 lesion voxels to confidently-negative (miss them).
        fn_logits = baseline_logits.clone()
        fn_logits[0, 0, 2, 2:6] = -10.0
        fn_loss = focal_tversky_loss(fn_logits, target)

        # FP case: flip 4 non-lesion voxels to confidently-positive
        # (spurious activation), the SAME voxel COUNT as the FN case.
        fp_logits = baseline_logits.clone()
        fp_logits[0, 0, 7, 2:6] = 10.0
        fp_loss = focal_tversky_loss(fp_logits, target)

        assert fn_loss > fp_loss > baseline_loss

    def test_increasing_false_negatives_raises_combo_loss_more_too(self) -> None:
        """Same property, through combo_loss (FTL + BCE) -- BCE is
        FN/FP-symmetric, so the asymmetry is diluted but must still hold
        (tversky_weight=1.0, bce_weight=0.5 by default -- the Tversky
        term still dominates).
        """
        h, w = 10, 10
        target = torch.zeros(1, 1, h, w)
        target[0, 0, 2:6, 2:6] = 1.0

        baseline_logits = torch.where(target > 0, 10.0, -10.0)
        fn_logits = baseline_logits.clone()
        fn_logits[0, 0, 2, 2:6] = -10.0
        fp_logits = baseline_logits.clone()
        fp_logits[0, 0, 7, 2:6] = 10.0

        assert combo_loss(fn_logits, target) > combo_loss(fp_logits, target)


# ---------------------------------------------------------------------------
# Positive-focused selection metric -- prefers a genuine lesion-detecting
# checkpoint over an empty-collapsed one (the "empty-reference" pitfall
# this implementation exists to fix).
# ---------------------------------------------------------------------------


class TestSelectionMetricPrefersLesionDetection:
    def test_new_metrics_prefer_the_lesion_finding_checkpoint_blended_dice_would_not(
        self,
    ) -> None:
        # A "collapsed" checkpoint: predicts empty everywhere. On a
        # mostly-healthy val set this scores a near-perfect blended Dice
        # (empty-vs-empty trivial matches dominate) but finds ZERO
        # lesions on the positive cases it should have found.
        collapsed = ValidationMetrics(
            blended_dice=0.90,
            mean_positive_dice=0.0,
            mean_positive_iou=0.0,
            detection_precision=0.0,
            detection_recall=0.0,
            detection_f1=0.0,
            negative_clean_rate=1.0,
            dice_x_clean=0.0,
            det_f1_gated_dice=0.0,
            n_positive=4,
            n_negative=16,
        )
        # A genuinely better checkpoint: finds most lesions reasonably
        # well and stays mostly (not perfectly) clean on healthy cases --
        # but its BLENDED Dice looks WORSE than the collapsed one's, a
        # realistic instance of the empty-reference pitfall this implementation's
        # background section itself measured (train.py's module
        # docstring, item 11).
        good = ValidationMetrics(
            blended_dice=0.74,
            mean_positive_dice=0.55,
            mean_positive_iou=0.40,
            detection_precision=0.80,
            detection_recall=0.75,
            detection_f1=0.77,
            negative_clean_rate=0.85,
            dice_x_clean=0.55 * 0.85,
            det_f1_gated_dice=0.60,
            n_positive=4,
            n_negative=16,
        )

        # Under the OLD selection metric (blended Dice), the collapsed
        # checkpoint would have WON -- exactly the failure this implementation
        # exists to fix.
        assert _select_metric_value(collapsed, "blended_dice") > _select_metric_value(
            good, "blended_dice"
        )

        # Under BOTH new positive-focused composites, the genuinely
        # better, lesion-finding checkpoint wins instead.
        assert _select_metric_value(good, "dice_x_clean") > _select_metric_value(
            collapsed, "dice_x_clean"
        )
        assert _select_metric_value(good, "det_f1_gated_dice") > _select_metric_value(
            collapsed, "det_f1_gated_dice"
        )

    def test_default_selection_metric_is_dice_x_clean(
        self, tmp_path: Path, shared_bundle_dirs: tuple[Path, Path]
    ) -> None:
        train_dir, val_dir = shared_bundle_dirs
        config = _tiny_train_config(
            train_dir=train_dir, val_dir=val_dir, run_root=tmp_path / "run"
        )
        assert config.selection_metric == "dice_x_clean"


# ---------------------------------------------------------------------------
# Cosine LR schedule with linear warmup -- shape.
# ---------------------------------------------------------------------------


class TestLrSchedule:
    def test_cosine_schedule_warmup_then_decay_shape(self) -> None:
        base_lr = 1e-3
        total_steps = 20
        warmup = 5
        lrs = [
            _lr_at_step(
                s,
                base_lr=base_lr,
                total_steps=total_steps,
                warmup_steps=warmup,
                schedule="cosine",
            )
            for s in range(total_steps)
        ]
        # Linear warmup: strictly increasing over [0, warmup).
        assert all(lrs[i] < lrs[i + 1] for i in range(warmup - 1))
        assert lrs[warmup - 1] == pytest.approx(base_lr, rel=1e-9)
        # Cosine decay: non-increasing over [warmup, total_steps).
        assert all(lrs[i] >= lrs[i + 1] for i in range(warmup, total_steps - 1))
        # The LAST index in range(total_steps) is total_steps - 1, one
        # step short of the schedule's own total_steps horizon, so its
        # cosine progress is close to but not exactly 1.0 -- it must be
        # small (near the tail of the decay), not exactly zero.
        assert 0.0 <= lrs[-1] < 1e-3

    def test_none_schedule_is_constant(self) -> None:
        base_lr = 1e-3
        vals = [
            _lr_at_step(
                s, base_lr=base_lr, total_steps=20, warmup_steps=5, schedule="none"
            )
            for s in (0, 5, 19)
        ]
        assert all(v == base_lr for v in vals)

    def test_zero_warmup_starts_at_full_lr(self) -> None:
        base_lr = 1e-3
        v0 = _lr_at_step(
            0, base_lr=base_lr, total_steps=10, warmup_steps=0, schedule="cosine"
        )
        assert v0 == pytest.approx(base_lr, rel=1e-9)

    def test_unknown_schedule_raises(self) -> None:
        with pytest.raises(ValueError):
            _lr_at_step(
                0, base_lr=1e-3, total_steps=10, warmup_steps=0, schedule="linear"
            )


# ---------------------------------------------------------------------------
# Exploratory feature bundle: generated/offline integration contracts.
# ---------------------------------------------------------------------------


class TestFeatureDefaults:
    def test_implicit_and_explicit_off_paths_match_exactly(
        self,
        tmp_path: Path,
        shared_cache_dirs: tuple[Path, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        train_cache, val_cache = shared_cache_dirs
        implicit = _tiny_cache_train_config(
            train_cache=train_cache,
            val_cache=val_cache,
            run_root=tmp_path / "implicit",
            seed=404,
        )
        explicit = replace(
            implicit,
            run_root=tmp_path / "explicit",
            soft_target=False,
            augment=False,
            ema_decay=None,
            secondary_selection_metric=None,
            hard_negative_mining=False,
        )

        original_loss = train_module._LOSS_FUNCTIONS["combo"]  # noqa: SLF001

        def hard_target_spy(
            logits: torch.Tensor,
            target: torch.Tensor,
            valid_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            assert bool(torch.logical_or(target == 0.0, target == 1.0).all())
            return original_loss(logits, target, valid_mask=valid_mask)

        def forbidden_augmentation(*args: object, **kwargs: object) -> None:
            raise AssertionError("augmentation was called with augment=False")

        monkeypatch.setitem(train_module._LOSS_FUNCTIONS, "combo", hard_target_spy)
        monkeypatch.setattr(train_module, "_augment_batch", forbidden_augmentation)

        implicit_result = train(implicit)
        explicit_result = train(explicit)
        implicit_payload = load_checkpoint(implicit_result.last_checkpoint_path)
        explicit_payload = load_checkpoint(explicit_result.last_checkpoint_path)

        assert _metrics_records(implicit_result.metrics_log_path) == _metrics_records(
            explicit_result.metrics_log_path
        )
        for name, tensor in implicit_payload.model_state_dict.items():
            assert torch.equal(tensor, explicit_payload.model_state_dict[name])
        for run_root in (implicit.run_root, explicit.run_root):
            assert not (run_root / "last_ema.pt").exists()
            assert not any(run_root.glob("best_ema*.pt"))
            assert not any(
                record["event"] == "hard_negative_mining"
                for record in _metrics_records(run_root / "metrics.jsonl")
            )


class TestResumeEarlyStoppingState:
    def test_split_resume_stops_at_same_epoch_as_continuous_run(
        self,
        tmp_path: Path,
        shared_cache_dirs: tuple[Path, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        train_cache, val_cache = shared_cache_dirs

        def install(values: list[float]) -> None:
            queue = list(values)

            def scripted(*args: object, **kwargs: object) -> ValidationMetrics:
                return _scripted_validation_metrics(queue.pop(0))

            monkeypatch.setattr(train_module, "_run_validation", scripted)

        continuous_config = _tiny_cache_train_config(
            train_cache=train_cache,
            val_cache=val_cache,
            run_root=tmp_path / "continuous",
            max_epochs=5,
            early_stop_patience=2,
            seed=505,
        )
        install([0.5, 0.4, 0.3])
        continuous = train(continuous_config)

        split_root = tmp_path / "split"
        first_config = replace(
            continuous_config,
            run_root=split_root,
            max_epochs=2,
        )
        install([0.5, 0.4])
        first = train(first_config)
        assert first.stopped_early is False

        install([0.3])
        resumed = resume(
            split_root,
            replace(first_config, max_epochs=5),
        )

        assert continuous.stopped_early is resumed.stopped_early is True
        assert continuous.final_epoch == resumed.final_epoch == 2
        assert continuous.best_epoch == resumed.best_epoch == 0
        assert continuous.best_val_metric == resumed.best_val_metric == 0.5


class TestSecondarySelectorCheckpoint:
    def test_raw_and_ema_secondary_payloads_name_the_selecting_metric(
        self,
        tmp_path: Path,
        shared_cache_dirs: tuple[Path, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        train_cache, val_cache = shared_cache_dirs
        queue = [
            _scripted_validation_metrics(0.9, secondary=0.1),  # raw epoch 0
            _scripted_validation_metrics(0.2, secondary=0.8),  # EMA epoch 0
            _scripted_validation_metrics(0.8, secondary=0.7),  # raw epoch 1
            _scripted_validation_metrics(0.3, secondary=0.6),  # EMA epoch 1
        ]

        def scripted(*args: object, **kwargs: object) -> ValidationMetrics:
            return queue.pop(0)

        monkeypatch.setattr(train_module, "_run_validation", scripted)
        run_root = tmp_path / "selectors"
        result = train(
            _tiny_cache_train_config(
                train_cache=train_cache,
                val_cache=val_cache,
                run_root=run_root,
                max_epochs=2,
                ema_decay=0.9,
                secondary_selection_metric="mean_positive_iou",
            )
        )
        assert not queue

        expected = {
            "best.pt": (0, "dice_x_clean", 0.9),
            "best_mean_positive_iou.pt": (1, "mean_positive_iou", 0.7),
            "best_ema.pt": (1, "dice_x_clean", 0.3),
            "best_ema_mean_positive_iou.pt": (0, "mean_positive_iou", 0.8),
        }
        for filename, (epoch, metric_name, metric_value) in expected.items():
            payload = load_checkpoint(run_root / filename)
            assert payload.epoch == epoch
            assert payload.best_val_metric_name == metric_name
            assert payload.best_val_metric == pytest.approx(metric_value)
        assert result.best_secondary_epoch == 1
        assert result.best_ema_secondary_epoch == 0


def _scripted_constrained_metrics(
    *, iou: float, precision: float, f1: float, clean: float = 1.0
) -> ValidationMetrics:
    """A :class:`ValidationMetrics` with independently controllable
    ``mean_positive_iou``/``detection_precision``/``detection_f1``/
    ``negative_clean_rate`` -- unlike :func:`_scripted_validation_metrics`
    (which pins precision/F1/clean at 1.0), this lets a test place an epoch
    on either side of item 20's legality floors.
    """
    return ValidationMetrics(
        blended_dice=iou,
        mean_positive_dice=iou,
        mean_positive_iou=iou,
        detection_precision=precision,
        detection_recall=1.0,
        detection_f1=f1,
        negative_clean_rate=clean,
        dice_x_clean=iou * clean,
        det_f1_gated_dice=iou,
        n_positive=1,
        n_negative=1,
    )


class TestConstrainedIouSelector:
    """Track A / P4 constrained-floor IoU checkpoint selector -- module
    docstring, item 20.
    """

    def test_disabled_by_default_writes_no_extra_checkpoint_or_events(
        self,
        tmp_path: Path,
        shared_cache_dirs: tuple[Path, Path],
    ) -> None:
        train_cache, val_cache = shared_cache_dirs
        run_root = tmp_path / "off_by_default"
        result = train(
            _tiny_cache_train_config(
                train_cache=train_cache,
                val_cache=val_cache,
                run_root=run_root,
                max_epochs=1,
            )
        )
        assert not (run_root / "best_constrained_iou.pt").exists()
        assert not (run_root / "best_ema_constrained_iou.pt").exists()
        assert result.best_constrained_iou_epoch is None
        assert result.best_constrained_iou_val_metric is None
        assert result.best_constrained_iou_checkpoint_path is None
        assert result.best_ema_constrained_iou_epoch is None
        records = _metrics_records(result.metrics_log_path)
        assert not any(
            r["event"]
            in {
                "constrained_iou_selection",
                "constrained_iou_selection_ema",
                "constrained_iou_selection_summary",
                "constrained_iou_selection_ema_summary",
            }
            for r in records
        )

    def test_selects_max_iou_epoch_among_legal_only_not_global_max(
        self,
        tmp_path: Path,
        shared_cache_dirs: tuple[Path, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Epoch 1 has the GLOBAL max mean_positive_iou (0.90) but fails the
        # default precision floor (0.901) -- an illegal peak, exactly
        # v6exp's raw 0.607 epoch-95 failure mode. Epoch 2 is the highest
        # -IoU LEGAL epoch (0.75) and must be the one selected -- NOT
        # epoch 0 (legal but lower IoU) and NOT epoch 1 (illegal, higher
        # IoU).
        queue = [
            _scripted_constrained_metrics(iou=0.60, precision=0.95, f1=0.90),
            _scripted_constrained_metrics(iou=0.90, precision=0.85, f1=0.95),
            _scripted_constrained_metrics(iou=0.75, precision=0.92, f1=0.87),
        ]

        def scripted(*args: object, **kwargs: object) -> ValidationMetrics:
            return queue.pop(0)

        monkeypatch.setattr(train_module, "_run_validation", scripted)
        run_root = tmp_path / "constrained_max_legal"
        result = train(
            _tiny_cache_train_config(
                train_cache=shared_cache_dirs[0],
                val_cache=shared_cache_dirs[1],
                run_root=run_root,
                max_epochs=3,
                constrained_iou_selection=True,
            )
        )
        assert not queue

        assert result.best_constrained_iou_epoch == 2
        assert result.best_constrained_iou_val_metric == pytest.approx(0.75)
        assert result.best_constrained_iou_checkpoint_path == (
            run_root / "best_constrained_iou.pt"
        )
        payload = load_checkpoint(run_root / "best_constrained_iou.pt")
        assert payload.epoch == 2
        assert payload.best_val_metric_name == "mean_positive_iou"
        assert payload.best_val_metric == pytest.approx(0.75)

        records = _metrics_records(result.metrics_log_path)
        selection_records = {
            r["epoch"]: r for r in records if r["event"] == "constrained_iou_selection"
        }
        assert selection_records[0]["legal"] is True
        assert selection_records[0]["improved"] is True
        assert selection_records[1]["legal"] is False
        assert selection_records[1]["improved"] is False
        assert selection_records[2]["legal"] is True
        assert selection_records[2]["improved"] is True
        assert not any(
            r["event"] == "constrained_iou_selection_summary" for r in records
        )

    def test_min_clean_floor_gates_legality_when_configured(
        self,
        tmp_path: Path,
        shared_cache_dirs: tuple[Path, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # constrained_iou_min_clean defaults to 0.0 (no clean-rate
        # constraint); with it raised to 0.70, an otherwise-legal, higher
        # -IoU epoch with a low clean rate must be rejected in favor of a
        # lower-IoU epoch that also clears the clean floor.
        queue = [
            _scripted_constrained_metrics(
                iou=0.80, precision=0.95, f1=0.90, clean=0.50
            ),
            _scripted_constrained_metrics(
                iou=0.65, precision=0.93, f1=0.88, clean=0.75
            ),
        ]

        def scripted(*args: object, **kwargs: object) -> ValidationMetrics:
            return queue.pop(0)

        monkeypatch.setattr(train_module, "_run_validation", scripted)
        run_root = tmp_path / "constrained_clean_floor"
        result = train(
            _tiny_cache_train_config(
                train_cache=shared_cache_dirs[0],
                val_cache=shared_cache_dirs[1],
                run_root=run_root,
                max_epochs=2,
                constrained_iou_selection=True,
                constrained_iou_min_clean=0.70,
            )
        )
        assert not queue
        assert result.best_constrained_iou_epoch == 1
        assert result.best_constrained_iou_val_metric == pytest.approx(0.65)

    def test_no_epoch_qualifies_writes_no_checkpoint_and_logs_summary(
        self,
        tmp_path: Path,
        shared_cache_dirs: tuple[Path, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        queue = [
            _scripted_constrained_metrics(iou=0.90, precision=0.80, f1=0.70),
            _scripted_constrained_metrics(iou=0.85, precision=0.70, f1=0.60),
        ]

        def scripted(*args: object, **kwargs: object) -> ValidationMetrics:
            return queue.pop(0)

        monkeypatch.setattr(train_module, "_run_validation", scripted)
        run_root = tmp_path / "constrained_never_legal"
        result = train(
            _tiny_cache_train_config(
                train_cache=shared_cache_dirs[0],
                val_cache=shared_cache_dirs[1],
                run_root=run_root,
                max_epochs=2,
                constrained_iou_selection=True,
            )
        )
        assert not queue
        assert result.best_constrained_iou_epoch is None
        assert result.best_constrained_iou_val_metric is None
        assert result.best_constrained_iou_checkpoint_path is None
        assert not (run_root / "best_constrained_iou.pt").exists()

        records = _metrics_records(result.metrics_log_path)
        assert all(
            r["legal"] is False
            for r in records
            if r["event"] == "constrained_iou_selection"
        )
        summary = [
            r for r in records if r["event"] == "constrained_iou_selection_summary"
        ]
        assert len(summary) == 1
        assert summary[0]["qualified"] is False

    def test_ema_variant_gated_by_its_own_metrics_independently(
        self,
        tmp_path: Path,
        shared_cache_dirs: tuple[Path, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Raw model: epoch 0 legal (iou 0.5), epoch 1 illegal (iou 0.9,
        # low precision). EMA model: epoch 0 illegal (iou 0.9, low
        # precision), epoch 1 legal (iou 0.6). The two best-constrained
        # -IoU epochs must diverge, proving the EMA branch reads
        # ema_val_metrics, never the raw val_metrics.
        queue = [
            _scripted_constrained_metrics(iou=0.5, precision=0.95, f1=0.90),  # raw e0
            _scripted_constrained_metrics(iou=0.9, precision=0.80, f1=0.70),  # ema e0
            _scripted_constrained_metrics(iou=0.9, precision=0.80, f1=0.70),  # raw e1
            _scripted_constrained_metrics(iou=0.6, precision=0.93, f1=0.88),  # ema e1
        ]

        def scripted(*args: object, **kwargs: object) -> ValidationMetrics:
            return queue.pop(0)

        monkeypatch.setattr(train_module, "_run_validation", scripted)
        run_root = tmp_path / "constrained_ema"
        result = train(
            _tiny_cache_train_config(
                train_cache=shared_cache_dirs[0],
                val_cache=shared_cache_dirs[1],
                run_root=run_root,
                max_epochs=2,
                ema_decay=0.9,
                constrained_iou_selection=True,
            )
        )
        assert not queue
        assert result.best_constrained_iou_epoch == 0
        assert result.best_constrained_iou_val_metric == pytest.approx(0.5)
        assert result.best_ema_constrained_iou_epoch == 1
        assert result.best_ema_constrained_iou_val_metric == pytest.approx(0.6)
        assert (run_root / "best_constrained_iou.pt").exists()
        assert (run_root / "best_ema_constrained_iou.pt").exists()


class TestSoftTargetRouting:
    @staticmethod
    def _fractional_cache_copy(source: Path, destination: Path) -> Path:
        shutil.copytree(source, destination)
        sample_path = destination / "sample_0000.npz"
        with np.load(sample_path, allow_pickle=False) as npz:
            payload = {name: np.array(npz[name], copy=True) for name in npz.files}
        payload["source_fraction"] = np.where(
            payload["target_mask"] >= 0.5,
            np.float32(0.6),
            np.float32(0.0),
        ).astype(np.float32)
        np.savez(sample_path, **payload)
        return destination

    def test_training_loss_receives_soft_or_hard_target_according_to_flag(
        self,
        tmp_path: Path,
        shared_cache_dirs: tuple[Path, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source_train_cache, val_cache = shared_cache_dirs
        train_cache = self._fractional_cache_copy(
            source_train_cache, tmp_path / "fractional_train"
        )
        captured: dict[str, torch.Tensor] = {}

        soft_original = train_module._LOSS_FUNCTIONS["soft_combo"]  # noqa: SLF001
        hard_original = train_module._LOSS_FUNCTIONS["combo"]  # noqa: SLF001

        def soft_spy(
            logits: torch.Tensor,
            target: torch.Tensor,
            valid_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            captured["soft"] = target.detach().cpu().clone()
            return soft_original(logits, target, valid_mask=valid_mask)

        def hard_spy(
            logits: torch.Tensor,
            target: torch.Tensor,
            valid_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            captured["hard"] = target.detach().cpu().clone()
            return hard_original(logits, target, valid_mask=valid_mask)

        monkeypatch.setitem(train_module._LOSS_FUNCTIONS, "soft_combo", soft_spy)
        monkeypatch.setitem(train_module._LOSS_FUNCTIONS, "combo", hard_spy)

        train(
            _tiny_cache_train_config(
                train_cache=train_cache,
                val_cache=val_cache,
                run_root=tmp_path / "soft_run",
                loss="soft_combo",
                soft_target=True,
            )
        )
        train(
            _tiny_cache_train_config(
                train_cache=train_cache,
                val_cache=val_cache,
                run_root=tmp_path / "hard_run",
                loss="combo",
                soft_target=False,
            )
        )

        assert bool((captured["soft"] == 0.6).any())
        assert not bool((captured["hard"] == 0.6).any())
        assert bool(
            torch.logical_or(captured["hard"] == 0.0, captured["hard"] == 1.0).all()
        )

    def test_soft_target_and_hnm_preserve_routing_and_mining_contract(
        self,
        tmp_path: Path,
        shared_cache_dirs: tuple[Path, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source_train_cache, val_cache = shared_cache_dirs
        train_cache = self._fractional_cache_copy(
            source_train_cache, tmp_path / "fractional_hnm_train"
        )
        captured_targets: list[torch.Tensor] = []
        original = train_module._LOSS_FUNCTIONS["soft_combo"]  # noqa: SLF001

        def spy(
            logits: torch.Tensor,
            target: torch.Tensor,
            valid_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            captured_targets.append(target.detach().cpu().clone())
            return original(logits, target, valid_mask=valid_mask)

        monkeypatch.setitem(train_module._LOSS_FUNCTIONS, "soft_combo", spy)
        run_root = tmp_path / "soft_hnm_run"
        config = _tiny_cache_train_config(
            train_cache=train_cache,
            val_cache=val_cache,
            run_root=run_root,
            max_epochs=2,
            loss="soft_combo",
            soft_target=True,
            hard_negative_mining=True,
            hard_negative_fraction=1.0,
            hard_negative_oversample_weight=3.0,
            hard_negative_warmup_epochs=1,
        )
        result = train(config)

        assert any(bool((target == 0.6).any()) for target in captured_targets)

        def reject_nonstandard_constant(token: str) -> None:
            raise AssertionError(f"non-standard JSON constant: {token}")

        for line in result.metrics_log_path.read_text().splitlines():
            json.loads(line, parse_constant=reject_nonstandard_constant)
        mining = [
            record
            for record in _metrics_records(result.metrics_log_path)
            if record["event"] == "hard_negative_mining"
        ]
        assert [record["sampling_mode_this_epoch"] for record in mining] == [
            "uniform",
            "mining",
        ]
        assert mining[0]["n_negative_total"] == 1
        assert mining[0]["n_negative_seen"] == 1
        assert mining[0]["n_hard_negatives_mined"] == 1
        assert mining[0]["mean_easy_negative_score"] is None
        assert mining[0]["score_null_reasons"] == {
            "mean_easy_negative_score": "all_observed_negatives_selected_as_hard"
        }
        assert result.global_step == 2

        with pytest.raises(
            CheckpointCompatibilityError, match="checkpoint-external state"
        ):
            resume(run_root, replace(config, max_epochs=3))

    def test_validation_uses_hard_mask_even_when_soft_training_is_enabled(
        self,
        shared_cache_dirs: tuple[Path, Path],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        train_cache, val_cache = shared_cache_dirs
        sample = CachedSampleDataset(train_cache)[0]
        contradictory = replace(
            sample,
            source_fraction=torch.zeros_like(sample.source_fraction),
        )
        config = _tiny_cache_train_config(
            train_cache=train_cache,
            val_cache=val_cache,
            run_root=tmp_path / "unused",
            soft_target=True,
            loss="soft_combo",
        )

        def forbidden_augmentation(*args: object, **kwargs: object) -> None:
            raise AssertionError("validation invoked training augmentation")

        monkeypatch.setattr(train_module, "_augment_batch", forbidden_augmentation)
        metrics = train_module._run_validation(  # noqa: SLF001
            build_model(ModelConfig(base_channels=4)),
            [contradictory],
            config,
            torch.device("cpu"),
            False,
        )
        assert metrics.n_positive == 1
        assert metrics.n_negative == 0


class TestAugmentationContract:
    def test_shared_geometry_binary_masks_and_soft_interpolation(
        self,
        tmp_path: Path,
        shared_bundle_dirs: tuple[Path, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        train_dir, val_dir = shared_bundle_dirs
        config = _tiny_train_config(
            train_dir=train_dir,
            val_dir=val_dir,
            run_root=tmp_path / "unused",
            augment=True,
            augment_rotation_deg=0.0,
            augment_translate_px=1.0,
            augment_scale_delta=0.0,
            augment_pet_gain_delta=0.0,
            augment_pet_bias=0.0,
            augment_ct_gain_delta=0.0,
            augment_ct_bias=0.0,
        )
        base = torch.zeros(1, 1, 8, 8)
        base[0, 0, 3:5, 3:5] = 1.0
        left = base.repeat(1, 10, 1, 1)
        right = left.clone()
        right[:, :5] = 0.0
        diff = left[:, :5] - right[:, :5]
        target = base.clone()
        valid = torch.ones_like(base)

        draws = [0.0, 1.0, 0.5, 0.5, 1.0, 0.0, 1.0, 0.0]

        def install_draws() -> None:
            queue = list(draws)

            def scripted_uniform(
                n: int,
                low: float,
                high: float,
                device: torch.device,
            ) -> torch.Tensor:
                return torch.full((n,), queue.pop(0), device=device)

            monkeypatch.setattr(train_module, "_uniform", scripted_uniform)

        install_draws()
        hard = _augment_batch(
            left, right, diff, target, valid, config=config, target_interp="nearest"
        )
        install_draws()
        soft = _augment_batch(
            left, right, diff, target, valid, config=config, target_interp="bilinear"
        )

        hard_left, hard_right, hard_diff, hard_target, hard_valid = hard
        assert torch.equal(hard_left[:, 5:], hard_right[:, 5:])
        assert torch.equal(hard_left[:, :5] - hard_right[:, :5], hard_diff)
        assert bool(torch.logical_or(hard_target == 0.0, hard_target == 1.0).all())
        assert bool(torch.logical_or(hard_valid == 0.0, hard_valid == 1.0).all())
        assert bool(((soft[3] > 0.0) & (soft[3] < 1.0)).any())

    def test_intensity_jitter_preserves_pet_difference_identity(
        self,
        tmp_path: Path,
        shared_bundle_dirs: tuple[Path, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        train_dir, val_dir = shared_bundle_dirs
        config = _tiny_train_config(
            train_dir=train_dir,
            val_dir=val_dir,
            run_root=tmp_path / "unused",
            augment=True,
            augment_rotation_deg=0.0,
            augment_translate_px=0.0,
            augment_scale_delta=0.0,
        )
        left = torch.zeros(1, 10, 6, 6)
        right = torch.zeros(1, 10, 6, 6)
        left[:, :5] = 1.0
        diff = left[:, :5] - right[:, :5]
        target = torch.zeros(1, 1, 6, 6)
        valid = torch.ones_like(target)
        # The negative PET bias clips the right view at zero. This boundary
        # case detects implementations that transform/clamp pet_diff
        # independently instead of deriving it from the final PET views.
        draws = [0.0, 1.0, 0.0, 0.0, 1.1, -0.03, 0.9, -0.02]
        queue = list(draws)

        def scripted_uniform(
            n: int, low: float, high: float, device: torch.device
        ) -> torch.Tensor:
            return torch.full((n,), queue.pop(0), device=device)

        monkeypatch.setattr(train_module, "_uniform", scripted_uniform)
        augmented = _augment_batch(left, right, diff, target, valid, config=config)
        aug_left, aug_right, aug_diff, aug_target, aug_valid = augmented
        torch.testing.assert_close(aug_diff, aug_left[:, :5] - aug_right[:, :5])
        assert torch.equal(aug_target, target)
        assert torch.equal(aug_valid, valid)


class TestEmaContract:
    def test_update_ema_uses_exact_polyak_arithmetic(self) -> None:
        ema = torch.nn.Linear(1, 1, bias=False)
        live = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            ema.weight.fill_(2.0)
            live.weight.fill_(6.0)
        _update_ema(ema, live, 0.75)
        assert ema.weight.item() == pytest.approx(3.0)
        assert live.weight.item() == pytest.approx(6.0)

    def test_ema_is_additive_and_does_not_change_raw_trajectory(
        self, tmp_path: Path, shared_cache_dirs: tuple[Path, Path]
    ) -> None:
        train_cache, val_cache = shared_cache_dirs
        raw_result = train(
            _tiny_cache_train_config(
                train_cache=train_cache,
                val_cache=val_cache,
                run_root=tmp_path / "raw",
                seed=606,
            )
        )
        ema_result = train(
            _tiny_cache_train_config(
                train_cache=train_cache,
                val_cache=val_cache,
                run_root=tmp_path / "ema",
                seed=606,
                ema_decay=0.9,
            )
        )
        raw_payload = load_checkpoint(raw_result.last_checkpoint_path)
        ema_raw_payload = load_checkpoint(ema_result.last_checkpoint_path)
        for name, tensor in raw_payload.model_state_dict.items():
            assert torch.equal(tensor, ema_raw_payload.model_state_dict[name])
        assert (tmp_path / "ema" / "last_ema.pt").is_file()
        assert (tmp_path / "ema" / "best_ema.pt").is_file()
        assert not (tmp_path / "raw" / "last_ema.pt").exists()


class TestHardNegativeMiningContract:
    def test_config_bounds_and_cache_requirement(
        self,
        tmp_path: Path,
        shared_bundle_dirs: tuple[Path, Path],
    ) -> None:
        train_dir, val_dir = shared_bundle_dirs
        base = {
            "train_dir": train_dir,
            "val_dir": val_dir,
            "run_root": tmp_path / "run",
        }
        with pytest.raises(TrainConfigError, match="requires train_cache_dir"):
            _tiny_train_config(**base, hard_negative_mining=True)
        for override in (
            {"hard_negative_fraction": 0.0},
            {"hard_negative_fraction": 1.1},
            {"hard_negative_oversample_weight": 0.5},
            {"hard_negative_warmup_epochs": -1},
            {"hard_negative_score_momentum": 0.0},
            {"hard_negative_score_momentum": 1.0},
        ):
            with pytest.raises(TrainConfigError):
                _tiny_train_config(**base, **override)

    def test_clipped_negative_score_excludes_positives_and_invalid_pixels(
        self,
    ) -> None:
        logits = torch.tensor(
            [
                [[[0.0, 0.0], [0.0, 0.0]]],
                [[[2.0, 2.0], [2.0, 100.0]]],
                [[[-2.0, -2.0], [-2.0, -100.0]]],
            ]
        )
        target = torch.zeros_like(logits)
        target[0, 0, 0, 0] = 1.0
        valid = torch.ones_like(logits)
        valid[1:, 0, 1, 1] = 0.0

        is_negative, scores = _per_sample_clipped_negative_score(logits, target, valid)
        assert is_negative.tolist() == [False, True, True]
        expected_high = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.full((3,), 2.0), torch.zeros(3)
        )
        expected_low = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.full((3,), -2.0), torch.zeros(3)
        )
        assert scores[1] == pytest.approx(float(expected_high))
        assert scores[2] == pytest.approx(float(expected_low))
        assert scores[1] > scores[2]

    def test_clipped_negative_score_documents_saturated_ties(self) -> None:
        logits = torch.tensor([[[[20.0]]], [[[100.0]]]])
        target = torch.zeros_like(logits)
        valid = torch.ones_like(logits)

        is_negative, scores = _per_sample_clipped_negative_score(logits, target, valid)

        assert is_negative.tolist() == [True, True]
        expected_cap = -math.log(1e-6)
        assert scores[0] == pytest.approx(expected_cap)
        assert scores[1] == pytest.approx(expected_cap)


class TestSoftLossContract:
    def test_proper_soft_losses_are_minimized_at_target_and_have_finite_gradients(
        self,
    ) -> None:
        target = torch.tensor([[[[0.25, 0.75]]]], dtype=torch.float32)
        optimum = torch.logit(target).detach().requires_grad_(True)
        perturbed = (torch.logit(target) + 0.8).detach().requires_grad_(True)

        assert soft_dice_semimetric_loss(optimum, target).item() == pytest.approx(
            0.0, abs=2e-7
        )
        assert soft_bce_loss(optimum, target) < soft_bce_loss(perturbed, target)
        assert soft_combo_loss(optimum, target) < soft_combo_loss(perturbed, target)

        loss = soft_combo_loss(optimum, target)
        loss.backward()
        assert optimum.grad is not None
        assert torch.isfinite(optimum.grad).all()

    def test_valid_mask_makes_invalid_soft_target_changes_inert(self) -> None:
        logits = torch.tensor([[[[0.2, -0.3]]]], requires_grad=True)
        target_a = torch.tensor([[[[0.25, 0.75]]]])
        target_b = torch.tensor([[[[0.25, 0.05]]]])
        valid = torch.tensor([[[[1.0, 0.0]]]])
        torch.testing.assert_close(
            soft_combo_loss(logits, target_a, valid),
            soft_combo_loss(logits, target_b, valid),
        )
