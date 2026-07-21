"""Tests for the Phase 4 lever B2/L4 deep-supervision feature (train-only
multi-scale aux heads on ``SiameseBilateralUNet``, ``combo_loss`` reused
unmodified at each aux scale).

Independence note: every assertion in this file is derived from the frozen
deep-supervision contract, with the two corrections the shipped implementation's
module docstrings record it follows exactly (not the spec's original
3-head/nearest-downsample draft): TWO aux heads at decoder scales
``{2, 4}`` only (the spec's x8 head is dropped), and hard-mask
downsampling by MAX-POOL (``F.max_pool2d`` -- "any positive pixel in the
block -> 1"), not nearest/strided sampling. Test cases follow the T1-T7
enumeration defined for this feature (a corrected restatement of the
spec's own Sec 5 table), not a reading of ``src/`` diffs.

Where a case needs an oracle independent of the implementation under
test, this file uses the recorded v6 model signature. A local v6 manifest
is checked against that value when available, and the optional checkpoint
test runs only when its local artifact exists.

CPU-only, generated-tensor/generated-bundle fixtures -- no ``Data/``
access anywhere in this file (matching ``tests/test_ml_train.py``'s own
stated convention, whose small fixture-builder helpers this file
deliberately duplicates rather than imports, per this project's own
small-primitive-duplication convention).
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

import src.vascutrace.ml.evaluate as evaluate_module
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
from src.vascutrace.geometry import GeometrySidecar
from src.vascutrace.ml.cache import precompute_synthetic_cache
from src.vascutrace.ml.checkpoint import load_checkpoint
from src.vascutrace.ml.dataset import DatasetConfig, Sample
from src.vascutrace.ml.model import ModelConfig, build_model, model_signature
from src.vascutrace.ml.tensor_schema import (
    IN_PLANE_HW,
    LEFT_VIEW_SHAPE,
    PET_DIFF_SHAPE,
    RIGHT_VIEW_SHAPE,
    TARGET_SHAPE,
)
from src.vascutrace.ml.train import TrainConfig, TrainConfigError
from src.vascutrace.ml.train import _max_pool_hard_target  # noqa: PLC2701 -- direct unit test
from src.vascutrace.ml.train import _min_pool_valid_mask  # noqa: PLC2701 -- direct unit test
from src.vascutrace.ml.train import _run_validation  # noqa: PLC2701 -- direct unit test

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_DIR = _ROOT / "configs"
H, W = IN_PLANE_HW

# The pre-existing, completed v6exp run -- a real project artifact that
# predates the deep-supervision feature entirely (its config has no
# deep_supervision field at all). Its recorded model_signature and its
# checkpoint's state_dict are used below as an independent oracle for
# T1's "byte-identical to v6" invariant, rather than a value copied from
# this feature's own implementation.
_V6EXP_MANIFEST = _ROOT / "runs" / "siamese_v6exp" / "manifest.json"
_V6EXP_CHECKPOINT = _ROOT / "runs" / "siamese_v6exp" / "last.pt"
_FROZEN_V6_SIGNATURE = "p6-siamese-unet-v1+p6-tensor-v1+cfg-0e0f8f2a1779"


def _frozen_v6_signature() -> str:
    if _V6EXP_MANIFEST.is_file():
        payload = json.loads(_V6EXP_MANIFEST.read_text())
        assert payload["model_signature"] == _FROZEN_V6_SIGNATURE
    return _FROZEN_V6_SIGNATURE


def _fixed_input(batch: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A deterministic (non-RNG-consuming) input triple with the network's
    frozen tensor-schema shapes, for forward-pass comparisons that must
    not depend on ambient global RNG state.
    """
    left = torch.linspace(-1.0, 1.0, batch * np.prod(LEFT_VIEW_SHAPE).item()).reshape(
        batch, *LEFT_VIEW_SHAPE
    )
    right = torch.linspace(1.0, -1.0, batch * np.prod(RIGHT_VIEW_SHAPE).item()).reshape(
        batch, *RIGHT_VIEW_SHAPE
    )
    diff = torch.linspace(-0.5, 0.5, batch * np.prod(PET_DIFF_SHAPE).item()).reshape(
        batch, *PET_DIFF_SHAPE
    )
    return left, right, diff


def _load_yaml(name: str) -> dict[str, object]:
    payload = yaml.safe_load((_CONFIG_DIR / name).read_text())
    assert isinstance(payload, dict)
    return payload


def _synthetic_sample(*, positive: bool) -> Sample:
    """A hand-built :class:`Sample` with the frozen tensor-schema shapes --
    bypasses the bundle/dataset/cache pipeline entirely so T3's validation
    -path test stays a fast, pure unit test.
    """
    target = torch.zeros(*TARGET_SHAPE)
    if positive:
        target[0, 60:66, 30:36] = 1.0
    return Sample(
        left_view=torch.zeros(*LEFT_VIEW_SHAPE),
        right_view=torch.zeros(*RIGHT_VIEW_SHAPE),
        pet_diff=torch.zeros(*PET_DIFF_SHAPE),
        target_mask=target,
        source_fraction=torch.zeros(*TARGET_SHAPE),
        valid_mask=torch.ones(*TARGET_SHAPE),
        raw_pet=torch.zeros(*TARGET_SHAPE),
        meta={"positive": positive},
    )


def _tiny_deep_sup_config(*, tmp_path: Path, **overrides) -> TrainConfig:
    """A minimal, dataset-free ``TrainConfig`` -- used only by tests that
    exercise ``__post_init__`` validation or ``_run_validation`` directly
    (neither reads ``train_bundle_dirs``/``val_bundle_dirs`` off disk), so
    the placeholder directories below deliberately do not need to exist.
    """
    defaults: dict = dict(
        train_bundle_dirs=(tmp_path / "unused_train_bundles",),
        val_bundle_dirs=(tmp_path / "unused_val_bundles",),
        run_root=tmp_path / "run",
        batch_size=1,
        device="cpu",
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


# ---------------------------------------------------------------------------
# Shared bundle fixture (duplicated from tests/test_ml_train.py's own
# _make_bundle/_two_bundle_dirs/shared_bundle_dirs pattern; needed only by
# the T5 real-train() spy test, which must exercise the actual training
# step's loss composition, not a hand-rolled reimplementation of it).
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


@pytest.fixture(scope="module")
def shared_bundle_dirs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("deepsup_bundles")
    train_dir = save_crop_bundle(
        _make_bundle(subject="SYNTH_SUBJECT_201", seed=11), root
    )
    val_dir = save_crop_bundle(_make_bundle(subject="SYNTH_SUBJECT_202", seed=12), root)
    return train_dir, val_dir


@pytest.fixture(scope="module")
def shared_deep_sup_cache_dirs(
    tmp_path_factory: pytest.TempPathFactory,
    shared_bundle_dirs: tuple[Path, Path],
) -> tuple[Path, Path]:
    train_dir, val_dir = shared_bundle_dirs
    root = tmp_path_factory.mktemp("deepsup_caches")
    train_cache = root / "train_cache"
    val_cache = root / "val_cache"
    cache_config = DatasetConfig(supersample=1)
    precompute_synthetic_cache(
        [train_dir],
        train_cache,
        n_positive_per_bundle=1,
        n_negative_per_bundle=1,
        seed=51,
        config=cache_config,
        num_workers=1,
    )
    precompute_synthetic_cache(
        [val_dir],
        val_cache,
        n_positive_per_bundle=1,
        n_negative_per_bundle=1,
        seed=52,
        config=cache_config,
        num_workers=1,
    )
    return train_cache, val_cache


def _tiny_train_config(
    *, train_dir: Path, val_dir: Path, run_root: Path, **overrides
) -> TrainConfig:
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
        limit_train_batches=1,
        limit_val_batches=1,
        log_every_n_steps=1,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


def _metrics_records(metrics_path: Path) -> list[dict]:
    return [json.loads(line) for line in metrics_path.read_text().splitlines() if line]


# ---------------------------------------------------------------------------
# T1 -- DEFAULT-OFF BYTE-IDENTICAL: the incumbent-protection invariant.
# ---------------------------------------------------------------------------


class TestT1DefaultOffByteIdentical:
    def test_forward_byte_identical_to_explicit_off(self) -> None:
        model_implicit = build_model(ModelConfig(seed=7))
        model_explicit = build_model(ModelConfig(seed=7, deep_supervision=False))
        model_implicit.eval()
        model_explicit.eval()

        left, right, diff = _fixed_input(batch=2)
        out_implicit = model_implicit(left, right, diff)
        out_explicit = model_explicit(left, right, diff)

        assert torch.equal(out_implicit, out_explicit)

    def test_model_signature_matches_frozen_v6_signature(self) -> None:
        frozen = _frozen_v6_signature()
        assert model_signature(ModelConfig()) == frozen
        assert model_signature(ModelConfig(deep_supervision=False)) == frozen
        # A default ModelConfig() must be indistinguishable, at the
        # signature level, from one that names the field explicitly.
        assert model_signature(ModelConfig()) == model_signature(
            ModelConfig(deep_supervision=False)
        )

    def test_v6_checkpoint_loads_strict_no_missing_or_unexpected_keys(self) -> None:
        if not _V6EXP_CHECKPOINT.exists():
            pytest.skip(f"real v6exp checkpoint artifact missing: {_V6EXP_CHECKPOINT}")
        payload = load_checkpoint(_V6EXP_CHECKPOINT)
        assert payload.model_signature == _frozen_v6_signature()

        model = build_model(ModelConfig())
        result = model.load_state_dict(payload.model_state_dict, strict=False)
        assert list(result.missing_keys) == []
        assert list(result.unexpected_keys) == []

        # strict=True must not raise given the same, already-proven-clean
        # state dict.
        model.load_state_dict(payload.model_state_dict, strict=True)

    def test_default_model_has_no_aux_head_modules_or_params(self) -> None:
        model = build_model(ModelConfig())
        assert model.aux_head2 is None
        assert model.aux_head3 is None
        assert not any("aux_head" in key for key in model.state_dict())
        assert not any("aux_head" in name for name, _ in model.named_parameters())


# ---------------------------------------------------------------------------
# T2 -- AUX SHAPES.
# ---------------------------------------------------------------------------


class TestT2AuxShapes:
    def test_aux_logit_shapes_at_scale_2_and_4(self) -> None:
        model = build_model(ModelConfig(base_channels=4, deep_supervision=True, seed=3))
        model.eval()
        batch = 2
        left, right, diff = _fixed_input(batch=batch)

        logits, aux_logits_by_scale = model(left, right, diff, return_aux=True)

        assert logits.shape == (batch, 1, *IN_PLANE_HW)
        assert set(aux_logits_by_scale.keys()) == {2, 4}
        assert aux_logits_by_scale[2].shape == (batch, 1, 72, 40)
        assert aux_logits_by_scale[4].shape == (batch, 1, 36, 20)

    def test_return_aux_true_without_deep_supervision_raises_value_error(self) -> None:
        model = build_model(
            ModelConfig(base_channels=4, deep_supervision=False, seed=3)
        )
        left, right, diff = _fixed_input(batch=1)
        with pytest.raises(ValueError, match="deep_supervision=True"):
            model(left, right, diff, return_aux=True)


# ---------------------------------------------------------------------------
# T3 -- EVAL INVARIANT: no aux leakage into validation/evaluation.
# ---------------------------------------------------------------------------


class TestT3EvalInvariant:
    def test_run_validation_source_never_requests_aux(self) -> None:
        source = inspect.getsource(train_module._run_validation)  # noqa: SLF001
        assert "return_aux" not in source

    def test_evaluate_module_source_never_requests_aux(self) -> None:
        source = Path(evaluate_module.__file__).read_text()
        assert "return_aux" not in source

    def test_run_validation_never_actually_passes_return_aux_true(
        self, tmp_path: Path
    ) -> None:
        """Behavioral spy: even a model BUILT WITH aux heads (so requesting
        them would succeed rather than raise) must never see
        ``return_aux=True`` from ``_run_validation``.
        """
        model = build_model(ModelConfig(base_channels=4, deep_supervision=True, seed=1))
        original_forward = model.forward

        def spying_forward(*args, **kwargs):
            requested_aux = kwargs.get("return_aux", False)
            if len(args) >= 4:
                requested_aux = requested_aux or bool(args[3])
            if requested_aux:
                raise AssertionError("validation path must never request aux logits")
            result = original_forward(*args, **kwargs)
            assert isinstance(result, torch.Tensor), (
                "eval-path forward must return a bare tensor, not a tuple"
            )
            return result

        model.forward = spying_forward  # direct instance-level spy

        samples = [_synthetic_sample(positive=True), _synthetic_sample(positive=False)]
        config = _tiny_deep_sup_config(tmp_path=tmp_path)

        metrics = _run_validation(model, samples, config, torch.device("cpu"), False)

        assert metrics.n_positive == 1
        assert metrics.n_negative == 1

    def test_run_validation_batches_never_carry_source_fraction(
        self, tmp_path: Path
    ) -> None:
        model = build_model(ModelConfig(base_channels=4, deep_supervision=True, seed=2))
        samples = [_synthetic_sample(positive=True)]
        config = _tiny_deep_sup_config(tmp_path=tmp_path)
        # _run_validation itself asserts "source_fraction" not in batch;
        # simply completing without raising is the observable here.
        _run_validation(model, samples, config, torch.device("cpu"), False)


# ---------------------------------------------------------------------------
# T4 -- MAX-POOL DOWNSAMPLING.
# ---------------------------------------------------------------------------


class TestT4MaxPoolDownsampling:
    def _small_lesion_target(self) -> torch.Tensor:
        target = torch.zeros(1, *TARGET_SHAPE)
        # A 4px blob at rows/cols 9-10 -- deliberately NOT aligned to any
        # scale-4 block's sampled corner (indices 0, 4, 8, 12, ...), so a
        # naive strided/nearest downsample at scale 4 misses it entirely.
        target[0, 0, 9:11, 9:11] = 1.0
        return target

    def test_small_lesion_stays_binary_and_survives_max_pool(self) -> None:
        target = self._small_lesion_target()
        for scale, expected_hw in ((2, (72, 40)), (4, (36, 20))):
            down = _max_pool_hard_target(target, scale)
            uniq = set(torch.unique(down).tolist())
            assert uniq <= {0.0, 1.0}
            assert down.shape[-2:] == expected_hw
            assert down.sum().item() > 0.0, (
                f"max-pool at scale={scale} lost the small lesion"
            )

    def test_naive_strided_downsample_at_scale_4_loses_the_same_lesion(self) -> None:
        """The Root-correction contrast: what motivated dropping nearest
        downsampling in favor of max-pool.
        """
        target = self._small_lesion_target()
        strided = target[:, :, ::4, ::4]
        assert strided.shape[-2:] == (36, 20)
        assert strided.sum().item() == 0.0

    def test_min_pool_valid_mask_stays_binary_and_is_conservative(self) -> None:
        valid = torch.ones(1, *TARGET_SHAPE)
        valid[0, 0, 0, 0] = 0.0  # a single invalid pixel in the first block
        for scale in (2, 4):
            down_valid = _min_pool_valid_mask(valid, scale)
            uniq = set(torch.unique(down_valid).tolist())
            assert uniq <= {0.0, 1.0}
            # the block containing the invalid source pixel is marked
            # invalid (conservative, whole-block-valid semantics) ...
            assert down_valid[0, 0, 0, 0].item() == 0.0
            # ... while every other, fully-valid block stays valid.
            assert down_valid[0, 0, 0, 1].item() == 1.0
            assert down_valid.sum().item() == down_valid.numel() - 1


# ---------------------------------------------------------------------------
# T5 -- combo_loss REUSE (no boundary_auxiliary_loss / edge-BCE; additive
# weighted-sum formula against the shipped combo_loss at every scale).
# ---------------------------------------------------------------------------


class TestT5ComboLossReuse:
    def test_training_step_calls_combo_loss_thrice_and_never_boundary_aux(
        self, shared_bundle_dirs: tuple[Path, Path], tmp_path: Path, monkeypatch
    ) -> None:
        train_dir, val_dir = shared_bundle_dirs
        real_combo = train_module._LOSS_FUNCTIONS["combo"]  # noqa: SLF001
        call_count = {"combo": 0}

        def spy_combo(*args, **kwargs):
            call_count["combo"] += 1
            return real_combo(*args, **kwargs)

        def forbidden_boundary_aux(*args, **kwargs):
            raise AssertionError(
                "deep supervision must never call boundary_auxiliary_loss"
            )

        # _LOSS_FUNCTIONS is looked up BY NAME at call time inside train(),
        # so patching the dict entry (not the module-level `combo_loss`
        # name, which _LOSS_FUNCTIONS already captured a direct reference
        # to at import time) is what actually intercepts every call site
        # -- both the main term and every aux-scale term go through this
        # same `loss_fn` binding.
        monkeypatch.setitem(train_module._LOSS_FUNCTIONS, "combo", spy_combo)  # noqa: SLF001
        monkeypatch.setattr(
            train_module, "boundary_auxiliary_loss", forbidden_boundary_aux
        )

        config = _tiny_train_config(
            train_dir=train_dir,
            val_dir=val_dir,
            run_root=tmp_path / "run",
            loss="combo",
            deep_supervision=True,
            deep_supervision_scales=(2, 4),
            deep_supervision_weights=(0.5, 0.25),
        )

        result = train_module.train(config)

        # main term (1) + one aux term per configured scale (2) per step.
        assert call_count["combo"] == 1 + len(config.deep_supervision_scales)

        records = _metrics_records(result.metrics_log_path)
        step_records = [r for r in records if r["event"] == "train_step"]
        assert step_records, "expected at least one train_step log record"
        record = step_records[0]
        assert record["deep_supervision"] is True
        assert record["L_aux"] == 0.0  # boundary-aux term inert (arm A)

        aux_terms = record["deep_sup_aux_terms"]
        assert set(aux_terms.keys()) == {"2", "4"}
        expected_deep_sup = 0.5 * aux_terms["2"] + 0.25 * aux_terms["4"]
        assert record["L_deep_sup"] == pytest.approx(expected_deep_sup, rel=1e-6)
        # total = combo(full) + sum_i w_i * combo(aux_i) -- boundary term
        # is 0.0 for this arm, so train_loss decomposes exactly.
        assert record["train_loss"] == pytest.approx(
            record["L_hard"] + record["L_deep_sup"], rel=1e-6
        )

    def test_default_off_step_has_zero_deep_sup_loss(
        self, shared_bundle_dirs: tuple[Path, Path], tmp_path: Path
    ) -> None:
        train_dir, val_dir = shared_bundle_dirs
        config = _tiny_train_config(
            train_dir=train_dir,
            val_dir=val_dir,
            run_root=tmp_path / "run_off",
            loss="combo",
        )
        assert config.deep_supervision is False

        result = train_module.train(config)
        records = _metrics_records(result.metrics_log_path)
        step_records = [r for r in records if r["event"] == "train_step"]
        assert step_records
        record = step_records[0]
        assert record["deep_supervision"] is False
        assert record["L_deep_sup"] == 0.0
        assert record["deep_sup_aux_terms"] == {}
        assert record["train_loss"] == pytest.approx(record["L_hard"], rel=1e-6)


# ---------------------------------------------------------------------------
# T6 -- CONFIG DELTA.
# ---------------------------------------------------------------------------


class TestT6ConfigDelta:
    _EXPECTED_CHANGED_KEYS = {
        "deep_supervision",
        "deep_supervision_scales",
        "deep_supervision_weights",
        "constrained_iou_selection",
        "constrained_iou_min_precision",
        "constrained_iou_min_f1",
        "constrained_iou_min_clean",
        "early_stop_patience",
    }

    def test_p4b2_deepsup_differs_from_v6exp_only_in_declared_fields(self) -> None:
        v6 = _load_yaml("train_siamese_v6exp.yaml")
        b2 = _load_yaml("train_siamese_p4b2_deepsup.yaml")
        changed = {k for k in set(v6) | set(b2) if v6.get(k) != b2.get(k)}
        assert changed == self._EXPECTED_CHANGED_KEYS

    def test_p4b2_deepsup_declared_field_values(self) -> None:
        b2 = _load_yaml("train_siamese_p4b2_deepsup.yaml")
        assert b2["deep_supervision"] is True
        assert b2["deep_supervision_scales"] == [2, 4]
        assert b2["deep_supervision_weights"] == [0.5, 0.25]
        assert b2["constrained_iou_selection"] is True
        assert b2["constrained_iou_min_precision"] == 0.901
        assert b2["constrained_iou_min_f1"] == 0.859
        assert b2["constrained_iou_min_clean"] == 0.0
        assert b2["early_stop_patience"] == 150

    def test_p4b2_deepsup_holds_loss_aug_hnm_batch_seed_fields(self) -> None:
        v6 = _load_yaml("train_siamese_v6exp.yaml")
        b2 = _load_yaml("train_siamese_p4b2_deepsup.yaml")
        held_fields = [
            "loss",
            "batch_size",
            "seed",
            "val_seed",
            "max_epochs",
            "lr",
            "weight_decay",
            "grad_clip_norm",
            "ema_decay",
            "selection_metric",
            "secondary_selection_metric",
            "augment",
            "augment_rotation_deg",
            "augment_translate_px",
            "augment_scale_delta",
            "augment_pet_gain_delta",
            "augment_pet_bias",
            "augment_ct_gain_delta",
            "augment_ct_bias",
            "hard_negative_mining",
            "hard_negative_fraction",
            "hard_negative_oversample_weight",
            "hard_negative_warmup_epochs",
            "hard_negative_score_momentum",
            "train_cache_dir",
            "val_cache_dir",
        ]
        for field in held_fields:
            assert v6.get(field) == b2.get(field), f"field {field!r} must be HELD"


# ---------------------------------------------------------------------------
# T7 -- CONFIG VALIDATION.
# ---------------------------------------------------------------------------


class TestT7ConfigValidation:
    def test_scale_weight_length_mismatch_raises_even_when_off(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(TrainConfigError, match="equal length"):
            _tiny_deep_sup_config(
                tmp_path=tmp_path,
                deep_supervision_scales=(2, 4),
                deep_supervision_weights=(0.5,),
            )

    def test_unsupported_scale_raises(self, tmp_path: Path) -> None:
        with pytest.raises(TrainConfigError, match="subset"):
            _tiny_deep_sup_config(
                tmp_path=tmp_path,
                deep_supervision_scales=(2, 8),
                deep_supervision_weights=(0.5, 0.25),
            )

    def test_unsupported_scale_3_raises(self, tmp_path: Path) -> None:
        with pytest.raises(TrainConfigError, match="subset"):
            _tiny_deep_sup_config(
                tmp_path=tmp_path,
                deep_supervision_scales=(3,),
                deep_supervision_weights=(0.5,),
            )

    def test_non_positive_weight_raises(self, tmp_path: Path) -> None:
        with pytest.raises(TrainConfigError, match=r"finite and > 0"):
            _tiny_deep_sup_config(
                tmp_path=tmp_path,
                deep_supervision_scales=(2,),
                deep_supervision_weights=(0.0,),
            )

    def test_negative_weight_raises(self, tmp_path: Path) -> None:
        with pytest.raises(TrainConfigError, match=r"finite and > 0"):
            _tiny_deep_sup_config(
                tmp_path=tmp_path,
                deep_supervision_scales=(2,),
                deep_supervision_weights=(-0.5,),
            )

    def test_deep_supervision_with_soft_target_raises(self, tmp_path: Path) -> None:
        with pytest.raises(TrainConfigError, match="soft_target=False"):
            _tiny_deep_sup_config(
                tmp_path=tmp_path,
                deep_supervision=True,
                soft_target=True,
                loss="soft_combo",
            )

    def test_valid_deep_supervision_config_round_trips(self, tmp_path: Path) -> None:
        config = _tiny_deep_sup_config(
            tmp_path=tmp_path,
            deep_supervision=True,
            deep_supervision_scales=[2, 4],
            deep_supervision_weights=[0.5, 0.25],
        )
        assert config.deep_supervision is True
        assert config.deep_supervision_scales == (2, 4)
        assert config.deep_supervision_weights == (0.5, 0.25)
        # model_config.deep_supervision must be derived True automatically.
        assert config.model_config.deep_supervision is True

    def test_p4b2_deepsup_config_file_parses_into_valid_train_config(
        self, tmp_path: Path, shared_deep_sup_cache_dirs: tuple[Path, Path]
    ) -> None:
        train_cache, val_cache = shared_deep_sup_cache_dirs
        payload = _load_yaml("train_siamese_p4b2_deepsup.yaml")
        payload = dict(payload)
        payload["train_bundle_dirs"] = ()
        payload["val_bundle_dirs"] = ()
        payload["train_cache_dir"] = train_cache
        payload["val_cache_dir"] = val_cache
        payload["run_root"] = tmp_path / "unused_deepsup_parse_check"
        payload.pop("model_config", None)
        payload.pop("dataset_config", None)
        config = TrainConfig(**payload)
        assert config.deep_supervision is True
        assert config.deep_supervision_scales == (2, 4)
        assert config.deep_supervision_weights == (0.5, 0.25)
        assert config.early_stop_patience == 150
        assert config.constrained_iou_selection is True
