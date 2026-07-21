"""Tests for the Phase 4 B3 additive soft-DML term (train-only, full-res
``soft_combo_loss`` added on top of the B2 [deep-supervision] base).

Independence note: every assertion in this file is derived from the frozen
B3 mechanism and the T1 through T7 case list, not from a reading of
``src/`` diffs. The exact
pinned formula this file tests against:

    total = combo(target_mask_full, logits_full, valid_full)              # B2 main (hard), unchanged
          + sum_i w_i * combo(maxpool(target_mask, i), aux_logits_i,
                               minpool(valid, i))                          # B2 deep-sup aux (hard), unchanged
          + beta * soft_combo(logits_full, source_fraction_full, valid_full)  # B3 ADDITIVE soft term, full-res only

B2's hard main + hard-aux terms are RETAINED unchanged (not replaced) --
B3 = B2 + one additive full-resolution soft term. ``soft_combo_loss`` is
reused unmodified; NOT edge-BCE, NOT a target replacement (that is the
separate, mutually exclusive ``soft_target=True`` path -- item 17). The
soft term is TRAIN-ONLY and full-resolution only (no multi-scale soft
supervision -- deferred to a future "B3.1").

``src/vascutrace/ml/train.py`` was read ONLY for exact function/dataclass
signatures (``TrainConfig`` field names, ``_run_validation``,
``_iter_val_batches``, ``_collate_samples``, ``soft_combo_loss``'s import
name) needed to construct valid calls -- never to decide what to assert.
T2 in particular is constructed adversarially: it would FAIL under a
hypothetical implementation that REPLACED a B2 hard term with the soft
term, even though such an implementation might still "work" and log a
plausible-looking ``L_soft``.

CPU-only, generated-bundle/generated-cache fixtures. No ``Data/`` access
or pre-existing cache is required. T6 replaces the YAML cache paths with
the generated cache pair before validating the remaining configuration.
"""

from __future__ import annotations

import inspect
import json
import math
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

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
from src.vascutrace.ml.cache import CachedSampleDataset, precompute_synthetic_cache
from src.vascutrace.ml.checkpoint import load_checkpoint
from src.vascutrace.ml.dataset import DatasetConfig, Sample
from src.vascutrace.ml.model import ModelConfig, build_model
from src.vascutrace.ml.tensor_schema import (
    IN_PLANE_HW,
    LEFT_VIEW_SHAPE,
    PET_DIFF_SHAPE,
    RIGHT_VIEW_SHAPE,
    TARGET_SHAPE,
)
from src.vascutrace.ml.train import TrainConfig, TrainConfigError

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_DIR = _ROOT / "configs"
H, W = IN_PLANE_HW

_BETA = 0.37  # a known, nonzero, non-1.0 soft_term_weight used across T2/T5


# ---------------------------------------------------------------------------
# Shared synthetic-fixture builders (deliberately duplicated from
# tests/test_ml_train.py / tests/test_ml_deep_supervision.py's own helpers,
# matching this project's small-primitive-duplication convention).
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


def _synthetic_sample(*, positive: bool) -> Sample:
    """A hand-built :class:`Sample` with the frozen tensor-schema shapes --
    bypasses the bundle/dataset/cache pipeline for the fast, pure-unit
    validation-path tests (T4).
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


def _load_yaml(name: str) -> dict[str, object]:
    payload = yaml.safe_load((_CONFIG_DIR / name).read_text())
    assert isinstance(payload, dict)
    return payload


def _metrics_records(metrics_path: Path) -> list[dict]:
    return [json.loads(line) for line in metrics_path.read_text().splitlines() if line]


def _first_train_step(metrics_path: Path) -> dict:
    for record in _metrics_records(metrics_path):
        if record["event"] == "train_step":
            return record
    raise AssertionError(f"no train_step record found in {metrics_path}")


@pytest.fixture(scope="module")
def shared_bundle_dirs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("b3_bundles")
    train_dir = save_crop_bundle(
        _make_bundle(subject="SYNTH_SUBJECT_301", seed=21), root
    )
    val_dir = save_crop_bundle(_make_bundle(subject="SYNTH_SUBJECT_302", seed=22), root)
    return train_dir, val_dir


_CACHE_TEST_CONFIG = DatasetConfig(supersample=1)


@pytest.fixture(scope="module")
def shared_cache_dirs(
    tmp_path_factory: pytest.TempPathFactory, shared_bundle_dirs: tuple[Path, Path]
) -> tuple[Path, Path]:
    """A real train/val cache PAIR built with ``precompute_synthetic_cache``
    (default ``has_source_fraction=True``) -- the artifact T2/T3/T5/T6/T7
    below need for a real ``soft_term_enabled=True`` run/config, rather
    than a hand-rolled manifest.
    """
    train_dir, val_dir = shared_bundle_dirs
    root = tmp_path_factory.mktemp("b3_caches")
    train_cache = root / "train_cache"
    val_cache = root / "val_cache"
    precompute_synthetic_cache(
        [train_dir],
        train_cache,
        n_positive_per_bundle=2,
        n_negative_per_bundle=2,
        seed=41,
        config=_CACHE_TEST_CONFIG,
        num_workers=1,
    )
    precompute_synthetic_cache(
        [val_dir],
        val_cache,
        n_positive_per_bundle=2,
        n_negative_per_bundle=2,
        seed=42,
        config=_CACHE_TEST_CONFIG,
        num_workers=1,
    )
    return train_cache, val_cache


@pytest.fixture(scope="module")
def no_soft_cache_dir(
    tmp_path_factory: pytest.TempPathFactory, shared_cache_dirs: tuple[Path, Path]
) -> Path:
    """A REAL cache copy whose manifest declares ``has_source_fraction =
    False`` -- a legacy-style cache T7's construction-time check must
    refuse. Built by copying the real cache T7 already proved supplies
    real ``source_fraction`` and flipping only the manifest flag (matching
    ``tests/test_ml_train.py``'s own legacy-manifest technique), so this
    is a real artifact, not an invented shape.
    """
    train_cache, _ = shared_cache_dirs
    dest = tmp_path_factory.mktemp("b3_no_soft_cache") / "train_cache"
    shutil.copytree(train_cache, dest)
    manifest_path = dest / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["has_source_fraction"] is True, (
        "sanity check: the source cache must really have source_fraction "
        "before this fixture flips it off"
    )
    manifest["has_source_fraction"] = False
    manifest_path.write_text(json.dumps(manifest))
    return dest


def _tiny_config(*, tmp_path: Path, **overrides: object) -> TrainConfig:
    """A minimal, dataset-free ``TrainConfig`` for ``__post_init__``
    validation-only tests -- neither ``train_bundle_dirs`` nor
    ``val_bundle_dirs`` is ever read off disk by these tests.
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


def _tiny_cache_config(
    *, train_cache: Path, val_cache: Path, run_root: Path, **overrides: object
) -> TrainConfig:
    """A small, fast, cache-backed ``TrainConfig`` with
    ``deep_supervision=True`` (the B2 base T1/T2 are testing B3 ON TOP
    OF) -- the caller flips ``soft_term_enabled``/``soft_term_weight``.
    """
    seed = int(overrides.pop("seed", 20260720))
    defaults: dict = dict(
        train_bundle_dirs=(),
        val_bundle_dirs=(),
        train_cache_dir=train_cache,
        val_cache_dir=val_cache,
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
        train_positive_fraction=0.5,
        val_positive_fraction=0.5,
        limit_train_batches=1,
        limit_val_batches=1,
        log_every_n_steps=1,
        lr_schedule="none",
        deep_supervision=True,
        deep_supervision_scales=(2, 4),
        deep_supervision_weights=(0.5, 0.25),
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


# ---------------------------------------------------------------------------
# T1 -- DEFAULT-OFF BYTE-IDENTICAL to B2 (incumbent-protection invariant).
# ---------------------------------------------------------------------------


class TestT1DefaultOffByteIdentical:
    def test_soft_term_fields_default_off(self, tmp_path: Path) -> None:
        config = _tiny_config(tmp_path=tmp_path)
        assert config.soft_term_enabled is False
        assert config.soft_term_weight == 0.0

    def test_implicit_and_explicit_off_are_byte_identical_checkpoints(
        self, shared_cache_dirs: tuple[Path, Path], tmp_path: Path
    ) -> None:
        train_cache, val_cache = shared_cache_dirs
        implicit = _tiny_cache_config(
            train_cache=train_cache,
            val_cache=val_cache,
            run_root=tmp_path / "implicit",
        )
        # Reuse the deep-sup config (deep_supervision=True) as the B2 base
        # -- B3-off must equal B2, not the pre-B2 baseline.
        assert implicit.deep_supervision is True
        assert implicit.soft_term_enabled is False  # never named -> default

        explicit = replace(
            implicit,
            run_root=tmp_path / "explicit",
            soft_term_enabled=False,
            soft_term_weight=0.0,
        )

        implicit_result = train_module.train(implicit)
        explicit_result = train_module.train(explicit)

        assert _metrics_records(implicit_result.metrics_log_path) == _metrics_records(
            explicit_result.metrics_log_path
        )

        implicit_payload = load_checkpoint(implicit_result.last_checkpoint_path)
        explicit_payload = load_checkpoint(explicit_result.last_checkpoint_path)
        assert set(implicit_payload.model_state_dict) == set(
            explicit_payload.model_state_dict
        )
        for name, tensor in implicit_payload.model_state_dict.items():
            assert torch.equal(tensor, explicit_payload.model_state_dict[name])

    def test_off_train_step_has_zero_soft_term_and_decomposes_to_b2_total(
        self, shared_cache_dirs: tuple[Path, Path], tmp_path: Path
    ) -> None:
        train_cache, val_cache = shared_cache_dirs
        config = _tiny_cache_config(
            train_cache=train_cache,
            val_cache=val_cache,
            run_root=tmp_path / "off_decomp",
        )
        result = train_module.train(config)
        step = _first_train_step(result.metrics_log_path)
        assert step["soft_term_enabled"] is False
        assert step["soft_term_weight"] == 0.0
        assert step["L_soft"] == 0.0
        assert step["soft_grad_norm"] is None
        assert step["hard_total_grad_norm"] is None
        assert step["soft_hard_grad_ratio"] is None
        # B2 total, unaffected by B3's own presence when off.
        assert step["train_loss"] == pytest.approx(
            step["L_hard"] + step["L_deep_sup"], rel=1e-6
        )


# ---------------------------------------------------------------------------
# T2 -- ADDITIVE (the core Root-ruling point): total = B2 total (unchanged)
# + beta * soft_combo_loss(...). Adversarially constructed to FAIL if a
# hard term were replaced rather than added to.
# ---------------------------------------------------------------------------


class TestT2Additive:
    def test_soft_term_adds_to_an_unchanged_b2_total(
        self, shared_cache_dirs: tuple[Path, Path], tmp_path: Path
    ) -> None:
        train_cache, val_cache = shared_cache_dirs
        config_off = _tiny_cache_config(
            train_cache=train_cache, val_cache=val_cache, run_root=tmp_path / "b2_off"
        )
        config_on = replace(
            config_off,
            run_root=tmp_path / "b3_on",
            soft_term_enabled=True,
            soft_term_weight=_BETA,
        )

        step_off = _first_train_step(train_module.train(config_off).metrics_log_path)
        step_on = _first_train_step(train_module.train(config_on).metrics_log_path)

        # Same seed/data/init both runs -> the B2 hard main term and hard
        # deep-sup aux terms must be UNCHANGED whether or not the soft term
        # is added (frozen B3 design: "B2's hard main + hard-aux terms are
        # RETAINED unchanged (not replaced)").
        assert step_on["L_hard"] == pytest.approx(step_off["L_hard"], rel=1e-6)
        assert step_on["L_deep_sup"] == pytest.approx(step_off["L_deep_sup"], rel=1e-6)
        assert step_on["deep_sup_aux_terms"]["2"] == pytest.approx(
            step_off["deep_sup_aux_terms"]["2"], rel=1e-6
        )
        assert step_on["deep_sup_aux_terms"]["4"] == pytest.approx(
            step_off["deep_sup_aux_terms"]["4"], rel=1e-6
        )

        # The soft term actually fired.
        assert step_on["soft_term_enabled"] is True
        assert step_on["soft_term_weight"] == _BETA
        assert math.isfinite(step_on["L_soft"])
        assert step_on["L_soft"] != 0.0

        b2_total = step_off["train_loss"]
        assert b2_total == pytest.approx(
            step_off["L_hard"] + step_off["L_deep_sup"], rel=1e-6
        )

        # THE core additive check: the B3 total must equal EXACTLY the
        # PRESERVED B2-off total plus beta * the raw soft term. A
        # hypothetical implementation that REPLACED a hard term with the
        # soft term (rather than adding to it) would break this equality
        # even if L_hard/L_deep_sup above happened to still get logged
        # with unchanged-looking values, because train_loss itself would
        # then not decompose this way.
        assert step_on["train_loss"] == pytest.approx(
            b2_total + _BETA * step_on["L_soft"], rel=1e-5
        )
        # Same decomposition using ONLY the on-run's own logged terms --
        # catches a bug where L_hard/L_deep_sup are logged correctly but
        # are NOT what was actually backpropagated into train_loss.
        assert step_on["train_loss"] == pytest.approx(
            step_on["L_hard"] + step_on["L_deep_sup"] + _BETA * step_on["L_soft"],
            rel=1e-5,
        )

    def test_zero_weight_soft_term_is_computed_but_inert_on_total(
        self, shared_cache_dirs: tuple[Path, Path], tmp_path: Path
    ) -> None:
        """beta=0.0 distinguishes 'additive with weight 0' from 'off': the
        soft term still runs (soft_term_enabled=True) but contributes
        exactly zero to the total -- linear scaling, not a special-cased
        skip.
        """
        train_cache, val_cache = shared_cache_dirs
        config = _tiny_cache_config(
            train_cache=train_cache,
            val_cache=val_cache,
            run_root=tmp_path / "beta0",
            soft_term_enabled=True,
            soft_term_weight=0.0,
        )
        step = _first_train_step(train_module.train(config).metrics_log_path)
        assert step["soft_term_enabled"] is True
        assert step["soft_term_weight"] == 0.0
        assert math.isfinite(step["L_soft"])
        assert step["train_loss"] == pytest.approx(
            step["L_hard"] + step["L_deep_sup"], rel=1e-6
        )


# ---------------------------------------------------------------------------
# T3 -- COMPATIBLE WITH DEEP SUPERVISION (distinct from soft_target, which
# IS mutually exclusive with deep_supervision).
# ---------------------------------------------------------------------------


class TestT3CompatibleWithDeepSupervision:
    def test_soft_term_enabled_with_deep_supervision_is_valid(
        self, shared_cache_dirs: tuple[Path, Path], tmp_path: Path
    ) -> None:
        train_cache, val_cache = shared_cache_dirs
        config = TrainConfig(
            train_bundle_dirs=(),
            val_bundle_dirs=(),
            train_cache_dir=train_cache,
            val_cache_dir=val_cache,
            run_root=tmp_path / "run",
            batch_size=1,
            device="cpu",
            deep_supervision=True,
            soft_term_enabled=True,
            soft_term_weight=0.25,
        )
        assert config.deep_supervision is True
        assert config.soft_term_enabled is True
        assert config.model_config.deep_supervision is True  # derived, unaffected

    def test_old_soft_target_with_deep_supervision_still_raises(
        self, tmp_path: Path
    ) -> None:
        """The distinct, still-standing item-17 mutual exclusion: unlike
        soft_term_enabled, soft_target REPLACES the training target and
        remains incompatible with deep_supervision.
        """
        with pytest.raises(TrainConfigError, match="soft_target=False"):
            _tiny_config(
                tmp_path=tmp_path,
                deep_supervision=True,
                soft_target=True,
                loss="soft_combo",
            )

    def test_soft_term_enabled_with_soft_target_raises_mutual_exclusion(
        self, shared_cache_dirs: tuple[Path, Path], tmp_path: Path
    ) -> None:
        train_cache, val_cache = shared_cache_dirs
        with pytest.raises(TrainConfigError, match="soft_target=False"):
            TrainConfig(
                train_bundle_dirs=(),
                val_bundle_dirs=(),
                train_cache_dir=train_cache,
                val_cache_dir=val_cache,
                run_root=tmp_path / "run2",
                batch_size=1,
                device="cpu",
                soft_term_enabled=True,
                soft_target=True,
                loss="soft_combo",
            )


# ---------------------------------------------------------------------------
# T4 -- TRAIN-ONLY / EVAL INVARIANT: source_fraction never reaches
# validation, regardless of soft_term_enabled.
# ---------------------------------------------------------------------------


class TestT4TrainOnlyEvalInvariant:
    def test_iter_val_batches_source_structurally_ignores_source_fraction(
        self,
    ) -> None:
        source = inspect.getsource(train_module._iter_val_batches)  # noqa: SLF001
        assert "source_fraction" not in source
        assert "include_source_fraction" not in source

    def test_run_validation_completes_with_soft_term_enabled_true(
        self, tmp_path: Path
    ) -> None:
        model = build_model(ModelConfig(base_channels=4, deep_supervision=True, seed=5))
        samples = [_synthetic_sample(positive=True), _synthetic_sample(positive=False)]
        # soft_term_enabled cannot legally be True without a real cache;
        # this dataset-free config exercises _run_validation directly
        # (which reads no soft_term_* field at all) to prove the eval path
        # structurally cannot vary with it.
        config = _tiny_config(tmp_path=tmp_path)
        metrics = train_module._run_validation(  # noqa: SLF001
            model, samples, config, torch.device("cpu"), False
        )
        assert metrics.n_positive == 1
        assert metrics.n_negative == 1

    def test_collate_samples_spy_include_source_fraction_always_false_when_soft_term_enabled(
        self,
        shared_cache_dirs: tuple[Path, Path],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Behavioral spy: with a REAL, valid ``soft_term_enabled=True``
        config, every ``_collate_samples`` call made from the validation
        path (``_iter_val_batches``) must still pass
        ``include_source_fraction=False`` -- source_fraction is attached
        to the TRAIN batch only.
        """
        train_cache, val_cache = shared_cache_dirs
        model = build_model(ModelConfig(base_channels=4, deep_supervision=True, seed=6))
        config = _tiny_cache_config(
            train_cache=train_cache,
            val_cache=val_cache,
            run_root=tmp_path / "val_spy",
            soft_term_enabled=True,
            soft_term_weight=0.2,
        )
        assert config.soft_term_enabled is True

        observed_flags: list[bool] = []
        original_collate = train_module._collate_samples  # noqa: SLF001

        def spying_collate(samples_arg, *, include_source_fraction=False):
            observed_flags.append(include_source_fraction)
            return original_collate(
                samples_arg, include_source_fraction=include_source_fraction
            )

        monkeypatch.setattr(train_module, "_collate_samples", spying_collate)  # noqa: SLF001

        val_dataset = CachedSampleDataset(val_cache)
        train_module._run_validation(  # noqa: SLF001
            model, val_dataset, config, torch.device("cpu"), False
        )

        assert observed_flags, "expected _iter_val_batches to call _collate_samples"
        assert all(flag is False for flag in observed_flags)

    def test_source_fraction_batch_reaches_train_step_when_enabled(
        self, shared_cache_dirs: tuple[Path, Path], tmp_path: Path
    ) -> None:
        """The train-side contrast to the val-side spy above: a real
        soft_term_enabled=True training step must actually have consumed
        source_fraction (observable indirectly via a finite L_soft --
        already covered by T2 -- and directly here via the source cache's
        own has_source_fraction=True manifest).
        """
        from src.vascutrace.ml.cache import cache_has_source_fraction

        train_cache, val_cache = shared_cache_dirs
        assert cache_has_source_fraction(train_cache) is True
        config = _tiny_cache_config(
            train_cache=train_cache,
            val_cache=val_cache,
            run_root=tmp_path / "train_side",
            soft_term_enabled=True,
            soft_term_weight=0.2,
        )
        step = _first_train_step(train_module.train(config).metrics_log_path)
        assert step["soft_term_enabled"] is True
        assert math.isfinite(step["L_soft"])


# ---------------------------------------------------------------------------
# T5 -- soft_combo_loss REUSE: full-res only, never boundary_auxiliary_loss.
# ---------------------------------------------------------------------------


class TestT5SoftComboReuse:
    def test_training_step_calls_soft_combo_loss_once_full_res_never_boundary_aux(
        self,
        shared_cache_dirs: tuple[Path, Path],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        train_cache, val_cache = shared_cache_dirs
        real_soft_combo = train_module.soft_combo_loss
        call_log: list[dict] = []

        def spy_soft_combo(logits, target, valid_mask=None, **kwargs):
            call_log.append(
                {
                    "logits_shape": tuple(logits.shape),
                    "target_shape": tuple(target.shape),
                    "valid_shape": (
                        tuple(valid_mask.shape) if valid_mask is not None else None
                    ),
                }
            )
            return real_soft_combo(logits, target, valid_mask=valid_mask, **kwargs)

        def forbidden_boundary_aux(*args, **kwargs):
            raise AssertionError(
                "B3 additive soft term must never call boundary_auxiliary_loss "
                "(edge-BCE) -- it reuses soft_combo_loss unmodified"
            )

        monkeypatch.setattr(train_module, "soft_combo_loss", spy_soft_combo)
        monkeypatch.setattr(
            train_module, "boundary_auxiliary_loss", forbidden_boundary_aux
        )

        config = _tiny_cache_config(
            train_cache=train_cache,
            val_cache=val_cache,
            run_root=tmp_path / "t5",
            soft_term_enabled=True,
            soft_term_weight=_BETA,
        )
        train_module.train(config)

        assert len(call_log) == 1, (
            "soft_combo_loss must be called exactly once per training step "
            "-- FULL-RES ONLY, no per-aux-scale calls (multi-scale soft "
            "supervision is deferred to a future B3.1)"
        )
        call = call_log[0]
        assert call["logits_shape"] == (1, 1, *IN_PLANE_HW)
        assert call["target_shape"] == (1, 1, *IN_PLANE_HW)
        assert call["valid_shape"] == (1, 1, *IN_PLANE_HW)

    def test_default_off_never_calls_soft_combo_loss(
        self,
        shared_cache_dirs: tuple[Path, Path],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        train_cache, val_cache = shared_cache_dirs

        def forbidden_soft_combo(*args, **kwargs):
            raise AssertionError(
                "soft_combo_loss must never be called when soft_term_enabled=False"
            )

        monkeypatch.setattr(train_module, "soft_combo_loss", forbidden_soft_combo)

        config = _tiny_cache_config(
            train_cache=train_cache, val_cache=val_cache, run_root=tmp_path / "t5_off"
        )
        assert config.soft_term_enabled is False
        train_module.train(config)  # must not raise / must not call the spy


# ---------------------------------------------------------------------------
# T6 -- CONFIG DELTA: b3_softdml differs from p4b2_deepsup only in
# {train_cache_dir, val_cache_dir, soft_term_enabled, soft_term_weight}.
# ---------------------------------------------------------------------------


class TestT6ConfigDelta:
    _EXPECTED_CHANGED_KEYS = {
        "train_cache_dir",
        "val_cache_dir",
        "soft_term_enabled",
        "soft_term_weight",
    }

    def test_b3_softdml_differs_from_p4b2_deepsup_only_in_declared_fields(
        self,
    ) -> None:
        b2 = _load_yaml("train_siamese_p4b2_deepsup.yaml")
        b3 = _load_yaml("train_siamese_b3_softdml.yaml")
        changed = {k for k in set(b2) | set(b3) if b2.get(k) != b3.get(k)}
        assert changed == self._EXPECTED_CHANGED_KEYS

    def test_b3_softdml_declared_field_values(self) -> None:
        b3 = _load_yaml("train_siamese_b3_softdml.yaml")
        assert b3["train_cache_dir"] == "data/processed/p6_cache_big_soft/train"
        assert b3["val_cache_dir"] == "data/processed/p6_cache_big_soft/val"
        assert b3["soft_term_enabled"] is True
        assert isinstance(b3["soft_term_weight"], (int, float))
        assert b3["soft_term_weight"] >= 0.0

    def test_b3_softdml_holds_deep_sup_selector_patience_loss_soft_target(
        self,
    ) -> None:
        b2 = _load_yaml("train_siamese_p4b2_deepsup.yaml")
        b3 = _load_yaml("train_siamese_b3_softdml.yaml")
        held_fields = [
            "deep_supervision",
            "deep_supervision_scales",
            "deep_supervision_weights",
            "constrained_iou_selection",
            "constrained_iou_min_precision",
            "constrained_iou_min_f1",
            "constrained_iou_min_clean",
            "early_stop_patience",
            "loss",
            "seed",
            "val_seed",
            "batch_size",
            "max_epochs",
            "lr",
            "weight_decay",
            "grad_clip_norm",
            "ema_decay",
            "selection_metric",
            "secondary_selection_metric",
            "augment",
            "hard_negative_mining",
        ]
        for field in held_fields:
            assert b2.get(field) == b3.get(field), f"field {field!r} must be HELD"
        # soft_target must stay unset/default False -- the additive path,
        # not the mutually-exclusive target-replacement path.
        assert b3.get("soft_target", False) is False
        assert b3.get("loss") == "combo"

    def test_b3_softdml_config_file_parses_into_valid_train_config(
        self, shared_cache_dirs: tuple[Path, Path]
    ) -> None:
        train_cache, val_cache = shared_cache_dirs
        payload = dict(_load_yaml("train_siamese_b3_softdml.yaml"))
        payload["train_bundle_dirs"] = ()
        payload["val_bundle_dirs"] = ()
        payload["train_cache_dir"] = train_cache
        payload["val_cache_dir"] = val_cache
        payload["run_root"] = "runs/_unused_b3_parse_check"
        payload.pop("model_config", None)
        payload.pop("dataset_config", None)
        config = TrainConfig(**payload)
        assert config.deep_supervision is True
        assert config.soft_term_enabled is True
        assert config.soft_target is False
        # soft_term_weight is beta*, set from the outcome-blind gradient-balance
        # probe (scripts/b3_grad_balance_probe.py) -- NOT a fixed literal. Assert
        # the contract (finite, strictly positive, and faithful to the YAML), so
        # re-running the probe cannot break this test on a value change alone.
        assert math.isfinite(config.soft_term_weight)
        assert config.soft_term_weight > 0.0
        assert config.soft_term_weight == pytest.approx(
            float(payload["soft_term_weight"])
        )


# ---------------------------------------------------------------------------
# T7 -- CONFIG VALIDATION.
# ---------------------------------------------------------------------------


class TestT7Validation:
    def test_negative_soft_term_weight_raises(self, tmp_path: Path) -> None:
        with pytest.raises(TrainConfigError, match=r"finite and >= 0"):
            _tiny_config(tmp_path=tmp_path, soft_term_weight=-0.1)

    def test_non_finite_soft_term_weight_raises(self, tmp_path: Path) -> None:
        with pytest.raises(TrainConfigError, match=r"finite and >= 0"):
            _tiny_config(tmp_path=tmp_path, soft_term_weight=float("nan"))

    def test_zero_soft_term_weight_is_valid_even_when_off(self, tmp_path: Path) -> None:
        config = _tiny_config(tmp_path=tmp_path, soft_term_weight=0.0)
        assert config.soft_term_weight == 0.0

    def test_soft_term_enabled_without_train_cache_dir_raises(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(TrainConfigError, match="requires train_cache_dir"):
            _tiny_config(tmp_path=tmp_path, soft_term_enabled=True)

    def test_soft_term_enabled_with_cache_missing_source_fraction_raises(
        self,
        no_soft_cache_dir: Path,
        shared_cache_dirs: tuple[Path, Path],
        tmp_path: Path,
    ) -> None:
        _, val_cache = shared_cache_dirs
        with pytest.raises(TrainConfigError, match="source_fraction propagation"):
            TrainConfig(
                train_bundle_dirs=(),
                val_bundle_dirs=(),
                train_cache_dir=no_soft_cache_dir,
                val_cache_dir=val_cache,
                run_root=tmp_path / "run",
                batch_size=1,
                device="cpu",
                soft_term_enabled=True,
                soft_term_weight=0.1,
            )

    def test_soft_term_enabled_with_real_source_fraction_cache_is_valid(
        self, shared_cache_dirs: tuple[Path, Path], tmp_path: Path
    ) -> None:
        train_cache, val_cache = shared_cache_dirs
        config = TrainConfig(
            train_bundle_dirs=(),
            val_bundle_dirs=(),
            train_cache_dir=train_cache,
            val_cache_dir=val_cache,
            run_root=tmp_path / "run2",
            batch_size=1,
            device="cpu",
            soft_term_enabled=True,
            soft_term_weight=0.1,
        )
        assert config.soft_term_enabled is True

    def test_soft_term_enabled_compatible_with_deep_supervision_but_soft_target_still_excluded(
        self, shared_cache_dirs: tuple[Path, Path], tmp_path: Path
    ) -> None:
        train_cache, val_cache = shared_cache_dirs
        ok = TrainConfig(
            train_bundle_dirs=(),
            val_bundle_dirs=(),
            train_cache_dir=train_cache,
            val_cache_dir=val_cache,
            run_root=tmp_path / "ok",
            batch_size=1,
            device="cpu",
            deep_supervision=True,
            soft_term_enabled=True,
            soft_term_weight=0.1,
        )
        assert ok.deep_supervision is True
        assert ok.soft_term_enabled is True

        with pytest.raises(TrainConfigError, match="soft_target=False"):
            TrainConfig(
                train_bundle_dirs=(),
                val_bundle_dirs=(),
                train_cache_dir=train_cache,
                val_cache_dir=val_cache,
                run_root=tmp_path / "bad",
                batch_size=1,
                device="cpu",
                deep_supervision=True,
                soft_target=True,
                loss="soft_combo",
                soft_term_enabled=True,
            )
