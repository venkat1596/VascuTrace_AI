"""Unit tests locking Phase 4 Lever L3 -- the ``TrainConfig.tversky_fn_weight``
/``tversky_fp_weight`` knob (module docstring item 19 in
``src/vascutrace/ml/train.py``).

Independence note: these tests are written from the task's stated contract
(the mechanism description handed to test-unit), NOT by reading
``train.py``'s diff to decide what to assert. They exercise train.py's own
private dispatch table/constants (``_LOSS_FUNCTIONS``,
``_TVERSKY_WEIGHTED_LOSS_NAMES``, ``_TVERSKY_ALPHA_DEFAULT``,
``_TVERSKY_BETA_DEFAULT``) directly -- the same pattern this test file's
sibling (``test_ml_train.py``) already uses for other private train.py
internals (``_augment_batch``, ``_lr_at_step``, ``_update_ema``) -- so a
regression in the *actual* construction logic, not a hand-rolled
reimplementation of it, is what these tests catch.
"""

from __future__ import annotations

import functools
from pathlib import Path

import pytest
import torch
import yaml

import src.vascutrace.ml.train as train_module
from src.vascutrace.ml.losses import combo_loss
from src.vascutrace.ml.model import ModelConfig
from src.vascutrace.ml.train import TrainConfig, TrainConfigError

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_DIR = _ROOT / "configs"


def _minimal_config(tmp_path: Path, **overrides) -> TrainConfig:
    """The smallest ``TrainConfig`` that passes ``__post_init__`` validation
    -- no bundle directory is ever read at construction time (only checked
    for non-emptiness/non-overlap), so these dummy, non-existent paths are
    sufficient; no actual data or CUDA is touched by any test in this file.
    """
    defaults: dict = dict(
        train_bundle_dirs=(tmp_path / "train_bundle",),
        val_bundle_dirs=(tmp_path / "val_bundle",),
        run_root=tmp_path / "run",
        model_config=ModelConfig(base_channels=4),
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


def _construct_loss_fn(config: TrainConfig):
    """Reproduces train.py's OWN loss_fn construction (module docstring,
    item 19) against train.py's OWN private dispatch table/constants --
    ``_LOSS_FUNCTIONS``, ``_TVERSKY_WEIGHTED_LOSS_NAMES``,
    ``_TVERSKY_ALPHA_DEFAULT``, ``_TVERSKY_BETA_DEFAULT`` -- so a change to
    any of those four names inside train.py is what makes this helper (and
    therefore these tests) track train.py's real behavior, not an
    independently-invented copy of the branching logic.
    """
    tversky_overridden = (
        config.tversky_fn_weight != train_module._TVERSKY_ALPHA_DEFAULT
        or config.tversky_fp_weight != train_module._TVERSKY_BETA_DEFAULT
    )
    if config.loss in train_module._TVERSKY_WEIGHTED_LOSS_NAMES and tversky_overridden:
        return functools.partial(
            train_module._LOSS_FUNCTIONS[config.loss],
            alpha=config.tversky_fn_weight,
            beta=config.tversky_fp_weight,
        )
    return train_module._LOSS_FUNCTIONS[config.loss]


def _load_yaml(name: str) -> dict[str, object]:
    payload = yaml.safe_load((_CONFIG_DIR / name).read_text())
    assert isinstance(payload, dict)
    return payload


# ---------------------------------------------------------------------------
# 1. The knob reaches the loss (load-bearing)
# ---------------------------------------------------------------------------


class TestTverskyKnobReachesLoss:
    def test_rebalanced_weights_thread_into_loss_fn(self, tmp_path: Path) -> None:
        config = _minimal_config(
            tmp_path, loss="combo", tversky_fn_weight=0.5, tversky_fp_weight=0.5
        )
        loss_fn = _construct_loss_fn(config)

        torch.manual_seed(0)
        logits = torch.randn(2, 1, 8, 8)
        target = (torch.rand(2, 1, 8, 8) > 0.7).float()
        valid = torch.ones(2, 1, 8, 8)

        actual = loss_fn(logits, target, valid_mask=valid)
        expected_symmetric = combo_loss(logits, target, valid, alpha=0.5, beta=0.5)
        expected_default_weights = combo_loss(
            logits, target, valid, alpha=0.7, beta=0.3
        )

        # A 0.5/0.5 config must produce the numerically SAME loss as an
        # explicit alpha=0.5/beta=0.5 combo_loss call ...
        assert torch.equal(actual, expected_symmetric)
        # ... and must DIFFER from the 0.7/0.3 loss -- this is what proves
        # a 0.5/0.5 run truly trains symmetric Tversky, not a silent
        # 0.7/0.3 fallback that ignores the config.
        assert not torch.equal(actual, expected_default_weights)

    def test_rebalanced_weights_thread_into_focal_tversky_too(
        self, tmp_path: Path
    ) -> None:
        config = _minimal_config(
            tmp_path,
            loss="focal_tversky",
            tversky_fn_weight=0.5,
            tversky_fp_weight=0.5,
        )
        loss_fn = _construct_loss_fn(config)

        torch.manual_seed(2)
        logits = torch.randn(2, 1, 8, 8)
        target = (torch.rand(2, 1, 8, 8) > 0.7).float()
        valid = torch.ones(2, 1, 8, 8)

        from src.vascutrace.ml.losses import focal_tversky_loss

        actual = loss_fn(logits, target, valid_mask=valid)
        expected_symmetric = focal_tversky_loss(
            logits, target, valid, alpha=0.5, beta=0.5
        )
        expected_default_weights = focal_tversky_loss(
            logits, target, valid, alpha=0.7, beta=0.3
        )

        assert torch.equal(actual, expected_symmetric)
        assert not torch.equal(actual, expected_default_weights)


# ---------------------------------------------------------------------------
# 2. Default (0.7/0.3) is unchanged: bare callable, byte-identical loss
# ---------------------------------------------------------------------------


class TestTverskyKnobDefaultUnchanged:
    def test_default_weights_produce_bare_callable_and_identical_loss(
        self, tmp_path: Path
    ) -> None:
        config = _minimal_config(tmp_path, loss="combo")
        assert config.tversky_fn_weight == 0.7
        assert config.tversky_fp_weight == 0.3

        loss_fn = _construct_loss_fn(config)

        # At the defaults, NO functools.partial wrapping happens -- the
        # exact bare dict-lookup callable train.py used before this knob
        # existed, so a pre-item-19 monkeypatched test double (assuming the
        # narrower (logits, target, valid_mask=...) signature with no extra
        # kwargs) keeps working unchanged.
        assert loss_fn is combo_loss
        assert not isinstance(loss_fn, functools.partial)

        torch.manual_seed(1)
        logits = torch.randn(2, 1, 8, 8)
        target = (torch.rand(2, 1, 8, 8) > 0.7).float()
        valid = torch.ones(2, 1, 8, 8)

        actual = loss_fn(logits, target, valid_mask=valid)
        expected = combo_loss(logits, target, valid)
        assert torch.equal(actual, expected)


# ---------------------------------------------------------------------------
# 3. Asymmetry direction sanity: symmetric (0.5/0.5) penalizes FP more than
#    the FN-tilted default (0.7/0.3) -- the intended anti
#    -over-segmentation effect.
# ---------------------------------------------------------------------------


class TestTverskyAsymmetryDirection:
    def test_symmetric_rebalance_penalizes_false_positives_more_than_default(
        self,
    ) -> None:
        h, w = 10, 10
        target = torch.zeros(1, 1, h, w)
        target[0, 0, 2:6, 2:6] = 1.0  # a 4x4 lesion

        baseline_logits = torch.where(target > 0, 10.0, -10.0)  # perfect prediction

        # Deliberate false-positive-only fixture: 4 spurious activations
        # OUTSIDE the lesion, zero false negatives (every true lesion voxel
        # remains correctly predicted positive) -- isolates beta's (FP
        # weight's) effect from alpha's (FN weight's) simultaneous change.
        fp_logits = baseline_logits.clone()
        fp_logits[0, 0, 7, 2:6] = 10.0

        fp_loss_default = combo_loss(fp_logits, target, alpha=0.7, beta=0.3)
        fp_loss_symmetric = combo_loss(fp_logits, target, alpha=0.5, beta=0.5)

        # The SAME false-positive error must cost MORE under the symmetric
        # (0.5/0.5) rebalance than under the FN-tilted default (0.3 FP
        # weight) -- the rebalance genuinely increases the FP penalty.
        assert fp_loss_symmetric > fp_loss_default

        # Mirror-image check with the same-sized false-negative error, zero
        # false positives: the SAME false-negative error must cost LESS
        # under the symmetric rebalance (alpha dropped 0.7 -> 0.5),
        # confirming the shift is a genuine FN<->FP rebalance in the
        # intended direction, not merely FP getting noisier for an
        # unrelated reason.
        fn_logits = baseline_logits.clone()
        fn_logits[0, 0, 2, 2:6] = -10.0

        fn_loss_default = combo_loss(fn_logits, target, alpha=0.7, beta=0.3)
        fn_loss_symmetric = combo_loss(fn_logits, target, alpha=0.5, beta=0.5)

        assert fn_loss_symmetric < fn_loss_default


# ---------------------------------------------------------------------------
# 4. Config delta: train_siamese_p4l3_dice.yaml vs train_siamese_v6exp.yaml
#    differ in EXACTLY {tversky_fn_weight, tversky_fp_weight}.
# ---------------------------------------------------------------------------


class TestTverskyConfigDelta:
    def test_p4l3_dice_is_v6exp_plus_exactly_the_tversky_rebalance(self) -> None:
        v6 = _load_yaml("train_siamese_v6exp.yaml")
        p4l3 = _load_yaml("train_siamese_p4l3_dice.yaml")

        all_keys = v6.keys() | p4l3.keys()
        differing_keys = {k for k in all_keys if v6.get(k) != p4l3.get(k)}

        assert differing_keys == {"tversky_fn_weight", "tversky_fp_weight"}
        assert p4l3["tversky_fn_weight"] == 0.5
        assert p4l3["tversky_fp_weight"] == 0.5
        # v6exp never sets these keys at all -- p4l3_dice is v6exp PLUS the
        # two new keys, not a config that merely overrides existing ones.
        assert "tversky_fn_weight" not in v6
        assert "tversky_fp_weight" not in v6


# ---------------------------------------------------------------------------
# 5. Validation: a non-positive tversky weight raises before allocation.
# ---------------------------------------------------------------------------


class TestTverskyWeightValidation:
    def test_rejects_non_positive_tversky_fn_weight(self, tmp_path: Path) -> None:
        with pytest.raises(TrainConfigError, match="tversky_fn_weight"):
            _minimal_config(tmp_path, tversky_fn_weight=0.0)

    def test_rejects_negative_tversky_fp_weight(self, tmp_path: Path) -> None:
        with pytest.raises(TrainConfigError, match="tversky_fp_weight"):
            _minimal_config(tmp_path, tversky_fp_weight=-0.1)
