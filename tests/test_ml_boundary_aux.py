"""Unit tests for the boundary-local A/B/C auxiliary-loss contract.

INDEPENDENCE NOTE: every assertion below is derived from the frozen formulas
and formulas alone -- ``F_i``/``H_i``/``M_i``/``W_i`` support definition,
per-sample normalization, exact-zero-on-no-support safety, the fixed
``batch_size`` batch reduction, and the B/C target-only isolation. This
file does NOT read ``src/vascutrace/ml/losses.py``'s internals to decide
what to assert -- only its public signature (``boundary_auxiliary_loss``/
``BoundaryAuxiliaryLoss``) is used as an entry point.

Research prototype -- these tests exercise a synthetic, EXPLORATORY
-- NOT CERTIFIABLE auxiliary loss (the boundary experiment §1/R8); they make no clinical
or performance claim.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from src.vascutrace.ml.dataset import Sample
from src.vascutrace.ml.losses import BoundaryAuxiliaryLoss, boundary_auxiliary_loss
from src.vascutrace.ml.train import _collate_samples, _iter_val_batches

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_DIR = _ROOT / "configs"


def _load_config(name: str) -> dict[str, object]:
    payload = yaml.safe_load((_CONFIG_DIR / name).read_text())
    assert isinstance(payload, dict)
    return payload


def _bce_with_logits(z: float, g: float) -> float:
    """Reference BCE-with-logits value, computed independently of the
    implementation, for building fixtures with known per-pixel loss
    values (plan §5's ``BCEWithLogits(z_ij, G_ij)``)."""
    import math

    return max(z, 0.0) - z * g + math.log(1.0 + math.exp(-abs(z)))


class TestSupport:
    """§5: ``W_i = M_i * 1[0 < F_i < 1]`` -- only valid, genuinely
    fractional pixels are support."""

    def test_boundary_count_matches_valid_and_strictly_fractional_pixels(self) -> None:
        # 8 pixels laid out 1x1x2x4 (B=1, C=1, H=2, W=4).
        f = torch.tensor([[[[0.3, 0.0, 1.0, 0.7], [0.5, 0.2, 0.0, 1.0]]]])
        valid = torch.tensor([[[[1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0]]]])
        # Expected support: (0,0)=0.3 valid -> yes; (0,1)=0.0 -> no (F==0);
        # (0,2)=1.0 -> no (F==1); (0,3)=0.7 valid -> yes;
        # (1,0)=0.5 but INVALID -> no; (1,1)=0.2 valid -> yes;
        # (1,2)=0.0 -> no; (1,3)=1.0 -> no.
        expected_support_count = 3
        target_mask = (f >= 0.5).to(torch.float32)
        logits = torch.zeros_like(f)

        result = boundary_auxiliary_loss(
            logits, f, target_mask, valid, target_mode="hard"
        )

        assert isinstance(result, BoundaryAuxiliaryLoss)
        assert result.boundary_count.item() == pytest.approx(expected_support_count)

    def test_valid_mask_zero_excludes_fractional_pixel(self) -> None:
        # A single fractional pixel (F=0.4) but valid_mask=0 must not count.
        f = torch.tensor([[[[0.4]]]])
        valid = torch.tensor([[[[0.0]]]])
        target_mask = (f >= 0.5).to(torch.float32)
        logits = torch.zeros_like(f)

        result = boundary_auxiliary_loss(
            logits, f, target_mask, valid, target_mode="hard"
        )
        assert result.boundary_count.item() == 0.0

    def test_exact_zero_and_one_fraction_excluded_even_when_valid(self) -> None:
        f = torch.tensor([[[[0.0, 1.0]]]])
        valid = torch.ones_like(f)
        target_mask = (f >= 0.5).to(torch.float32)
        logits = torch.zeros_like(f)

        result = boundary_auxiliary_loss(
            logits, f, target_mask, valid, target_mode="hard"
        )
        assert result.boundary_count.item() == 0.0


class TestExactZeroOnNoSupport:
    """§5: 'exact differentiable zero' when a sample's support is empty --
    the most important correctness invariant."""

    def test_empty_support_sample_yields_exact_zero_loss(self) -> None:
        # Single-sample batch, all pixels F in {0, 1} -> empty support.
        f = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
        valid = torch.ones_like(f)
        target_mask = (f >= 0.5).to(torch.float32)
        logits = torch.randn_like(f)

        result = boundary_auxiliary_loss(
            logits, f, target_mask, valid, target_mode="fraction"
        )

        assert result.loss.item() == 0.0

    def test_empty_support_sample_yields_finite_zero_gradient(self) -> None:
        f = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
        valid = torch.ones_like(f)
        target_mask = (f >= 0.5).to(torch.float32)
        logits = torch.randn_like(f, requires_grad=True)

        result = boundary_auxiliary_loss(
            logits, f, target_mask, valid, target_mode="fraction"
        )
        result.loss.backward()

        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()
        assert torch.equal(logits.grad, torch.zeros_like(logits.grad))

    def test_empty_support_sample_within_mixed_batch_still_exact_zero_and_finite(
        self,
    ) -> None:
        # A batch of two samples: sample 0 has support, sample 1 has none.
        # Verify sample 1's own contribution path stays finite (no NaN
        # propagation into the batch-reduced loss/gradient).
        f = torch.tensor(
            [
                [[[0.3, 0.7], [0.6, 0.2]]],  # sample 0: has fractional pixels
                [[[0.0, 1.0], [1.0, 0.0]]],  # sample 1: no fractional pixels
            ]
        )
        valid = torch.ones_like(f)
        target_mask = (f >= 0.5).to(torch.float32)
        logits = torch.randn_like(f, requires_grad=True)

        result = boundary_auxiliary_loss(
            logits, f, target_mask, valid, target_mode="fraction"
        )

        assert torch.isfinite(result.loss)
        result.loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()


class TestFixedBatchSizeDenominator:
    """§5: 'Reduce with the same final batch denominator as the hard
    objective... An ineligible sample contributes exact zero auxiliary, so
    the effective auxiliary weight cannot silently increase when a batch
    happens to contain fewer boundary-bearing samples.'"""

    def _build_batch(self, n_ineligible_extra: int) -> tuple[torch.Tensor, ...]:
        # Sample 0: has support with nonzero BCE (logit=1, hard target=1
        # at a fractional pixel -> BCE != 0). Sample 1: no support.
        base_f = [
            [[[0.7, 0.2]]],  # sample 0: fractional pixels -> support
            [[[0.0, 1.0]]],  # sample 1: no support
        ]
        base_logits = [
            [[[1.0, 1.0]]],
            [[[1.0, 1.0]]],
        ]
        for _ in range(n_ineligible_extra):
            base_f.append([[[0.0, 1.0]]])
            base_logits.append([[[1.0, 1.0]]])

        f = torch.tensor(base_f)
        logits = torch.tensor(base_logits)
        valid = torch.ones_like(f)
        target_mask = (f >= 0.5).to(torch.float32)
        return logits, f, target_mask, valid

    def test_adding_ineligible_sample_strictly_decreases_reduced_aux(self) -> None:
        logits_n, f_n, h_n, m_n = self._build_batch(n_ineligible_extra=0)
        logits_n1, f_n1, h_n1, m_n1 = self._build_batch(n_ineligible_extra=1)

        loss_n = boundary_auxiliary_loss(
            logits_n, f_n, h_n, m_n, target_mode="hard"
        ).loss
        loss_n1 = boundary_auxiliary_loss(
            logits_n1, f_n1, h_n1, m_n1, target_mode="hard"
        ).loss

        assert loss_n.item() > 0.0, "fixture must have nonzero eligible contribution"
        assert loss_n1.item() < loss_n.item(), (
            "appending an all-ineligible sample must strictly shrink the "
            "batch-reduced aux (fixed batch_size denominator grows while "
            "the numerator is unchanged)"
        )


class TestTargetStrategyIsolation:
    """§5 candidate matrix (plan §4): B uses ``target_mask``, C uses
    ``source_fraction``; identical support/weighting/normalization, only
    the target field differs."""

    def test_hard_and_fraction_targets_differ_on_a_genuine_boundary_pixel(self) -> None:
        # F=0.7 -> H=1 (hard rounds to 1). Use logit=1.0 (chosen to avoid
        # z=0's BCE-is-target-invariant degeneracy) so hard-target BCE and
        # fraction-target BCE are numerically different.
        f = torch.tensor([[[[0.7]]]])
        valid = torch.ones_like(f)
        target_mask = (f >= 0.5).to(torch.float32)
        logits = torch.tensor([[[[1.0]]]])

        hard_result = boundary_auxiliary_loss(
            logits, f, target_mask, valid, target_mode="hard"
        )
        frac_result = boundary_auxiliary_loss(
            logits, f, target_mask, valid, target_mode="fraction"
        )

        expected_hard = _bce_with_logits(1.0, 1.0)
        expected_frac = _bce_with_logits(1.0, 0.7)
        assert expected_hard != pytest.approx(expected_frac)
        assert hard_result.loss.item() == pytest.approx(expected_hard, abs=1e-5)
        assert frac_result.loss.item() == pytest.approx(expected_frac, abs=1e-5)
        assert hard_result.loss.item() != pytest.approx(frac_result.loss.item())

    def test_hard_and_fraction_coincide_when_support_is_empty(self) -> None:
        f = torch.tensor([[[[0.0, 1.0]]]])
        valid = torch.ones_like(f)
        target_mask = (f >= 0.5).to(torch.float32)
        logits = torch.randn_like(f)

        hard_result = boundary_auxiliary_loss(
            logits, f, target_mask, valid, target_mode="hard"
        )
        frac_result = boundary_auxiliary_loss(
            logits, f, target_mask, valid, target_mode="fraction"
        )

        assert hard_result.loss.item() == 0.0
        assert frac_result.loss.item() == 0.0
        assert hard_result.loss.item() == frac_result.loss.item()


class TestPerSampleNormalization:
    """§5: ``L_aux_i = numerator_i / denominator_i`` -- normalized by that
    SAMPLE's own support count, not by crop size, so a tiny-lesion sample
    is not diluted relative to a large one."""

    def test_same_per_pixel_bce_different_support_size_same_l_aux(self) -> None:
        # Sample A: a small 1x2 crop, both pixels in support, same BCE
        # value each (logit=1, hard target=1).
        f_a = torch.tensor([[[[0.7, 0.6]]]])
        logits_a = torch.tensor([[[[1.0, 1.0]]]])
        valid_a = torch.ones_like(f_a)
        target_a = (f_a >= 0.5).to(torch.float32)

        # Sample B: a bigger 2x3 crop, 6 pixels ALL in support, same
        # per-pixel BCE value (same logit=1, same hard target=1).
        f_b = torch.tensor([[[[0.6, 0.7, 0.55], [0.9, 0.51, 0.65]]]])
        logits_b = torch.full_like(f_b, 1.0)
        valid_b = torch.ones_like(f_b)
        target_b = (f_b >= 0.5).to(torch.float32)

        loss_a = boundary_auxiliary_loss(
            logits_a, f_a, target_a, valid_a, target_mode="hard"
        ).loss
        loss_b = boundary_auxiliary_loss(
            logits_b, f_b, target_b, valid_b, target_mode="hard"
        ).loss

        # Both single-sample (batch_size=1) calls, so the returned batch
        # -reduced loss IS that sample's own L_aux_i.
        assert loss_a.item() == pytest.approx(loss_b.item(), abs=1e-5)
        assert loss_a.item() == pytest.approx(_bce_with_logits(1.0, 1.0), abs=1e-5)


class TestConfigIsolation:
    """§5/§4: B and C differ ONLY in ``boundary_aux_target``; A has the
    auxiliary off; all three share the soft cache and the unchanged hard
    ``combo`` primary objective."""

    def test_b_and_c_differ_in_exactly_one_key(self) -> None:
        b = _load_config("train_siamese_p3_B.yaml")
        c = _load_config("train_siamese_p3_C.yaml")

        assert set(b.keys()) == set(c.keys())
        differing_keys = {k for k in b if b[k] != c[k]}
        assert differing_keys == {"boundary_aux_target"}
        assert b["boundary_aux_target"] == "hard"
        assert c["boundary_aux_target"] == "fraction"

    def test_a_disables_aux_and_otherwise_matches_b_and_c(self) -> None:
        a = _load_config("train_siamese_p3_A.yaml")
        b = _load_config("train_siamese_p3_B.yaml")
        c = _load_config("train_siamese_p3_C.yaml")

        assert a["boundary_aux_target"] == "none"

        exempt = {"lambda_boundary", "boundary_aux_target"}
        shared_a = {k: v for k, v in a.items() if k not in exempt}
        shared_b = {k: v for k, v in b.items() if k not in exempt}
        shared_c = {k: v for k, v in c.items() if k not in exempt}
        assert shared_a == shared_b == shared_c

    def test_all_three_arms_on_soft_cache_hard_combo_primary_objective(self) -> None:
        for name in (
            "train_siamese_p3_A.yaml",
            "train_siamese_p3_B.yaml",
            "train_siamese_p3_C.yaml",
        ):
            cfg = _load_config(name)
            assert "soft" in str(cfg["train_cache_dir"])
            assert "soft" in str(cfg["val_cache_dir"])
            assert cfg["loss"] == "combo"
            assert cfg["soft_target"] is False


class TestEvalIsolation:
    """§5/train.py module docstring item 17: ``source_fraction`` must
    never reach the validation path (hard-mask-only evaluation
    invariant) -- a leak would let the fractional field silently
    influence checkpoint selection, which the boundary experiment §5's frozen contrasts
    (C-A/C-B/B-A) never authorize."""

    @staticmethod
    def _make_sample(seed: float) -> Sample:
        h, w, k = 4, 4, 1
        return Sample(
            left_view=torch.zeros((2 * k, h, w)),
            right_view=torch.zeros((2 * k, h, w)),
            pet_diff=torch.zeros((k, h, w)),
            target_mask=torch.zeros((1, h, w)),
            source_fraction=torch.full((1, h, w), seed),
            valid_mask=torch.ones((1, h, w)),
            raw_pet=torch.zeros((1, h, w)),
            meta={"positive": False},
        )

    def test_collate_default_excludes_source_fraction(self) -> None:
        samples = [self._make_sample(0.3), self._make_sample(0.6)]
        batch = _collate_samples(samples)
        assert "source_fraction" not in batch

    def test_iter_val_batches_never_yields_source_fraction(self) -> None:
        samples = [
            self._make_sample(0.3),
            self._make_sample(0.6),
            self._make_sample(0.9),
        ]
        saw_any_batch = False
        for batch in _iter_val_batches(samples, batch_size=2):
            saw_any_batch = True
            assert "source_fraction" not in batch, (
                "source_fraction must never reach the validation path "
                "(hard-mask-only evaluation invariant, the boundary experiment -- train.py "
                "module docstring item 17)"
            )
        assert saw_any_batch
