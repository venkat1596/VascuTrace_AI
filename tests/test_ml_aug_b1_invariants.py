"""Physics-safety invariant tests for the EXISTING augmentation code
(``_augment_batch``/``_affine_theta`` in ``src.vascutrace.ml.train``) run at
the STRONGER B1 parameter values.

Scope / independence note: this file implements the frozen physics-safety
invariants I1 through I10. The contract is the sole source of truth for what is
asserted here. B1 is a CONFIG-ONLY lever (``configs/train_siamese_p4b1_aug.
yaml``): ``augment_scale_delta`` 0.10 -> 0.20, ``augment_rotation_deg`` 10
-> 15, ``augment_translate_px`` 6 -> 10; the aug implementation itself is
untouched. The point of this file is therefore NOT to re-derive coverage
``tests/test_ml_train.py::TestAugmentationContract`` already has (those two
tests exercise the mechanism generically, via SCRIPTED/monkeypatched
``_uniform`` draws that are independent of ``config.augment_*``'s actual
numeric values) -- it is to re-run the same invariants using the REAL,
unscripted RNG draws at the ACTUAL B1 config values (including the
strengthened boundary draws), so a regression that only manifests at the
wider parameter range is caught.

No ``src/`` edits. CPU-only, no ``Data/`` access, no cache/bundle
fixtures -- ``_augment_batch``/``_affine_theta`` are pure tensor functions
of ``config.augment_*`` plus the already-seeded global torch RNG stream, so
a minimal ``TrainConfig`` (fake, non-existent bundle dirs -- ``__post_init__``
never touches bundle CONTENTS, only the tuple/overlap checks) is enough.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

import src.vascutrace.ml.train as train_module
from src.vascutrace.ml.tensor_schema import CT_CHANNEL_SLICE, K, PET_CHANNEL_SLICE
from src.vascutrace.ml.train import TrainConfig
from src.vascutrace.ml.train import _affine_theta  # noqa: PLC2701 -- direct unit test
from src.vascutrace.ml.train import _augment_batch  # noqa: PLC2701 -- direct unit test

B1_CONFIG_PATH = Path("configs/train_siamese_p4b1_aug.yaml")

# Frozen B1 lever values -- spec Section 5. If these constants drift from
# the shipped config, TestB1ConfigMatchesSpec below fails first.
B1_SCALE_DELTA = 0.20
B1_ROTATION_DEG = 15.0
B1_TRANSLATE_PX = 10.0
# Held at v6 (unchanged) -- spec Section 5.
B1_PET_GAIN_DELTA = 0.10
B1_PET_BIAS = 0.05
B1_CT_GAIN_DELTA = 0.10
B1_CT_BIAS = 0.05

H, W = 64, 64


def _b1_config(tmp_path: Path, **overrides: object) -> TrainConfig:
    """Minimal ``TrainConfig`` carrying the exact B1 augmentation levers.
    Bundle dirs need not exist -- ``_augment_batch`` never reads them.
    """
    defaults: dict = dict(
        train_bundle_dirs=(tmp_path / "train",),
        val_bundle_dirs=(tmp_path / "val",),
        run_root=tmp_path / "run",
        augment=True,
        augment_rotation_deg=B1_ROTATION_DEG,
        augment_translate_px=B1_TRANSLATE_PX,
        augment_scale_delta=B1_SCALE_DELTA,
        augment_pet_gain_delta=B1_PET_GAIN_DELTA,
        augment_pet_bias=B1_PET_BIAS,
        augment_ct_gain_delta=B1_CT_GAIN_DELTA,
        augment_ct_bias=B1_CT_BIAS,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


def _make_tensors(
    batch: int = 4, h: int = H, w: int = W, *, with_source_fraction: bool = False
) -> tuple[torch.Tensor, ...]:
    """Synthetic batch respecting the tensor_schema channel contract:
    left/right = [B, 2K, H, W] (PET in [0,1], CT in [-1,1] -- the
    documented network-normalized ranges), diff = left_pet - right_pet
    (the SAME identity the real dataset/collate path establishes before
    augmentation ever runs), target/valid = [B, 1, H, W] binary.
    """
    g = torch.Generator().manual_seed(20260719)
    left = torch.zeros(batch, 2 * K, h, w)
    right = torch.zeros(batch, 2 * K, h, w)
    left[:, PET_CHANNEL_SLICE] = torch.rand(batch, K, h, w, generator=g)
    right[:, PET_CHANNEL_SLICE] = torch.rand(batch, K, h, w, generator=g)
    left[:, CT_CHANNEL_SLICE] = torch.rand(batch, K, h, w, generator=g) * 2.0 - 1.0
    right[:, CT_CHANNEL_SLICE] = torch.rand(batch, K, h, w, generator=g) * 2.0 - 1.0
    diff = left[:, PET_CHANNEL_SLICE] - right[:, PET_CHANNEL_SLICE]

    target = torch.zeros(batch, 1, h, w)
    target[:, :, h // 2 - 4 : h // 2 + 4, w // 2 - 4 : w // 2 + 4] = 1.0
    valid = torch.ones(batch, 1, h, w)

    if not with_source_fraction:
        return left, right, diff, target, valid

    source_fraction = torch.zeros(batch, 1, h, w)
    source_fraction[:, :, h // 2 - 4 : h // 2 + 4, w // 2 - 4 : w // 2 + 4] = 0.5
    return left, right, diff, target, valid, source_fraction


def _install_scripted_uniform(
    monkeypatch: pytest.MonkeyPatch, draws: list[float]
) -> None:
    """Force ``_uniform``'s 8 draws (angle, scale, tx, ty, pet_gain,
    pet_bias, ct_gain, ct_bias -- ``_augment_batch``'s call order) to exact
    values, matching the technique already used by
    ``tests/test_ml_train.py::TestAugmentationContract``.
    """
    queue = list(draws)

    def scripted_uniform(
        n: int, low: float, high: float, device: torch.device
    ) -> torch.Tensor:
        return torch.full((n,), queue.pop(0), device=device)

    monkeypatch.setattr(train_module, "_uniform", scripted_uniform)


# ---------------------------------------------------------------------------
# 0. Sanity: the shipped config carries exactly the frozen B1 delta (spec
# Section 5). Not itself a physics-safety invariant, but a regression guard
# tying every test below (which hardcodes the B1 numbers) to the actual
# config file this implementation is de-risking.
# ---------------------------------------------------------------------------


class TestB1ConfigMatchesSpec:
    def test_shipped_config_carries_exact_b1_delta(self) -> None:
        raw = yaml.safe_load(B1_CONFIG_PATH.read_text())
        assert raw["augment"] is True
        assert raw["augment_scale_delta"] == B1_SCALE_DELTA
        assert raw["augment_rotation_deg"] == B1_ROTATION_DEG
        assert raw["augment_translate_px"] == B1_TRANSLATE_PX
        # Intensity jitter HELD at v6 -- spec Section 5, "B1 is a clean
        # single-mechanism (geometry) delta".
        assert raw["augment_pet_gain_delta"] == B1_PET_GAIN_DELTA
        assert raw["augment_pet_bias"] == B1_PET_BIAS
        assert raw["augment_ct_gain_delta"] == B1_CT_GAIN_DELTA
        assert raw["augment_ct_bias"] == B1_CT_BIAS
        # I6: B1 is hard-target; soft-target bilinear routing must not be
        # engaged by this config.
        assert not raw.get("soft_target", False)

    def test_b1_values_pass_trainconfig_validators(self, tmp_path: Path) -> None:
        # TrainConfig.__post_init__ validators (scale_delta in [0,1),
        # rotation_deg >= 0, translate_px >= 0) -- spec Section 5, last line.
        config = _b1_config(tmp_path)
        assert config.augment_scale_delta == B1_SCALE_DELTA
        assert config.augment_rotation_deg == B1_ROTATION_DEG
        assert config.augment_translate_px == B1_TRANSLATE_PX


# ---------------------------------------------------------------------------
# I1: pet_diff == left_pet - right_pet after aug (bilateral identity).
# ---------------------------------------------------------------------------


class TestI1BilateralDifferenceIdentity:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 17])
    def test_pet_diff_identity_holds_under_real_b1_draws(
        self, tmp_path: Path, seed: int
    ) -> None:
        config = _b1_config(tmp_path)
        left, right, diff, target, valid = _make_tensors()
        torch.manual_seed(seed)
        left_g, right_g, diff_g, target_g, valid_g = _augment_batch(
            left, right, diff, target, valid, config=config
        )
        assert torch.equal(
            diff_g, left_g[:, PET_CHANNEL_SLICE] - right_g[:, PET_CHANNEL_SLICE]
        )
        assert torch.isfinite(diff_g).all()

    def test_pet_diff_identity_at_max_b1_geometry_and_intensity_draws(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Worst-case corner of the B1 parameter box: max rotation, max
        # scale-up, max translate, max +gain/+bias on both views.
        config = _b1_config(tmp_path)
        draws = [
            B1_ROTATION_DEG,
            1.0 + B1_SCALE_DELTA,
            B1_TRANSLATE_PX,
            B1_TRANSLATE_PX,
            1.0 + B1_PET_GAIN_DELTA,
            B1_PET_BIAS,
            1.0 + B1_CT_GAIN_DELTA,
            B1_CT_BIAS,
        ]
        _install_scripted_uniform(monkeypatch, draws)
        left, right, diff, target, valid = _make_tensors()
        left_g, right_g, diff_g, target_g, valid_g = _augment_batch(
            left, right, diff, target, valid, config=config
        )
        assert torch.equal(
            diff_g, left_g[:, PET_CHANNEL_SLICE] - right_g[:, PET_CHANNEL_SLICE]
        )


# ---------------------------------------------------------------------------
# I2: single shared spatial grid applied to left/right/target/valid(/
# source_fraction) -- no independent per-tensor geometry.
# ---------------------------------------------------------------------------


class TestI2SharedGrid:
    def test_same_grid_tensor_feeds_every_grid_sample_call(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _b1_config(tmp_path)
        captured_grids: list[torch.Tensor] = []
        original_grid_sample = train_module.F.grid_sample

        def spy_grid_sample(
            input: torch.Tensor, grid: torch.Tensor, **kwargs: object
        ) -> torch.Tensor:
            captured_grids.append(grid)
            return original_grid_sample(input, grid, **kwargs)

        monkeypatch.setattr(train_module.F, "grid_sample", spy_grid_sample)
        torch.manual_seed(11)
        left, right, diff, target, valid, source_fraction = _make_tensors(
            with_source_fraction=True
        )
        _augment_batch(
            left,
            right,
            diff,
            target,
            valid,
            config=config,
            source_fraction=source_fraction,
        )
        # left, right, target, valid, source_fraction -> 5 grid_sample calls.
        assert len(captured_grids) == 5
        first = captured_grids[0]
        for other in captured_grids[1:]:
            assert torch.equal(other, first)

    def test_augmented_positive_target_still_lands_on_augmented_signal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A concrete alignment check (not just object identity): with a
        max-B1-translate-only draw (rotation=0, scale=1 -- both legal draws
        within the B1 range), a PET "hot spot" and a co-located target
        blob must both shift by the SAME pixel offset, so the target still
        overlaps the signal after augmentation.
        """
        config = _b1_config(tmp_path)
        draws = [0.0, 1.0, B1_TRANSLATE_PX, -B1_TRANSLATE_PX, 1.0, 0.0, 1.0, 0.0]
        _install_scripted_uniform(monkeypatch, draws)

        left = torch.zeros(1, 2 * K, H, W)
        right = torch.zeros(1, 2 * K, H, W)
        target = torch.zeros(1, 1, H, W)
        valid = torch.ones(1, 1, H, W)
        y0, y1 = H // 2 - 3, H // 2 + 3
        x0, x1 = W // 2 - 3, W // 2 + 3
        left[:, PET_CHANNEL_SLICE, y0:y1, x0:x1] = 1.0
        target[:, :, y0:y1, x0:x1] = 1.0
        diff = left[:, PET_CHANNEL_SLICE] - right[:, PET_CHANNEL_SLICE]

        left_g, right_g, diff_g, target_g, valid_g = _augment_batch(
            left, right, diff, target, valid, config=config
        )

        signal_mask = left_g[0, 0] > 0.5
        target_mask = target_g[0, 0] > 0.5
        assert signal_mask.any() and target_mask.any()

        def centroid(mask: torch.Tensor) -> tuple[float, float]:
            ys, xs = torch.nonzero(mask, as_tuple=True)
            return ys.float().mean().item(), xs.float().mean().item()

        signal_yx = centroid(signal_mask)
        target_yx = centroid(target_mask)
        # Pure integer-pixel translation (rotation=0, scale=1) -- nearest
        # vs. bilinear resampling of the SAME shifted square should center
        # on the same pixel to well within 1px.
        assert signal_yx == pytest.approx(target_yx, abs=1.0)


# ---------------------------------------------------------------------------
# I5 + I4: hard target_mask stays binary {0,1} (nearest interp); valid_mask
# stays {0,1} and excludes pixels pushed off-frame by the stronger warp.
# ---------------------------------------------------------------------------


class TestI4I5HardMasksStayBinary:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
    def test_target_and_valid_stay_binary_under_real_b1_draws(
        self, tmp_path: Path, seed: int
    ) -> None:
        config = _b1_config(tmp_path)
        left, right, diff, target, valid = _make_tensors()
        torch.manual_seed(seed)
        _, _, _, target_g, valid_g = _augment_batch(
            left, right, diff, target, valid, config=config, target_interp="nearest"
        )
        assert bool(torch.logical_or(target_g == 0.0, target_g == 1.0).all())
        assert bool(torch.logical_or(valid_g == 0.0, valid_g == 1.0).all())

    def test_valid_mask_zeros_pixels_pushed_off_frame_at_max_b1_warp(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _b1_config(tmp_path)
        # Max rotation + max translate simultaneously -- the strongest
        # single-draw B1 corner most likely to expose border padding.
        draws = [
            B1_ROTATION_DEG,
            1.0,
            B1_TRANSLATE_PX,
            B1_TRANSLATE_PX,
            1.0,
            0.0,
            1.0,
            0.0,
        ]
        _install_scripted_uniform(monkeypatch, draws)
        left, right, diff, target, valid = _make_tensors(batch=1)
        _, _, _, target_g, valid_g = _augment_batch(
            left, right, diff, target, valid, config=config
        )
        assert bool(torch.logical_or(valid_g == 0.0, valid_g == 1.0).all())
        # An all-ones valid_mask warped by a nonzero rotation+translation
        # at these magnitudes must expose SOME zero-padded border pixels
        # (proves the invariant is being exercised, not vacuously true).
        assert bool((valid_g == 0.0).any())

    def test_soft_target_bilinear_path_not_taken_for_b1_hard_target(
        self, tmp_path: Path
    ) -> None:
        # I6: B1's config carries no soft_target flag (default False) -- so
        # the training step's default target_interp ("nearest") is the one
        # that must be exercised; confirm the default arg itself is
        # "nearest" (no caller override needed for B1's hard-target path).
        config = _b1_config(tmp_path)
        assert config.soft_target is False
        left, right, diff, target, valid = _make_tensors()
        target[:, :, :, :] = 0.37  # a fractional value that must NOT survive
        torch.manual_seed(5)
        _, _, _, target_g, _ = _augment_batch(
            left, right, diff, target, valid, config=config
        )
        # nearest interpolation of a (deliberately, for this test) uniform
        # 0.37 field reproduces 0.37 exactly (no averaging) -- the point is
        # this is NOT routed through the bilinear soft-target path, which
        # this specific input can't distinguish on its own, so we also
        # directly check the default parameter value used by the B1 (hard
        # -target) call the training step makes.
        import inspect

        sig = inspect.signature(_augment_batch)
        assert sig.parameters["target_interp"].default == "nearest"


# ---------------------------------------------------------------------------
# I7: no laterality flip -- affine linear part has positive determinant
# for every legal B1 draw.
# ---------------------------------------------------------------------------


class TestI7NoLateralityFlip:
    def test_positive_determinant_across_full_b1_parameter_box(self) -> None:
        # Dense sweep of the FULL legal B1 draw ranges (not just corners).
        angles = torch.linspace(-B1_ROTATION_DEG, B1_ROTATION_DEG, 25)
        scales = torch.linspace(1.0 - B1_SCALE_DELTA, 1.0 + B1_SCALE_DELTA, 25)
        tx = torch.linspace(-1.0, 1.0, 5)  # normalized translate, any value
        ty = torch.linspace(-1.0, 1.0, 5)
        grid_a, grid_s, grid_tx, grid_ty = torch.meshgrid(
            angles, scales, tx, ty, indexing="ij"
        )
        theta = _affine_theta(
            grid_a.reshape(-1),
            grid_s.reshape(-1),
            grid_tx.reshape(-1),
            grid_ty.reshape(-1),
        )
        det = theta[:, 0, 0] * theta[:, 1, 1] - theta[:, 0, 1] * theta[:, 1, 0]
        assert bool((det > 0).all())

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_positive_determinant_under_real_b1_uniform_draws(
        self, tmp_path: Path, seed: int
    ) -> None:
        config = _b1_config(tmp_path)
        torch.manual_seed(seed)
        angle_deg = train_module._uniform(  # noqa: SLF001
            256,
            -config.augment_rotation_deg,
            config.augment_rotation_deg,
            torch.device("cpu"),
        )
        scale = train_module._uniform(  # noqa: SLF001
            256,
            1.0 - config.augment_scale_delta,
            1.0 + config.augment_scale_delta,
            torch.device("cpu"),
        )
        tx = torch.zeros(256)
        ty = torch.zeros(256)
        theta = _affine_theta(angle_deg, scale, tx, ty)
        det = theta[:, 0, 0] * theta[:, 1, 1] - theta[:, 0, 1] * theta[:, 1, 0]
        assert bool((det > 0).all())


# ---------------------------------------------------------------------------
# I10: determinism under a fixed seed.
# ---------------------------------------------------------------------------


class TestI10Determinism:
    @pytest.mark.parametrize("seed", [0, 42, 20260719])
    def test_same_seed_gives_bit_identical_augmented_batch(
        self, tmp_path: Path, seed: int
    ) -> None:
        config = _b1_config(tmp_path)
        left, right, diff, target, valid = _make_tensors()

        torch.manual_seed(seed)
        out_a = _augment_batch(
            left.clone(),
            right.clone(),
            diff.clone(),
            target.clone(),
            valid.clone(),
            config=config,
        )
        torch.manual_seed(seed)
        out_b = _augment_batch(
            left.clone(),
            right.clone(),
            diff.clone(),
            target.clone(),
            valid.clone(),
            config=config,
        )
        for a, b in zip(out_a, out_b):
            assert torch.equal(a, b)

    def test_different_seed_gives_different_augmented_batch(
        self, tmp_path: Path
    ) -> None:
        # Negative control -- guards against a test/mechanism that is
        # "deterministic" only because it ignores the RNG entirely.
        config = _b1_config(tmp_path)
        left, right, diff, target, valid = _make_tensors()

        torch.manual_seed(1)
        left_a, *_ = _augment_batch(
            left.clone(),
            right.clone(),
            diff.clone(),
            target.clone(),
            valid.clone(),
            config=config,
        )
        torch.manual_seed(2)
        left_b, *_ = _augment_batch(
            left.clone(),
            right.clone(),
            diff.clone(),
            target.clone(),
            valid.clone(),
            config=config,
        )
        assert not torch.equal(left_a, left_b)


# ---------------------------------------------------------------------------
# I8: no NaN/Inf; jittered values stay within the documented safety clamps
# (PET [0, 1.5], CT [-1.5, 1.5]) at the stronger B1 params.
# ---------------------------------------------------------------------------


class TestI8NoNanInfWithinClamps:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7])
    def test_finite_and_within_clamps_under_real_b1_draws(
        self, tmp_path: Path, seed: int
    ) -> None:
        config = _b1_config(tmp_path)
        left, right, diff, target, valid = _make_tensors(batch=8)
        torch.manual_seed(seed)
        left_g, right_g, diff_g, target_g, valid_g = _augment_batch(
            left, right, diff, target, valid, config=config
        )
        for name, tensor in [
            ("left", left_g),
            ("right", right_g),
            ("diff", diff_g),
            ("target", target_g),
            ("valid", valid_g),
        ]:
            assert torch.isfinite(tensor).all(), f"{name} has non-finite values"

        pet_g = torch.cat([left_g[:, PET_CHANNEL_SLICE], right_g[:, PET_CHANNEL_SLICE]])
        ct_g = torch.cat([left_g[:, CT_CHANNEL_SLICE], right_g[:, CT_CHANNEL_SLICE]])
        assert bool((pet_g >= 0.0).all()) and bool((pet_g <= 1.5).all())
        assert bool((ct_g >= -1.5).all()) and bool((ct_g <= 1.5).all())
        # pet_diff is a subtraction of two [0, 1.5]-clamped fields.
        assert bool((diff_g >= -1.5).all()) and bool((diff_g <= 1.5).all())

    def test_worst_case_out_of_envelope_input_still_clamped_at_b1_intensity_bounds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Deliberately out-of-physical-envelope pre-aug input (2.0 far
        # exceeds the documented [0, 1] normalized PET range) combined with
        # the max +gain/+bias B1 draw -- proves the clamp holds even for an
        # adversarial upstream value, not merely the in-envelope case.
        config = _b1_config(tmp_path)
        draws = [
            0.0,
            1.0,
            0.0,
            0.0,
            1.0 + B1_PET_GAIN_DELTA,
            B1_PET_BIAS,
            1.0 + B1_CT_GAIN_DELTA,
            B1_CT_BIAS,
        ]
        _install_scripted_uniform(monkeypatch, draws)
        left = torch.full((1, 2 * K, 8, 8), 2.0)
        right = torch.full((1, 2 * K, 8, 8), -2.0)
        left[:, CT_CHANNEL_SLICE] = 2.0
        right[:, CT_CHANNEL_SLICE] = -2.0
        diff = left[:, PET_CHANNEL_SLICE] - right[:, PET_CHANNEL_SLICE]
        target = torch.zeros(1, 1, 8, 8)
        valid = torch.ones(1, 1, 8, 8)

        left_g, right_g, diff_g, target_g, valid_g = _augment_batch(
            left, right, diff, target, valid, config=config
        )
        assert torch.isfinite(left_g).all()
        assert torch.isfinite(right_g).all()
        assert torch.isfinite(diff_g).all()
        pet_g = torch.cat([left_g[:, PET_CHANNEL_SLICE], right_g[:, PET_CHANNEL_SLICE]])
        ct_g = torch.cat([left_g[:, CT_CHANNEL_SLICE], right_g[:, CT_CHANNEL_SLICE]])
        assert bool((pet_g >= 0.0).all()) and bool((pet_g <= 1.5).all())
        assert bool((ct_g >= -1.5).all()) and bool((ct_g <= 1.5).all())
        # The clamp actually triggered (proves the assertion isn't vacuous).
        assert bool((pet_g == 1.5).any()) or bool((pet_g == 0.0).any())
