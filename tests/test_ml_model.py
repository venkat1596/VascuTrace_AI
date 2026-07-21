"""Tests for ``src.vascutrace.ml.model`` (VascuTrace Phase 6).

CPU-only, generated-tensor fixtures -- no ``Data/`` access anywhere in this
file. See ``src/vascutrace/ml/model.py``'s module docstring for the
architecture design rationale and paper citations these tests exercise.
"""

import time

import pytest
import torch

from src.vascutrace.ml.model import (
    RESEARCH_PROTOTYPE_WARNING,
    ModelConfig,
    SharedEncoder,
    build_model,
    dice_bce_loss,
    dice_score,
    model_signature,
)
from src.vascutrace.ml.tensor_schema import (
    IN_PLANE_HW,
    K,
    LEFT_VIEW_SHAPE,
    PET_DIFF_SHAPE,
    RIGHT_VIEW_SHAPE,
    TARGET_SHAPE,
)

H, W = IN_PLANE_HW


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _random_batch(
    batch_size: int, generator: torch.Generator
) -> tuple[torch.Tensor, ...]:
    """Random, schema-shaped ``(left_view, right_view, pet_diff, target)``."""
    left = torch.randn(batch_size, *LEFT_VIEW_SHAPE, generator=generator)
    right = torch.randn(batch_size, *RIGHT_VIEW_SHAPE, generator=generator)
    diff = torch.randn(batch_size, *PET_DIFF_SHAPE, generator=generator)
    target = (torch.rand(batch_size, *TARGET_SHAPE, generator=generator) > 0.9).float()
    return left, right, diff, target


def _make_easy_case(
    center_y: int, center_x: int, size: int
) -> tuple[torch.Tensor, ...]:
    """One fully-deterministic (no RNG) "easy" Siamese case: a smoothly
    varying, side-symmetric PET/CT background plus one square "hot" lesion
    patch inserted into ``left_view``'s PET channels only (``right_view``
    keeps the plain background, mimicking the physically-reflected
    contralateral side with no lesion). ``pet_diff`` is the exact
    left-minus-right PET difference the dataset builder would compute, and
    ``target`` marks the same patch on the center slice. This directly
    instantiates ``tensor_schema.py``'s framing: "A unilateral synthetic
    lesion breaks left/right symmetry, so it is visible in the paired
    views and their difference."

    Returns ``(left_view, right_view, pet_diff, target)``, each shaped per
    ``tensor_schema`` (no batch dim).
    """
    yy = torch.arange(H, dtype=torch.float32).view(H, 1) / H
    xx = torch.arange(W, dtype=torch.float32).view(1, W) / W
    pet_bg = 0.05 + 0.03 * torch.sin(2 * torch.pi * yy * 3) * torch.cos(
        2 * torch.pi * xx * 2
    )
    ct_bg = torch.full((H, W), 0.2) + 0.02 * torch.cos(2 * torch.pi * yy * 2)

    left_pet = pet_bg.unsqueeze(0).repeat(K, 1, 1).clone()
    right_pet = pet_bg.unsqueeze(0).repeat(K, 1, 1).clone()
    half = size // 2
    y0, y1 = center_y - half, center_y + half
    x0, x1 = center_x - half, center_x + half
    left_pet[:, y0:y1, x0:x1] += 0.6  # unilateral hot lesion, left side only

    left_ct = ct_bg.unsqueeze(0).repeat(K, 1, 1).clone()
    right_ct = ct_bg.unsqueeze(0).repeat(K, 1, 1).clone()

    left_view = torch.cat([left_pet, left_ct], dim=0)
    right_view = torch.cat([right_pet, right_ct], dim=0)
    pet_diff = left_pet - right_pet

    target = torch.zeros(*TARGET_SHAPE)
    target[0, y0:y1, x0:x1] = 1.0
    return left_view, right_view, pet_diff, target


def _stack_easy_cases() -> tuple[torch.Tensor, ...]:
    """The two fixed easy cases (different location/size) as one batch of 2."""
    case_a = _make_easy_case(center_y=60, center_x=30, size=12)
    case_b = _make_easy_case(center_y=90, center_x=50, size=10)
    left = torch.stack([case_a[0], case_b[0]])
    right = torch.stack([case_a[1], case_b[1]])
    diff = torch.stack([case_a[2], case_b[2]])
    target = torch.stack([case_a[3], case_b[3]])
    return left, right, diff, target


# ---------------------------------------------------------------------------
# 1. Schema conformance
# ---------------------------------------------------------------------------


def test_forward_schema_conformance() -> None:
    gen = torch.Generator().manual_seed(0)
    left, right, _diff, _target = _random_batch(batch_size=2, generator=gen)
    assert left.shape == (2, *LEFT_VIEW_SHAPE)
    assert right.shape == (2, *RIGHT_VIEW_SHAPE)

    model = build_model()
    model.eval()
    with torch.no_grad():
        logits = model(left, right)

    assert logits.shape == (2, *TARGET_SHAPE)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


def test_forward_schema_conformance_with_pet_diff() -> None:
    gen = torch.Generator().manual_seed(1)
    left, right, diff, _target = _random_batch(batch_size=2, generator=gen)
    assert diff.shape == (2, *PET_DIFF_SHAPE)

    model = build_model()
    model.eval()
    with torch.no_grad():
        logits = model(left, right, diff)

    assert logits.shape == (2, *TARGET_SHAPE)
    assert torch.isfinite(logits).all()


def test_forward_rejects_malformed_input_shape() -> None:
    model = build_model()
    bad_left = torch.randn(2, LEFT_VIEW_SHAPE[0] + 1, *IN_PLANE_HW)
    right = torch.randn(2, *RIGHT_VIEW_SHAPE)
    with pytest.raises(ValueError):
        model(bad_left, right)


# ---------------------------------------------------------------------------
# 2. Finite forward/loss/backward/gradients
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_pet_diff", [True, False])
def test_finite_loss_and_full_gradient_coverage(use_pet_diff: bool) -> None:
    gen = torch.Generator().manual_seed(2)
    left, right, diff, target = _random_batch(batch_size=2, generator=gen)

    model = build_model(ModelConfig(seed=20260716))
    model.train()

    logits = model(left, right, diff if use_pet_diff else None)
    loss = dice_bce_loss(logits, target)

    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)

    loss.backward()

    param_names = [name for name, _ in model.named_parameters()]
    assert len(param_names) > 0

    none_grad_names = [name for name, p in model.named_parameters() if p.grad is None]
    assert none_grad_names == [], (
        f"parameters with no gradient (unused-parameter surprise): {none_grad_names}"
    )

    non_finite_grad_names = [
        name
        for name, p in model.named_parameters()
        if p.grad is not None and not torch.isfinite(p.grad).all()
    ]
    assert non_finite_grad_names == [], (
        f"parameters with NaN/Inf gradient: {non_finite_grad_names}"
    )


# ---------------------------------------------------------------------------
# 3. Determinism
# ---------------------------------------------------------------------------


def test_same_seed_same_input_yields_identical_output() -> None:
    cfg = ModelConfig(seed=20260716, base_channels=8)
    model_a = build_model(cfg)
    model_b = build_model(cfg)

    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    assert state_a.keys() == state_b.keys()
    for name in state_a:
        assert torch.equal(state_a[name], state_b[name]), (
            f"seeded init mismatch at {name}"
        )

    gen = torch.Generator().manual_seed(999)
    left, right, diff, _target = _random_batch(batch_size=2, generator=gen)

    model_a.eval()
    model_b.eval()

    # Single-threaded to remove any theoretical run-to-run nondeterminism
    # from multi-threaded CPU reduction ordering, isolating the assertion
    # to genuine weight/architecture determinism.
    prior_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.no_grad():
            out_a = model_a(left, right, diff)
            out_b = model_b(left, right, diff)
    finally:
        torch.set_num_threads(prior_threads)

    assert torch.allclose(out_a, out_b, atol=1e-5, rtol=1e-5)


def test_different_seed_yields_different_weights() -> None:
    model_a = build_model(ModelConfig(seed=1, base_channels=8))
    model_b = build_model(ModelConfig(seed=2, base_channels=8))
    state_a, state_b = model_a.state_dict(), model_b.state_dict()
    any_different = any(
        not torch.equal(state_a[name], state_b[name]) for name in state_a
    )
    assert any_different, "different seeds produced identical weights"


# ---------------------------------------------------------------------------
# 4. Weight sharing (the encoder is genuinely one shared instance)
# ---------------------------------------------------------------------------


def test_exactly_one_encoder_instance() -> None:
    model = build_model()
    encoder_instances = [m for m in model.modules() if isinstance(m, SharedEncoder)]
    assert len(encoder_instances) == 1, (
        "expected exactly one SharedEncoder submodule (Siamese branches must "
        f"share one set of weights); found {len(encoder_instances)}"
    )
    assert encoder_instances[0] is model.encoder


def test_encoder_is_called_once_per_branch_with_shared_weights() -> None:
    model = build_model()
    model.eval()

    calls: list[torch.Tensor] = []

    def _hook(_module: torch.nn.Module, inputs: tuple, _output: object) -> None:
        calls.append(inputs[0])

    handle = model.encoder.register_forward_hook(_hook)
    try:
        gen = torch.Generator().manual_seed(3)
        left, right, _diff, _target = _random_batch(batch_size=2, generator=gen)
        with torch.no_grad():
            model(left, right)
    finally:
        handle.remove()

    # The single shared nn.Module instance must be invoked exactly twice --
    # once per Siamese branch -- through model.encode() (see model.py,
    # SiameseBilateralUNet.forward).
    assert len(calls) == 2
    assert torch.equal(calls[0], left)
    assert torch.equal(calls[1], right)


def test_identical_inputs_produce_identical_encoder_features() -> None:
    """If left_view == right_view exactly, the shared encoder -- being a
    pure, deterministic function called with the *same* parameters for
    both branches -- must produce bit-identical feature pyramids for both
    branches. This is only true if there is genuinely one shared weight
    set, not two independently-initialized encoders that happen to start
    from the same seed (which test_same_seed_same_input_yields_identical_
    output alone would not distinguish from architecture-level sharing).
    """
    model = build_model()
    model.eval()

    gen = torch.Generator().manual_seed(4)
    x = torch.randn(2, *LEFT_VIEW_SHAPE, generator=gen)

    with torch.no_grad():
        feats_left = model.encode(x)
        feats_right = model.encode(x)

    assert len(feats_left) == len(feats_right) == 5
    for level, (f_l, f_r) in enumerate(zip(feats_left, feats_right, strict=True)):
        assert torch.equal(f_l, f_r), (
            f"encoder feature mismatch on identical input at level {level}"
        )


# ---------------------------------------------------------------------------
# 5. Easy-overfit gate (critical)
# ---------------------------------------------------------------------------


def test_easy_overfit_gate_reaches_dice_0_95_within_1000_steps() -> None:
    """Per the project's learned-model promotion contract
    ("Learned-model promotion": "Stop architecture work if two fixed easy
    cases do not reach Dice at least 0.95 within 1,000 steps"), this test
    IS that gate. A small-but-real instance of the exact same
    SiameseBilateralUNet architecture (base_channels=4 instead of the
    default 16, purely for CPU wall-clock speed -- no architectural
    change) must drive both fixed easy cases to Dice >= 0.95 within 1000
    Adam steps, and in well under the "keep it CPU-fast" budget.
    """
    left, right, diff, target = _stack_easy_cases()

    model = build_model(ModelConfig(seed=20260716, base_channels=4))
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    max_steps = 1000
    check_every = 10
    dice_case_a = dice_case_b = 0.0
    start = time.time()
    step = 0
    for step in range(1, max_steps + 1):
        optimizer.zero_grad()
        logits = model(left, right, diff)
        loss = dice_bce_loss(logits, target)
        loss.backward()
        optimizer.step()

        if step % check_every == 0:
            with torch.no_grad():
                dice_case_a = dice_score(logits[0:1], target[0:1]).item()
                dice_case_b = dice_score(logits[1:2], target[1:2]).item()
            if dice_case_a >= 0.95 and dice_case_b >= 0.95:
                break
    elapsed = time.time() - start

    assert dice_case_a >= 0.95, (
        f"case A only reached Dice {dice_case_a:.4f} after {step} steps"
    )
    assert dice_case_b >= 0.95, (
        f"case B only reached Dice {dice_case_b:.4f} after {step} steps"
    )
    assert step <= max_steps
    assert elapsed < 25.0, (
        f"easy-overfit gate took {elapsed:.1f}s, exceeding the CPU-fast budget"
    )


# ---------------------------------------------------------------------------
# 6. Loss masking
# ---------------------------------------------------------------------------


def test_valid_mask_excludes_invalid_voxels_from_loss() -> None:
    gen = torch.Generator().manual_seed(5)
    logits = torch.randn(2, *TARGET_SHAPE, generator=gen)
    target_a = (torch.rand(2, *TARGET_SHAPE, generator=gen) > 0.7).float()

    # target_b differs from target_a ONLY inside the excluded patch.
    target_b = target_a.clone()
    y0, y1, x0, x1 = 40, 60, 20, 40
    target_b[:, :, y0:y1, x0:x1] = 1.0 - target_a[:, :, y0:y1, x0:x1]

    valid_mask = torch.ones(2, *TARGET_SHAPE)
    valid_mask[:, :, y0:y1, x0:x1] = 0.0

    loss_a_masked = dice_bce_loss(logits, target_a, valid_mask=valid_mask)
    loss_b_masked = dice_bce_loss(logits, target_b, valid_mask=valid_mask)
    assert torch.allclose(loss_a_masked, loss_b_masked, atol=1e-6), (
        "corrupting only the masked-out region changed the masked loss -- valid_mask is not excluding it"
    )

    loss_a_unmasked = dice_bce_loss(logits, target_a)
    loss_b_unmasked = dice_bce_loss(logits, target_b)
    assert not torch.allclose(loss_a_unmasked, loss_b_unmasked, atol=1e-4), (
        "corrupting the region had no effect even without masking -- test fixture is not discriminating"
    )


def test_valid_mask_all_zero_is_finite_not_nan() -> None:
    gen = torch.Generator().manual_seed(6)
    logits = torch.randn(2, *TARGET_SHAPE, generator=gen)
    target = (torch.rand(2, *TARGET_SHAPE, generator=gen) > 0.5).float()
    valid_mask = torch.zeros(2, *TARGET_SHAPE)

    loss = dice_bce_loss(logits, target, valid_mask=valid_mask)
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Supporting API surface: dice_score, model_signature, compactness, boundary
# ---------------------------------------------------------------------------


def test_dice_score_perfect_and_empty_cases() -> None:
    target = torch.zeros(1, *TARGET_SHAPE)
    target[0, 0, 10:20, 10:20] = 1.0

    perfect_logits = torch.where(target > 0, torch.tensor(10.0), torch.tensor(-10.0))
    assert dice_score(perfect_logits, target).item() == pytest.approx(1.0, abs=1e-4)

    empty_target = torch.zeros(1, *TARGET_SHAPE)
    empty_logits = torch.full_like(empty_target, -10.0)
    assert dice_score(empty_logits, empty_target).item() == pytest.approx(1.0, abs=1e-6)


def test_dice_score_from_logits_flag_matches_manual_sigmoid() -> None:
    gen = torch.Generator().manual_seed(7)
    logits = torch.randn(2, *TARGET_SHAPE, generator=gen)
    target = (torch.rand(2, *TARGET_SHAPE, generator=gen) > 0.5).float()

    from_logits_score = dice_score(logits, target, from_logits=True)
    score = torch.sigmoid(logits)
    from_score_value = dice_score(score, target, from_logits=False)
    assert torch.allclose(from_logits_score, from_score_value)


def test_model_signature_stable_and_config_sensitive() -> None:
    sig_default_a = model_signature()
    sig_default_b = model_signature(ModelConfig())
    assert sig_default_a == sig_default_b

    sig_other = model_signature(ModelConfig(base_channels=32))
    assert sig_other != sig_default_a

    # seed must NOT affect the signature (architecture-only provenance).
    sig_seed_a = model_signature(ModelConfig(seed=1))
    sig_seed_b = model_signature(ModelConfig(seed=2))
    assert sig_seed_a == sig_seed_b == sig_default_a


def test_model_is_compact() -> None:
    model = build_model()
    total_params = sum(p.numel() for p in model.parameters())
    assert 0 < total_params < 8_000_000, (
        f"model has {total_params} params -- expected a compact (few-M) model"
    )


def test_research_prototype_warning_is_reused_not_duplicated() -> None:
    from src.vascutrace.geometry import RESEARCH_PROTOTYPE_WARNING as GEOMETRY_WARNING

    assert RESEARCH_PROTOTYPE_WARNING is GEOMETRY_WARNING
    assert "simulated vascular-like" in RESEARCH_PROTOTYPE_WARNING
