"""Phase 4 lever B3 -- outcome-blind gradient-balance probe (soft_term_weight).

Research prototype. Trained and evaluated using simulated vascular-like
abnormalities, not confirmed human post-angioplasty lesions.

EXPLORATORY -- NOT CERTIFIABLE. Implements the frozen beta-selection rule:

    "beta is set by the outcome-blind gradient-balance probe: beta* =
    0.5 / r_median where r = |g_soft|/|g_hard| (hard = the full B2 total)
    at fixed init on real train batches; accept iff scaled median in
    [0.2, 0.8]. (A valid beta exists by linear scaling unless r_median is
    0/non-finite -> then report + stop.)"

At FIXED init (``model_init_seed``, ``sampler_seed`` below -- the SAME two
constants ``scripts/p3_lambda_probe.py`` uses, for continuity across these
gradient probes), in ``model.train()`` mode, float32, AMP OFF, NO
optimizer step, on real train batches drawn from a cache with
``has_source_fraction=True`` (e.g. ``data/processed/p6_cache_big_soft/train``),
this script builds the production model with ``ModelConfig(deep_supervision=
True)`` and computes, per batch, w.r.t. the SAME shared decoder+head
parameter subset ``train.py``'s own online drift monitor uses
(``train._shared_decoder_head_params`` -- ``up4``/``up3``/``up2``/``up1``/
``diff_stem``/``diff_fuse``/``head``):

    g_hard = grad of [combo_loss(logits, target_mask, valid_mask)
                       + sum_i weight_i * combo_loss(aux_logits_i,
                             maxpool(target_mask, i), minpool(valid_mask, i))]
             w.r.t. shared decoder+head params            # the full B2 total
    g_soft = grad of soft_combo_loss(logits, source_fraction, valid_mask)
             w.r.t. shared decoder+head params             # RAW, pre-beta

    r = |g_soft| / max(|g_hard|, 1e-12)

Reports ``r_median`` (plus IQR/p95 as non-gating diagnostics) and
``beta_star = 0.5 / r_median``, which satisfies ``beta_star * r_median =
0.5`` EXACTLY (i.e. the "scaled median in [0.2, 0.8]" acceptance check is a
tautology once ``beta_star`` is defined at all. It is reported as an
explicit sanity re-check rather than an
implicit one). If ``r_median`` is exactly ``0.0`` or non-finite (no
qualifying batch produced two comparable, non-degenerate gradients), this
script prints ``B3_PROBE: NO_QUALIFYING_BETA`` and stops -- it never
silently substitutes a fallback beta.

DEVIATIONS FROM `scripts/p3_lambda_probe.py`'s PROTOCOL (documented
honestly, per this project's own convention -- an engineering-grade probe
is acceptable):

  * p3's protocol draws a STRATIFIED 16-boundary-bearing + 16-zero
    -boundary batch composition (the boundary experiment Sec 5's own table). The B3 Root
    ruling's probe instruction states no such composition requirement --
    only "real train batches from p6_cache_big_soft" -- so this script
    draws batches via a PLAIN seeded permutation over the whole cache (no
    positive/negative stratification), taking ``n_batches * batch_size``
    samples without replacement (or fewer, if the cache is smaller,
    reporting the shortfall rather than padding/repeating silently).
  * p3 computes separate head-only and decoder-only ratios (against two
    distinct thresholds). This probe computes ONE ratio against ONE
    combined "shared decoder+head" parameter set, as specified by the B3
    protocol and to match ``train.py``'s own online drift-monitor
    denominator exactly (so the offline probe number and the online
    per-step number are directly comparable).
  * p3 sweeps a FIXED, discrete lambda grid and selects the smallest
    qualifying candidate. B3's beta is instead solved for in closed form
    (``beta_star = 0.5 / r_median``) -- there is no grid to sweep; the
    B3 protocol describes this as "a valid beta exists by
    linear scaling," not a grid search.
  * ``model_init_seed``/``sampler_seed`` default to the SAME two
    constants p3 uses (``20260721``/``20260720``) for continuity and
    reproducibility.

Usage (run only after the operator authorizes local data and compute use):

    uv run python -m scripts.b3_grad_balance_probe \\
        --train-cache-dir data/processed/p6_cache_big_soft/train \\
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.vascutrace.geometry import RESEARCH_PROTOTYPE_WARNING
from src.vascutrace.ml.cache import CachedSampleDataset
from src.vascutrace.ml.dataset import Sample
from src.vascutrace.ml.losses import combo_loss, soft_combo_loss
from src.vascutrace.ml.model import ModelConfig, build_model
from src.vascutrace.ml.train import (  # noqa: PLC2701 -- direct reuse, documented deviation note above
    _max_pool_hard_target,
    _min_pool_valid_mask,
    _safe_grad_l2_norm,
    _shared_decoder_head_params,
    seed_everything,
)

# Same two constants scripts/p3_lambda_probe.py uses -- see module
# docstring, deviation notes.
DEFAULT_MODEL_INIT_SEED = 20260721
DEFAULT_SAMPLER_SEED = 20260720
DEFAULT_N_BATCHES = 16
DEFAULT_BATCH_SIZE = 32

# The B3 protocol's frozen deep-supervision scale/weight pair (matches
# TrainConfig's own defaults and configs/train_siamese_p4b2_deepsup.yaml /
# configs/train_siamese_b3_softdml.yaml, both of which HOLD these values
# unchanged from B2).
DEEP_SUP_SCALES: tuple[int, ...] = (2, 4)
DEEP_SUP_WEIGHTS: tuple[float, ...] = (0.5, 0.25)

# B3 protocol: "accept iff scaled median in [0.2, 0.8]" -- see module
# docstring for why this is a tautology once beta_star is defined at all
# (beta_star * r_median == 0.5 exactly by construction); reported as an
# explicit re-check rather than left implicit.
ACCEPT_LOW = 0.2
ACCEPT_HIGH = 0.8
TARGET_SCALED_MEDIAN = 0.5


def _stack_batch(
    samples: list[Sample], device: torch.device
) -> dict[str, torch.Tensor]:
    """Same tensor keys :func:`train._collate_samples` produces (plus
    ``source_fraction``, always) -- a small standalone re-implementation,
    mirroring ``scripts/p3_lambda_probe.py``'s own ``_stack_batch``, so
    this script has no import-time dependency on that private function.
    """
    return {
        "left_view": torch.stack([s.left_view for s in samples]).to(device),
        "right_view": torch.stack([s.right_view for s in samples]).to(device),
        "pet_diff": torch.stack([s.pet_diff for s in samples]).to(device),
        "target_mask": torch.stack([s.target_mask for s in samples]).to(device),
        "valid_mask": torch.stack([s.valid_mask for s in samples]).to(device),
        "source_fraction": torch.stack([s.source_fraction for s in samples]).to(device),
    }


def _build_batches(
    dataset: CachedSampleDataset,
    *,
    sampler_seed: int,
    n_batches: int,
    batch_size: int,
    device: torch.device,
) -> tuple[list[dict[str, torch.Tensor]], dict[str, Any]]:
    """Up to ``n_batches`` fixed batches of ``batch_size`` samples each,
    drawn WITHOUT replacement via a seeded permutation over the whole
    cache -- see module docstring, deviation notes (no positive/negative
    stratification, unlike p3's own protocol).
    """
    rng = np.random.default_rng(sampler_seed)
    order = rng.permutation(len(dataset))

    max_full_batches = len(order) // max(1, batch_size)
    achievable_batches = min(n_batches, max_full_batches)

    composition: dict[str, Any] = {
        "requested_n_batches": n_batches,
        "requested_batch_size": batch_size,
        "n_samples_available": len(order),
        "achieved_n_batches": achievable_batches,
        "shortfall": achievable_batches < n_batches,
    }

    batches: list[dict[str, torch.Tensor]] = []
    for b in range(achievable_batches):
        idx = order[b * batch_size : (b + 1) * batch_size]
        samples = [dataset[int(i)] for i in idx]
        batches.append(_stack_batch(samples, device))

    return batches, composition


def _percentile_higher(values: list[float], q: float) -> float:
    """Empirical percentile with ``method="higher"`` -- same convention
    ``scripts/p3_lambda_probe.py`` uses (verified against this project's
    installed numpy version, which supports the ``method`` kwarg)."""
    return float(
        np.percentile(np.asarray(values, dtype=np.float64), q, method="higher")
    )


def run_probe(
    *,
    train_cache_dir: Path,
    device: torch.device,
    model_init_seed: int = DEFAULT_MODEL_INIT_SEED,
    sampler_seed: int = DEFAULT_SAMPLER_SEED,
    n_batches: int = DEFAULT_N_BATCHES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    deep_sup_scales: tuple[int, ...] = DEEP_SUP_SCALES,
    deep_sup_weights: tuple[float, ...] = DEEP_SUP_WEIGHTS,
) -> dict[str, Any]:
    """Run the full probe; returns the JSON-serializable report dict this
    module's ``main()`` prints. Kept separate from ``main()`` so a caller
    (or a test) can invoke it directly without going through argparse/
    stdout.
    """
    dataset = CachedSampleDataset(train_cache_dir)
    batches, composition = _build_batches(
        dataset,
        sampler_seed=sampler_seed,
        n_batches=n_batches,
        batch_size=batch_size,
        device=device,
    )
    if not batches:
        raise RuntimeError(
            "b3_grad_balance_probe: zero usable batches -- the cache at "
            f"{train_cache_dir} does not contain at least one batch of "
            f"{batch_size} samples (composition={composition!r})"
        )

    # Identical initialization for every batch -- the B3 protocol's "fixed
    # init on real train batches." Built ONCE, never re-seeded or updated
    # (no optimizer step -- the B3 protocol's "no optimizer step").
    seed_everything(model_init_seed)
    model = build_model(ModelConfig(deep_supervision=True)).to(device)
    model.train()

    shared_params = _shared_decoder_head_params(model)

    per_batch: list[dict[str, Any]] = []
    for batch in batches:
        logits, aux_logits_by_scale = model(
            batch["left_view"], batch["right_view"], batch["pet_diff"], return_aux=True
        )

        # g_hard -- the full B2 total: hard main + hard deep-sup aux terms,
        # the SAME combo_loss call at each scale train.py's own training
        # step uses (module docstring, "hard = the full B2 total").
        hard_total = combo_loss(
            logits, batch["target_mask"], valid_mask=batch["valid_mask"]
        )
        for scale, weight in zip(deep_sup_scales, deep_sup_weights, strict=True):
            aux_target = _max_pool_hard_target(batch["target_mask"], scale)
            aux_valid = _min_pool_valid_mask(batch["valid_mask"], scale)
            aux_term = combo_loss(
                aux_logits_by_scale[scale], aux_target, valid_mask=aux_valid
            )
            hard_total = hard_total + weight * aux_term

        # g_soft -- the RAW (pre-beta) B3 additive term, full-resolution
        # only, exactly as train.py's own training step computes it.
        soft_term = soft_combo_loss(
            logits, batch["source_fraction"], valid_mask=batch["valid_mask"]
        )

        hard_grad = torch.autograd.grad(
            hard_total, shared_params, retain_graph=True, allow_unused=True
        )
        hard_norm = _safe_grad_l2_norm(hard_grad)
        soft_grad = torch.autograd.grad(
            soft_term, shared_params, retain_graph=True, allow_unused=True
        )
        soft_norm = _safe_grad_l2_norm(soft_grad)

        if hard_norm is None or soft_norm is None:
            ratio = math.nan
        else:
            ratio = soft_norm / max(hard_norm, 1e-12)

        per_batch.append(
            {
                "hard_total_norm": hard_norm,
                "soft_norm": soft_norm,
                "ratio": ratio,
            }
        )

    finite_ratios = [r["ratio"] for r in per_batch if math.isfinite(r["ratio"])]
    r_median = (
        float(np.median(np.asarray(finite_ratios, dtype=np.float64)))
        if finite_ratios
        else None
    )
    r_p25 = _percentile_higher(finite_ratios, 25.0) if finite_ratios else None
    r_p75 = _percentile_higher(finite_ratios, 75.0) if finite_ratios else None
    r_p95 = _percentile_higher(finite_ratios, 95.0) if finite_ratios else None

    beta_star: float | None = None
    scaled_median: float | None = None
    accepted = False
    if r_median is not None and math.isfinite(r_median) and r_median > 0.0:
        beta_star = TARGET_SCALED_MEDIAN / r_median
        scaled_median = beta_star * r_median
        accepted = ACCEPT_LOW <= scaled_median <= ACCEPT_HIGH

    return {
        "research_prototype_warning": RESEARCH_PROTOTYPE_WARNING,
        "model_init_seed": model_init_seed,
        "sampler_seed": sampler_seed,
        "deep_sup_scales": list(deep_sup_scales),
        "deep_sup_weights": list(deep_sup_weights),
        "composition": composition,
        "n_batches_used": len(batches),
        "n_finite_ratios": len(finite_ratios),
        "r_median": r_median,
        "r_iqr": [r_p25, r_p75] if (r_p25 is not None and r_p75 is not None) else None,
        "r_p95": r_p95,
        "beta_star": beta_star,
        "scaled_median": scaled_median,
        "accepted": accepted,
        "per_batch": per_batch,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-cache-dir",
        type=Path,
        default=Path("data/processed/p6_cache_big_soft/train"),
        help="Cache with source_fraction propagation (has_source_fraction=True).",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--model-init-seed", type=int, default=DEFAULT_MODEL_INIT_SEED)
    parser.add_argument("--sampler-seed", type=int, default=DEFAULT_SAMPLER_SEED)
    parser.add_argument("--n-batches", type=int, default=DEFAULT_N_BATCHES)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "b3_grad_balance_probe: --device cuda requested but "
            "torch.cuda.is_available() is False -- VascuTrace never "
            "silently falls back to CPU."
        )

    report = run_probe(
        train_cache_dir=args.train_cache_dir,
        device=device,
        model_init_seed=args.model_init_seed,
        sampler_seed=args.sampler_seed,
        n_batches=args.n_batches,
        batch_size=args.batch_size,
    )
    print(json.dumps(report, indent=2, allow_nan=True))

    if report["beta_star"] is None:
        print("B3_PROBE: NO_QUALIFYING_BETA")
    else:
        print(
            f"B3_PROBE: beta_star={report['beta_star']:.6g} "
            f"(r_median={report['r_median']:.6g}, "
            f"accepted={report['accepted']})"
        )


if __name__ == "__main__":
    main()
