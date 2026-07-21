"""Outcome-blind lambda_boundary gradient-ratio probe.

Research prototype. Trained and evaluated using simulated vascular-like
abnormalities, not confirmed human post-angioplasty lesions.

EXPLORATORY -- NOT CERTIFIABLE. This script implements the production-gradient half of the
frozen outcome-blind lambda rule. It covers the
rule (the second of its two outcome-blind inputs: "sixteen fixed,
train-only, batch-size-32 cache batches passed through the production
model at identical initialization for final-head and shared-decoder
gradients"). The FIRST input -- the 135 generated logit/autodiff
scenarios -- is a separate, dedicated-test-file deliverable (E3-S2/E3-S4's
generated-fixture gates), not this script's job.

Selects the SMALLEST lambda_boundary in {0.03, 0.10, 0.30, 1.00} such
that, for BOTH arm B ("hard") and arm C ("fraction"):

    1. median final-head    aux:hard gradient ratio >= 0.10
    2. p95    shared-decoder aux:hard gradient ratio <= 0.50
    3. max    shared-decoder aux:hard gradient ratio <= 1.00
    4. every ratio is finite

If no candidate qualifies, prints exactly
``LAMBDA_PROBE: NO_QUALIFYING_LAMBDA (E3-S4 stop)``. Under the frozen
mechanism-gate stop rule, a failed gate is a
scientific result, not a retry reason. This script never widens the grid,
selects from validation performance, or falls back to lambda=1.

DEVIATIONS FROM THE FROZEN PROTOCOL:

  * ``sample order = SHA256-sort of (cache_content_digest, sample_index,
    sampler_seed)`` is NOT reproduced literally. This script instead uses
    ``numpy.random.default_rng(sampler_seed).permutation(...)`` over each
    composition pool (see below) -- a simpler, still-deterministic-given-
    the-seed ordering that does not depend on re-deriving the cache's own
    content digest inside this standalone script. A caller who needs the
    exact SHA256-sort ordering must implement it separately; this script
    documents, not hides, the gap.
  * ``logit norm support = W > 0 positions only`` from the production
    -gradient protocol table) is NOT computed by this script. It is not
    one of the four selection criteria listed here
    (median final-head ratio, p95/max shared-decoder ratio, finiteness),
    and its exact intended construction is not fully specified by the
    protocol available to this script. Rather than guess at an
    unspecified quantity, this script omits it and flags the omission
    here.
  * "boundary-bearing-positive" / "zero-boundary-negative" batch
    composition ("batch composition = 16 boundary-bearing
    positives + 16 zero-boundary negatives") is operationalized purely
    from each cached sample's own boundary support ``W = valid_mask *
    1[0 < source_fraction < 1]`` (the SAME support definition
    ``losses.boundary_auxiliary_loss`` uses): "boundary-bearing" =
    ``sum(W) > 0`` for that sample, "zero-boundary" = ``sum(W) == 0``.
    This is a stated interpretation, not a re-derivation of the cache's
    ``meta["positive"]``/negative-sample convention (which the project's
    own ``train.py`` docstring, item 11, documents as having previously
    disagreed with actual target content) -- the two conventions likely
    coincide in the common case (a "positive" simulated-lesion sample
    almost always has some fractional-boundary pixels; a true negative/
    sham sample has ``source_fraction`` identically zero and therefore no
    boundary support) but are not asserted identical here.
  * If the cache does not contain enough distinct samples of either class
    to fill 16 batches x 16-per-class without replacement, this script
    builds as many FULL batches as the smaller class supports and reports
    the actual achieved composition in the printed JSON (``composition``
    key) rather than silently padding, repeating, or shrinking the
    per-batch composition below 16 without saying so.
  * Runs at whatever ``--device``/``--amp`` the caller passes; the
    protocol's own "AMP off" requirement is enforced by this script
    refusing ``--amp`` entirely (there is no flag for it) -- forward/loss/
    grad here are always plain float32, autocast never entered.

Usage (run only after the operator authorizes local data and compute use):

    uv run python -m scripts.p3_lambda_probe \\
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
from src.vascutrace.ml.losses import boundary_auxiliary_loss, combo_loss
from src.vascutrace.ml.model import ModelConfig, build_model
from src.vascutrace.ml.train import _safe_grad_l2_norm, seed_everything  # noqa: PLC2701

# Frozen grid: "lambda_boundary in {0.03, 0.10, 0.30, 1.00}".
# Never widen this grid based on results ("Do not widen the
# grid, choose lambda from validation performance, switch automatically to
# lambda=1").
LAMBDA_GRID: tuple[float, ...] = (0.03, 0.10, 0.30, 1.00)

# the boundary experiment Sec 5's frozen production-gradient protocol constants.
DEFAULT_MODEL_INIT_SEED = 20260721
DEFAULT_SAMPLER_SEED = 20260720
DEFAULT_N_BATCHES = 16
DEFAULT_BATCH_POSITIVE = 16
DEFAULT_BATCH_NEGATIVE = 16

# Frozen selection thresholds.
MEDIAN_HEAD_RATIO_MIN = 0.10
P95_DECODER_RATIO_MAX = 0.50
MAX_DECODER_RATIO_MAX = 1.00

_TARGET_MODES: tuple[str, ...] = ("hard", "fraction")

_DECODER_ATTRS: tuple[str, ...] = (
    "up4",
    "up3",
    "up2",
    "up1",
    "diff_stem",
    "diff_fuse",
)


def _stack_batch(
    samples: list[Sample], device: torch.device
) -> dict[str, torch.Tensor]:
    """Same tensor keys :func:`train._collate_samples` produces (plus
    ``source_fraction``, always) -- a small standalone re-implementation so
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


def _boundary_support_count(sample: Sample) -> int:
    """``sum(W)`` for one sample -- the SAME support definition
    ``losses.boundary_auxiliary_loss`` uses (``valid_mask * 1[0 <
    source_fraction < 1]``). Used only to classify a sample as
    "boundary-bearing" (``> 0``) or "zero-boundary" (``== 0``) for batch
    composition -- see this module's own docstring, deviation notes.
    """
    f = sample.source_fraction
    m = sample.valid_mask
    support = (f > 0.0) & (f < 1.0) & (m >= 0.5)
    return int(support.sum().item())


def _classify_cache(
    dataset: CachedSampleDataset,
) -> tuple[list[int], list[int]]:
    """Scan every cached sample once; return
    ``(boundary_bearing_indices, zero_boundary_indices)``."""
    boundary_bearing: list[int] = []
    zero_boundary: list[int] = []
    for i in range(len(dataset)):
        sample = dataset[i]
        if _boundary_support_count(sample) > 0:
            boundary_bearing.append(i)
        else:
            zero_boundary.append(i)
    return boundary_bearing, zero_boundary


def _build_batches(
    dataset: CachedSampleDataset,
    *,
    sampler_seed: int,
    n_batches: int,
    batch_positive: int,
    batch_negative: int,
    device: torch.device,
) -> tuple[list[dict[str, torch.Tensor]], dict[str, Any]]:
    """Build up to ``n_batches`` fixed batches, each composed of
    ``batch_positive`` boundary-bearing samples + ``batch_negative``
    zero-boundary samples, drawn WITHOUT replacement from the cache (see
    module docstring, deviation notes, for the ``numpy`` permutation used
    in place of the protocol's exact SHA256-sort ordering). Returns the batch
    list plus a ``composition`` diagnostic dict documenting what was
    actually achievable.
    """
    boundary_bearing, zero_boundary = _classify_cache(dataset)

    rng = np.random.default_rng(sampler_seed)
    boundary_order = rng.permutation(len(boundary_bearing))
    zero_order = rng.permutation(len(zero_boundary))
    boundary_bearing_shuffled = [boundary_bearing[i] for i in boundary_order]
    zero_boundary_shuffled = [zero_boundary[i] for i in zero_order]

    max_full_batches_by_positive = len(boundary_bearing_shuffled) // max(
        1, batch_positive
    )
    max_full_batches_by_negative = len(zero_boundary_shuffled) // max(1, batch_negative)
    achievable_batches = min(
        n_batches, max_full_batches_by_positive, max_full_batches_by_negative
    )

    composition: dict[str, Any] = {
        "requested_n_batches": n_batches,
        "requested_batch_positive": batch_positive,
        "requested_batch_negative": batch_negative,
        "n_boundary_bearing_available": len(boundary_bearing_shuffled),
        "n_zero_boundary_available": len(zero_boundary_shuffled),
        "achieved_n_batches": achievable_batches,
        "shortfall": achievable_batches < n_batches,
    }

    batches: list[dict[str, torch.Tensor]] = []
    for b in range(achievable_batches):
        pos_indices = boundary_bearing_shuffled[
            b * batch_positive : (b + 1) * batch_positive
        ]
        neg_indices = zero_boundary_shuffled[
            b * batch_negative : (b + 1) * batch_negative
        ]
        samples = [dataset[i] for i in pos_indices + neg_indices]
        batches.append(_stack_batch(samples, device))

    return batches, composition


def _percentile_higher(values: list[float], q: float) -> float:
    """Empirical percentile with ``method="higher"`` (the protocol's
    definition) -- ``numpy.percentile``'s ``method`` kwarg name (verified
    against this project's installed numpy version; older numpy releases
    spell this kwarg ``interpolation`` instead, not used here since the
    project's own ``pyproject.toml``-pinned numpy supports ``method``).
    """
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
    batch_positive: int = DEFAULT_BATCH_POSITIVE,
    batch_negative: int = DEFAULT_BATCH_NEGATIVE,
    lambda_grid: tuple[float, ...] = LAMBDA_GRID,
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
        batch_positive=batch_positive,
        batch_negative=batch_negative,
        device=device,
    )
    if not batches:
        raise RuntimeError(
            "p3_lambda_probe: zero usable batches -- the cache at "
            f"{train_cache_dir} does not contain both a boundary-bearing "
            f"and a zero-boundary sample (composition={composition!r})"
        )

    # Identical initialization for every batch/lambda/arm -- frozen protocol,
    # "identical initialization". Built ONCE, never re-seeded or updated
    # (no optimizer step -- frozen protocol, "no optimizer step").
    seed_everything(model_init_seed)
    model = build_model(ModelConfig()).to(device)
    model.train()

    head_params = [p for p in model.head.parameters() if p.requires_grad]
    decoder_params: list[torch.Tensor] = []
    for attr in _DECODER_ATTRS:
        decoder_params.extend(
            p for p in getattr(model, attr).parameters() if p.requires_grad
        )

    # Per-batch, per-mode raw gradient norms -- computed ONCE (independent
    # of lambda; lambda only rescales the ratio afterward), reused for
    # every lambda candidate.
    per_batch: list[dict[str, Any]] = []
    for batch in batches:
        logits = model(batch["left_view"], batch["right_view"], batch["pet_diff"])
        hard_loss = combo_loss(
            logits, batch["target_mask"], valid_mask=batch["valid_mask"]
        )

        hard_head_grad = torch.autograd.grad(
            hard_loss, head_params, retain_graph=True, allow_unused=True
        )
        hard_decoder_grad = torch.autograd.grad(
            hard_loss, decoder_params, retain_graph=True, allow_unused=True
        )
        hard_head_norm = _safe_grad_l2_norm(hard_head_grad)
        hard_decoder_norm = _safe_grad_l2_norm(hard_decoder_grad)

        per_mode: dict[str, Any] = {}
        for mode in _TARGET_MODES:
            aux_result = boundary_auxiliary_loss(
                logits,
                batch["source_fraction"],
                batch["target_mask"],
                batch["valid_mask"],
                target_mode=mode,
            )
            aux_head_grad = torch.autograd.grad(
                aux_result.loss, head_params, retain_graph=True, allow_unused=True
            )
            aux_decoder_grad = torch.autograd.grad(
                aux_result.loss, decoder_params, retain_graph=True, allow_unused=True
            )
            per_mode[mode] = {
                "aux_head_norm": _safe_grad_l2_norm(aux_head_grad),
                "aux_decoder_norm": _safe_grad_l2_norm(aux_decoder_grad),
                "boundary_count": aux_result.boundary_count.item(),
                "boundary_fraction": aux_result.boundary_fraction.item(),
            }

        per_batch.append(
            {
                "hard_head_norm": hard_head_norm,
                "hard_decoder_norm": hard_decoder_norm,
                "modes": per_mode,
            }
        )

    # Now sweep the frozen lambda grid -- the protocol's ratio denominator =
    # max(hard-gradient norm, 1e-12).
    table: dict[str, Any] = {}
    selected_lambda: float | None = None
    for lam in lambda_grid:
        lam_key = f"{lam:g}"
        table[lam_key] = {}
        lambda_qualifies = True
        for mode in _TARGET_MODES:
            head_ratios: list[float] = []
            decoder_ratios: list[float] = []
            all_finite = True
            for record in per_batch:
                hard_head = record["hard_head_norm"]
                hard_decoder = record["hard_decoder_norm"]
                aux_head = record["modes"][mode]["aux_head_norm"]
                aux_decoder = record["modes"][mode]["aux_decoder_norm"]
                if None in (hard_head, hard_decoder, aux_head, aux_decoder):
                    all_finite = False
                    head_ratios.append(math.nan)
                    decoder_ratios.append(math.nan)
                    continue
                head_ratio = lam * aux_head / max(hard_head, 1e-12)
                decoder_ratio = lam * aux_decoder / max(hard_decoder, 1e-12)
                if not (math.isfinite(head_ratio) and math.isfinite(decoder_ratio)):
                    all_finite = False
                head_ratios.append(head_ratio)
                decoder_ratios.append(decoder_ratio)

            finite_head = [v for v in head_ratios if math.isfinite(v)]
            finite_decoder = [v for v in decoder_ratios if math.isfinite(v)]
            median_head = (
                float(np.median(np.asarray(finite_head, dtype=np.float64)))
                if finite_head
                else None
            )
            p95_decoder = (
                _percentile_higher(finite_decoder, 95.0) if finite_decoder else None
            )
            max_decoder = max(finite_decoder) if finite_decoder else None

            mode_qualifies = (
                all_finite
                and median_head is not None
                and median_head >= MEDIAN_HEAD_RATIO_MIN
                and p95_decoder is not None
                and p95_decoder <= P95_DECODER_RATIO_MAX
                and max_decoder is not None
                and max_decoder <= MAX_DECODER_RATIO_MAX
            )
            lambda_qualifies = lambda_qualifies and mode_qualifies

            table[lam_key][mode] = {
                "median_final_head_ratio": median_head,
                "p95_shared_decoder_ratio": p95_decoder,
                "max_shared_decoder_ratio": max_decoder,
                "all_finite": all_finite,
                "qualifies": mode_qualifies,
                "raw_head_ratios": head_ratios,
                "raw_decoder_ratios": decoder_ratios,
            }

        table[lam_key]["qualifies"] = lambda_qualifies
        if lambda_qualifies and selected_lambda is None:
            # Smallest qualifying lambda -- lambda_grid is ascending, and
            # this is the FIRST qualifying candidate encountered.
            selected_lambda = lam

    return {
        "research_prototype_warning": RESEARCH_PROTOTYPE_WARNING,
        "model_init_seed": model_init_seed,
        "sampler_seed": sampler_seed,
        "composition": composition,
        "lambda_grid": list(lambda_grid),
        "selected_lambda": selected_lambda,
        "table": table,
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
    parser.add_argument("--batch-positive", type=int, default=DEFAULT_BATCH_POSITIVE)
    parser.add_argument("--batch-negative", type=int, default=DEFAULT_BATCH_NEGATIVE)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "p3_lambda_probe: --device cuda requested but torch.cuda.is_available() "
            "is False -- VascuTrace never silently falls back to CPU."
        )

    report = run_probe(
        train_cache_dir=args.train_cache_dir,
        device=device,
        model_init_seed=args.model_init_seed,
        sampler_seed=args.sampler_seed,
        n_batches=args.n_batches,
        batch_positive=args.batch_positive,
        batch_negative=args.batch_negative,
    )
    print(json.dumps(report, indent=2, allow_nan=True))

    if report["selected_lambda"] is None:
        print("LAMBDA_PROBE: NO_QUALIFYING_LAMBDA (E3-S4 stop)")
    else:
        print(f"LAMBDA_PROBE: selected_lambda={report['selected_lambda']:g}")


if __name__ == "__main__":
    main()
