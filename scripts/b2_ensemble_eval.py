"""OFF-CRITICAL-PATH IoU stretch experiment -- prediction-map ensembling of
the two independently-trained B2 deep-supervision checkpoints (seed 20260716
and seed 20260722).

Research prototype. Trained and evaluated using simulated vascular-like
abnormalities, not confirmed human post-angioplasty lesions.

EXPLORATORY -- NOT CERTIFIABLE. This script does not retrain anything and
does not alter Tversky, edge-BCE, or soft-target training settings.

HYPOTHESIS
============================================================================
Two independently-trained B2 checkpoints (``runs/siamese_p4b2_deepsup/
best_constrained_iou.pt``, seed 20260716; ``runs/siamese_p4b2_deepsup_seed2/
best_constrained_iou.pt``, seed 20260722) are available. Averaging their
predicted abnormality-score maps before thresholding (a standard, no-retrain
ensembling technique -- Dietterich, 2000, "Ensemble Methods in Machine
Learning", LNCS 1857) has not been tried on this project and is not a
closed lever (closed levers are all TRAINING-time changes: soft-global
targets, boundary-auxiliary BCE, and Tversky FN/FP rebalancing). This
script tests whether averaging raises legal
mean positive IoU on the frozen val cache.

WHY THIS SCRIPT REUSES ``evaluate.py`` INTERNALS RATHER THAN REIMPLEMENTING
ANY METRIC MATH
============================================================================
The evaluation protocol requires reuse of the metric functions from
``src/vascutrace/ml/metrics.py`` rather than reimplementation. This script
goes one step further than importing the *public*
``metrics.py`` functions -- it imports ``evaluate.py``'s private
per-sample/aggregation helpers (``_process_sample``, ``_aggregate``,
``_run_inference``, ``_SampleRecord``, ``_subject_clustered_bootstrap``,
``_stat_mean_positive_iou``) directly. Every one of these is already
exercised by the project's official single-checkpoint evaluation path
(``evaluate_checkpoint``); the ONLY thing this script changes is what
score map is handed to ``_process_sample`` per sample -- a
per-model-averaged ``score`` array instead of a single model's ``score`` --
so the comparison between "single model" and "ensemble" rows in the
scoreboard below is guaranteed apples-to-apples: identical threshold,
identical component-matching code, identical bootstrap code, differing
ONLY in whether ``score`` came from one model or several. Importing
underscore-prefixed names from a sibling module is intentional here (not
an accident this script papers over) -- each import site carries a
``# noqa: PLC2701`` acknowledging the private-member-access lint rule this
``scripts/p3_lambda_probe.py`` already established the same
pattern for.

WHY THIS SCRIPT DOES NOT ATTEMPT TEST-TIME AUGMENTATION (TTA)
============================================================================
The evaluation specification permits laterality-safe TTA (small translations/scales)
ONLY if the transform provably preserves ``pet_diff``'s relationship to
``left_view``/``right_view`` and never swaps laterality (explicitly: no
horizontal flip -- this is a bilateral Siamese model, and flipping would
swap left/right semantics). ``tensor_schema.py``'s own docstring (line 68)
defines ``pet_diff`` as "network-normalized left PET minus right PET" --
a quantity produced by the CACHE-BUILDING pipeline's own PET normalization
(mean/std or SUV-percentile statistics computed over the ORIGINAL,
untransformed crop), not a value this eval-only script can recompute from
the already-cached ``left_view``/``right_view`` tensors alone. Applying a
translation or scale to the cached ``left_view``/``right_view`` tensors
post-hoc and expecting ``pet_diff`` to remain valid would silently violate
the identity the model was trained under UNLESS ``pet_diff`` is
independently re-derived through the exact same normalization pipeline --
that pipeline lives in the data/cache-building code (``src/vascutrace/
data/...``), which is out of scope for a read-only, no-retrain,
evaluation-only script and risks reintroducing exactly the kind of
normalization-mismatch bug this project's own ``dataset.py`` module
docstring (item 6) already documents a prior real instance of. The
evaluation protocol requires skipping TTA when preservation of
``pet_diff == left_view - right_view`` and laterality cannot be guaranteed.
This script therefore skips TTA.

USAGE
============================================================================
    uv run python -m scripts.b2_ensemble_eval --device cuda
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.vascutrace.geometry import RESEARCH_PROTOTYPE_WARNING
from src.vascutrace.ml.cache import CachedSampleDataset
from src.vascutrace.ml.checkpoint import CheckpointError, load_checkpoint
from src.vascutrace.ml.evaluate import (  # noqa: PLC2701 -- see module docstring
    DEFAULT_IN_PLANE_SPACING_MM,
    BootstrapCI,
    EvaluationError,
    NegativeMetrics,
    PositiveMetrics,
    _aggregate,
    _process_sample,
    _run_inference,
    _SampleRecord,
    _stat_mean_positive_iou,
    _subject_clustered_bootstrap,
    evaluate_checkpoint,
)
from src.vascutrace.ml.metrics import (
    DEFAULT_LESION_IOU_THRESHOLD,
    DEFAULT_MIN_COMPONENT_SIZE,
    DEFAULT_SCORE_THRESHOLD,
)
from src.vascutrace.ml.model import build_model
from src.vascutrace.ml.tensor_schema import TENSOR_SCHEMA_VERSION

# ---------------------------------------------------------------------------
# Frozen protocol constants -- must match src/vascutrace/ml/evaluate.py's
# own defaults exactly, or the single-vs-ensemble comparison is invalid
# (evaluation specification: "match the project's official evaluation exactly").
# ---------------------------------------------------------------------------

VAL_CACHE_DIR = Path("data/processed/p6_cache_big/val")

SEED1_CONSTRAINED = Path("runs/siamese_p4b2_deepsup/best_constrained_iou.pt")
SEED1_EMA_CONSTRAINED = Path("runs/siamese_p4b2_deepsup/best_ema_constrained_iou.pt")
SEED2_CONSTRAINED = Path("runs/siamese_p4b2_deepsup_seed2/best_constrained_iou.pt")
SEED2_EMA_CONSTRAINED = Path(
    "runs/siamese_p4b2_deepsup_seed2/best_ema_constrained_iou.pt"
)

# Historical aggregate reference values this script's wiring check must
# reproduce before the ensemble result is interpreted.
EXPECTED_SEED1_IOU = 0.6149
EXPECTED_SEED2_IOU = 0.6097
WIRING_CHECK_TOLERANCE = 0.001

# Frozen legal floors (evaluation specification; matches configs/train_siamese_
# p4b2_deepsup*.yaml's own constrained_iou_min_precision/min_f1).
LEGAL_MIN_PRECISION = 0.901
LEGAL_MIN_F1 = 0.859

# Comparison anchors quoted in the evaluation specification.
B0_LEGAL_IOU = 0.597
SINGLE_B2_MEAN_IOU = (EXPECTED_SEED1_IOU + EXPECTED_SEED2_IOU) / 2.0  # 0.6123
ASPIRATIONAL_IOU = 0.70

DEFAULT_BOOTSTRAP_RESAMPLES = 2000
DEFAULT_BOOTSTRAP_SEED = 0


# ---------------------------------------------------------------------------
# Model loading -- mirrors the implementation's own VERIFIED load/forward path
# exactly (load_checkpoint -> build_model -> load_state_dict -> eval()),
# plus the same schema-compatibility guards evaluate_checkpoint applies.
# ---------------------------------------------------------------------------


def _load_model(
    checkpoint_path: Path, device: str, expected_crop_schema_version: str
) -> torch.nn.Module:
    payload = load_checkpoint(checkpoint_path)  # CheckpointError propagates
    if payload.tensor_schema_version != TENSOR_SCHEMA_VERSION:
        raise EvaluationError(
            f"{checkpoint_path}: tensor_schema_version="
            f"{payload.tensor_schema_version!r} != current code's "
            f"{TENSOR_SCHEMA_VERSION!r}"
        )
    if payload.crop_schema_version != expected_crop_schema_version:
        raise EvaluationError(
            f"{checkpoint_path}: crop_schema_version="
            f"{payload.crop_schema_version!r} != val cache's "
            f"{expected_crop_schema_version!r}"
        )
    model = build_model(payload.model_config)
    model.load_state_dict(payload.model_state_dict)
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Ensembling -- average abnormality-score maps, post-sigmoid and pre-
# threshold), then feed the SAME evaluate.py per-sample scorer every
# single-checkpoint report already uses. No metric math lives below this
# point that evaluate.py/metrics.py did not already define and test.
# ---------------------------------------------------------------------------


def _ensemble_records(
    models: list[torch.nn.Module],
    dataset: CachedSampleDataset,
    device: str,
    *,
    score_threshold: float,
    iou_threshold: float,
    spacing_mm: tuple[float, float],
    min_component_size: int,
) -> list[_SampleRecord]:
    records: list[_SampleRecord] = []
    for i in range(len(dataset)):
        sample = dataset[i]
        probs: list[np.ndarray] = []
        target = valid = None
        for model in models:
            score, target, valid = _run_inference(model, sample, device)
            probs.append(score)
        ensemble_prob = np.mean(np.stack(probs, axis=0), axis=0)
        records.append(
            _process_sample(
                ensemble_prob,
                target,
                valid,
                sample.meta,
                score_threshold=score_threshold,
                iou_threshold=iou_threshold,
                spacing_mm=spacing_mm,
                min_component_size=min_component_size,
            )
        )
    return records


@dataclass(frozen=True)
class ScoreboardRow:
    """One scoreboard row -- the same six columns as the project's own
    B2-deepsup scoreboard note (IoU, recall, precision, F1, clean,
    activation) plus miss k/78 and the legal-floor verdict.
    """

    name: str
    n_positive: int
    n_negative: int
    mean_iou: float
    recall: float
    precision: float
    f1: float
    clean: float
    activation: float
    miss_k: int
    n_lesions_total: int
    legal: bool

    def to_line(self) -> str:
        return (
            f"{self.name:<48s} IoU={self.mean_iou:.4f} "
            f"recall={self.recall:.4f} precision={self.precision:.4f} "
            f"F1={self.f1:.4f} clean={self.clean:.4f} "
            f"activation={self.activation:.4f} "
            f"miss={self.miss_k}/{self.n_lesions_total} "
            f"legal={'YES' if self.legal else 'no'}"
        )


def _row_from_aggregate(
    name: str, positives: PositiveMetrics, negatives: NegativeMetrics
) -> ScoreboardRow:
    legal = (
        positives.lesion_precision >= LEGAL_MIN_PRECISION
        and positives.lesion_f1 >= LEGAL_MIN_F1
    )
    return ScoreboardRow(
        name=name,
        n_positive=positives.n_positive_samples,
        n_negative=negatives.n_negative_samples,
        mean_iou=positives.mean_iou,
        recall=positives.positive_detection_fraction,
        precision=positives.lesion_precision,
        f1=positives.lesion_f1,
        clean=negatives.pct_fully_clean,
        activation=1.0 - negatives.pct_fully_clean,
        miss_k=positives.n_lesions_missed,
        n_lesions_total=positives.n_lesions_total,
        legal=legal,
    )


# ---------------------------------------------------------------------------
# Paired, subject-clustered bootstrap on the ENSEMBLE-vs-SINGLE IoU delta.
# Reuses evaluate.py's OWN _subject_clustered_bootstrap + _stat_mean_
# positive_iou verbatim -- see module docstring for why this is not a
# metric reimplementation: _stat_mean_positive_iou is generic over any
# `.iou` field a _SampleRecord carries, so handing it per-sample DELTAS
# (ensemble_iou - single_iou) instead of raw IoUs makes the SAME tested
# bootstrap code compute a delta CI instead of a point-estimate CI, with
# zero new statistical code written here beyond delta subtraction.
# ---------------------------------------------------------------------------


def _paired_delta_bootstrap(
    treatment_records: list[_SampleRecord],
    reference_records: list[_SampleRecord],
    *,
    n_resamples: int,
    seed: int,
) -> BootstrapCI:
    assert len(treatment_records) == len(reference_records)
    delta_records: list[_SampleRecord] = []
    for t, r in zip(treatment_records, reference_records, strict=True):
        if not (t.target_nonempty and r.target_nonempty):
            continue
        assert t.subject == r.subject, (
            "treatment/reference record order mismatch -- same dataset, "
            "same index, must be the same sample/subject"
        )
        if t.iou is None or r.iou is None:
            continue
        delta_records.append(
            _SampleRecord(
                subject=t.subject,
                meta_positive=True,
                target_nonempty=True,
                iou=t.iou - r.iou,
                dice=t.iou - r.iou,
            )
        )
    return _subject_clustered_bootstrap(
        delta_records, _stat_mean_positive_iou, n_resamples=n_resamples, seed=seed
    )


# ---------------------------------------------------------------------------
# Wiring check -- reproduce the two documented single-model legal IoU
# numbers on THIS harness before trusting any ensemble comparison (task
# brief: "if you cannot reproduce them, STOP and report -- the comparison
# is invalid otherwise").
# ---------------------------------------------------------------------------


def _wiring_check(device: str, n_bootstrap: int, bootstrap_seed: int) -> dict[str, Any]:
    report1 = evaluate_checkpoint(
        SEED1_CONSTRAINED,
        VAL_CACHE_DIR,
        device=device,
        iou_threshold=DEFAULT_LESION_IOU_THRESHOLD,
        score_threshold=DEFAULT_SCORE_THRESHOLD,
        spacing_mm=DEFAULT_IN_PLANE_SPACING_MM,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
        min_component_size=DEFAULT_MIN_COMPONENT_SIZE,
    )
    report2 = evaluate_checkpoint(
        SEED2_CONSTRAINED,
        VAL_CACHE_DIR,
        device=device,
        iou_threshold=DEFAULT_LESION_IOU_THRESHOLD,
        score_threshold=DEFAULT_SCORE_THRESHOLD,
        spacing_mm=DEFAULT_IN_PLANE_SPACING_MM,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
        min_component_size=DEFAULT_MIN_COMPONENT_SIZE,
    )

    seed1_iou = report1.positives.mean_iou
    seed2_iou = report2.positives.mean_iou
    seed1_ok = abs(seed1_iou - EXPECTED_SEED1_IOU) <= WIRING_CHECK_TOLERANCE
    seed2_ok = abs(seed2_iou - EXPECTED_SEED2_IOU) <= WIRING_CHECK_TOLERANCE

    return {
        "seed1_report": report1,
        "seed2_report": report2,
        "seed1_iou": seed1_iou,
        "seed2_iou": seed2_iou,
        "seed1_ok": seed1_ok,
        "seed2_ok": seed2_ok,
        "passed": seed1_ok and seed2_ok,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=DEFAULT_BOOTSTRAP_RESAMPLES,
        help="Subject-clustered bootstrap resamples (0 disables).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--skip-ensemble4",
        action="store_true",
        help="Skip the 4-model (raw + EMA constrained, both seeds) ensemble.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "b2_ensemble_eval: --device cuda requested but "
            "torch.cuda.is_available() is False -- VascuTrace never "
            "silently falls back to CPU."
        )

    print(f"> {RESEARCH_PROTOTYPE_WARNING}")
    print("> EXPLORATORY IoU stretch experiment -- prediction ensembling.")
    print(f"> device={device}, val cache={VAL_CACHE_DIR}")
    print()

    # -- 1. Wiring check ----------------------------------------------------
    print("=" * 78)
    print("STEP 1: wiring check -- reproduce the two documented single-model")
    print("legal IoU numbers before trusting anything downstream.")
    print("=" * 78)
    try:
        wiring = _wiring_check(device, args.n_bootstrap, args.seed)
    except (EvaluationError, CheckpointError) as exc:
        print(f"WIRING CHECK FAILED TO RUN: {type(exc).__name__}: {exc}")
        return 1

    print(
        f"seed-1 ({SEED1_CONSTRAINED}): IoU={wiring['seed1_iou']:.5f} "
        f"(expected {EXPECTED_SEED1_IOU:.4f} +/- {WIRING_CHECK_TOLERANCE}) "
        f"-> {'MATCH' if wiring['seed1_ok'] else 'MISMATCH'}"
    )
    print(
        f"seed-2 ({SEED2_CONSTRAINED}): IoU={wiring['seed2_iou']:.5f} "
        f"(expected {EXPECTED_SEED2_IOU:.4f} +/- {WIRING_CHECK_TOLERANCE}) "
        f"-> {'MATCH' if wiring['seed2_ok'] else 'MISMATCH'}"
    )
    if not wiring["passed"]:
        print()
        print(
            "WIRING CHECK FAILED -- the harness does not reproduce the "
            "documented single-model numbers. STOPPING: the ensemble "
            "comparison below would be meaningless against a harness that "
            "cannot even reproduce the known single-model baseline."
        )
        return 1
    print()
    print("Wiring check PASSED -- proceeding to ensemble experiments.")
    print()

    # -- 2. Load models, build dataset once ----------------------------------
    dataset = CachedSampleDataset(VAL_CACHE_DIR, enforce_manifest_class_content=False)
    crop_schema_version = dataset.manifest.get("crop_schema_version")

    model_seed1 = _load_model(SEED1_CONSTRAINED, device, crop_schema_version)
    model_seed2 = _load_model(SEED2_CONSTRAINED, device, crop_schema_version)

    # Single-model records, reusing the exact same evaluate.py per-sample
    # scorer the ensemble path below uses -- lets us both build the
    # scoreboard's single-model rows AND supply the paired-delta bootstrap's
    # "reference" records from ONE dataset pass per model (not
    # re-deriving them from the wiring-check EvaluationReports, which do not
    # expose per-sample records).
    print("Building single-model per-sample records (seed-1, seed-2)...")
    records_seed1 = _ensemble_records(
        [model_seed1],
        dataset,
        device,
        score_threshold=DEFAULT_SCORE_THRESHOLD,
        iou_threshold=DEFAULT_LESION_IOU_THRESHOLD,
        spacing_mm=DEFAULT_IN_PLANE_SPACING_MM,
        min_component_size=DEFAULT_MIN_COMPONENT_SIZE,
    )
    records_seed2 = _ensemble_records(
        [model_seed2],
        dataset,
        device,
        score_threshold=DEFAULT_SCORE_THRESHOLD,
        iou_threshold=DEFAULT_LESION_IOU_THRESHOLD,
        spacing_mm=DEFAULT_IN_PLANE_SPACING_MM,
        min_component_size=DEFAULT_MIN_COMPONENT_SIZE,
    )
    pos1, neg1, _ = _aggregate(records_seed1)
    pos2, neg2, _ = _aggregate(records_seed2)
    row_seed1 = _row_from_aggregate("single: seed-1 (best_constrained_iou)", pos1, neg1)
    row_seed2 = _row_from_aggregate("single: seed-2 (best_constrained_iou)", pos2, neg2)

    # Cross-check: these per-sample-derived aggregates must match the
    # wiring-check EvaluationReports exactly (same protocol, same data,
    # only the code PATH differs -- evaluate_checkpoint's own dataset loop
    # vs this script's _ensemble_records loop with n_models=1).
    assert abs(row_seed1.mean_iou - wiring["seed1_iou"]) < 1e-9, (
        "seed-1 per-sample-record IoU disagrees with evaluate_checkpoint's "
        "own IoU -- internal inconsistency, do not trust downstream numbers"
    )
    assert abs(row_seed2.mean_iou - wiring["seed2_iou"]) < 1e-9, (
        "seed-2 per-sample-record IoU disagrees with evaluate_checkpoint's "
        "own IoU -- internal inconsistency, do not trust downstream numbers"
    )

    # -- 3. ENSEMBLE-2 --------------------------------------------------------
    print("Building ENSEMBLE-2 (seed-1 + seed-2, both best_constrained_iou)...")
    records_ens2 = _ensemble_records(
        [model_seed1, model_seed2],
        dataset,
        device,
        score_threshold=DEFAULT_SCORE_THRESHOLD,
        iou_threshold=DEFAULT_LESION_IOU_THRESHOLD,
        spacing_mm=DEFAULT_IN_PLANE_SPACING_MM,
        min_component_size=DEFAULT_MIN_COMPONENT_SIZE,
    )
    pos_ens2, neg_ens2, _ = _aggregate(records_ens2)
    row_ens2 = _row_from_aggregate("ENSEMBLE-2 (seed-1 + seed-2)", pos_ens2, neg_ens2)

    rows = [row_seed1, row_seed2, row_ens2]
    records_by_name: dict[str, list[_SampleRecord]] = {
        row_seed1.name: records_seed1,
        row_seed2.name: records_seed2,
        row_ens2.name: records_ens2,
    }

    # -- 4. ENSEMBLE-4 (optional) --------------------------------------------
    if (
        not args.skip_ensemble4
        and SEED1_EMA_CONSTRAINED.exists()
        and SEED2_EMA_CONSTRAINED.exists()
    ):
        print("Building ENSEMBLE-4 (+ both seeds' best_ema_constrained_iou)...")
        model_seed1_ema = _load_model(
            SEED1_EMA_CONSTRAINED, device, crop_schema_version
        )
        model_seed2_ema = _load_model(
            SEED2_EMA_CONSTRAINED, device, crop_schema_version
        )
        records_ens4 = _ensemble_records(
            [model_seed1, model_seed2, model_seed1_ema, model_seed2_ema],
            dataset,
            device,
            score_threshold=DEFAULT_SCORE_THRESHOLD,
            iou_threshold=DEFAULT_LESION_IOU_THRESHOLD,
            spacing_mm=DEFAULT_IN_PLANE_SPACING_MM,
            min_component_size=DEFAULT_MIN_COMPONENT_SIZE,
        )
        pos_ens4, neg_ens4, _ = _aggregate(records_ens4)
        row_ens4 = _row_from_aggregate(
            "ENSEMBLE-4 (+ both EMA constrained)", pos_ens4, neg_ens4
        )
        rows.append(row_ens4)
        records_by_name[row_ens4.name] = records_ens4
    else:
        print("Skipping ENSEMBLE-4 (--skip-ensemble4 or EMA checkpoints missing).")

    # -- 5. TTA -- explicitly skipped, documented -----------------------------
    print()
    print("TTA: SKIPPED. See module docstring 'WHY THIS SCRIPT DOES NOT")
    print("ATTEMPT TEST-TIME AUGMENTATION' -- pet_diff is a network-")
    print("normalized quantity produced by the cache-building pipeline, not")
    print("recomputable from cached left_view/right_view alone; a")
    print("translation/scale applied post-hoc to the cached tensors cannot")
    print("be guaranteed to preserve that normalization without re-deriving")
    print("it through code outside this script's scope. No flip attempted")
    print("under any circumstance (would swap left/right laterality).")
    print()

    # -- 6. Scoreboard --------------------------------------------------------
    print("=" * 78)
    print("STEP 2: scoreboard (single models vs ensembles)")
    print("=" * 78)
    header = (
        f"{'Model':<48s} {'IoU':>8s} {'Recall':>8s} {'Prec':>8s} {'F1':>8s} "
        f"{'Clean':>8s} {'Activ':>8s} {'Miss':>8s} {'Legal':>6s}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row.name:<48s} {row.mean_iou:8.4f} {row.recall:8.4f} "
            f"{row.precision:8.4f} {row.f1:8.4f} {row.clean:8.4f} "
            f"{row.activation:8.4f} "
            f"{row.miss_k:>3d}/{row.n_lesions_total:<3d} "
            f"{'YES' if row.legal else 'no':>6s}"
        )
    print()

    # -- 7. Legality + delta vs B0 / single-B2-mean / 0.70 --------------------
    print("=" * 78)
    print("STEP 3: legality and distance to reference points")
    print("=" * 78)
    print(f"B0 legal IoU (reference)           = {B0_LEGAL_IOU:.4f}")
    print(
        f"Single-B2 mean-of-two-seeds IoU    = {SINGLE_B2_MEAN_IOU:.4f} "
        f"(seed-1={EXPECTED_SEED1_IOU:.4f}, seed-2={EXPECTED_SEED2_IOU:.4f})"
    )
    print(f"Aspirational IoU                   = {ASPIRATIONAL_IOU:.4f}")
    print()
    for row in rows:
        delta_b0 = row.mean_iou - B0_LEGAL_IOU
        delta_single_mean = row.mean_iou - SINGLE_B2_MEAN_IOU
        dist_070 = ASPIRATIONAL_IOU - row.mean_iou
        print(
            f"{row.name:<48s} vs B0={delta_b0:+.4f}  "
            f"vs single-B2-mean={delta_single_mean:+.4f}  "
            f"dist-to-0.70={dist_070:+.4f}  legal={'YES' if row.legal else 'no'}"
        )
    print()

    # -- 8. Paired, subject-clustered bootstrap on ensemble-vs-single delta --
    print("=" * 78)
    print("STEP 4: subject-clustered bootstrap CI on ENSEMBLE-vs-SINGLE IoU")
    print("delta (paired per-sample, delta = ensemble - single)")
    print("=" * 78)
    ensemble_rows = [r for r in rows if r.name.startswith("ENSEMBLE")]
    delta_results: dict[str, dict[str, BootstrapCI]] = {}
    for erow in ensemble_rows:
        e_records = records_by_name[erow.name]
        delta_results[erow.name] = {}
        for single_name, single_records in (
            (row_seed1.name, records_seed1),
            (row_seed2.name, records_seed2),
        ):
            ci = _paired_delta_bootstrap(
                e_records,
                single_records,
                n_resamples=args.n_bootstrap,
                seed=args.seed,
            )
            delta_results[erow.name][single_name] = ci
            print(
                f"delta({erow.name} - {single_name}): "
                f"{ci.point_estimate:+.4f}  "
                f"95% CI [{ci.ci_low:+.4f}, {ci.ci_high:+.4f}]  "
                f"(n_resamples={ci.n_resamples})"
            )
    print()

    # -- 9. Verdict -------------------------------------------------------------
    print("=" * 78)
    print("STEP 5: honest verdict")
    print("=" * 78)
    best_ensemble = max(ensemble_rows, key=lambda r: r.mean_iou, default=None)
    if best_ensemble is None:
        print("No ensemble row was built -- nothing to verdict on.")
    else:
        best_single = max(row_seed1.mean_iou, row_seed2.mean_iou)
        gain_over_best_single = best_ensemble.mean_iou - best_single
        gain_over_mean_single = best_ensemble.mean_iou - SINGLE_B2_MEAN_IOU
        print(
            f"Best ensemble: {best_ensemble.name}, legal IoU = "
            f"{best_ensemble.mean_iou:.4f} "
            f"({'LEGAL' if best_ensemble.legal else 'NOT LEGAL'}: "
            f"precision={best_ensemble.precision:.4f} "
            f"(floor {LEGAL_MIN_PRECISION}), f1={best_ensemble.f1:.4f} "
            f"(floor {LEGAL_MIN_F1}))"
        )
        print(
            f"Gain vs best single model ({best_single:.4f}): "
            f"{gain_over_best_single:+.4f}"
        )
        print(
            f"Gain vs single-B2 mean-of-two-seeds ({SINGLE_B2_MEAN_IOU:.4f}): "
            f"{gain_over_mean_single:+.4f}"
        )
        print(
            f"Distance to aspirational 0.70: {ASPIRATIONAL_IOU - best_ensemble.mean_iou:+.4f}"
        )
        if gain_over_best_single < 0.01:
            print(
                "HONEST READ: gain over the best single model is < 0.01 IoU "
                "-- this is NOT a meaningful gain by this note's own "
                "threshold. No 'IoU ceiling' claim is made. Ensembling is "
                "not recommended for banking as a production lever on this "
                "evidence alone."
            )
        else:
            print(
                "HONEST READ: gain over the best single model is >= 0.01 "
                "IoU. Check the paired bootstrap CI above (Step 4) before "
                "treating this as decisive -- a CI that includes zero means "
                "the point estimate alone is not sufficient evidence."
            )
        if not best_ensemble.legal:
            print(
                "LEGALITY WARNING: the best ensemble by raw IoU does NOT "
                "clear the frozen legal floors (precision >= "
                f"{LEGAL_MIN_PRECISION}, F1 >= {LEGAL_MIN_F1}) -- an "
                "illegal candidate's IoU number is not eligible for "
                "promotion/banking under this project's own selection rule."
            )
    print()
    print(
        "Note: this is a SINGLE val set of 78 positives over 7 subject "
        "clusters (not independent draws) -- treat all point estimates and "
        "CIs above as bounded by that sample size, not as population-level "
        "claims. No 'IoU ceiling' claim is made anywhere in this report."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
