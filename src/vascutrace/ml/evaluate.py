"""Segmentation + detection evaluation of a trained checkpoint against a
precomputed sample cache (VascuTrace Phase 6).

RESEARCH_PROTOTYPE_WARNING
---------------------------------------------------------------------------
Research prototype. Trained and evaluated using simulated vascular-like
abnormalities, not confirmed human post-angioplasty lesions.
---------------------------------------------------------------------------

Implementation notes
============================================================================
This module answers the question every reader of a segmentation paper asks
of its results table: "is this one blended number hiding a model that is
actually bad at one of the two things that matter -- tracing a lesion it
found, or noticing a lesion is there at all -- behind a good score on the
other?" :func:`evaluate_checkpoint` runs a trained checkpoint over every
sample in a precomputed cache (``cache.py``'s
:class:`~src.vascutrace.ml.cache.CachedSampleDataset`) and reports
POSITIVES and NEGATIVES in entirely separate sections, using only the pure
metric functions in ``metrics.py`` -- never a single averaged "accuracy".

1. Positives and negatives are separated because they measure different
   failure modes, and averaging them together hides both
   ------------------------------------------------------------------------
   A healthy (negative) sample has an empty target by construction --
   voxelwise "accuracy" on it is dominated by trivially-correct background
   and says nothing about segmentation quality; what actually matters there
   is whether the model stayed quiet (false-activation COUNTS -- see item 5
   below on why this report never calls that a "clinical false positive").
   A lesion-bearing (positive) sample's Dice/IoU/detection/boundary-quality
   numbers say nothing about false activation elsewhere. Blending the two
   into one "mean Dice over the whole cache" (as a naive evaluation script
   would) lets a model that is simply cautious everywhere (rarely predicts
   positive at all) look artificially strong, because most of the cache's
   samples are negative (manifest.json's own ``total_positive``/
   ``total_negative`` counts are never close to 1:1 by construction --
   ``prepare-synthetic``'s ``n_positive_per_bundle``/``n_negative_per_bundle``
   are set independently). This module's report structure makes that
   failure mode impossible to hide.

2. "Positive" vs "negative" is decided by ACTUAL target content, not the
   cache's ``meta["positive"]`` intent flag
   ------------------------------------------------------------------------
   ``dataset.py``'s own module docstring (item 6) documents a real,
   previously-fixed bug where an "intended-positive" sample's candidate
   ``center_z`` selection could land in a genuinely empty-target overshoot
   band. This module does not assume that class of bug can never recur (or
   recur in some other form): every sample's target mask is checked
   directly (``target_mask >= score_threshold``, valid-region-masked) to
   decide which report bucket it belongs in, and any sample where
   ``meta["positive"]`` disagrees with the sample's ACTUAL target content
   is counted and surfaced as an explicit warning
   (:attr:`EvaluationReport.n_meta_positive_actual_empty` /
   :attr:`EvaluationReport.n_meta_negative_actual_nonempty`) rather than
   silently trusted. This is itself a useful QA signal for a reviewer: a
   nonzero count here means the synthetic-lesion cache has samples whose
   generation intent and actual content disagree.

3. Subject-clustered bootstrap -- why resampling SAMPLES (not subjects)
   would overstate confidence
   ------------------------------------------------------------------------
   Multiple cache samples come from the same subject (``manifest.json``'s
   ``per_bundle_counts``: several positive/negative samples per bundle, and
   a subject typically contributes both a "Test" and "Retest" bundle -- see
   the real ``p6_cache/val`` manifest's ``bundle_identities``). Samples from
   the same subject share that subject's PET/CT noise characteristics,
   anatomy, and reconstruction quality, so they are NOT independent draws;
   a plain per-sample bootstrap would treat 8 correlated samples from one
   subject as 8 independent data points and report a confidence interval
   far narrower than the data actually supports. :func:`_subject_clustered_
   bootstrap` instead resamples whole SUBJECTS (``meta["subject"]``, drawn
   with replacement) and includes every one of that subject's samples each
   time it is drawn -- the standard cluster/block bootstrap for grouped
   data (Efron & Tibshirani, 1993, *An Introduction to the Bootstrap*,
   Chapman & Hall, Ch. 3, "the bootstrap ... resampling the clusters, not
   the individual observations, when observations within a cluster are
   correlated"). The 95% CI is the [2.5th, 97.5th] percentile of the
   resampled-statistic distribution (the percentile bootstrap; ibid., Ch.
   13).

4. HD95/ASSD need a physical voxel spacing the cache does not itself store
   -- the documented default and how it was verified
   ------------------------------------------------------------------------
   ``Sample.meta`` (``dataset.py``'s :func:`~src.vascutrace.ml.dataset.
   build_sample`) intentionally carries only ``subject``/``session``/
   ``center_z``/``positive``/``side``/``sim_params``/schema-version keys --
   never ``pet_spacing_mm`` (that field lives on the upstream
   :class:`~src.vascutrace.data.contract.CropBundle`, which the cache
   deliberately never re-reads at evaluation time -- the whole point of the
   cache is to decouple evaluation from the raw bundle tree). Rather than
   silently assume a spacing, this module's default,
   :data:`DEFAULT_IN_PLANE_SPACING_MM` ``= (1.65, 1.65)``, was verified
   directly against the artifact: EVERY one of the 96 bundle directories
   under ``data/processed/p2/crops/p2-crop-v2`` as checked on 2026-07-16
   reports ``pet_spacing_mm == (1.65, 1.65, 2.0)`` (in-plane X/Y spacing is
   isotropic and, empirically, dataset-wide constant for this single
   -scanner QUADRA cohort) -- not the ``tensor_schema.py``-documented 2.0mm
   Z-axis spacing, which is a DIFFERENT axis. This is a documented,
   overridable default (``--spacing-mm`` on the CLI,
   ``spacing_mm=`` on :func:`evaluate_checkpoint`), not a silently
   hardcoded constant -- if a future crop build ever mixes reconstructions
   with different in-plane spacing, HD95/ASSD computed under this default
   would be wrong for those bundles, and re-verifying this constant against
   the bundle tree (the check above) is the first thing to redo.

5. Negatives report false-ACTIVATION COUNTS, never "clinical false
   positives"
   ------------------------------------------------------------------------
   This project's own scientific-boundary contract (``RESEARCH_PROTOTYPE_
   WARNING``, carried through every artifact this codebase produces) means
   no number here may imply a claim about real patient outcomes. A
   predicted-positive connected component on a healthy (empty-target) cache
   sample is therefore always reported as a "false-activation component" or
   "false-positive component" count -- a description of what the SEGMENTATION
   MODEL did on SIMULATED data -- never phrased as a clinical false-positive
   rate, which would imply a diagnostic claim this project is not making.

6. Never printing subject identifiers
   ------------------------------------------------------------------------
   Every field on :class:`EvaluationReport` and every line
   :meth:`EvaluationReport.to_markdown`/:meth:`~EvaluationReport.to_dict`
   produce is an aggregate count, mean, median, or CI over the whole cache
   -- no per-sample ``meta["subject"]``/``meta["session"]`` value is ever
   surfaced, matching ``cache.py``'s own "never prints or logs a bundle
   identity string itself" convention (module docstring, item 6) and this
   project's local-only, gitignored-artifact handling of subject
   identifiers (``src.vascutrace.data.split.save_subject_split``'s
   docstring).

7. The selection-metric suggestion
   ------------------------------------------------------------------------
   :attr:`EvaluationReport.selection_metric_value` is
   ``mean_positive_dice * negative_pct_fully_clean`` -- a SUGGESTION (never
   authoritative; the raw components remain fully visible in the report) for
   picking among checkpoints. A pure positive-Dice selection criterion (as
   the training loop's own ``best_val_metric``/``val_dice`` already uses --
   see ``train.py``) never penalizes a checkpoint that is trigger-happy on
   healthy cases; multiplying by the fraction of fully-clean negative cases
   ("did it also stay quiet where there is nothing to find") makes a
   checkpoint that trades positive segmentation quality for indiscriminate
   activation score visibly worse on this composite, without discarding
   the plain positive-Dice number a reader may prefer instead.

8. ``min_component_size`` -- a config-gated exploratory predicted-component
   pixel filter, default OFF
   ------------------------------------------------------------------------
   :func:`evaluate_checkpoint` accepts an optional ``min_component_size``
   (pixel-count; default ``metrics.DEFAULT_MIN_COMPONENT_SIZE == 0``, i.e.
   no filtering, so prior metric values are numerically preserved unless a
   caller opts in) and forwards it unchanged to every ``metrics.py`` call
   this module makes (:func:`~src.vascutrace.ml.metrics.dice`,
   :func:`~src.vascutrace.ml.metrics.iou_jaccard`,
   :func:`~src.vascutrace.ml.metrics.hd95`,
   :func:`~src.vascutrace.ml.metrics.assd`,
   :func:`~src.vascutrace.ml.metrics.lesion_detected`,
   :func:`~src.vascutrace.ml.metrics.lesion_component_confusion`,
   :func:`~src.vascutrace.ml.metrics.false_positive_components`) -- see
   ``metrics.py``'s own module docstring, item 8, for the full algorithm
   and the FDG-PET/CT research contract. The contract requires a
   3-D physical-volume gate; this 2-D pixel count is only an analogue.
   This module's contribution is
   purely plumbing: exposing the SAME already-defined, already-tested
   ``metrics.py`` parameter through :func:`evaluate_checkpoint`'s public
   signature and the CLI (``--min-component-size``), and recording the
   value used on :class:`EvaluationReport` (``min_component_size`` field)
   so a report is fully reproducible from its own printed configuration --
   no new component-filtering math lives in this module.
============================================================================
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.vascutrace.geometry import RESEARCH_PROTOTYPE_WARNING
from src.vascutrace.ml.cache import CachedSampleDataset, CacheSchemaError
from src.vascutrace.ml.checkpoint import CheckpointError, load_checkpoint
from src.vascutrace.ml.dataset import Sample
from src.vascutrace.ml.metrics import (
    DEFAULT_LESION_IOU_THRESHOLD,
    DEFAULT_MIN_COMPONENT_SIZE,
    DEFAULT_SCORE_THRESHOLD,
    assd,
    connected_components,
    dice,
    hd95,
    iou_jaccard,
    lesion_component_confusion,
    lesion_detected,
    precision_recall_f_beta,
)
from src.vascutrace.ml.model import abnormality_score, build_model
from src.vascutrace.ml.tensor_schema import TENSOR_SCHEMA_VERSION

__all__ = [
    "RESEARCH_PROTOTYPE_WARNING",
    "EVALUATE_MODULE_VERSION",
    "DEFAULT_IN_PLANE_SPACING_MM",
    "DEFAULT_N_BOOTSTRAP",
    "EvaluationError",
    "BootstrapCI",
    "PositiveMetrics",
    "NegativeMetrics",
    "EvaluationReport",
    "evaluate_checkpoint",
    "build_parser",
    "main",
]

EVALUATE_MODULE_VERSION = "p6-evaluate-v1"

# Empirically-verified, dataset-wide-constant in-plane (X, Y) PET voxel
# spacing (mm) -- see module docstring, item 4, for how this was verified
# and why the cache cannot supply it directly. Override via --spacing-mm
# (CLI) / spacing_mm= (evaluate_checkpoint) if the underlying crop build
# ever changes.
DEFAULT_IN_PLANE_SPACING_MM: tuple[float, float] = (1.65, 1.65)

DEFAULT_N_BOOTSTRAP = 2000
_BOOTSTRAP_CI_LOW_PCT = 2.5
_BOOTSTRAP_CI_HIGH_PCT = 97.5


class EvaluationError(RuntimeError):
    """Raised for an evaluate.py-level structural problem: a checkpoint/
    cache schema mismatch or an unsatisfiable device request. Distinct from
    :class:`~src.vascutrace.ml.checkpoint.CheckpointError` (checkpoint file
    I/O) and :class:`~src.vascutrace.ml.cache.CacheSchemaError` (cache
    manifest I/O), both of which this module also lets propagate
    unmodified where they originate.
    """


# ---------------------------------------------------------------------------
# Per-sample record -- module-internal (not part of the public report
# schema; EvaluationReport carries only aggregates -- see module docstring,
# item 6).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SampleRecord:
    subject: str
    meta_positive: bool
    target_nonempty: bool

    dice: float | None = None
    iou: float | None = None
    detected: bool | None = None
    hd95_mm: float | None = None
    assd_mm: float | None = None
    lesion_tp: int = 0
    lesion_fp: int = 0
    lesion_fn: int = 0

    fp_components: int | None = None
    fp_voxels: int | None = None


def _run_inference(
    model: torch.nn.Module, sample: Sample, device: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One sample's forward pass -> ``(score [H, W], target [H, W], valid
    [H, W])`` numpy arrays. ``score`` is
    :func:`~src.vascutrace.ml.model.abnormality_score`'s ``[0, 1]``
    monotonic sigmoid transform of the raw logit (never a calibrated
    calibrated probability; see ``model.py`` for that boundary),
    left un-thresholded so every ``metrics.py`` call downstream applies its
    own explicit ``threshold``.
    """
    left = sample.left_view.unsqueeze(0).to(device)
    right = sample.right_view.unsqueeze(0).to(device)
    pet_diff = sample.pet_diff.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(left, right, pet_diff)
        score = abnormality_score(logits)
    score_np = score.squeeze(0).squeeze(0).cpu().numpy()
    target_np = sample.target_mask.squeeze(0).numpy()
    valid_np = sample.valid_mask.squeeze(0).numpy()
    return score_np, target_np, valid_np


def _process_sample(
    score: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    meta: dict[str, Any],
    *,
    score_threshold: float,
    iou_threshold: float,
    spacing_mm: tuple[float, float],
    min_component_size: int = DEFAULT_MIN_COMPONENT_SIZE,
) -> _SampleRecord:
    """Compute every per-sample metric this report needs, for exactly ONE
    of the two branches (positive/lesion-bearing or negative/healthy) --
    see module docstring, item 2, for why the branch is chosen from ACTUAL
    target content, not ``meta["positive"]``. ``min_component_size`` (module
    docstring, item 8; default ``0`` == no filtering) is forwarded to every
    ``metrics.py`` call that forms PREDICTED components -- both the
    detection-matching (:func:`~src.vascutrace.ml.metrics.lesion_detected`/
    :func:`~src.vascutrace.ml.metrics.lesion_component_confusion`) and the
    IoU/Dice mask (:func:`~src.vascutrace.ml.metrics.dice`/
    :func:`~src.vascutrace.ml.metrics.iou_jaccard`) branches, plus the
    negative-sample false-activation count AND voxel count (both computed
    from the same filtered :func:`~src.vascutrace.ml.metrics.
    connected_components` call below, so ``fp_components`` and
    ``fp_voxels`` never disagree about which components survived
    filtering).
    """
    subject = str(meta["subject"])
    meta_positive = bool(meta["positive"])

    target_bin = (target >= score_threshold) & (valid >= 0.5)
    target_nonempty = bool(target_bin.any())

    if target_nonempty:
        d = dice(
            score,
            target,
            threshold=score_threshold,
            valid_mask=valid,
            min_component_size=min_component_size,
        )
        i = iou_jaccard(
            score,
            target,
            threshold=score_threshold,
            valid_mask=valid,
            min_component_size=min_component_size,
        )
        detected = lesion_detected(
            score,
            target,
            iou_threshold=iou_threshold,
            threshold=score_threshold,
            valid_mask=valid,
            min_component_size=min_component_size,
        )
        counts = lesion_component_confusion(
            score,
            target,
            iou_threshold=iou_threshold,
            threshold=score_threshold,
            valid_mask=valid,
            min_component_size=min_component_size,
        )
        hd95_mm: float | None = None
        assd_mm: float | None = None
        if detected:
            # HD95/ASSD only on found lesions -- an undetected lesion has no
            # meaningful boundary correspondence to measure (see metrics.py's
            # own empty/undefined-mask policy and this module docstring's
            # framing). Use the same filtered prediction as Dice, IoU, and
            # component matching so one EvaluationReport never mixes two
            # different operating points.
            hd95_mm = hd95(
                score,
                target,
                spacing_mm,
                threshold=score_threshold,
                valid_mask=valid,
                min_component_size=min_component_size,
            )
            assd_mm = assd(
                score,
                target,
                spacing_mm,
                threshold=score_threshold,
                valid_mask=valid,
                min_component_size=min_component_size,
            )
        return _SampleRecord(
            subject=subject,
            meta_positive=meta_positive,
            target_nonempty=True,
            dice=d,
            iou=i,
            detected=detected,
            hd95_mm=hd95_mm,
            assd_mm=assd_mm,
            lesion_tp=counts.tp,
            lesion_fp=counts.fp,
            lesion_fn=counts.fn,
        )

    # A single connected_components(...) call (rather than
    # false_positive_components(...) for the count plus a separate raw
    # pixel sum for fp_voxels) so the "fully clean" component count and the
    # false-activation VOXEL count are computed from the SAME
    # min_component_size-filtered label set -- with filtering on, a raw
    # (score >= threshold) pixel sum would double-count voxels belonging to
    # components this same call already dropped, silently disagreeing with
    # fp_components == 0 ("fully clean").
    fp_labeled, fp_components = connected_components(
        score,
        threshold=score_threshold,
        valid_mask=valid,
        min_component_size=min_component_size,
    )
    fp_voxels = int(np.sum(fp_labeled > 0))
    return _SampleRecord(
        subject=subject,
        meta_positive=meta_positive,
        target_nonempty=False,
        fp_components=fp_components,
        fp_voxels=fp_voxels,
    )


# ---------------------------------------------------------------------------
# Aggregate statistic functions -- each is Sequence[_SampleRecord] -> float,
# so the SAME function computes both the point estimate (over all records)
# and every bootstrap resample's statistic (see
# _subject_clustered_bootstrap).
# ---------------------------------------------------------------------------


def _stat_mean_positive_dice(records: Sequence[_SampleRecord]) -> float:
    vals = [r.dice for r in records if r.target_nonempty and r.dice is not None]
    return float(np.mean(vals)) if vals else float("nan")


def _stat_mean_positive_iou(records: Sequence[_SampleRecord]) -> float:
    vals = [r.iou for r in records if r.target_nonempty and r.iou is not None]
    return float(np.mean(vals)) if vals else float("nan")


def _stat_positive_detection_fraction(records: Sequence[_SampleRecord]) -> float:
    tp = sum(r.lesion_tp for r in records if r.target_nonempty)
    fn = sum(r.lesion_fn for r in records if r.target_nonempty)
    denom = tp + fn
    return float(tp / denom) if denom > 0 else float("nan")


def _stat_pct_fully_clean(records: Sequence[_SampleRecord]) -> float:
    vals = [
        r.fp_components == 0
        for r in records
        if not r.target_nonempty and r.fp_components is not None
    ]
    return float(np.mean(vals)) if vals else float("nan")


def _stat_mean_fp_components(records: Sequence[_SampleRecord]) -> float:
    vals = [
        r.fp_components
        for r in records
        if not r.target_nonempty and r.fp_components is not None
    ]
    return float(np.mean(vals)) if vals else float("nan")


def _stat_selection_metric(records: Sequence[_SampleRecord]) -> float:
    """See module docstring, item 7."""
    mean_dice = _stat_mean_positive_dice(records)
    pct_clean = _stat_pct_fully_clean(records)
    if math.isnan(mean_dice) or math.isnan(pct_clean):
        return float("nan")
    return mean_dice * pct_clean


# ---------------------------------------------------------------------------
# Subject-clustered bootstrap -- see module docstring, item 3.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapCI:
    point_estimate: float
    ci_low: float
    ci_high: float
    n_resamples: int


def _subject_clustered_bootstrap(
    records: Sequence[_SampleRecord],
    statistic_fn: Callable[[Sequence[_SampleRecord]], float],
    *,
    n_resamples: int,
    seed: int,
) -> BootstrapCI:
    point_estimate = statistic_fn(records)

    subjects = sorted({r.subject for r in records})
    if n_resamples <= 0 or len(subjects) < 2:
        # Degenerate: no bootstrap requested, or too few independent
        # clusters to resample meaningfully -- report the point estimate
        # only (0 resamples signals this to the report reader, rather than
        # a misleadingly tight CI computed on nothing).
        return BootstrapCI(
            point_estimate=point_estimate,
            ci_low=point_estimate,
            ci_high=point_estimate,
            n_resamples=0,
        )

    by_subject: dict[str, list[_SampleRecord]] = defaultdict(list)
    for r in records:
        by_subject[r.subject].append(r)

    rng = np.random.default_rng(seed)
    values = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        drawn_subjects = rng.choice(subjects, size=len(subjects), replace=True)
        resampled: list[_SampleRecord] = []
        for s in drawn_subjects:
            resampled.extend(by_subject[s])
        values[b] = statistic_fn(resampled)

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return BootstrapCI(
            point_estimate=point_estimate,
            ci_low=float("nan"),
            ci_high=float("nan"),
            n_resamples=n_resamples,
        )
    ci_low = float(np.percentile(finite, _BOOTSTRAP_CI_LOW_PCT))
    ci_high = float(np.percentile(finite, _BOOTSTRAP_CI_HIGH_PCT))
    return BootstrapCI(
        point_estimate=point_estimate,
        ci_low=ci_low,
        ci_high=ci_high,
        n_resamples=n_resamples,
    )


# ---------------------------------------------------------------------------
# Report schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PositiveMetrics:
    """Aggregate metrics over positive simulated-source samples only."""

    n_positive_samples: int

    mean_dice: float
    median_dice: float
    std_dice: float
    pct_dice_above_0_5: float

    mean_iou: float
    median_iou: float

    n_lesions_total: int
    n_lesions_detected: int
    n_lesions_missed: int
    positive_detection_fraction: float
    lesion_precision: float
    lesion_f1: float
    lesion_f2: float

    n_found_for_surface_metrics: int
    mean_hd95_found_mm: float | None
    median_hd95_found_mm: float | None
    mean_assd_found_mm: float | None
    median_assd_found_mm: float | None


@dataclass(frozen=True)
class NegativeMetrics:
    """Aggregate metrics over healthy (negative, empty-target) samples
    only -- false-ACTIVATION counts, never "clinical false positives" (see
    module docstring, item 5).
    """

    n_negative_samples: int
    pct_fully_clean: float
    mean_fp_components_per_case: float
    median_fp_components_per_case: float
    total_fp_components: int
    mean_fp_voxels_per_case: float


@dataclass(frozen=True)
class EvaluationReport:
    """The full evaluation report for one checkpoint on one cache. Every
    field is an aggregate over the whole cache -- see module docstring,
    item 6, "never printing subject identifiers".
    """

    checkpoint_path: str
    cache_dir: str
    device: str
    score_threshold: float
    iou_threshold: float
    spacing_mm: tuple[float, float]
    n_bootstrap: int
    min_component_size: int

    n_samples: int
    n_positive: int
    n_negative: int
    n_meta_positive_actual_empty: int
    n_meta_negative_actual_nonempty: int

    positives: PositiveMetrics
    negatives: NegativeMetrics
    bootstrap: dict[str, BootstrapCI]

    selection_metric_name: str
    selection_metric_value: float
    selection_metric_formula: str

    model_signature: str
    checkpoint_tensor_schema_version: str
    checkpoint_best_val_metric: float | None
    checkpoint_best_val_metric_name: str

    warnings: tuple[str, ...]
    research_prototype_warning: str
    generated_at: str

    def to_dict(self) -> dict[str, Any]:
        """Plain, JSON-serializable dict -- aggregate/non-identifying only
        (see module docstring, item 6).

        Nested ``lesion_*`` and ``positive_detection_fraction`` keys describe
        synthetic-source component metrics. ``metric_scope`` binds their
        nonclinical meaning.
        """
        payload = asdict(self)
        payload["metric_scope"] = (
            "simulated-source component detection in synthetic-abnormality "
            "samples; not clinical sensitivity or specificity"
        )
        payload["legacy_metric_field_names"] = (
            "nested lesion_* keys are compatibility names for synthetic-source "
            "component metrics"
        )
        return payload

    def to_markdown(self) -> str:
        """Human-readable aggregate report -- see :func:`main` for the CLI
        that prints and optionally writes this."""
        lines: list[str] = []
        lines.append(f"# VascuTrace evaluation report ({EVALUATE_MODULE_VERSION})")
        lines.append("")
        lines.append(f"> {self.research_prototype_warning}")
        lines.append("")
        lines.append(f"- Checkpoint: `{self.checkpoint_path}`")
        lines.append(f"- Cache: `{self.cache_dir}`")
        lines.append(f"- Device: `{self.device}`")
        lines.append(f"- Model signature: `{self.model_signature}`")
        lines.append(
            f"- Checkpoint best val metric: {self.checkpoint_best_val_metric} "
            f"({self.checkpoint_best_val_metric_name})"
        )
        lines.append(
            f"- abnormality_score_threshold={self.score_threshold}, "
            f"iou_threshold={self.iou_threshold}, "
            f"spacing_mm={self.spacing_mm}, min_component_size={self.min_component_size}"
        )
        lines.append(f"- Generated: {self.generated_at}")
        lines.append("")
        lines.append(
            f"Total samples: {self.n_samples} "
            f"(positive={self.n_positive}, negative={self.n_negative})"
        )
        if self.n_meta_positive_actual_empty or self.n_meta_negative_actual_nonempty:
            lines.append(
                f"- QA flag: {self.n_meta_positive_actual_empty} meta-positive "
                "sample(s) had an actually-empty target; "
                f"{self.n_meta_negative_actual_nonempty} meta-negative sample(s) "
                "had actual target content. The positive/negative split above "
                "follows ACTUAL target content, not the cache's intent flag "
                "(see evaluate.py's module docstring, item 2)."
            )
        lines.append("")

        p = self.positives
        lines.append("## Positive simulated-source samples")
        lines.append(f"- n = {p.n_positive_samples}")
        lines.append(
            f"- Dice: mean={p.mean_dice:.4f}, median={p.median_dice:.4f}, "
            f"std={p.std_dice:.4f}"
        )
        lines.append(f"- IoU: mean={p.mean_iou:.4f}, median={p.median_iou:.4f}")
        pct_str = (
            f"{p.pct_dice_above_0_5:.1%}"
            if not math.isnan(p.pct_dice_above_0_5)
            else "n/a"
        )
        lines.append(f"- % positive samples with Dice > 0.5: {pct_str}")
        lines.append(
            "- Synthetic-source component detection: "
            f"{p.n_lesions_detected}/{p.n_lesions_total} matched "
            f"(positive_detection_fraction={p.positive_detection_fraction:.4f}), "
            f"precision={p.lesion_precision:.4f}, F1={p.lesion_f1:.4f}, "
            f"F2={p.lesion_f2:.4f}"
        )
        if p.mean_hd95_found_mm is not None:
            lines.append(
                f"- Surface distance (matched synthetic-source components only, "
                f"n={p.n_found_for_surface_metrics}): "
                f"HD95 mean={p.mean_hd95_found_mm:.2f}mm "
                f"median={p.median_hd95_found_mm:.2f}mm; "
                f"ASSD mean={p.mean_assd_found_mm:.2f}mm "
                f"median={p.median_assd_found_mm:.2f}mm"
            )
        else:
            lines.append(
                "- Surface distance: n/a (no matched synthetic-source components)"
            )
        lines.append("")

        n = self.negatives
        lines.append("## Negative target-empty background samples")
        lines.append(f"- n = {n.n_negative_samples}")
        clean_str = (
            f"{n.pct_fully_clean:.1%}" if not math.isnan(n.pct_fully_clean) else "n/a"
        )
        lines.append(f"- % fully clean (zero false-activation components): {clean_str}")
        lines.append(
            f"- False-activation components per case: "
            f"mean={n.mean_fp_components_per_case:.3f}, "
            f"median={n.median_fp_components_per_case:.3f}, "
            f"total={n.total_fp_components}"
        )
        lines.append(
            f"- False-activation voxels per case: mean={n.mean_fp_voxels_per_case:.2f}"
        )
        lines.append("")

        lines.append(
            f"## Bootstrap 95% CIs (subject-clustered, n_resamples={self.n_bootstrap})"
        )
        for name, ci in self.bootstrap.items():
            lines.append(
                f"- {name}: {ci.point_estimate:.4f}  "
                f"[{ci.ci_low:.4f}, {ci.ci_high:.4f}]  (n_resamples={ci.n_resamples})"
            )
        lines.append("")

        lines.append("## Selection metric (suggestion, not authoritative)")
        lines.append(
            f"- {self.selection_metric_name} = {self.selection_metric_value:.4f}"
        )
        lines.append(f"- formula: {self.selection_metric_formula}")
        lines.append("")

        if self.warnings:
            lines.append("## Warnings")
            for w in self.warnings:
                lines.append(f"- {w}")
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate(
    records: Sequence[_SampleRecord],
) -> tuple[PositiveMetrics, NegativeMetrics, list[str]]:
    warnings: list[str] = []
    positive_records = [r for r in records if r.target_nonempty]
    negative_records = [r for r in records if not r.target_nonempty]

    dice_vals = [r.dice for r in positive_records if r.dice is not None]
    iou_vals = [r.iou for r in positive_records if r.iou is not None]

    lesion_tp = sum(r.lesion_tp for r in positive_records)
    lesion_fp = sum(r.lesion_fp for r in positive_records)
    lesion_fn = sum(r.lesion_fn for r in positive_records)
    prf1 = precision_recall_f_beta(lesion_tp, lesion_fp, lesion_fn, beta=1.0)
    prf2 = precision_recall_f_beta(lesion_tp, lesion_fp, lesion_fn, beta=2.0)

    found_hd95 = [
        r.hd95_mm
        for r in positive_records
        if r.detected and r.hd95_mm is not None and math.isfinite(r.hd95_mm)
    ]
    found_assd = [
        r.assd_mm
        for r in positive_records
        if r.detected and r.assd_mm is not None and math.isfinite(r.assd_mm)
    ]
    n_detected_but_undefined_surface = sum(
        1
        for r in positive_records
        if r.detected and (r.hd95_mm is None or not math.isfinite(r.hd95_mm))
    )
    if n_detected_but_undefined_surface:
        warnings.append(
            f"{n_detected_but_undefined_surface} detected lesion(s) had an "
            "undefined (non-finite) surface distance -- excluded from the "
            "HD95/ASSD aggregate."
        )

    positives = PositiveMetrics(
        n_positive_samples=len(positive_records),
        mean_dice=float(np.mean(dice_vals)) if dice_vals else float("nan"),
        median_dice=float(np.median(dice_vals)) if dice_vals else float("nan"),
        std_dice=float(np.std(dice_vals, ddof=1)) if len(dice_vals) > 1 else 0.0,
        pct_dice_above_0_5=(
            float(np.mean([d > 0.5 for d in dice_vals])) if dice_vals else float("nan")
        ),
        mean_iou=float(np.mean(iou_vals)) if iou_vals else float("nan"),
        median_iou=float(np.median(iou_vals)) if iou_vals else float("nan"),
        n_lesions_total=lesion_tp + lesion_fn,
        n_lesions_detected=lesion_tp,
        n_lesions_missed=lesion_fn,
        positive_detection_fraction=prf1.recall,
        lesion_precision=prf1.precision,
        lesion_f1=prf1.f_beta,
        lesion_f2=prf2.f_beta,
        n_found_for_surface_metrics=len(found_hd95),
        mean_hd95_found_mm=float(np.mean(found_hd95)) if found_hd95 else None,
        median_hd95_found_mm=float(np.median(found_hd95)) if found_hd95 else None,
        mean_assd_found_mm=float(np.mean(found_assd)) if found_assd else None,
        median_assd_found_mm=float(np.median(found_assd)) if found_assd else None,
    )

    fp_component_vals = [
        r.fp_components for r in negative_records if r.fp_components is not None
    ]
    fp_voxel_vals = [r.fp_voxels for r in negative_records if r.fp_voxels is not None]
    negatives = NegativeMetrics(
        n_negative_samples=len(negative_records),
        pct_fully_clean=(
            float(np.mean([v == 0 for v in fp_component_vals]))
            if fp_component_vals
            else float("nan")
        ),
        mean_fp_components_per_case=(
            float(np.mean(fp_component_vals)) if fp_component_vals else float("nan")
        ),
        median_fp_components_per_case=(
            float(np.median(fp_component_vals)) if fp_component_vals else float("nan")
        ),
        total_fp_components=int(sum(fp_component_vals)),
        mean_fp_voxels_per_case=(
            float(np.mean(fp_voxel_vals)) if fp_voxel_vals else float("nan")
        ),
    )

    return positives, negatives, warnings


# ---------------------------------------------------------------------------
# evaluate_checkpoint
# ---------------------------------------------------------------------------


def evaluate_checkpoint(
    checkpoint_path: Path,
    cache_dir: Path,
    *,
    device: str = "cpu",
    iou_threshold: float = DEFAULT_LESION_IOU_THRESHOLD,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    spacing_mm: tuple[float, float] = DEFAULT_IN_PLANE_SPACING_MM,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    bootstrap_seed: int = 0,
    min_component_size: int = DEFAULT_MIN_COMPONENT_SIZE,
) -> EvaluationReport:
    """Load ``checkpoint_path``, run it over every sample in ``cache_dir``,
    and return an :class:`EvaluationReport`. See module docstring for the
    full design (positives/negatives separation, subject-clustered
    bootstrap, spacing default, and item 8 for ``min_component_size``).
    ``min_component_size`` defaults to ``0`` (no filtering) -- passing the
    default preserves the prior unfiltered metric calculation.
    """
    checkpoint_path = Path(checkpoint_path)
    cache_dir = Path(cache_dir)

    if not isinstance(min_component_size, Integral) or isinstance(
        min_component_size, bool
    ):
        raise EvaluationError(
            "min_component_size must be a non-negative integer, got "
            f"{min_component_size!r}"
        )
    min_component_size = int(min_component_size)
    if min_component_size < 0:
        raise EvaluationError(
            f"min_component_size must be >= 0, got {min_component_size}"
        )

    if device == "cuda" and not torch.cuda.is_available():
        raise EvaluationError(
            f"device={device!r} requested but torch.cuda.is_available() is False"
        )

    payload = load_checkpoint(checkpoint_path)  # CheckpointError propagates unmodified
    if payload.tensor_schema_version != TENSOR_SCHEMA_VERSION:
        raise EvaluationError(
            f"checkpoint tensor_schema_version={payload.tensor_schema_version!r} != "
            f"current code's {TENSOR_SCHEMA_VERSION!r} -- refusing to evaluate a "
            "schema-incompatible checkpoint"
        )

    # Evaluation deliberately audits generation-intent/actual-target mismatches
    # (module docstring, item 2). All other cache bindings remain fail-closed.
    dataset = CachedSampleDataset(
        cache_dir, enforce_manifest_class_content=False
    )  # CacheSchemaError propagates unmodified
    # (CachedSampleDataset's own constructor already raises CacheSchemaError on a
    # missing/incompatible manifest or zero sample_*.npz files -- non-emptiness is
    # therefore already guaranteed by this point.)
    if dataset.manifest.get("crop_schema_version") != payload.crop_schema_version:
        raise EvaluationError(
            f"cache crop_schema_version={dataset.manifest.get('crop_schema_version')!r} "
            f"!= checkpoint crop_schema_version={payload.crop_schema_version!r} -- "
            "this checkpoint was not trained against a compatible crop build"
        )

    model = build_model(payload.model_config)
    model.load_state_dict(payload.model_state_dict)
    model.to(device)
    model.eval()

    records: list[_SampleRecord] = []
    for i in range(len(dataset)):
        sample = dataset[i]
        score, target, valid = _run_inference(model, sample, device)
        records.append(
            _process_sample(
                score,
                target,
                valid,
                sample.meta,
                score_threshold=score_threshold,
                iou_threshold=iou_threshold,
                spacing_mm=spacing_mm,
                min_component_size=min_component_size,
            )
        )

    n_meta_positive_actual_empty = sum(
        1 for r in records if r.meta_positive and not r.target_nonempty
    )
    n_meta_negative_actual_nonempty = sum(
        1 for r in records if not r.meta_positive and r.target_nonempty
    )

    positives, negatives, agg_warnings = _aggregate(records)
    warnings_list = list(agg_warnings)
    if n_meta_positive_actual_empty or n_meta_negative_actual_nonempty:
        warnings_list.append(
            f"{n_meta_positive_actual_empty} sample(s) labeled positive in the "
            f"cache manifest have an actually-empty target; "
            f"{n_meta_negative_actual_nonempty} sample(s) labeled negative have "
            "actual target content -- see module docstring, item 2."
        )

    n_unique_subjects = len({r.subject for r in records})
    if n_bootstrap > 0 and n_unique_subjects < 2:
        warnings_list.append(
            f"only {n_unique_subjects} unique subject(s) present -- "
            "subject-clustered bootstrap CIs are degenerate (point estimate only, "
            "n_resamples reported as 0)"
        )

    bootstrap: dict[str, BootstrapCI] = {
        "positive_mean_dice": _subject_clustered_bootstrap(
            records,
            _stat_mean_positive_dice,
            n_resamples=n_bootstrap,
            seed=bootstrap_seed,
        ),
        "positive_mean_iou": _subject_clustered_bootstrap(
            records,
            _stat_mean_positive_iou,
            n_resamples=n_bootstrap,
            seed=bootstrap_seed + 1,
        ),
        "positive_detection_fraction": _subject_clustered_bootstrap(
            records,
            _stat_positive_detection_fraction,
            n_resamples=n_bootstrap,
            seed=bootstrap_seed + 2,
        ),
        "negative_pct_fully_clean": _subject_clustered_bootstrap(
            records,
            _stat_pct_fully_clean,
            n_resamples=n_bootstrap,
            seed=bootstrap_seed + 3,
        ),
        "negative_mean_fp_components": _subject_clustered_bootstrap(
            records,
            _stat_mean_fp_components,
            n_resamples=n_bootstrap,
            seed=bootstrap_seed + 4,
        ),
        "selection_metric": _subject_clustered_bootstrap(
            records,
            _stat_selection_metric,
            n_resamples=n_bootstrap,
            seed=bootstrap_seed + 5,
        ),
    }

    return EvaluationReport(
        checkpoint_path=str(checkpoint_path),
        cache_dir=str(cache_dir),
        device=device,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
        spacing_mm=tuple(spacing_mm),
        n_bootstrap=n_bootstrap,
        min_component_size=min_component_size,
        n_samples=len(records),
        n_positive=positives.n_positive_samples,
        n_negative=negatives.n_negative_samples,
        n_meta_positive_actual_empty=n_meta_positive_actual_empty,
        n_meta_negative_actual_nonempty=n_meta_negative_actual_nonempty,
        positives=positives,
        negatives=negatives,
        bootstrap=bootstrap,
        selection_metric_name="dice_x_clean",
        selection_metric_value=_stat_selection_metric(records),
        selection_metric_formula="mean_positive_dice * negative_pct_fully_clean",
        model_signature=payload.model_signature,
        checkpoint_tensor_schema_version=payload.tensor_schema_version,
        checkpoint_best_val_metric=payload.best_val_metric,
        checkpoint_best_val_metric_name=payload.best_val_metric_name,
        warnings=tuple(warnings_list),
        research_prototype_warning=RESEARCH_PROTOTYPE_WARNING,
        generated_at=datetime.now(UTC).isoformat(),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vascutrace-ml-evaluate",
        description=(
            "Evaluate a VascuTrace checkpoint's segmentation + detection "
            "accuracy on a precomputed sample cache."
        ),
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to a checkpoint .pt file"
    )
    parser.add_argument(
        "--cache", required=True, help="Path to a p6-cache-v1 cache directory"
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--iou-threshold", type=float, default=DEFAULT_LESION_IOU_THRESHOLD
    )
    parser.add_argument(
        "--score-threshold", type=float, default=DEFAULT_SCORE_THRESHOLD
    )
    parser.add_argument(
        "--spacing-mm",
        type=float,
        nargs=2,
        default=list(DEFAULT_IN_PLANE_SPACING_MM),
        metavar=("SPACING_H", "SPACING_W"),
        help="In-plane (H, W) PET voxel spacing in mm for HD95/ASSD (see module docstring)",
    )
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap resampling seed")
    parser.add_argument(
        "--min-component-size",
        type=int,
        default=DEFAULT_MIN_COMPONENT_SIZE,
        help=(
            "Minimum PREDICTED-component pixel count to survive scoring "
            "(default 0 == no filtering, numerically preserving prior "
            "metric behavior). Applied to detection-matching, IoU/Dice, and "
            "surface-distance prediction masks, and the negative-sample "
            "false-activation "
            "count -- see module docstring, item 8, and metrics.py's own "
            "module docstring, item 8, for the full contract citation."
        ),
    )
    parser.add_argument(
        "--out", default=None, help="Write the markdown report to this path as well"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        report = evaluate_checkpoint(
            Path(args.checkpoint),
            Path(args.cache),
            device=args.device,
            iou_threshold=args.iou_threshold,
            score_threshold=args.score_threshold,
            spacing_mm=(args.spacing_mm[0], args.spacing_mm[1]),
            n_bootstrap=args.n_bootstrap,
            bootstrap_seed=args.seed,
            min_component_size=args.min_component_size,
        )
    except (EvaluationError, CheckpointError, CacheSchemaError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    markdown = report.to_markdown()
    print(markdown)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown)
        print(f"\nWrote report to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
