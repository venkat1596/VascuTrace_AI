"""Pure segmentation + detection metric functions (VascuTrace Phase 6).

RESEARCH_PROTOTYPE_WARNING
---------------------------------------------------------------------------
Research prototype. Trained and evaluated using simulated vascular-like
abnormalities, not confirmed human post-angioplasty lesions.
---------------------------------------------------------------------------

Implementation notes
============================================================================
This module answers the question a segmentation-paper reviewer always asks
first: "Dice alone is not enough -- what does this model actually get right
and wrong, at the pixel level AND at the clinically-meaningful lesion
level?" Every function here is a pure ``numpy``/``torch`` -> ``float`` (or
small dataclass) computation with no model, dataset, or I/O dependency, so
each one is directly testable against hand-computed known arrays (see
``tests/test_ml_metrics.py``).

1. Two families of metric, because they answer different questions
   ------------------------------------------------------------------------
   *Voxelwise* metrics (:func:`dice`, :func:`iou_jaccard`,
   :func:`positive_recall`, :func:`precision_ppv`, :func:`specificity`,
   :func:`pixel_accuracy`, :func:`hd95`, :func:`assd`) score how well the
   predicted region overlaps/bounds the true region, pixel by pixel or
   boundary point by boundary point. *Lesion/component-level* metrics
   (:func:`connected_components`, :func:`lesion_detected`,
   :func:`lesion_component_confusion`, :func:`false_positive_components`)
   score a coarser, arguably more clinically relevant question: "did the
   model notice the lesion was there at all?", independent of how precisely
   its boundary was traced. A model can have mediocre voxelwise Dice on a
   correctly-located lesion (imprecise boundary) yet perfect lesion-level
   detection, or the reverse (a spuriously high voxelwise accuracy on an
   all-healthy case that is actually a `pixel_accuracy` artifact of a
   overwhelmingly-background image) -- reporting both families, never just
   one, is this module's central honesty commitment (see also
   ``evaluate.py``'s module docstring, which separates the two families
   into distinct report sections rather than blending them into one
   "accuracy" number).

2. Threshold discipline -- every function takes an abnormality-score map or a binary
   mask, and thresholds internally
   ------------------------------------------------------------------------
   Every ``pred`` argument in this module may be either a raw ``[0, 1]``
   score map (e.g. ``abnormality_score(logits)`` from
   ``model.py``) or an already-binary ``{0, 1}`` mask; every function
   applies ``pred >= threshold`` (default ``0.5``) itself before computing
   anything overlap-based, matching the fixed-threshold-Dice convention
   ``model.py``'s own :func:`~src.vascutrace.ml.model.dice_score` already
   establishes in this project (thresholding a raw sigmoid score at
   0.5, not a raw logit -- ``sigmoid(0) == 0.5``, so thresholding logits
   directly at 0.5 would silently be wrong; callers are expected to have
   already applied :func:`~src.vascutrace.ml.model.abnormality_score` if
   starting from logits). ``target`` is thresholded the same way (defensive
   -- ground-truth masks are already exactly ``{0.0, 1.0}`` by construction
   in ``tensor_schema.py``, so this is a no-op for well-formed data, but
   keeps every function total/safe if handed a soft target).

3. ``valid_mask`` -- excluding out-of-FOV pixels correctly, not just
   zeroing them
   ------------------------------------------------------------------------
   Every function accepts an optional ``valid_mask`` (the same
   ``[H, W]``/``[1, H, W]`` center-slice valid-PET mask
   ``tensor_schema.py`` defines). For the *positive-region-only* metrics
   (:func:`dice`, :func:`iou_jaccard`, the connected-component/lesion
   functions), masking is implemented as ``pred_bin & valid`` /
   ``target_bin & valid`` before the overlap arithmetic -- correct because
   Dice/IoU/component-detection only ever sum over *positive* pixels, so an
   invalid pixel simply never contributes to either set. For the
   *four-way-confusion* metrics (:func:`positive_recall`,
   :func:`precision_ppv`, :func:`specificity`, :func:`pixel_accuracy`),
   AND-masking into ``pred_bin``/``target_bin`` would be silently WRONG: an
   invalid pixel would present as ``pred=False, target=False`` and get
   counted as a spurious true negative, inflating both specificity and
   pixel accuracy by every invalid pixel in the image. This module
   therefore computes those four metrics from a single, separately
   -implemented :func:`_pixel_confusion` helper that carries ``valid_mask``
   as an explicit *inclusion* mask applied to all four of TP/FP/FN/TN
   simultaneously (``tests/test_ml_metrics.py``'s
   ``TestVoxelwiseConfusionMetrics::test_valid_mask_excludes_invalid_pixel_from_true_negative_count``
   constructs a case where the naive AND-into-prediction approach and the
   correct inclusion-mask approach give different, checkable answers).

4. Surface distances (:func:`hd95`, :func:`assd`) -- the standard
   erosion + Euclidean-distance-transform algorithm
   ------------------------------------------------------------------------
   Both functions implement the widely-used "boundary erosion + exact
   Euclidean distance transform" algorithm for computing the (one
   -directional) set of nearest-boundary-point distances, exactly as
   popularized by the ``medpy.metric.binary`` reference implementation
   (Maier et al., ``medpy``, function ``__surface_distances``: erode each
   binary mask with a face-connectivity structuring element
   (``connectivity=1``) to get its boundary shell, take the Euclidean
   distance transform (scaled by voxel ``spacing``, via
   ``scipy.ndimage.distance_transform_edt``) of the *complement of the
   other mask's boundary shell*, then read that distance field off at each
   of this mask's own boundary pixels). Using the distance transform of the
   boundary's complement (not the whole mask's complement) is what makes
   this correct for boundary points that lie *inside* the other mask's
   interior, not just for points fully outside it -- see
   :func:`_surface_distances`'s own docstring for the full argument.
   ``connectivity=1`` (face/edge neighbors only -- a "thin", one-voxel
   -shell boundary) is deliberately DIFFERENT from the
   :func:`connected_components` family's ``connectivity`` (full/maximal --
   8-connectivity in 2D, 26-connectivity in 3D, per this implementation's own
   "LESION/component level (26-connectivity)" instruction): the two serve
   opposite purposes -- boundary extraction wants the thinnest correct
   shell (so a diagonal "corner" pixel is not spuriously excluded from the
   boundary by counting a diagonal neighbor as adjacent), while
   component labeling wants the MOST generous connectivity so that two
   diagonally-touching foreground blobs are correctly treated as one
   lesion rather than artificially split into two.

   - :func:`hd95` reports the symmetric 95th-percentile Hausdorff distance
     -- ``max(P95(d(pred -> target)), P95(d(target -> pred)))`` -- the
     "take the max of the two one-directional 95th percentiles" convention
     established as the standard medical-segmentation-challenge metric by
     Menze et al., 2015, *The Multimodal Brain Tumor Image Segmentation
     Benchmark (BRATS)*, IEEE Trans. Medical Imaging, Sec. III.C.2
     ("Boundary evaluation"): "we compute the 95th percentile of these
     distances instead of the maximum ... to be more robust with respect
     to small outlier regions" (the classical, non-robustified Hausdorff
     distance itself traces to Huttenlocher, Klanderman & Rucklidge, 1993,
     *Comparing Images Using the Hausdorff Distance*, IEEE Trans. PAMI).
   - :func:`assd` reports the (mean-of-means) average symmetric surface
     distance -- ``(mean(d(pred -> target)) + mean(d(target -> pred))) / 2``
     -- following the SLIVER07 liver-segmentation-challenge convention
     (Heimann et al., 2009, *Comparison and Evaluation of Methods for Liver
     Segmentation From CT Datasets*, IEEE Trans. Medical Imaging, Sec.
     II.B.5 "Average symmetric surface distance").

5. Lesion/component-level detection -- the LiTS-style "did the model find
   it" IoU-overlap criterion
   ------------------------------------------------------------------------
   :func:`lesion_component_confusion` labels ``pred``/``target`` into
   connected components (26-connectivity in 3D / 8-connectivity in 2D --
   :func:`connected_components`'s own ``scipy.ndimage.
   generate_binary_structure(ndim, ndim)`` maximal-connectivity structuring
   element), then matches each ground-truth component to its best
   -overlapping predicted component by component-level IoU
   (``|gt ∩ pred| / |gt ∪ pred|``); a ground-truth component with a best
   -match IoU ``>= iou_threshold`` is a detection (TP), otherwise a miss
   (FN); a predicted component that is never any ground-truth component's
   best match at that threshold is a false-positive component (FP).
   Matching each GT component to only its single best-overlapping
   predicted component (rather than a full bipartite/Hungarian assignment)
   is a deliberate simplification, correct for this project's synthetic
   -lesion regime where lesions are simulated as isolated, well-separated
   blobs (``simulation/anomaly.py``) -- documented here rather than
   silently assumed. This overlap-threshold detection criterion mirrors the
   lesion-level detection metric defined for the Liver Tumor Segmentation
   Benchmark (Bilic et al., 2019, *The Liver Tumor Segmentation Benchmark
   (LiTS)*, arXiv:1901.04056, Sec. 4.2 "Detection Metrics": a predicted
   lesion counts as detected if the overlap between the prediction and the
   corresponding ground truth lesion is non-zero / above a set threshold).
   :data:`DEFAULT_LESION_IOU_THRESHOLD` (``0.1``) is a low bar deliberately
   -- consistent with this implementation's own default -- because the question this
   metric asks is "did the model notice the lesion at all", not "did it
   trace its boundary precisely" (:func:`dice`/:func:`iou_jaccard` already
   answer the boundary-precision question for the lesions that *are*
   found).

6. Precision/Recall/F-beta -- the classical retrieval formula, generic
   over both pixel-level and lesion-level TP/FP/FN counts
   ------------------------------------------------------------------------
   :func:`precision_recall_f_beta` implements the standard
   ``F_beta = (1 + beta^2) * P * R / (beta^2 * P + R)`` weighted harmonic
   mean (van Rijsbergen, 1979, *Information Retrieval*, 2nd ed., Ch. 7,
   "the effectiveness measure ... F"), taking plain integer
   ``(tp, fp, fn)`` counts rather than raw masks -- this is what lets one
   function serve both a single sample's lesion-level counts (this
   module's own :func:`lesion_component_confusion`) and
   ``evaluate.py``'s dataset-level SUMMED lesion counts across every
   positive sample, without duplicating the F-beta arithmetic in two
   places. ``beta=2`` (F2) weights recall (missing a lesion) twice as
   heavily as precision (a spurious detection) relative to F1 -- the
   standard choice when false negatives are considered costlier than false
   positives, as is conventionally assumed for a lesion-detection screening
   task.

8. ``min_component_size`` -- a config-gated exploratory predicted-component
   pixel filter, default OFF
   ------------------------------------------------------------------------
   Every prediction-region metric in this module
   (:func:`connected_components`, :func:`dice`, :func:`iou_jaccard`,
   :func:`hd95`, :func:`assd`,
   :func:`lesion_component_confusion`, :func:`lesion_detected`,
   :func:`false_positive_components`) accepts an optional
   ``min_component_size`` (pixel/voxel COUNT; default
   :data:`DEFAULT_MIN_COMPONENT_SIZE` ``= 0``, i.e. no filtering). When
   ``> 0``, PREDICTED connected components (never the ground-truth
   ``target``) with fewer than ``min_component_size`` pixels are dropped --
   zeroed out of the thresholded prediction mask -- before any
   overlap/matching arithmetic runs (:func:`_filter_min_component_size`).
   This is a 2-D/array-count analogue of the project evaluation contract's
   3-D component-volume gate. This module's 2D center-slice eval
   path has no physical (mm^3) voxel volume to weight by (see
   ``evaluate.py``'s own module docstring, item 4, on why the cache never
   carries a spacing this module could use for that), so
   ``min_component_size`` here is the PIXEL-COUNT analogue of the
   contract's "minimum physical component volume" gate -- same
   post-thresholding-pre-scoring position in the pipeline, same intent
   (suppress small spurious activations), different unit. Filtering is
   applied to the PREDICTION only, matching the read-only exploratory sweep
   this parameter productionizes
   (``runs/diagnostics/iou_2026-07-17/v6_component_size_sweep_raw.json``,
   ``v6exp_component_size_sweep_2026-07-19.md``): the ground-truth
   ``target`` is never filtered. Default ``0`` numerically preserves the
   prior unfiltered mask behavior -- ``_filter_min_component_size``
   short-circuits by returning the same binary array, spending zero extra
   ``scipy.ndimage`` work on the default path.

9. Empty-mask policy -- documented, not silently 0/0
   ------------------------------------------------------------------------
   - :func:`dice`/:func:`iou_jaccard`: both-empty (``pred`` AND ``target``
     have zero foreground pixels after thresholding/masking) -> ``1.0``
     (a perfect, if vacuous, match -- this project's own established
     convention; see ``model.py``'s :func:`~src.vascutrace.ml.model.
     dice_score` docstring, "rather than the 0/0 NaN a naive implementation
     would produce"). Exactly-one-empty needs no special case: the standard
     formula already gives ``0.0`` (intersection is necessarily 0, and the
     denominator is not). ``evaluate.py`` is responsible for EXCLUDING
     true-negative (empty-target) samples from the "positive-only Dice"
     aggregate entirely, rather than letting their trivial ``1.0`` inflate
     it -- this module only supplies the correctly-defined per-sample
     number; the decision of which samples belong in which aggregate is an
     ``evaluate.py``-level honesty policy, not a ``metrics.py`` one.
   - :func:`hd95`/:func:`assd`: both-empty -> ``0.0`` (no boundary
     disagreement is possible when neither region exists). Exactly-one
     -empty -> ``float("inf")`` (no boundary correspondence is defined --
     signals "undefined", not "perfect" or "zero"; ``evaluate.py`` only
     ever calls these on samples where :func:`lesion_detected` is already
     ``True``, i.e. both masks are known non-empty and overlapping, so this
     branch is a defensive contract for direct callers of this module, not
     the code path ``evaluate.py`` itself exercises in its "found lesions"
     aggregate).
============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import torch
from scipy import ndimage

from src.vascutrace.geometry import RESEARCH_PROTOTYPE_WARNING

__all__ = [
    "RESEARCH_PROTOTYPE_WARNING",
    "DEFAULT_SCORE_THRESHOLD",
    "DEFAULT_LESION_IOU_THRESHOLD",
    "DEFAULT_SURFACE_CONNECTIVITY",
    "DEFAULT_MIN_COMPONENT_SIZE",
    "PixelConfusion",
    "DetectionCounts",
    "PrecisionRecallF",
    "dice",
    "iou_jaccard",
    "positive_recall",
    "precision_ppv",
    "specificity",
    "pixel_accuracy",
    "hd95",
    "assd",
    "connected_components",
    "lesion_detected",
    "lesion_component_confusion",
    "false_positive_components",
    "precision_recall_f_beta",
]

# Matches model.py's dice_score default abnormality-score threshold (see module
# docstring, item 2).
DEFAULT_SCORE_THRESHOLD: float = 0.5

# Matches this implementation's own default lesion-detection IoU threshold (see
# module docstring, item 5).
DEFAULT_LESION_IOU_THRESHOLD: float = 0.1

# Face-connectivity (connectivity=1) for boundary/surface extraction only
# -- see module docstring, item 4, for why this differs from the
# connected-components family's maximal connectivity.
DEFAULT_SURFACE_CONNECTIVITY: int = 1

# Minimum PREDICTED-component pixel count to survive scoring -- see module
# docstring, item 8. Default 0 == no filtering == every function below is
# numerically identical to its prior unfiltered behavior.
DEFAULT_MIN_COMPONENT_SIZE: int = 0


ArrayLike = np.ndarray | torch.Tensor


# ---------------------------------------------------------------------------
# Shared array-prep helpers
# ---------------------------------------------------------------------------


def _asarray(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _prepare_float(x: ArrayLike) -> np.ndarray:
    """Convert to a float64 numpy array and squeeze a leading singleton
    "channel" axis (``[1, H, W]`` -> ``[H, W]``), matching this implementation's
    stated ``[H, W]`` or ``[1, H, W]`` input contract. A genuine 3D volume
    (``[D, H, W]`` with ``D > 1``) is left as-is -- every function in this
    module is ndim-generic, so a caller handing in stacked-slice 3D volumes
    gets genuine 26-connectivity component labeling and 3D surface
    distances "for free" (see module docstring, item 5), even though this
    project's own 2.5D samples only ever exercise the 2D path.
    """
    arr = _asarray(x).astype(np.float64, copy=False)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim not in (2, 3):
        raise ValueError(
            "expected a 2D [H, W] or 3D [D, H, W] array (optionally with a "
            f"leading singleton channel, [1, H, W]); got shape {arr.shape}"
        )
    return arr


def _check_same_shape(a: np.ndarray, b: np.ndarray, *, names: tuple[str, str]) -> None:
    if a.shape != b.shape:
        raise ValueError(f"{names[0]} shape {a.shape} != {names[1]} shape {b.shape}")


def _prepare_valid(valid_mask: ArrayLike | None, like: np.ndarray) -> np.ndarray:
    """Boolean inclusion mask, defaulting to all-``True`` when
    ``valid_mask`` is not supplied.
    """
    if valid_mask is None:
        return np.ones_like(like, dtype=bool)
    valid_arr = _prepare_float(valid_mask) >= 0.5
    _check_same_shape(valid_arr, like, names=("valid_mask", "pred/target"))
    return valid_arr


def _filter_min_component_size(
    binary: np.ndarray, min_component_size: int
) -> np.ndarray:
    """Zero out connected components of ``binary`` with fewer than
    ``min_component_size`` pixels -- see module docstring, item 8, for the
    full contract citation and rationale. ``min_component_size == 0`` is a
    true no-op: returns ``binary`` UNCHANGED (same object, not a copy) so
    the default path never pays for a labeling pass it doesn't need and is
    numerically identical to this module's pre-filter mask behavior.

    Labels with maximal connectivity (matching :func:`connected_components`
    -- 8-connectivity in 2D, 26-connectivity in 3D), computes each label's
    pixel count via ``np.bincount``, and keeps only labels whose count
    is ``>= min_component_size`` (label ``0``, the background, is always
    dropped). This is the exact algorithm the read-only sweep this
    parameter productionizes used (see module docstring, item 8) --
    reproduced here as the single, tested implementation every
    component-forming function in this module now shares, rather than a
    scratch wrapper duplicated per call site.
    """
    if not isinstance(min_component_size, Integral) or isinstance(
        min_component_size, bool
    ):
        raise ValueError(
            "min_component_size must be a non-negative integer, got "
            f"{min_component_size!r}"
        )
    min_component_size = int(min_component_size)
    if min_component_size < 0:
        raise ValueError(f"min_component_size must be >= 0, got {min_component_size}")
    if min_component_size == 0:
        return binary
    structure = ndimage.generate_binary_structure(binary.ndim, binary.ndim)
    labeled, n = ndimage.label(binary, structure=structure)
    if n == 0:
        return binary
    sizes = np.bincount(labeled.ravel(), minlength=n + 1)
    keep = sizes >= min_component_size
    keep[0] = False  # background (label 0) is never a "component" to keep
    return keep[labeled]


def _prepare_pair_bin(
    pred: ArrayLike,
    target: ArrayLike,
    threshold: float,
    valid_mask: ArrayLike | None,
    min_component_size: int = DEFAULT_MIN_COMPONENT_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Binary ``(pred, target)`` pair, AND-masked by ``valid_mask``. Safe
    for POSITIVE-REGION-ONLY metrics (Dice/IoU/components) -- see module
    docstring, item 3, for why this masking style is wrong for the
    four-way-confusion metrics (:func:`_pixel_confusion` handles those
    separately, correctly). ``min_component_size`` (module docstring, item
    8) is applied to ``pred`` ONLY, after valid-masking -- the ground-truth
    ``target`` is never size-filtered.
    """
    pred_arr = _prepare_float(pred)
    target_arr = _prepare_float(target)
    _check_same_shape(pred_arr, target_arr, names=("pred", "target"))
    pred_bin = pred_arr >= threshold
    target_bin = target_arr >= threshold
    valid_arr = _prepare_valid(valid_mask, pred_bin)
    pred_bin = pred_bin & valid_arr
    target_bin = target_bin & valid_arr
    pred_bin = _filter_min_component_size(pred_bin, min_component_size)
    return pred_bin, target_bin


# ---------------------------------------------------------------------------
# Dice / IoU -- see module docstring, item 9, for the empty-mask policy.
# ---------------------------------------------------------------------------


def dice(
    pred: ArrayLike,
    target: ArrayLike,
    *,
    threshold: float = DEFAULT_SCORE_THRESHOLD,
    valid_mask: ArrayLike | None = None,
    min_component_size: int = DEFAULT_MIN_COMPONENT_SIZE,
) -> float:
    """Hard Sorensen-Dice coefficient, ``2*|P ∩ G| / (|P| + |G|)``. Both
    -empty -> ``1.0`` (see module docstring, item 9). ``min_component_size``
    (module docstring, item 8; default ``0`` == no filtering) drops
    PREDICTED connected components smaller than the cutoff before scoring.
    """
    pred_bin, target_bin = _prepare_pair_bin(
        pred, target, threshold, valid_mask, min_component_size
    )
    p = int(pred_bin.sum())
    g = int(target_bin.sum())
    if p == 0 and g == 0:
        return 1.0
    intersection = int(np.logical_and(pred_bin, target_bin).sum())
    return float(2.0 * intersection / (p + g))


def iou_jaccard(
    pred: ArrayLike,
    target: ArrayLike,
    *,
    threshold: float = DEFAULT_SCORE_THRESHOLD,
    valid_mask: ArrayLike | None = None,
    min_component_size: int = DEFAULT_MIN_COMPONENT_SIZE,
) -> float:
    """Hard Jaccard index / IoU, ``|P ∩ G| / |P ∪ G|``. Both-empty ->
    ``1.0`` -- chosen so the identity ``IoU = Dice / (2 - Dice)`` holds
    exactly in the empty-vs-empty case too (``Dice=1`` -> ``1/(2-1)=1``),
    not just in the generic non-empty case (see module docstring, item 9,
    and ``tests/test_ml_metrics.py``'s
    ``TestDiceIouKnownArrays::test_dice_iou_identity_holds``).
    ``min_component_size`` (module docstring, item 8; default ``0`` == no
    filtering) drops PREDICTED connected components smaller than the
    cutoff before scoring.
    """
    pred_bin, target_bin = _prepare_pair_bin(
        pred, target, threshold, valid_mask, min_component_size
    )
    union = int(np.logical_or(pred_bin, target_bin).sum())
    if union == 0:
        return 1.0
    intersection = int(np.logical_and(pred_bin, target_bin).sum())
    return float(intersection / union)


# ---------------------------------------------------------------------------
# Voxelwise confusion metrics -- see module docstring, item 3, for why
# these are derived from one correctly-masked helper, not
# ``_prepare_pair_bin``.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PixelConfusion:
    """Voxelwise TP/FP/FN/TN counts within the valid region only."""

    tp: int
    fp: int
    fn: int
    tn: int


def _pixel_confusion(
    pred: ArrayLike,
    target: ArrayLike,
    threshold: float,
    valid_mask: ArrayLike | None,
) -> PixelConfusion:
    pred_arr = _prepare_float(pred)
    target_arr = _prepare_float(target)
    _check_same_shape(pred_arr, target_arr, names=("pred", "target"))
    pred_bin = pred_arr >= threshold
    target_bin = target_arr >= threshold
    valid_arr = _prepare_valid(valid_mask, pred_bin)

    tp = int(np.sum(pred_bin & target_bin & valid_arr))
    fp = int(np.sum(pred_bin & ~target_bin & valid_arr))
    fn = int(np.sum(~pred_bin & target_bin & valid_arr))
    tn = int(np.sum(~pred_bin & ~target_bin & valid_arr))
    return PixelConfusion(tp=tp, fp=fp, fn=fn, tn=tn)


def positive_recall(
    pred: ArrayLike,
    target: ArrayLike,
    *,
    threshold: float = DEFAULT_SCORE_THRESHOLD,
    valid_mask: ArrayLike | None = None,
) -> float:
    """Voxelwise sensitivity / recall, ``TP / (TP + FN)``. ``nan`` if the
    (valid-region) target has no positive pixels at all -- undefined, not
    silently ``0`` or ``1``.
    """
    c = _pixel_confusion(pred, target, threshold, valid_mask)
    denom = c.tp + c.fn
    return float(c.tp / denom) if denom > 0 else float("nan")


def precision_ppv(
    pred: ArrayLike,
    target: ArrayLike,
    *,
    threshold: float = DEFAULT_SCORE_THRESHOLD,
    valid_mask: ArrayLike | None = None,
) -> float:
    """Voxelwise precision / positive predictive value, ``TP / (TP + FP)``.
    ``nan`` if the (valid-region) prediction has no positive pixels at all.
    """
    c = _pixel_confusion(pred, target, threshold, valid_mask)
    denom = c.tp + c.fp
    return float(c.tp / denom) if denom > 0 else float("nan")


def specificity(
    pred: ArrayLike,
    target: ArrayLike,
    *,
    threshold: float = DEFAULT_SCORE_THRESHOLD,
    valid_mask: ArrayLike | None = None,
) -> float:
    """Voxelwise specificity, ``TN / (TN + FP)``. ``nan`` if the
    (valid-region) target has no negative pixels at all.
    """
    c = _pixel_confusion(pred, target, threshold, valid_mask)
    denom = c.tn + c.fp
    return float(c.tn / denom) if denom > 0 else float("nan")


def pixel_accuracy(
    pred: ArrayLike,
    target: ArrayLike,
    *,
    threshold: float = DEFAULT_SCORE_THRESHOLD,
    valid_mask: ArrayLike | None = None,
) -> float:
    """Voxelwise accuracy, ``(TP + TN) / (valid pixel count)``. ``nan`` if
    ``valid_mask`` excludes every pixel.
    """
    c = _pixel_confusion(pred, target, threshold, valid_mask)
    denom = c.tp + c.fp + c.fn + c.tn
    return float((c.tp + c.tn) / denom) if denom > 0 else float("nan")


# ---------------------------------------------------------------------------
# Surface distances (HD95, ASSD) -- see module docstring, item 4.
# ---------------------------------------------------------------------------


def _normalize_spacing(
    spacing: float | tuple[float, ...], ndim: int
) -> tuple[float, ...]:
    if isinstance(spacing, int | float):
        return (float(spacing),) * ndim
    spacing_t = tuple(float(s) for s in spacing)
    if len(spacing_t) != ndim:
        raise ValueError(
            f"spacing must have {ndim} entries (one per array axis) or be a "
            f"single scalar; got {spacing_t}"
        )
    return spacing_t


def _surface_distances(
    pred_bin: np.ndarray,
    target_bin: np.ndarray,
    spacing: tuple[float, ...],
    connectivity: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Directed surface-distance arrays ``(d(pred -> target), d(target ->
    pred))``. Preconditions: both ``pred_bin`` and ``target_bin`` are
    non-empty (callers -- :func:`hd95`/:func:`assd` -- special-case the
    empty cases before reaching here).

    Algorithm (medpy's ``__surface_distances``; see module docstring, item
    4): erode each mask to get its boundary shell, take the Euclidean
    distance transform of the COMPLEMENT OF THE OTHER MASK'S BOUNDARY
    SHELL (not the complement of the whole other mask), then index that
    distance field at this mask's own boundary pixels. Using the boundary
    shell's complement (rather than the whole mask's complement) is what
    makes a boundary point that happens to lie deep inside the other mask's
    interior get its correct (nonzero) distance to that mask's own
    boundary, instead of the wrong "0, because I'm already inside" answer
    a naive ``distance_transform_edt(~other_mask)`` would give for such a
    point.
    """
    structure = ndimage.generate_binary_structure(pred_bin.ndim, connectivity)
    pred_border = pred_bin ^ ndimage.binary_erosion(
        pred_bin, structure=structure, border_value=0
    )
    target_border = target_bin ^ ndimage.binary_erosion(
        target_bin, structure=structure, border_value=0
    )

    dt_from_target_border = ndimage.distance_transform_edt(
        ~target_border, sampling=spacing
    )
    dt_from_pred_border = ndimage.distance_transform_edt(~pred_border, sampling=spacing)

    d_pred_to_target = dt_from_target_border[pred_border]
    d_target_to_pred = dt_from_pred_border[target_border]
    return d_pred_to_target, d_target_to_pred


def hd95(
    pred: ArrayLike,
    target: ArrayLike,
    spacing: float | tuple[float, ...],
    *,
    threshold: float = DEFAULT_SCORE_THRESHOLD,
    valid_mask: ArrayLike | None = None,
    percentile: float = 95.0,
    connectivity: int = DEFAULT_SURFACE_CONNECTIVITY,
    min_component_size: int = DEFAULT_MIN_COMPONENT_SIZE,
) -> float:
    """Symmetric ``percentile``-th Hausdorff distance in the physical units
    of ``spacing`` (default the 95th percentile -- see module docstring,
    item 4): ``max(P_k(d(pred -> target)), P_k(d(target -> pred)))``.
    Empty-mask policy: both empty -> ``0.0``; exactly one empty ->
    ``float("inf")`` (see module docstring, item 9).
    ``min_component_size`` filters only predicted components before the
    surface is formed, matching Dice/IoU/component evaluation.
    """
    pred_bin, target_bin = _prepare_pair_bin(
        pred, target, threshold, valid_mask, min_component_size
    )
    spacing_t = _normalize_spacing(spacing, pred_bin.ndim)

    pred_empty = not pred_bin.any()
    target_empty = not target_bin.any()
    if pred_empty and target_empty:
        return 0.0
    if pred_empty or target_empty:
        return float("inf")

    d_p2t, d_t2p = _surface_distances(pred_bin, target_bin, spacing_t, connectivity)
    p95_p2t = float(np.percentile(d_p2t, percentile)) if d_p2t.size else 0.0
    p95_t2p = float(np.percentile(d_t2p, percentile)) if d_t2p.size else 0.0
    return max(p95_p2t, p95_t2p)


def assd(
    pred: ArrayLike,
    target: ArrayLike,
    spacing: float | tuple[float, ...],
    *,
    threshold: float = DEFAULT_SCORE_THRESHOLD,
    valid_mask: ArrayLike | None = None,
    connectivity: int = DEFAULT_SURFACE_CONNECTIVITY,
    min_component_size: int = DEFAULT_MIN_COMPONENT_SIZE,
) -> float:
    """Average symmetric surface distance (mean-of-means; see module
    docstring, item 4): ``(mean(d(pred -> target)) + mean(d(target ->
    pred))) / 2``. Empty-mask policy identical to :func:`hd95`.
    ``min_component_size`` has the same prediction-only semantics as
    :func:`hd95`.
    """
    pred_bin, target_bin = _prepare_pair_bin(
        pred, target, threshold, valid_mask, min_component_size
    )
    spacing_t = _normalize_spacing(spacing, pred_bin.ndim)

    pred_empty = not pred_bin.any()
    target_empty = not target_bin.any()
    if pred_empty and target_empty:
        return 0.0
    if pred_empty or target_empty:
        return float("inf")

    d_p2t, d_t2p = _surface_distances(pred_bin, target_bin, spacing_t, connectivity)
    asd_p2t = float(np.mean(d_p2t))
    asd_t2p = float(np.mean(d_t2p))
    return (asd_p2t + asd_t2p) / 2.0


# ---------------------------------------------------------------------------
# Connected components / lesion-level detection -- see module docstring,
# item 5.
# ---------------------------------------------------------------------------


def connected_components(
    mask: ArrayLike,
    *,
    threshold: float = DEFAULT_SCORE_THRESHOLD,
    valid_mask: ArrayLike | None = None,
    min_component_size: int = DEFAULT_MIN_COMPONENT_SIZE,
) -> tuple[np.ndarray, int]:
    """Label ``mask``'s thresholded positive region with MAXIMAL
    connectivity -- 8-connectivity in 2D, 26-connectivity in 3D
    (``scipy.ndimage.generate_binary_structure(ndim, ndim)``; see module
    docstring, item 5). Returns ``(labeled_array, n_components)``.

    ``min_component_size`` (module docstring, item 8; default ``0`` == no
    filtering) drops components smaller than the cutoff BEFORE the final
    labeling pass that produces the returned ``labeled_array`` -- so
    ``n_components`` already reflects the post-filter count, and the
    returned label ids are a fresh consecutive ``1..n_components``
    relabeling (label identity itself carries no meaning to any caller in
    this module -- every downstream use compares boolean masks, never raw
    label values -- so relabeling after filtering is safe).
    """
    arr = _prepare_float(mask)
    binary = arr >= threshold
    valid_arr = _prepare_valid(valid_mask, binary)
    binary = binary & valid_arr
    binary = _filter_min_component_size(binary, min_component_size)
    structure = ndimage.generate_binary_structure(binary.ndim, binary.ndim)
    labeled, n = ndimage.label(binary, structure=structure)
    return labeled, int(n)


@dataclass(frozen=True)
class DetectionCounts:
    """Lesion/component-level TP/FP/FN for one sample -- see
    :func:`lesion_component_confusion`.
    """

    tp: int
    fp: int
    fn: int
    n_gt_components: int
    n_pred_components: int


def lesion_component_confusion(
    pred: ArrayLike,
    target: ArrayLike,
    *,
    iou_threshold: float = DEFAULT_LESION_IOU_THRESHOLD,
    threshold: float = DEFAULT_SCORE_THRESHOLD,
    valid_mask: ArrayLike | None = None,
    min_component_size: int = DEFAULT_MIN_COMPONENT_SIZE,
) -> DetectionCounts:
    """Match each ground-truth connected component to its single
    best-overlapping predicted component by component-level IoU; a match
    ``>= iou_threshold`` is a detection (TP), otherwise the ground-truth
    component is a miss (FN); a predicted component that is never any
    ground-truth component's best match at that threshold is a
    false-positive component (FP). See module docstring, item 5, for the
    LiTS-style overlap-threshold rationale and the documented
    best-match-only (non-bipartite) simplification. ``min_component_size``
    (module docstring, item 8; default ``0`` == no filtering) drops
    PREDICTED connected components smaller than the cutoff BEFORE matching
    -- both the detection-matching TP/FN/FP counts below are computed on
    the filtered prediction; the ground-truth ``target`` is never filtered.
    """
    pred_arr = _prepare_float(pred)
    target_arr = _prepare_float(target)
    _check_same_shape(pred_arr, target_arr, names=("pred", "target"))
    pred_bin = pred_arr >= threshold
    target_bin = target_arr >= threshold
    valid_arr = _prepare_valid(valid_mask, pred_bin)
    pred_bin = pred_bin & valid_arr
    target_bin = target_bin & valid_arr
    pred_bin = _filter_min_component_size(pred_bin, min_component_size)

    structure = ndimage.generate_binary_structure(pred_bin.ndim, pred_bin.ndim)
    pred_labeled, n_pred = ndimage.label(pred_bin, structure=structure)
    target_labeled, n_gt = ndimage.label(target_bin, structure=structure)

    pred_matched = np.zeros(n_pred + 1, dtype=bool)
    n_gt_detected = 0

    for gt_id in range(1, n_gt + 1):
        gt_mask = target_labeled == gt_id
        candidate_ids = np.unique(pred_labeled[gt_mask])
        candidate_ids = candidate_ids[candidate_ids != 0]
        if candidate_ids.size == 0:
            # No predicted component touches this GT component at all -> a
            # miss regardless of iou_threshold (in particular, NOT a "0.0 >=
            # 0.0" spurious detection when iou_threshold == 0.0).
            continue

        best_iou = 0.0
        best_pred_id: int | None = None
        for pred_id in candidate_ids:
            pred_mask = pred_labeled == pred_id
            intersection = int(np.logical_and(gt_mask, pred_mask).sum())
            union = int(np.logical_or(gt_mask, pred_mask).sum())
            iou = intersection / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best_pred_id = int(pred_id)

        if best_iou >= iou_threshold:
            n_gt_detected += 1
            if best_pred_id is not None:
                pred_matched[best_pred_id] = True

    tp = n_gt_detected
    fn = n_gt - n_gt_detected
    fp = n_pred - int(pred_matched[1:].sum())

    return DetectionCounts(
        tp=tp, fp=fp, fn=fn, n_gt_components=n_gt, n_pred_components=n_pred
    )


def lesion_detected(
    pred: ArrayLike,
    target: ArrayLike,
    *,
    iou_threshold: float = DEFAULT_LESION_IOU_THRESHOLD,
    threshold: float = DEFAULT_SCORE_THRESHOLD,
    valid_mask: ArrayLike | None = None,
    min_component_size: int = DEFAULT_MIN_COMPONENT_SIZE,
) -> bool:
    """``True`` iff at least one ground-truth lesion component in
    ``target`` is matched by an overlapping predicted component at
    ``iou_threshold`` (see :func:`lesion_component_confusion`) -- "did the
    model find the lesion at all". Requires ``target`` to have at least one
    (valid-region) foreground pixel -- raises :class:`ValueError` on an
    empty target (a healthy/negative sample has no lesion to have found;
    use :func:`false_positive_components` for that case instead).
    ``min_component_size`` (module docstring, item 8; default ``0`` == no
    filtering) is forwarded unchanged to :func:`lesion_component_confusion`.
    """
    target_arr = _prepare_float(target)
    valid_arr = _prepare_valid(valid_mask, target_arr)
    target_bin_check = (target_arr >= threshold) & valid_arr
    if not target_bin_check.any():
        raise ValueError(
            "lesion_detected(...) requires a target with at least one "
            "(valid-region) foreground pixel -- a known lesion-bearing "
            "sample; got an empty target. Use false_positive_components(...) "
            "for a negative/healthy sample instead."
        )
    counts = lesion_component_confusion(
        pred,
        target,
        iou_threshold=iou_threshold,
        threshold=threshold,
        valid_mask=valid_mask,
        min_component_size=min_component_size,
    )
    return counts.tp > 0


def false_positive_components(
    pred: ArrayLike,
    *,
    threshold: float = DEFAULT_SCORE_THRESHOLD,
    valid_mask: ArrayLike | None = None,
    min_component_size: int = DEFAULT_MIN_COMPONENT_SIZE,
) -> int:
    """Connected-component count of ``pred``'s thresholded positive
    region. Intended for a NEGATIVE/healthy sample (empty ground truth),
    where every predicted-positive component is by definition a false
    activation -- this function does not itself inspect a target mask (a
    healthy sample has none to check against); the caller is responsible
    for only invoking it on a sample it already knows is healthy.
    ``min_component_size`` (module docstring, item 8; default ``0`` == no
    filtering) is forwarded unchanged to :func:`connected_components`.
    """
    _, n = connected_components(
        pred,
        threshold=threshold,
        valid_mask=valid_mask,
        min_component_size=min_component_size,
    )
    return n


# ---------------------------------------------------------------------------
# Precision / recall / F-beta -- see module docstring, item 6.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrecisionRecallF:
    precision: float
    recall: float
    f_beta: float
    beta: float


def precision_recall_f_beta(
    tp: int, fp: int, fn: int, *, beta: float = 1.0
) -> PrecisionRecallF:
    """Standard precision/recall/F-beta from plain integer counts (see
    module docstring, item 6) -- reusable for a single sample's lesion
    -level counts (:func:`lesion_component_confusion`) or for
    ``evaluate.py``'s dataset-level SUMMED counts across every positive
    sample. ``precision``/``recall`` are ``nan`` on an undefined (0
    denominator) ratio; ``f_beta`` is ``0.0`` when ``tp == 0`` with BOTH
    ``fp > 0`` and ``fn > 0`` (precision and recall are both finite
    -but-zero), and ``nan`` whenever either ``precision`` or ``recall``
    is itself ``nan`` (i.e. ``tp + fp == 0`` or ``tp + fn == 0``).
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

    if not (np.isfinite(precision) and np.isfinite(recall)):
        f_beta = float("nan")
    else:
        beta_sq = beta**2
        denom = beta_sq * precision + recall
        f_beta = (1 + beta_sq) * precision * recall / denom if denom > 0 else 0.0

    return PrecisionRecallF(
        precision=precision, recall=recall, f_beta=f_beta, beta=beta
    )
