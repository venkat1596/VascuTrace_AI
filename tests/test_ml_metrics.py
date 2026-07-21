"""Tests for ``src.vascutrace.ml.metrics``.

CPU-only, hand-computed known arrays -- no ``Data/`` or checkpoint access
anywhere in this file (that lives in ``evaluate.py``'s own manually-run
``--checkpoint``/``--cache`` CLI path, not in pytest). Every non-trivial
expected value below is either hand-derived in a comment or cross-checked
against a SECOND, independently-implemented reference computation (see
``TestSurfaceDistances``'s brute-force ``cdist`` cross-check) so this file
never merely re-asserts what ``metrics.py`` itself computes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy.ndimage import binary_erosion, generate_binary_structure
from scipy.spatial.distance import cdist

from src.vascutrace.ml.metrics import (
    RESEARCH_PROTOTYPE_WARNING,
    DetectionCounts,
    PixelConfusion,
    assd,
    connected_components,
    dice,
    false_positive_components,
    hd95,
    iou_jaccard,
    lesion_component_confusion,
    lesion_detected,
    pixel_accuracy,
    precision_ppv,
    precision_recall_f_beta,
    positive_recall,
    specificity,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _square_mask(
    shape: tuple[int, int], row_slice: slice, col_slice: slice
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.float32)
    mask[row_slice, col_slice] = 1.0
    return mask


# ---------------------------------------------------------------------------
# 0. Module-level sanity
# ---------------------------------------------------------------------------


def test_research_prototype_warning_reexported() -> None:
    assert "simulated" in RESEARCH_PROTOTYPE_WARNING.lower()


# ---------------------------------------------------------------------------
# 1. Dice / IoU on known overlapping squares + the Dice<->IoU identity
# ---------------------------------------------------------------------------


class TestDiceIouKnownArrays:
    # target: rows[2:6) x cols[2:6) = 4x4 = 16 px.
    # pred:   rows[2:6) x cols[4:8) = 4x4 = 16 px, shifted right by 2 cols.
    # intersection: rows[2:6) x cols[4:6) = 4x2 = 8 px.
    # union = 16 + 16 - 8 = 24.
    # Dice = 2*8 / (16+16) = 16/32 = 0.5
    # IoU  = 8/24 = 1/3
    def _overlapping_squares(self) -> tuple[np.ndarray, np.ndarray]:
        shape = (10, 10)
        target = _square_mask(shape, slice(2, 6), slice(2, 6))
        pred = _square_mask(shape, slice(2, 6), slice(4, 8))
        return pred, target

    def test_dice_known_value(self) -> None:
        pred, target = self._overlapping_squares()
        assert dice(pred, target) == pytest.approx(0.5)

    def test_iou_known_value(self) -> None:
        pred, target = self._overlapping_squares()
        assert iou_jaccard(pred, target) == pytest.approx(1.0 / 3.0)

    def test_dice_iou_identity_holds_on_overlapping_squares(self) -> None:
        pred, target = self._overlapping_squares()
        d = dice(pred, target)
        i = iou_jaccard(pred, target)
        assert i == pytest.approx(d / (2.0 - d))

    def test_dice_iou_identity_holds_on_random_binary_masks(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(25):
            pred = (rng.random((16, 12)) > 0.5).astype(np.float32)
            target = (rng.random((16, 12)) > 0.5).astype(np.float32)
            d = dice(pred, target)
            i = iou_jaccard(pred, target)
            assert i == pytest.approx(d / (2.0 - d), abs=1e-9)

    def test_perfect_match_gives_dice_and_iou_one(self) -> None:
        shape = (8, 8)
        mask = _square_mask(shape, slice(1, 5), slice(1, 5))
        assert dice(mask, mask) == pytest.approx(1.0)
        assert iou_jaccard(mask, mask) == pytest.approx(1.0)

    def test_disjoint_masks_give_dice_and_iou_zero(self) -> None:
        shape = (8, 8)
        pred = _square_mask(shape, slice(0, 2), slice(0, 2))
        target = _square_mask(shape, slice(6, 8), slice(6, 8))
        assert dice(pred, target) == pytest.approx(0.0)
        assert iou_jaccard(pred, target) == pytest.approx(0.0)

    def test_threshold_parameter_applied_to_probability_input(self) -> None:
        # A soft score map where only pixels >= 0.5 should count.
        shape = (4, 4)
        pred_prob = np.zeros(shape, dtype=np.float32)
        pred_prob[0:2, 0:2] = 0.9  # above threshold
        pred_prob[2:4, 2:4] = 0.3  # below threshold -- should NOT count
        target = _square_mask(shape, slice(0, 2), slice(0, 2))
        assert dice(pred_prob, target) == pytest.approx(1.0)

    def test_leading_singleton_channel_dim_accepted(self) -> None:
        pred, target = self._overlapping_squares()
        pred_c = pred[np.newaxis, ...]
        target_c = target[np.newaxis, ...]
        assert dice(pred_c, target_c) == pytest.approx(dice(pred, target))

    def test_torch_tensor_input_matches_numpy(self) -> None:
        pred, target = self._overlapping_squares()
        d_np = dice(pred, target)
        i_np = iou_jaccard(pred, target)
        d_torch = dice(torch.from_numpy(pred), torch.from_numpy(target))
        i_torch = iou_jaccard(torch.from_numpy(pred), torch.from_numpy(target))
        assert d_torch == pytest.approx(d_np)
        assert i_torch == pytest.approx(i_np)

    def test_shape_mismatch_raises(self) -> None:
        pred = np.zeros((4, 4), dtype=np.float32)
        target = np.zeros((5, 5), dtype=np.float32)
        with pytest.raises(ValueError):
            dice(pred, target)


# ---------------------------------------------------------------------------
# 2. Empty-target / empty-pred policy (dice, iou)
# ---------------------------------------------------------------------------


class TestEmptyMaskPolicy:
    def test_both_empty_dice_and_iou_are_one(self) -> None:
        shape = (5, 5)
        empty = np.zeros(shape, dtype=np.float32)
        assert dice(empty, empty) == pytest.approx(1.0)
        assert iou_jaccard(empty, empty) == pytest.approx(1.0)

    def test_pred_nonempty_target_empty_gives_zero(self) -> None:
        shape = (5, 5)
        empty = np.zeros(shape, dtype=np.float32)
        pred = _square_mask(shape, slice(0, 2), slice(0, 2))
        assert dice(pred, empty) == pytest.approx(0.0)
        assert iou_jaccard(pred, empty) == pytest.approx(0.0)

    def test_pred_empty_target_nonempty_gives_zero(self) -> None:
        shape = (5, 5)
        empty = np.zeros(shape, dtype=np.float32)
        target = _square_mask(shape, slice(0, 2), slice(0, 2))
        assert dice(empty, target) == pytest.approx(0.0)
        assert iou_jaccard(empty, target) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 3. Voxelwise confusion metrics on a known hand-counted grid
# ---------------------------------------------------------------------------


class TestVoxelwiseConfusionMetrics:
    # 4x4 grid (16 px total).
    # target 1s at: (0,0) (0,1) (1,0) (1,1)                      -> 4 px
    # pred   1s at: (0,0) (1,0) (1,1) (2,1)                      -> 4 px
    # TP = {(0,0), (1,0), (1,1)}                                  -> 3
    # FN = {(0,1)}                                                 -> 1
    # FP = {(2,1)}                                                 -> 1
    # TN = remaining 11 cells                                     -> 11
    def _grid(self) -> tuple[np.ndarray, np.ndarray]:
        target = np.zeros((4, 4), dtype=np.float32)
        for r, c in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            target[r, c] = 1.0
        pred = np.zeros((4, 4), dtype=np.float32)
        for r, c in [(0, 0), (1, 0), (1, 1), (2, 1)]:
            pred[r, c] = 1.0
        return pred, target

    def test_positive_recall_and_precision_known_values(self) -> None:
        pred, target = self._grid()
        assert positive_recall(pred, target) == pytest.approx(3.0 / 4.0)
        assert precision_ppv(pred, target) == pytest.approx(3.0 / 4.0)

    def test_specificity_known_value(self) -> None:
        pred, target = self._grid()
        assert specificity(pred, target) == pytest.approx(11.0 / 12.0)

    def test_pixel_accuracy_known_value(self) -> None:
        pred, target = self._grid()
        assert pixel_accuracy(pred, target) == pytest.approx(14.0 / 16.0)

    def test_positive_recall_nan_when_target_has_no_positives(self) -> None:
        shape = (4, 4)
        empty_target = np.zeros(shape, dtype=np.float32)
        pred = _square_mask(shape, slice(0, 2), slice(0, 2))
        assert math.isnan(positive_recall(pred, empty_target))

    def test_precision_nan_when_pred_has_no_positives(self) -> None:
        shape = (4, 4)
        empty_pred = np.zeros(shape, dtype=np.float32)
        target = _square_mask(shape, slice(0, 2), slice(0, 2))
        assert math.isnan(precision_ppv(empty_pred, target))

    def test_specificity_nan_when_target_is_all_positive(self) -> None:
        shape = (4, 4)
        all_positive_target = np.ones(shape, dtype=np.float32)
        pred = np.ones(shape, dtype=np.float32)
        assert math.isnan(specificity(pred, all_positive_target))

    def test_pixel_accuracy_nan_when_valid_mask_excludes_everything(self) -> None:
        pred, target = self._grid()
        valid = np.zeros((4, 4), dtype=np.float32)
        assert math.isnan(pixel_accuracy(pred, target, valid_mask=valid))

    def test_valid_mask_excludes_invalid_pixel_from_true_negative_count(self) -> None:
        """Regression guard for the exact bug described in metrics.py's own
        module docstring, item 3: AND-masking a valid_mask into pred/target
        BEFORE computing the confusion counts silently turns an excluded
        pixel into a spurious true negative (since False & False == False
        either way), which would leave pixel_accuracy/specificity
        UNCHANGED by valid_mask -- the wrong answer. The correct
        implementation must actually shrink the denominator.
        """
        pred, target = self._grid()
        full_accuracy = pixel_accuracy(pred, target)
        full_specificity = specificity(pred, target)

        # Exclude exactly one known true-negative cell, (3, 3).
        valid = np.ones((4, 4), dtype=np.float32)
        valid[3, 3] = 0.0

        masked_accuracy = pixel_accuracy(pred, target, valid_mask=valid)
        masked_specificity = specificity(pred, target, valid_mask=valid)

        # Correct: TN drops from 11 to 10, denominator from 16 to 15 (accuracy)
        # and from 12 to 11 (specificity) -- both values must actually move.
        assert masked_accuracy == pytest.approx(13.0 / 15.0)
        assert masked_specificity == pytest.approx(10.0 / 11.0)
        assert masked_accuracy != pytest.approx(full_accuracy)
        assert masked_specificity != pytest.approx(full_specificity)

    def test_valid_mask_does_not_affect_dice_or_iou_when_excluding_pure_background(
        self,
    ) -> None:
        # Dice/IoU only ever sum positive-region pixels, so excluding a
        # true-negative background pixel must be a no-op for them (unlike
        # pixel_accuracy/specificity above).
        pred, target = self._grid()
        valid = np.ones((4, 4), dtype=np.float32)
        valid[3, 3] = 0.0
        assert dice(pred, target, valid_mask=valid) == pytest.approx(dice(pred, target))
        assert iou_jaccard(pred, target, valid_mask=valid) == pytest.approx(
            iou_jaccard(pred, target)
        )

    def test_confusion_metrics_torch_tensor_parity(self) -> None:
        pred, target = self._grid()
        pred_t, target_t = torch.from_numpy(pred), torch.from_numpy(target)
        assert positive_recall(pred_t, target_t) == pytest.approx(
            positive_recall(pred, target)
        )
        assert pixel_accuracy(pred_t, target_t) == pytest.approx(
            pixel_accuracy(pred, target)
        )

    def test_pixel_confusion_dataclass_fields(self) -> None:
        c = PixelConfusion(tp=3, fp=1, fn=1, tn=11)
        assert (c.tp, c.fp, c.fn, c.tn) == (3, 1, 1, 11)


# ---------------------------------------------------------------------------
# 4. Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_repeated_calls_are_bit_identical(self) -> None:
        rng = np.random.default_rng(7)
        pred = rng.random((20, 16)).astype(np.float32)
        target = (rng.random((20, 16)) > 0.6).astype(np.float32)

        results_a = (
            dice(pred, target),
            iou_jaccard(pred, target),
            positive_recall(pred, target),
            precision_ppv(pred, target),
            pixel_accuracy(pred, target),
        )
        results_b = (
            dice(pred, target),
            iou_jaccard(pred, target),
            positive_recall(pred, target),
            precision_ppv(pred, target),
            pixel_accuracy(pred, target),
        )
        assert results_a == results_b


# ---------------------------------------------------------------------------
# 5. Surface distances (HD95, ASSD)
# ---------------------------------------------------------------------------


def _reference_surface_distances(
    pred: np.ndarray, target: np.ndarray, spacing: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Independent brute-force reference: extract boundary pixels via the
    same erosion definition metrics.py uses, but then compute all pairwise
    distances explicitly (``scipy.spatial.distance.cdist``) instead of the
    Euclidean-distance-transform shortcut ``metrics._surface_distances``
    uses -- a genuinely different code path for cross-checking correctness.
    """
    structure = generate_binary_structure(pred.ndim, 1)
    pred_border = pred ^ binary_erosion(pred, structure=structure, border_value=0)
    target_border = target ^ binary_erosion(target, structure=structure, border_value=0)

    scale = np.asarray(spacing, dtype=np.float64)
    pred_pts = np.argwhere(pred_border) * scale
    target_pts = np.argwhere(target_border) * scale

    d_matrix = cdist(pred_pts, target_pts)
    d_pred_to_target = d_matrix.min(axis=1)
    d_target_to_pred = d_matrix.min(axis=0)
    return d_pred_to_target, d_target_to_pred


class TestSurfaceDistances:
    def test_single_pixel_masks_exact_euclidean_distance(self) -> None:
        # target at (10, 10), pred at (13, 14) -> dy=3, dx=4.
        # anisotropic spacing (2.0, 1.0) mm/px:
        # distance = sqrt((3*2.0)^2 + (4*1.0)^2) = sqrt(36 + 16) = sqrt(52)
        shape = (50, 50)
        target = np.zeros(shape, dtype=np.float32)
        target[10, 10] = 1.0
        pred = np.zeros(shape, dtype=np.float32)
        pred[13, 14] = 1.0
        spacing = (2.0, 1.0)
        expected = math.sqrt(52.0)

        assert hd95(pred, target, spacing) == pytest.approx(expected)
        assert assd(pred, target, spacing) == pytest.approx(expected)

    def test_isotropic_spacing_axis_aligned_offset(self) -> None:
        # Purely along one axis: distance = dx * spacing_x exactly.
        shape = (30, 30)
        target = np.zeros(shape, dtype=np.float32)
        target[5, 5] = 1.0
        pred = np.zeros(shape, dtype=np.float32)
        pred[5, 11] = 1.0  # dx = 6
        spacing = 1.5  # scalar spacing broadcast to both axes
        expected = 6 * 1.5

        assert hd95(pred, target, spacing) == pytest.approx(expected)
        assert assd(pred, target, spacing) == pytest.approx(expected)

    def test_both_empty_gives_zero(self) -> None:
        shape = (10, 10)
        empty = np.zeros(shape, dtype=np.float32)
        assert hd95(empty, empty, spacing=1.0) == pytest.approx(0.0)
        assert assd(empty, empty, spacing=1.0) == pytest.approx(0.0)

    def test_one_empty_gives_infinity(self) -> None:
        shape = (10, 10)
        empty = np.zeros(shape, dtype=np.float32)
        nonempty = _square_mask(shape, slice(2, 4), slice(2, 4))
        assert hd95(nonempty, empty, spacing=1.0) == math.inf
        assert hd95(empty, nonempty, spacing=1.0) == math.inf
        assert assd(nonempty, empty, spacing=1.0) == math.inf
        assert assd(empty, nonempty, spacing=1.0) == math.inf

    def test_cross_check_against_brute_force_reference(self) -> None:
        shape = (30, 30)
        target = _square_mask(shape, slice(5, 15), slice(5, 12))
        pred = _square_mask(shape, slice(8, 20), slice(9, 18))
        spacing = (1.3, 0.9)

        d_p2t, d_t2p = _reference_surface_distances(
            pred.astype(bool), target.astype(bool), spacing
        )
        expected_hd95 = max(
            float(np.percentile(d_p2t, 95)), float(np.percentile(d_t2p, 95))
        )
        expected_assd = (float(d_p2t.mean()) + float(d_t2p.mean())) / 2.0

        assert hd95(pred, target, spacing) == pytest.approx(expected_hd95)
        assert assd(pred, target, spacing) == pytest.approx(expected_assd)

    def test_percentile_100_matches_brute_force_max_hausdorff(self) -> None:
        shape = (30, 30)
        target = _square_mask(shape, slice(5, 15), slice(5, 12))
        pred = _square_mask(shape, slice(8, 20), slice(9, 18))
        spacing = (1.0, 1.0)

        d_p2t, d_t2p = _reference_surface_distances(
            pred.astype(bool), target.astype(bool), spacing
        )
        expected_max_hausdorff = max(float(d_p2t.max()), float(d_t2p.max()))

        assert hd95(pred, target, spacing, percentile=100.0) == pytest.approx(
            expected_max_hausdorff
        )

    def test_hd95_le_hd100_monotonicity(self) -> None:
        shape = (30, 30)
        target = _square_mask(shape, slice(5, 15), slice(5, 12))
        pred = _square_mask(shape, slice(8, 20), slice(9, 18))
        spacing = (1.0, 1.0)
        hd95_val = hd95(pred, target, spacing, percentile=95.0)
        hd100_val = hd95(pred, target, spacing, percentile=100.0)
        assert hd95_val <= hd100_val + 1e-9

    def test_wrong_spacing_length_raises(self) -> None:
        shape = (10, 10)
        mask = _square_mask(shape, slice(2, 4), slice(2, 4))
        with pytest.raises(ValueError):
            hd95(mask, mask, spacing=(1.0, 1.0, 1.0))


# ---------------------------------------------------------------------------
# 6. Connected components / lesion-level detection
# ---------------------------------------------------------------------------


class TestConnectedComponentsAndDetection:
    def test_single_blob_component_count(self) -> None:
        shape = (20, 20)
        mask = _square_mask(shape, slice(5, 9), slice(5, 9))
        _, n = connected_components(mask)
        assert n == 1

    def test_two_disjoint_blobs_counted_separately(self) -> None:
        shape = (20, 20)
        mask = _square_mask(shape, slice(0, 2), slice(0, 2))
        mask += _square_mask(shape, slice(15, 17), slice(15, 17))
        _, n = connected_components(mask)
        assert n == 2

    def test_diagonally_touching_blobs_merge_under_full_connectivity(self) -> None:
        # Two 1-pixel blobs touching only at a corner (diagonal neighbors)
        # must merge into ONE component under maximal (8-)connectivity --
        # see module docstring item 5's "26-connectivity in 3D /
        # 8-connectivity in 2D" contract.
        shape = (10, 10)
        mask = np.zeros(shape, dtype=np.float32)
        mask[5, 5] = 1.0
        mask[6, 6] = 1.0  # diagonal neighbor of (5, 5)
        _, n = connected_components(mask)
        assert n == 1

    def test_perfectly_overlapping_lesion_is_detected(self) -> None:
        shape = (20, 20)
        target = _square_mask(shape, slice(5, 9), slice(5, 9))
        pred = target.copy()
        assert lesion_detected(pred, target) is True
        counts = lesion_component_confusion(pred, target)
        assert counts == DetectionCounts(
            tp=1, fp=0, fn=0, n_gt_components=1, n_pred_components=1
        )

    def test_zero_overlap_lesion_is_a_miss_regardless_of_threshold(self) -> None:
        shape = (20, 20)
        target = _square_mask(shape, slice(2, 4), slice(2, 4))
        pred = _square_mask(shape, slice(15, 17), slice(15, 17))
        assert lesion_detected(pred, target, iou_threshold=0.0) is False
        counts = lesion_component_confusion(pred, target, iou_threshold=0.0)
        assert counts.tp == 0
        assert counts.fn == 1
        assert counts.fp == 1  # the stray pred blob matches nothing

    def test_detection_exactly_at_iou_threshold_boundary(self) -> None:
        # target: rows[0:4) cols[0:4) = 16 px.
        # pred:   rows[0:4) cols[2:6) = 16 px, shifted right by 2.
        # intersection = rows[0:4) cols[2:4) = 4x2 = 8; union = 24.
        # IoU = 8/24 = 1/3 exactly.
        shape = (10, 10)
        target = _square_mask(shape, slice(0, 4), slice(0, 4))
        pred = _square_mask(shape, slice(0, 4), slice(2, 6))
        exact_iou = 1.0 / 3.0

        assert lesion_detected(pred, target, iou_threshold=exact_iou) is True
        assert lesion_detected(pred, target, iou_threshold=exact_iou + 1e-6) is False

    def test_lesion_detected_raises_on_empty_target(self) -> None:
        shape = (10, 10)
        empty_target = np.zeros(shape, dtype=np.float32)
        pred = _square_mask(shape, slice(2, 4), slice(2, 4))
        with pytest.raises(ValueError):
            lesion_detected(pred, empty_target)

    def test_false_positive_components_on_clean_healthy_case(self) -> None:
        shape = (20, 20)
        pred = np.zeros(shape, dtype=np.float32)
        assert false_positive_components(pred) == 0

    def test_false_positive_components_counts_stray_blobs(self) -> None:
        shape = (20, 20)
        pred = _square_mask(shape, slice(0, 2), slice(0, 2))
        pred += _square_mask(shape, slice(10, 12), slice(10, 12))
        pred += _square_mask(shape, slice(15, 17), slice(2, 4))
        assert false_positive_components(pred) == 3

    def test_valid_mask_excludes_component_outside_valid_region(self) -> None:
        shape = (20, 20)
        pred = _square_mask(shape, slice(0, 2), slice(0, 2))
        pred += _square_mask(shape, slice(10, 12), slice(10, 12))
        valid = np.ones(shape, dtype=np.float32)
        valid[8:14, 8:14] = 0.0  # zero out the second blob's region
        assert false_positive_components(pred, valid_mask=valid) == 1

    def test_multi_lesion_case_tp_fp_fn_counts(self) -> None:
        shape = (30, 30)
        target = np.zeros(shape, dtype=np.float32)
        target[2:5, 2:5] = 1.0  # GT lesion 1 -- will be found
        target[20:23, 20:23] = 1.0  # GT lesion 2 -- will be missed

        pred = np.zeros(shape, dtype=np.float32)
        pred[2:5, 2:5] = 1.0  # matches GT lesion 1 exactly
        pred[15:17, 15:17] = 1.0  # spurious extra blob -> FP

        counts = lesion_component_confusion(pred, target)
        assert counts.n_gt_components == 2
        assert counts.n_pred_components == 2
        assert counts.tp == 1
        assert counts.fn == 1
        assert counts.fp == 1


# ---------------------------------------------------------------------------
# 6b. min_component_size -- config-gated PREDICTED-component pixel-count
# filter (default 0 == no filtering). See metrics.py's module docstring,
# item 8, for the project evaluation contract's minimum physical
# component-volume gate.
# ---------------------------------------------------------------------------


class TestMinComponentSizeFilter:
    # Fixture shared by every test below: one LARGE (4x4 = 16px) component
    # and one SMALL (2x2 = 4px) component, far apart (never touching, even
    # diagonally, so they are never merged under 8-connectivity).
    _SHAPE = (20, 20)
    _LARGE_SLICE = (slice(2, 6), slice(2, 6))  # 16 px
    _SMALL_SLICE = (slice(15, 17), slice(15, 17))  # 4 px
    _LARGE_SIZE = 16
    _SMALL_SIZE = 4

    def _pred_large_and_small(self) -> np.ndarray:
        pred = _square_mask(self._SHAPE, *self._LARGE_SLICE)
        pred += _square_mask(self._SHAPE, *self._SMALL_SLICE)
        return pred

    # -- connected_components ------------------------------------------------

    def test_default_min_component_size_zero_is_a_no_op(self) -> None:
        # The default (0) must reproduce the pre-filter behavior exactly --
        # both components present, byte-identical to calling with no
        # min_component_size argument at all.
        pred = self._pred_large_and_small()
        labeled_default, n_default = connected_components(pred)
        labeled_explicit_zero, n_explicit_zero = connected_components(
            pred, min_component_size=0
        )
        assert n_default == n_explicit_zero == 2
        assert np.array_equal(labeled_default > 0, labeled_explicit_zero > 0)
        assert np.array_equal(labeled_default > 0, pred > 0)

    def test_small_component_kept_when_cutoff_equals_its_exact_size(self) -> None:
        # min_component_size uses ">=" -- a component whose size EQUALS the
        # cutoff survives (this is the documented boundary semantics, see
        # metrics.py's _filter_min_component_size docstring).
        pred = self._pred_large_and_small()
        _, n = connected_components(pred, min_component_size=self._SMALL_SIZE)
        assert n == 2

    def test_small_component_dropped_one_pixel_above_its_size(self) -> None:
        # The core acceptance case: at cutoff = small_size + 1, the small
        # (4px) component is dropped and the large (16px) component
        # survives untouched.
        pred = self._pred_large_and_small()
        labeled, n = connected_components(pred, min_component_size=self._SMALL_SIZE + 1)
        assert n == 1
        kept_mask = labeled > 0
        assert int(kept_mask.sum()) == self._LARGE_SIZE
        assert np.array_equal(kept_mask, pred >= 1.0) is False  # small dropped
        expected = np.zeros(self._SHAPE, dtype=bool)
        expected[self._LARGE_SLICE] = True
        assert np.array_equal(kept_mask, expected)

    def test_cutoff_above_large_size_drops_everything(self) -> None:
        pred = self._pred_large_and_small()
        _, n = connected_components(pred, min_component_size=self._LARGE_SIZE + 1)
        assert n == 0

    # -- false_positive_components (negative/healthy-sample path) -----------

    def test_false_positive_components_drops_small_blob_at_cutoff(self) -> None:
        pred = self._pred_large_and_small()
        assert false_positive_components(pred) == 2  # unfiltered baseline
        assert (
            false_positive_components(pred, min_component_size=self._SMALL_SIZE + 1)
            == 1
        )

    # -- dice / iou_jaccard: filtering applies to PRED only, target untouched

    def test_dice_and_iou_ignore_filtered_out_pred_component(self) -> None:
        # target == the large component only; pred has both.  Unfiltered,
        # the spurious small pred blob inflates the union and drags
        # dice/iou down (a stand-in for "false activation elsewhere in the
        # frame"). Filtered at small_size+1, the small blob no longer
        # contributes to either metric -- both should equal the "pred ==
        # target exactly" answer of 1.0.
        target = _square_mask(self._SHAPE, *self._LARGE_SLICE)
        pred = self._pred_large_and_small()

        unfiltered_dice = dice(pred, target)
        unfiltered_iou = iou_jaccard(pred, target)
        assert unfiltered_dice < 1.0
        assert unfiltered_iou < 1.0

        filtered_dice = dice(pred, target, min_component_size=self._SMALL_SIZE + 1)
        filtered_iou = iou_jaccard(
            pred, target, min_component_size=self._SMALL_SIZE + 1
        )
        assert filtered_dice == pytest.approx(1.0)
        assert filtered_iou == pytest.approx(1.0)

    def test_min_component_size_never_filters_the_target(self) -> None:
        # A SMALL ground-truth lesion must never be dropped by
        # min_component_size -- only the prediction is filtered (module
        # docstring, item 8).
        target = self._pred_large_and_small()  # both a large + a small GT lesion
        pred = target.copy()  # perfect prediction of both

        filtered_dice = dice(pred, target, min_component_size=self._SMALL_SIZE + 1)
        filtered_iou = iou_jaccard(
            pred, target, min_component_size=self._SMALL_SIZE + 1
        )
        # pred's small component is dropped -> it no longer overlaps the
        # (unfiltered) small target component -> imperfect overlap, NOT 1.0
        # (proves the target side was never itself filtered to compensate).
        assert filtered_dice < 1.0
        assert filtered_iou < 1.0

    # -- lesion_component_confusion: filtering removes a spurious FP blob --

    def test_lesion_component_confusion_drops_small_fp_blob_at_cutoff(self) -> None:
        target = _square_mask(self._SHAPE, *self._LARGE_SLICE)  # one real lesion
        pred = self._pred_large_and_small()  # matches it + one spurious small blob

        unfiltered = lesion_component_confusion(pred, target)
        assert unfiltered.tp == 1
        assert unfiltered.fn == 0
        assert unfiltered.fp == 1
        assert unfiltered.n_pred_components == 2

        filtered = lesion_component_confusion(
            pred, target, min_component_size=self._SMALL_SIZE + 1
        )
        assert filtered.tp == 1
        assert filtered.fn == 0
        assert filtered.fp == 0
        assert filtered.n_pred_components == 1

    def test_lesion_detected_forwards_min_component_size(self) -> None:
        # A real lesion small enough to itself be filtered away as PREDICTED
        # content must become a miss once filtered, even though it would be
        # detected unfiltered (the tp>0 -> tp==0 flip is the sharpest
        # behavioral signature that the parameter reaches this function).
        target = _square_mask(self._SHAPE, *self._SMALL_SLICE)
        pred = target.copy()
        assert lesion_detected(pred, target) is True
        assert (
            lesion_detected(pred, target, min_component_size=self._SMALL_SIZE + 1)
            is False
        )

    def test_surface_metrics_use_the_same_filtered_prediction(self) -> None:
        target = _square_mask(self._SHAPE, *self._LARGE_SLICE)
        pred = self._pred_large_and_small()

        # The distant small component worsens both surface metrics before
        # filtering. Once removed, prediction and target are identical.
        assert hd95(pred, target, spacing=1.0) > 0.0
        assert assd(pred, target, spacing=1.0) > 0.0
        assert hd95(
            pred,
            target,
            spacing=1.0,
            min_component_size=self._SMALL_SIZE + 1,
        ) == pytest.approx(0.0)
        assert assd(
            pred,
            target,
            spacing=1.0,
            min_component_size=self._SMALL_SIZE + 1,
        ) == pytest.approx(0.0)

    def test_negative_min_component_size_is_rejected(self) -> None:
        pred = self._pred_large_and_small()
        with pytest.raises(ValueError, match="min_component_size must be >= 0"):
            connected_components(pred, min_component_size=-1)


# ---------------------------------------------------------------------------
# 7. Precision / Recall / F-beta from plain counts
# ---------------------------------------------------------------------------


class TestPrecisionRecallFBeta:
    def test_known_counts(self) -> None:
        # tp=3, fp=1, fn=1 -> precision=0.75, recall=0.75
        # F1 = 2*P*R/(P+R) = 2*0.75*0.75/1.5 = 1.125/1.5 = 0.75
        result = precision_recall_f_beta(tp=3, fp=1, fn=1, beta=1.0)
        assert result.precision == pytest.approx(0.75)
        assert result.recall == pytest.approx(0.75)
        assert result.f_beta == pytest.approx(0.75)

    def test_f2_weights_recall_more_than_precision(self) -> None:
        # tp=2, fp=0 (precision=1.0), fn=2 (recall=0.5).
        # F1 = 2*1*0.5/1.5 = 0.6667
        # F2 = 5*1*0.5/(4*1+0.5) = 2.5/4.5 = 0.5556 -- weighted toward recall,
        # so F2 < F1 here (recall is the weaker of the two).
        f1 = precision_recall_f_beta(tp=2, fp=0, fn=2, beta=1.0).f_beta
        f2 = precision_recall_f_beta(tp=2, fp=0, fn=2, beta=2.0).f_beta
        assert f1 == pytest.approx(2.0 / 3.0)
        assert f2 == pytest.approx(2.5 / 4.5)
        assert f2 < f1

    def test_perfect_detection(self) -> None:
        result = precision_recall_f_beta(tp=5, fp=0, fn=0)
        assert result.precision == pytest.approx(1.0)
        assert result.recall == pytest.approx(1.0)
        assert result.f_beta == pytest.approx(1.0)

    def test_zero_tp_with_both_fp_and_fn_gives_finite_zero_f_beta(self) -> None:
        result = precision_recall_f_beta(tp=0, fp=2, fn=3)
        assert result.precision == pytest.approx(0.0)
        assert result.recall == pytest.approx(0.0)
        assert result.f_beta == pytest.approx(0.0)

    def test_undefined_precision_gives_nan(self) -> None:
        # tp=0, fp=0 -> precision undefined (0/0), so f_beta must also be nan.
        result = precision_recall_f_beta(tp=0, fp=0, fn=3)
        assert math.isnan(result.precision)
        assert math.isnan(result.f_beta)

    def test_undefined_recall_gives_nan(self) -> None:
        result = precision_recall_f_beta(tp=0, fp=3, fn=0)
        assert math.isnan(result.recall)
        assert math.isnan(result.f_beta)

    def test_reusable_on_lesion_component_confusion_counts(self) -> None:
        shape = (30, 30)
        target = np.zeros(shape, dtype=np.float32)
        target[2:5, 2:5] = 1.0
        target[20:23, 20:23] = 1.0
        pred = np.zeros(shape, dtype=np.float32)
        pred[2:5, 2:5] = 1.0
        pred[15:17, 15:17] = 1.0

        counts = lesion_component_confusion(pred, target)
        result = precision_recall_f_beta(counts.tp, counts.fp, counts.fn)
        assert result.precision == pytest.approx(0.5)
        assert result.recall == pytest.approx(0.5)
        assert result.f_beta == pytest.approx(0.5)
