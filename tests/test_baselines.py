"""Tests for the transparent 26-connectivity threshold baseline detector
(VascuTrace Phase 5, ``src/vascutrace/baselines/threshold.py``).

Every fixture in this file is either a small, hand-computable, generated
array, or a P3-generated synthetic case (``simulate_vascular_anomaly``) --
nothing here reads ``Data/``, the network, CUDA, or any external service.

Most unit-level tests build tiny hand-crafted arrays directly and drive them
through ``DetectionMode.SUV`` (a pass-through mode with no bilateral
reflection at all) or the low-level ``label_components``/
``match_components_to_source`` functions, so the FROZEN 26-connectivity,
matching/tie-rule, F2, and freeze-ceiling logic can be exercised precisely
and deterministically without also depending on the reflection/geometry
machinery. The dedicated end-to-end test at the bottom of this file
exercises the full pipeline (``DetectionMode.ASYMMETRY`` with a real
physical reflection affine) against P3-generated positive and untouched
cases, per this implementation's required "generate positive synthetic cases with
P3 + untouched cases, run detect->match->F2->freeze" test.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import label as scipy_label

from src.vascutrace.geometry import GridGeometry
from src.vascutrace.simulation.anomaly import (
    AnomalySimulationParams,
    simulate_vascular_anomaly,
)
from src.vascutrace.baselines.threshold import (
    RESEARCH_PROTOTYPE_WARNING,
    CONNECTIVITY_26,
    UNTOUCHED_MEAN_COMPONENT_CEILING,
    BaselineCase,
    DetectionMode,
    FreezeResult,
    PositiveCaseOutcome,
    UntouchedCaseOutcome,
    aggregate_f2,
    compute_detection_map,
    evaluate_positive_case,
    evaluate_positive_case_at_threshold,
    evaluate_untouched_case,
    evaluate_untouched_case_at_threshold,
    f2_score,
    freeze_threshold,
    label_components,
    match_components_to_source,
)

# ---------------------------------------------------------------------------
# Fixture helpers (generated arrays only)
# ---------------------------------------------------------------------------


def _make_geometry(
    shape: tuple[int, int, int], affine: np.ndarray | None = None
) -> GridGeometry:
    """Build a :class:`GridGeometry` directly from a known shape/affine pair
    (no NIfTI header, no validation pass), matching ``test_quantification.py``'s
    own fixture convention.
    """
    if affine is None:
        affine = np.eye(4)
        affine[0, 3] = -(shape[0] - 1) / 2.0  # grid symmetric about world x == 0
    affine = np.asarray(affine, dtype=np.float64)
    corners = np.array(
        [
            [i, j, k]
            for i in (0, shape[0] - 1)
            for j in (0, shape[1] - 1)
            for k in (0, shape[2] - 1)
        ],
        dtype=np.float64,
    )
    homogeneous = np.concatenate([corners, np.ones((8, 1))], axis=1)
    world_corners = (affine @ homogeneous.T).T[:, :3]
    spacing = tuple(float(x) for x in np.linalg.norm(affine[:3, :3], axis=0))
    return GridGeometry(
        shape=tuple(int(s) for s in shape),
        affine=affine,
        spacing=spacing,
        units="mm",
        world_bounds_min=world_corners.min(axis=0),
        world_bounds_max=world_corners.max(axis=0),
    )


def _mirror_affine_through_x(offset: float = 0.0) -> np.ndarray:
    """A physical mirror reflecting world x through the plane ``x == offset``,
    matching ``test_quantification.py``'s own ``_mirror_affine_through_x``.
    """
    m = np.eye(4)
    m[0, 0] = -1.0
    m[0, 3] = 2.0 * offset
    return m


_SHAPE = (6, 4, 4)
_GEOMETRY = _make_geometry(_SHAPE)
_REFLECTION_X0 = _mirror_affine_through_x(offset=0.0)


def _suv_case(
    pet: np.ndarray, ground_truth_mask: np.ndarray | None = None
) -> BaselineCase:
    """A :class:`BaselineCase` for ``DetectionMode.SUV`` unit tests: the
    reflection affine is accepted for interface uniformity but never used in
    SUV mode, so a fixed valid mirror suffices for every such case.
    """
    return BaselineCase(
        pet_suvbw=pet.astype(np.float32),
        geometry=_GEOMETRY,
        reflection_affine=_REFLECTION_X0,
        ground_truth_mask=ground_truth_mask,
    )


def _flat_background(
    shape: tuple[int, int, int], value: float = 1.0, noise: float = 0.03, seed: int = 0
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (value + noise * rng.standard_normal(shape)).astype(np.float32)


# ---------------------------------------------------------------------------
# RESEARCH_PROTOTYPE_WARNING / scientific boundary
# ---------------------------------------------------------------------------


def test_research_prototype_warning_exact_text() -> None:
    assert RESEARCH_PROTOTYPE_WARNING == (
        "Research prototype. Trained and evaluated using simulated vascular-like "
        "abnormalities, not confirmed human post-angioplasty lesions."
    )


def test_connectivity_26_is_full_3x3x3_all_ones() -> None:
    assert CONNECTIVITY_26.shape == (3, 3, 3)
    assert np.array_equal(CONNECTIVITY_26, np.ones((3, 3, 3)))


def test_untouched_ceiling_is_frozen_at_quarter_component() -> None:
    assert UNTOUCHED_MEAN_COMPONENT_CEILING == 0.25


# ---------------------------------------------------------------------------
# Detection map: physical reflection, not np.flip
# ---------------------------------------------------------------------------


def test_asymmetry_detection_map_isolates_ipsilateral_excess() -> None:
    """A bump placed only on the high-x (world x > 0) side of a grid
    symmetric about ``x == 0`` produces a detection map that is exactly the
    bump's excess over background on that side, and exactly zero on the
    mirrored (low-x) side -- proving the reflection is a genuine physical
    mirror (voxel ``i`` <-> voxel ``shape[0]-1-i``), not merely a coincidence
    of the fixture.
    """
    pet = np.ones(_SHAPE, dtype=np.float32)
    pet[4:6, 1:3, 1:3] = 5.0  # high-x half (voxel index 4,5 -> world x = 1.5, 2.5)

    detection_map = compute_detection_map(pet, _GEOMETRY, _REFLECTION_X0)

    assert detection_map.dtype == np.float32
    assert detection_map.shape == _SHAPE
    np.testing.assert_allclose(detection_map[4:6, 1:3, 1:3], 4.0, atol=1e-5)
    # Mirrored (low-x) side: mirror voxel i=4 -> shape[0]-1-i = 1, i=5 -> 0.
    np.testing.assert_allclose(detection_map[0:2, 1:3, 1:3], 0.0, atol=1e-5)
    assert np.all(detection_map >= 0.0)  # positive (ipsilateral-excess) side only


def test_asymmetry_detection_map_negative_side_clamped_to_zero() -> None:
    """A voxel whose own SUV is *less* than its mirror's must contribute
    exactly 0.0 to the map (clamped, never left negative).
    """
    pet = np.ones(_SHAPE, dtype=np.float32)
    pet[0:2, 1:3, 1:3] = 5.0  # excess is on the *low*-x side this time

    detection_map = compute_detection_map(pet, _GEOMETRY, _REFLECTION_X0)

    # High-x side (the mirror of the low-x excess) would be *negative*
    # (its own SUV 1.0 minus its mirror's 5.0 == -4.0) without clamping.
    np.testing.assert_allclose(detection_map[4:6, 1:3, 1:3], 0.0, atol=1e-5)
    np.testing.assert_allclose(detection_map[0:2, 1:3, 1:3], 4.0, atol=1e-5)


def test_suv_mode_is_plain_passthrough_no_reflection() -> None:
    """``DetectionMode.SUV`` is a plain-SUV-threshold ablation: the raw PET
    array unchanged, with no bilateral comparison (unlike ``ASYMMETRY``,
    this mode does not zero out the region that ``ASYMMETRY`` would zero).
    """
    pet = np.ones(_SHAPE, dtype=np.float32)
    pet[0:2, 1:3, 1:3] = 5.0

    detection_map = compute_detection_map(
        pet, _GEOMETRY, _REFLECTION_X0, mode=DetectionMode.SUV
    )

    np.testing.assert_allclose(detection_map, pet, atol=1e-6)


def test_malformed_reflection_affine_is_rejected_in_asymmetry_mode() -> None:
    not_a_mirror = np.eye(4)  # identity: not orientation-reversing
    pet = np.ones(_SHAPE, dtype=np.float32)
    with pytest.raises(ValueError):
        compute_detection_map(
            pet, _GEOMETRY, not_a_mirror, mode=DetectionMode.ASYMMETRY
        )


def test_pet_shape_mismatch_is_rejected() -> None:
    wrong_shape_pet = np.ones((_SHAPE[0] + 1, *_SHAPE[1:]), dtype=np.float32)
    with pytest.raises(ValueError):
        compute_detection_map(wrong_shape_pet, _GEOMETRY, _REFLECTION_X0)


# ---------------------------------------------------------------------------
# FROZEN 26-connectivity correctness (diagonal-adjacent voxels are ONE
# component -- proves 26-, not 6-, connectivity)
# ---------------------------------------------------------------------------


def test_diagonal_adjacent_voxels_are_one_component_under_26_connectivity() -> None:
    detection_map = np.zeros((5, 5, 5), dtype=np.float32)
    detection_map[1, 1, 1] = 1.0
    detection_map[2, 2, 2] = 1.0  # touches (1,1,1) only at a shared corner

    labeled_map, component_count = label_components(detection_map, threshold=0.5)

    assert component_count == 1
    assert labeled_map[1, 1, 1] == labeled_map[2, 2, 2]
    assert labeled_map[1, 1, 1] != 0


def test_diagonal_adjacent_voxels_are_two_components_under_6_connectivity() -> None:
    """The contrasting case, proving the frozen choice is genuinely
    26-connectivity and not merely scipy's default: the *same* two-voxel
    configuration under scipy's default (6-connectivity, face-adjacency
    only) structuring element yields TWO components.
    """
    binary = np.zeros((5, 5, 5), dtype=bool)
    binary[1, 1, 1] = True
    binary[2, 2, 2] = True

    _labeled_default, count_default = scipy_label(binary)  # default structure = 6-conn

    assert count_default == 2  # contrast with the 26-conn result above (count == 1)


def test_edge_adjacent_voxels_are_one_component_under_26_connectivity() -> None:
    """Edge-adjacent (differ in exactly two axes by 1, share an edge, not a
    face) voxels are also connected under 26-connectivity.
    """
    detection_map = np.zeros((5, 5, 5), dtype=np.float32)
    detection_map[1, 1, 1] = 1.0
    detection_map[1, 2, 2] = 1.0  # shares an edge, not a face

    _labeled_map, component_count = label_components(detection_map, threshold=0.5)
    assert component_count == 1


def test_threshold_is_strict_greater_than() -> None:
    # Two well-separated (non-adjacent) voxels, one at exactly the
    # threshold, so "strict >" vs. ">=" changes the voxel *count*, not just
    # a merge/split of components.
    detection_map = np.zeros((1, 1, 4), dtype=np.float32)
    detection_map[0, 0, 0] = 0.5
    detection_map[0, 0, 3] = 0.6

    labeled_at_equal, count_at_equal = label_components(detection_map, threshold=0.5)
    labeled_above, count_above = label_components(detection_map, threshold=0.4)

    assert count_at_equal == 1  # only the 0.6 voxel clears a strict > 0.5
    assert labeled_at_equal[0, 0, 0] == 0  # the 0.5 voxel itself is unlabeled
    assert count_above == 2  # both 0.5 and 0.6 clear > 0.4 (still separate)
    assert labeled_above[0, 0, 0] != 0
    assert labeled_above[0, 0, 3] != 0


# ---------------------------------------------------------------------------
# One-source matching + the FROZEN tie rule (deterministic)
# ---------------------------------------------------------------------------


def test_single_overlapping_component_matches() -> None:
    detection_map = np.zeros((5, 5, 5), dtype=np.float32)
    detection_map[1, 1, 1] = 2.0
    labeled_map, count = label_components(detection_map, threshold=0.5)
    gt_mask = np.zeros((5, 5, 5), dtype=bool)
    gt_mask[1, 1, 1] = True

    matched, unmatched = match_components_to_source(
        detection_map, labeled_map, count, gt_mask
    )

    assert matched == 1
    assert unmatched == ()


def test_no_overlapping_component_is_a_miss_and_all_components_unmatched() -> None:
    detection_map = np.zeros((5, 5, 5), dtype=np.float32)
    detection_map[0, 0, 0] = 2.0
    detection_map[4, 4, 4] = 3.0
    labeled_map, count = label_components(detection_map, threshold=0.5)
    gt_mask = np.zeros((5, 5, 5), dtype=bool)
    gt_mask[2, 2, 2] = True  # neither component touches this

    matched, unmatched = match_components_to_source(
        detection_map, labeled_map, count, gt_mask
    )

    assert matched is None
    assert count == 2
    assert set(unmatched) == {1, 2}  # both visible, counted -- never dropped


def test_tie_rule_prefers_greatest_intersection_voxel_count() -> None:
    """Two separate components both overlap the GT mask; the one with more
    overlapping voxels wins regardless of its peak detection value.
    """
    detection_map = np.zeros((6, 6, 6), dtype=np.float32)
    # Component A: 2 voxels overlapping GT, low peak value.
    detection_map[0, 0, 0] = 0.6
    detection_map[0, 0, 1] = 0.6
    # Component B: 1 voxel overlapping GT, but a much higher peak value.
    detection_map[5, 5, 5] = 100.0
    labeled_map, count = label_components(detection_map, threshold=0.5)
    assert count == 2

    gt_mask = np.zeros((6, 6, 6), dtype=bool)
    gt_mask[0, 0, 0] = True
    gt_mask[0, 0, 1] = True
    gt_mask[5, 5, 5] = True

    matched, unmatched = match_components_to_source(
        detection_map, labeled_map, count, gt_mask
    )

    winning_label = labeled_map[0, 0, 0]
    assert matched == winning_label
    assert len(unmatched) == 1


def test_tie_rule_second_level_prefers_greatest_component_peak_score() -> None:
    """Two single-voxel components both overlap the GT mask with an
    identical (1-voxel) intersection count; the tie is broken by the
    component's peak detection-map value.
    """
    detection_map = np.zeros((6, 6, 6), dtype=np.float32)
    detection_map[0, 0, 0] = 0.6  # lower score
    detection_map[5, 5, 5] = 3.0  # higher score
    labeled_map, count = label_components(detection_map, threshold=0.5)
    assert count == 2

    gt_mask = np.zeros((6, 6, 6), dtype=bool)
    gt_mask[0, 0, 0] = True
    gt_mask[5, 5, 5] = True

    matched, unmatched = match_components_to_source(
        detection_map, labeled_map, count, gt_mask
    )

    assert matched == labeled_map[5, 5, 5]
    assert unmatched == (int(labeled_map[0, 0, 0]),)


def test_tie_rule_third_level_prefers_lowest_label_id_deterministically() -> None:
    """Two single-voxel components, identical (1-voxel) overlap AND
    identical peak score: the final, fully deterministic tie-break is the
    lowest label id (== lexicographically smallest native voxel index for
    this project's installed SciPy -- verified by this test itself, since
    the lexicographically-earlier voxel (0,0,0) is the one that wins).
    """
    detection_map = np.zeros((6, 6, 6), dtype=np.float32)
    detection_map[0, 0, 0] = 1.0
    detection_map[5, 5, 5] = 1.0  # identical score, identical overlap size
    labeled_map, count = label_components(detection_map, threshold=0.5)
    assert count == 2
    first_label = int(labeled_map[0, 0, 0])
    second_label = int(labeled_map[5, 5, 5])
    assert first_label < second_label  # sanity: confirms raster-scan label order

    gt_mask = np.zeros((6, 6, 6), dtype=bool)
    gt_mask[0, 0, 0] = True
    gt_mask[5, 5, 5] = True

    matched, unmatched = match_components_to_source(
        detection_map, labeled_map, count, gt_mask
    )

    assert matched == first_label
    assert unmatched == (second_label,)


def test_matching_is_deterministic_across_repeated_calls() -> None:
    detection_map = np.zeros((6, 6, 6), dtype=np.float32)
    detection_map[0, 0, 0] = 0.6
    detection_map[0, 0, 1] = 0.6
    detection_map[5, 5, 5] = 100.0
    labeled_map, count = label_components(detection_map, threshold=0.5)
    gt_mask = np.zeros((6, 6, 6), dtype=bool)
    gt_mask[0, 0, 0] = True
    gt_mask[0, 0, 1] = True
    gt_mask[5, 5, 5] = True

    results = [
        match_components_to_source(detection_map, labeled_map, count, gt_mask)
        for _ in range(5)
    ]
    assert all(r == results[0] for r in results)


def test_evaluate_positive_case_at_threshold_combines_label_and_match() -> None:
    """The public 'layer B' entry point (already-built detection map,
    threshold applied here) produces the exact same TP/FN/FP/labels as
    manually chaining ``label_components`` + ``match_components_to_source``.
    """
    detection_map = np.zeros((6, 6, 6), dtype=np.float32)
    detection_map[0, 0, 0] = 2.0  # matches
    detection_map[5, 5, 5] = 2.0  # spurious, unmatched
    gt_mask = np.zeros((6, 6, 6), dtype=bool)
    gt_mask[0, 0, 0] = True

    outcome = evaluate_positive_case_at_threshold(detection_map, gt_mask, threshold=1.0)

    labeled_map, count = label_components(detection_map, threshold=1.0)
    matched, unmatched = match_components_to_source(
        detection_map, labeled_map, count, gt_mask
    )
    assert outcome == PositiveCaseOutcome(
        tp=1,
        fn=0,
        fp=1,
        component_count=2,
        matched_label=matched,
        unmatched_labels=unmatched,
    )


def test_evaluate_untouched_case_at_threshold_reports_component_count_only() -> None:
    detection_map = np.zeros((6, 6, 6), dtype=np.float32)
    detection_map[0, 0, 0] = 2.0
    detection_map[5, 5, 5] = 2.0

    outcome = evaluate_untouched_case_at_threshold(detection_map, threshold=1.0)

    assert outcome == UntouchedCaseOutcome(component_count=2, component_labels=(1, 2))


def test_empty_component_set_has_no_match() -> None:
    detection_map = np.zeros((4, 4, 4), dtype=np.float32)
    labeled_map, count = label_components(detection_map, threshold=0.5)
    gt_mask = np.zeros((4, 4, 4), dtype=bool)
    gt_mask[1, 1, 1] = True

    matched, unmatched = match_components_to_source(
        detection_map, labeled_map, count, gt_mask
    )
    assert matched is None
    assert unmatched == ()


# ---------------------------------------------------------------------------
# Unmatched components remain visible / counted as FP (never dropped)
# ---------------------------------------------------------------------------


def test_unmatched_components_counted_as_fp_and_remain_visible() -> None:
    """One component matches the GT source; two others do not overlap it at
    all (spurious extra detections). All three must be visible in the
    outcome, and the two non-matching ones must both count toward FP.
    """
    pet = np.zeros(_SHAPE, dtype=np.float32)
    pet[0, 0, 0] = 2.0  # will match the GT source
    pet[3, 0, 0] = 2.0  # spurious, unrelated component
    pet[5, 3, 3] = 2.0  # spurious, unrelated component
    gt_mask = np.zeros(_SHAPE, dtype=np.int8)
    gt_mask[0, 0, 0] = 1

    case = _suv_case(pet, ground_truth_mask=gt_mask)
    outcome = evaluate_positive_case(case, threshold=1.0, mode=DetectionMode.SUV)

    assert outcome.tp == 1
    assert outcome.fn == 0
    assert outcome.fp == 2
    assert outcome.component_count == 3
    assert outcome.matched_label is not None
    assert len(outcome.unmatched_labels) == 2
    # Both unmatched labels are genuinely present in the outcome, not merely
    # counted: they must differ from the matched label.
    assert outcome.matched_label not in outcome.unmatched_labels


def test_no_components_detected_is_a_false_negative() -> None:
    pet = np.zeros(_SHAPE, dtype=np.float32)  # nothing above threshold
    gt_mask = np.zeros(_SHAPE, dtype=np.int8)
    gt_mask[0, 0, 0] = 1

    case = _suv_case(pet, ground_truth_mask=gt_mask)
    outcome = evaluate_positive_case(case, threshold=1.0, mode=DetectionMode.SUV)

    assert outcome == PositiveCaseOutcome(
        tp=0, fn=1, fp=0, component_count=0, matched_label=None, unmatched_labels=()
    )


def test_evaluate_positive_case_requires_ground_truth_mask() -> None:
    case = _suv_case(np.zeros(_SHAPE, dtype=np.float32), ground_truth_mask=None)
    with pytest.raises(ValueError):
        evaluate_positive_case(case, threshold=1.0, mode=DetectionMode.SUV)


def test_untouched_case_outcome_never_labeled_false_positive() -> None:
    """Contract point 7: untouched-scan detections are activation/component
    counts, never "false positives" -- ``UntouchedCaseOutcome`` has no ``fp``
    field at all.
    """
    assert not hasattr(UntouchedCaseOutcome, "fp")
    assert set(UntouchedCaseOutcome.__dataclass_fields__) == {
        "component_count",
        "component_labels",
    }

    pet = np.zeros(_SHAPE, dtype=np.float32)
    pet[0, 0, 0] = 2.0
    case = _suv_case(pet)
    outcome = evaluate_untouched_case(case, threshold=1.0, mode=DetectionMode.SUV)
    assert outcome.component_count == 1
    assert outcome.component_labels == (1,)


# ---------------------------------------------------------------------------
# Exact, FROZEN F2 = 5*TP / (5*TP + 4*FN + FP)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tp, fn, fp, expected",
    [
        (1, 0, 0, 1.0),
        (1, 0, 2, 5.0 / 7.0),
        (0, 1, 0, 0.0),
        (2, 0, 0, 1.0),
        (0, 0, 1, 0.0),
        (3, 1, 2, 15.0 / 21.0),
    ],
)
def test_f2_score_exact_hand_constructed_cases(
    tp: int, fn: int, fp: int, expected: float
) -> None:
    assert f2_score(tp, fn, fp) == pytest.approx(expected)


def test_f2_score_degenerate_zero_denominator_is_explicit_zero_not_nan() -> None:
    value = f2_score(0, 0, 0)
    assert value == 0.0
    assert not np.isnan(value)


def test_aggregate_f2_pools_outcomes() -> None:
    outcomes = [
        PositiveCaseOutcome(
            tp=1, fn=0, fp=1, component_count=2, matched_label=1, unmatched_labels=(2,)
        ),
        PositiveCaseOutcome(
            tp=0, fn=1, fp=0, component_count=0, matched_label=None, unmatched_labels=()
        ),
        PositiveCaseOutcome(
            tp=1, fn=0, fp=0, component_count=1, matched_label=1, unmatched_labels=()
        ),
    ]
    # tp=2, fn=1, fp=1 -> F2 = 10 / (10 + 4 + 1) = 10/15
    assert aggregate_f2(outcomes) == pytest.approx(10.0 / 15.0)


# ---------------------------------------------------------------------------
# Determinism (module-wide)
# ---------------------------------------------------------------------------


def test_detection_map_is_deterministic() -> None:
    pet = np.ones(_SHAPE, dtype=np.float32)
    pet[4:6, 1:3, 1:3] = 5.0
    first = compute_detection_map(pet, _GEOMETRY, _REFLECTION_X0)
    second = compute_detection_map(pet, _GEOMETRY, _REFLECTION_X0)
    assert np.array_equal(first, second)


def test_full_pipeline_is_deterministic_across_repeated_calls() -> None:
    pet = np.zeros(_SHAPE, dtype=np.float32)
    pet[0, 0, 0] = 2.0
    pet[3, 0, 0] = 2.0
    gt_mask = np.zeros(_SHAPE, dtype=np.int8)
    gt_mask[0, 0, 0] = 1
    case = _suv_case(pet, ground_truth_mask=gt_mask)

    outcomes = [
        evaluate_positive_case(case, threshold=1.0, mode=DetectionMode.SUV)
        for _ in range(5)
    ]
    assert all(o == outcomes[0] for o in outcomes)


# ---------------------------------------------------------------------------
# Validation-only freeze: picks the F2-maximizing feasible T
# ---------------------------------------------------------------------------


def _tiny_freeze_fixture() -> tuple[list[BaselineCase], list[BaselineCase]]:
    """Three positive cases (source always detectable above every candidate
    threshold below) and four untouched cases (clean, nothing above any
    candidate threshold) -- built with ``DetectionMode.SUV`` hand arrays so
    the freeze sweep itself is fully hand-verifiable.
    """
    positive_cases = []
    for i in range(3):
        pet = np.zeros(_SHAPE, dtype=np.float32)
        pet[3, 1, 1] = 10.0  # source, well above every candidate threshold
        gt_mask = np.zeros(_SHAPE, dtype=np.int8)
        gt_mask[3, 1, 1] = 1
        positive_cases.append(_suv_case(pet, ground_truth_mask=gt_mask))

    untouched_cases = [_suv_case(np.zeros(_SHAPE, dtype=np.float32)) for _ in range(4)]
    return positive_cases, untouched_cases


def test_freeze_threshold_picks_f2_maximizing_feasible_threshold() -> None:
    positive_cases, untouched_cases = _tiny_freeze_fixture()

    result = freeze_threshold(
        positive_cases,
        untouched_cases,
        candidate_thresholds=[1.0, 3.0, 5.0, 9.9],
        mode=DetectionMode.SUV,
    )

    assert result.feasible is True
    # Every candidate is feasible (untouched cases are all-clean) and every
    # candidate perfectly detects the source (F2 == 1.0 for all of them):
    # the FROZEN tie-break then prefers the *highest* threshold.
    assert result.frozen_threshold == 9.9
    assert result.frozen_f2 == pytest.approx(1.0)
    assert result.frozen_untouched_mean_components == pytest.approx(0.0)
    assert len(result.swept_thresholds) == 4
    for point in result.swept_thresholds:
        assert point.ceiling_satisfied is True


def test_freeze_threshold_ceiling_excludes_overly_permissive_thresholds() -> None:
    """A low threshold that also picks up untouched-scan noise must be
    excluded from consideration even if it achieves perfect F2 on positives.
    """
    positive_cases, _ = _tiny_freeze_fixture()
    noisy_untouched_cases = []
    for i in range(4):
        pet = np.zeros(_SHAPE, dtype=np.float32)
        pet[0, 0, 0] = 2.0  # spurious activation on every untouched scan
        noisy_untouched_cases.append(_suv_case(pet))

    # threshold=1.0 detects both the source (10.0) and the untouched noise
    # (2.0): mean_components == 1.0 > ceiling -> infeasible.
    # threshold=5.0 detects only the source (10.0), not the noise (2.0):
    # mean_components == 0.0 <= ceiling -> feasible.
    result = freeze_threshold(
        positive_cases,
        noisy_untouched_cases,
        candidate_thresholds=[1.0, 5.0],
        mode=DetectionMode.SUV,
    )

    assert result.feasible is True
    assert result.frozen_threshold == 5.0
    low_point = next(p for p in result.swept_thresholds if p.threshold == 1.0)
    high_point = next(p for p in result.swept_thresholds if p.threshold == 5.0)
    assert low_point.ceiling_satisfied is False
    assert low_point.untouched_mean_components == pytest.approx(1.0)
    assert high_point.ceiling_satisfied is True
    assert high_point.untouched_mean_components == pytest.approx(0.0)


def test_freeze_threshold_no_feasible_threshold_is_structured_not_relaxed() -> None:
    """Every candidate threshold, however good its positive-side F2, is
    infeasible because a persistent spurious activation on the untouched
    scans is never below the ceiling for any tested T. The ceiling must NOT
    be relaxed to force a number.
    """
    positive_cases, _ = _tiny_freeze_fixture()
    always_active_untouched_cases = []
    for i in range(4):
        pet = np.zeros(_SHAPE, dtype=np.float32)
        pet[0, 0, 0] = 10.0  # matches the positive source's own magnitude
        always_active_untouched_cases.append(_suv_case(pet))

    result = freeze_threshold(
        positive_cases,
        always_active_untouched_cases,
        candidate_thresholds=[1.0, 5.0],
        mode=DetectionMode.SUV,
    )

    assert result.feasible is False
    assert result.frozen_threshold is None
    assert result.frozen_f2 is None
    assert result.frozen_untouched_mean_components is None
    assert result.frozen_untouched_activation_rate is None
    assert result.reason is not None
    assert "NO_FEASIBLE_THRESHOLD" in result.reason
    # Full sweep record is still returned for audit -- infeasibility is
    # reported, not hidden.
    assert len(result.swept_thresholds) == 2
    assert all(not p.ceiling_satisfied for p in result.swept_thresholds)
    # Every swept candidate had a *perfect* positive F2 -- proving this is
    # genuinely "infeasible despite good F2", not "infeasible because F2 was
    # also bad".
    assert all(p.positive_f2 == pytest.approx(1.0) for p in result.swept_thresholds)


def test_freeze_threshold_is_deterministic() -> None:
    positive_cases, untouched_cases = _tiny_freeze_fixture()
    results = [
        freeze_threshold(
            positive_cases,
            untouched_cases,
            candidate_thresholds=[1.0, 3.0, 5.0, 9.9],
            mode=DetectionMode.SUV,
        )
        for _ in range(3)
    ]
    assert all(r == results[0] for r in results)


def test_freeze_threshold_requires_at_least_one_positive_and_untouched_case() -> None:
    positive_cases, untouched_cases = _tiny_freeze_fixture()
    with pytest.raises(ValueError):
        freeze_threshold([], untouched_cases, candidate_thresholds=[1.0])
    with pytest.raises(ValueError):
        freeze_threshold(positive_cases, [], candidate_thresholds=[1.0])


def test_freeze_result_feasible_false_never_carries_a_frozen_value() -> None:
    """A structural guard on the no-feasible-threshold contract itself:
    every ``frozen_*`` field is ``None`` exactly when ``feasible`` is
    ``False`` -- there is no code path that could report a number "just in
    case" alongside an infeasible verdict.
    """
    infeasible = FreezeResult(
        feasible=False,
        frozen_threshold=None,
        frozen_f2=None,
        frozen_untouched_mean_components=None,
        frozen_untouched_activation_rate=None,
        reason="NO_FEASIBLE_THRESHOLD: example",
        swept_thresholds=(),
    )
    assert infeasible.frozen_threshold is None
    assert infeasible.frozen_f2 is None
    assert infeasible.frozen_untouched_mean_components is None
    assert infeasible.frozen_untouched_activation_rate is None


# ---------------------------------------------------------------------------
# End-to-end: P3-generated positive cases + untouched cases, full pipeline
# ---------------------------------------------------------------------------


def _e2e_geometry() -> GridGeometry:
    shape = (24, 16, 16)
    return _make_geometry(shape)


def _e2e_positive_case(geometry: GridGeometry, seed: int) -> BaselineCase:
    """One P3-generated positive case: a real synthetic vascular source
    inserted on the high-world-x side of a grid symmetric about world
    ``x == 0``, via ``simulate_vascular_anomaly`` (P3).
    """
    background = _flat_background(geometry.shape, seed=seed)
    contralateral_mask = np.zeros(geometry.shape, dtype=bool)
    contralateral_mask[1:9, :, :] = True  # low-world-x baseline-sampling corridor
    centerline = np.array([[5.0, 8.0, 8.0], [9.0, 8.0, 8.0]])  # high-world-x (5..9mm)
    params = AnomalySimulationParams(
        side="right",
        radius_mm=2.0,
        length_mm=4.0,
        uptake_multiplier=3.0,
        blur_fwhm_mm=2.0,
        heterogeneity=0.1,
        pet_ct_shift_mm=(0.0, 0.0, 0.0),
        seed=seed,
    )
    result = simulate_vascular_anomaly(
        background, geometry, centerline, contralateral_mask, params
    )
    return BaselineCase(
        pet_suvbw=result.synthetic_pet,
        geometry=geometry,
        reflection_affine=_mirror_affine_through_x(offset=0.0),
        ground_truth_mask=result.ground_truth_mask,
    )


def _e2e_untouched_case(geometry: GridGeometry, seed: int) -> BaselineCase:
    background = _flat_background(geometry.shape, seed=seed)
    return BaselineCase(
        pet_suvbw=background,
        geometry=geometry,
        reflection_affine=_mirror_affine_through_x(offset=0.0),
        ground_truth_mask=None,
    )


def test_end_to_end_p3_generated_cases_detect_match_f2_freeze() -> None:
    """Generate positive synthetic cases with P3 and untouched cases, run
    the full detect -> match -> F2 -> freeze pipeline, and OBSERVE the
    frozen threshold + metrics.
    """
    geometry = _e2e_geometry()
    positive_cases = [_e2e_positive_case(geometry, seed=s) for s in range(4)]
    untouched_cases = [_e2e_untouched_case(geometry, seed=100 + s) for s in range(4)]

    # Sanity: every positive case actually has a nonempty ground-truth
    # source (P3's own contract), and detection maps are well-formed.
    for case in positive_cases:
        assert case.ground_truth_mask is not None
        assert case.ground_truth_mask.sum() > 0
        detection_map = compute_detection_map(
            case.pet_suvbw, case.geometry, case.reflection_affine
        )
        assert detection_map.shape == geometry.shape
        assert np.all(detection_map >= 0.0)
        assert detection_map.max() > 0.5  # the inserted source is visible

    result = freeze_threshold(
        positive_cases,
        untouched_cases,
        candidate_thresholds=[0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5],
    )

    # OBSERVE the frozen threshold + metrics (this is the fresh
    # run-and-observe artifact this required test asks for -- not merely a
    # green assertion, but concrete printed evidence of what the pipeline
    # actually produced on this run).
    print("Frozen result:", result.feasible, result.frozen_threshold, result.frozen_f2)
    for point in result.swept_thresholds:
        print(" ", point)

    assert result.feasible is True
    assert result.frozen_threshold is not None
    assert result.frozen_threshold > 0.1  # the too-permissive low end is excluded
    assert result.frozen_f2 == pytest.approx(
        1.0
    )  # every source detected, no spurious FPs
    assert result.frozen_untouched_mean_components == pytest.approx(0.0)
    assert result.frozen_untouched_activation_rate == pytest.approx(0.0)

    # Direct, independent per-case observation at the frozen threshold
    # (re-derives the same TP/FN/FP the freeze already computed, from the
    # public per-case API, as a cross-check).
    outcomes = [
        evaluate_positive_case(case, result.frozen_threshold) for case in positive_cases
    ]
    assert all(o.tp == 1 and o.fn == 0 and o.fp == 0 for o in outcomes)
    untouched_outcomes = [
        evaluate_untouched_case(case, result.frozen_threshold)
        for case in untouched_cases
    ]
    assert all(o.component_count == 0 for o in untouched_outcomes)


def test_end_to_end_default_threshold_grid_without_explicit_candidates() -> None:
    """``freeze_threshold`` also works with its internally-built default
    threshold grid (no ``candidate_thresholds`` supplied).
    """
    geometry = _e2e_geometry()
    positive_cases = [_e2e_positive_case(geometry, seed=s) for s in range(2)]
    untouched_cases = [_e2e_untouched_case(geometry, seed=200 + s) for s in range(2)]

    result = freeze_threshold(positive_cases, untouched_cases)

    assert len(result.swept_thresholds) > 0
    if result.feasible:
        assert result.frozen_threshold is not None
        assert result.frozen_f2 is not None
    else:
        assert result.reason is not None
