"""Deterministic reference baselines and synthetic detectability experiments."""

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import numpy as np

from vascutrace.contracts import StrictModel


class BaselineMetrics(StrictModel):
    model: str
    dice: float
    positive_voxel_recall: float
    background_voxel_activation_rate: float
    detected: bool


class ExperimentRow(StrictModel):
    radius_mm: float
    uptake_multiplier: float
    blur_fwhm_mm: float
    source_offset_mm: float
    seed: int
    pet_only: BaselineMetrics
    pet_ct_non_siamese: BaselineMetrics


class ExperimentSummary(StrictModel):
    experiment_id: str
    rows: list[ExperimentRow]
    results_json_path: Path
    results_csv_path: Path
    research_only: bool = True


def _synthetic_sample(
    radius_mm: float,
    uptake_multiplier: float,
    blur_fwhm_mm: float,
    source_offset_mm: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    z, y, x = np.indices((8, 32, 32))
    anatomy = np.exp(-(((x - 16) / 12) ** 2 + ((y - 16) / 14) ** 2))
    pet = (0.72 + 0.16 * anatomy + rng.normal(0, 0.012, anatomy.shape)).astype(
        np.float32
    )
    ct = (-850 + 900 * anatomy).astype(np.float32)
    radius_px = max(radius_mm / 2.0, 1.0)
    shift_px = source_offset_mm / 2.0
    mask = (
        ((x - (22 + shift_px)) ** 2 + (y - 18) ** 2 <= radius_px**2)
        & (z >= 2)
        & (z <= 5)
    )
    recovery = radius_mm / (radius_mm + blur_fwhm_mm / 2.0)
    pet[mask] += np.float32(0.58 * (uptake_multiplier - 1.0) * recovery)
    return pet, ct, mask


def _score_prediction(
    name: str, prediction: np.ndarray, truth: np.ndarray
) -> BaselineMetrics:
    intersection = int(np.logical_and(prediction, truth).sum())
    predicted = int(prediction.sum())
    positive = int(truth.sum())
    negative = truth.size - positive
    dice = 2 * intersection / (predicted + positive) if predicted + positive else 1.0
    positive_voxel_recall = intersection / positive if positive else 1.0
    background_activations = int(np.logical_and(prediction, ~truth).sum())
    return BaselineMetrics(
        model=name,
        dice=dice,
        positive_voxel_recall=positive_voxel_recall,
        background_voxel_activation_rate=background_activations / negative
        if negative
        else 0.0,
        detected=positive_voxel_recall >= 0.5 and dice >= 0.1,
    )


def run_reference_baselines(
    pet: np.ndarray, ct: np.ndarray, truth: np.ndarray
) -> tuple[BaselineMetrics, BaselineMetrics]:
    """Evaluate frozen deterministic PET-only and PET+CT threshold references."""

    control = pet[:, :, :16]
    threshold = float(control.mean() + 3.0 * control.std())
    pet_prediction = pet > threshold
    pet_ct_prediction = pet_prediction & (ct > -780) & (ct < 200)
    return (
        _score_prediction("pet_only_threshold_v1", pet_prediction, truth),
        _score_prediction("pet_ct_non_siamese_threshold_v1", pet_ct_prediction, truth),
    )


def run_detectability_experiment(
    output_root: Path | str = "outputs/experiments",
    *,
    radii_mm: list[float] | None = None,
    uptake_multipliers: list[float] | None = None,
    blur_fwhm_mm: list[float] | None = None,
    source_offset_mm: list[float] | None = None,
    seeds: list[int] | None = None,
) -> ExperimentSummary:
    radii_mm = radii_mm or [2.0, 4.0, 6.0, 8.0]
    uptake_multipliers = uptake_multipliers or [1.2, 1.4, 1.6, 2.0]
    blur_fwhm_mm = blur_fwhm_mm or [3.0, 5.0, 7.0]
    source_offset_mm = source_offset_mm or [0.0, 2.0, 4.0]
    seeds = seeds or [182]
    rows: list[ExperimentRow] = []
    for radius, uptake, blur, shift, seed in itertools.product(
        radii_mm, uptake_multipliers, blur_fwhm_mm, source_offset_mm, seeds
    ):
        pet, ct, truth = _synthetic_sample(radius, uptake, blur, shift, seed)
        pet_only, pet_ct = run_reference_baselines(pet, ct, truth)
        rows.append(
            ExperimentRow(
                radius_mm=radius,
                uptake_multiplier=uptake,
                blur_fwhm_mm=blur,
                source_offset_mm=shift,
                seed=seed,
                pet_only=pet_only,
                pet_ct_non_siamese=pet_ct,
            )
        )

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    experiment_id = "synthetic_detectability_reference_v1"
    json_path = root / f"{experiment_id}.json"
    csv_path = root / f"{experiment_id}.csv"
    json_path.write_text(
        json.dumps([row.model_dump(mode="json") for row in rows], indent=2) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "radius_mm",
                "uptake_multiplier",
                "blur_fwhm_mm",
                "source_offset_mm",
                "seed",
                "pet_only_dice",
                "pet_only_detected",
                "pet_ct_dice",
                "pet_ct_detected",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "radius_mm": row.radius_mm,
                    "uptake_multiplier": row.uptake_multiplier,
                    "blur_fwhm_mm": row.blur_fwhm_mm,
                    "source_offset_mm": row.source_offset_mm,
                    "seed": row.seed,
                    "pet_only_dice": row.pet_only.dice,
                    "pet_only_detected": row.pet_only.detected,
                    "pet_ct_dice": row.pet_ct_non_siamese.dice,
                    "pet_ct_detected": row.pet_ct_non_siamese.detected,
                }
            )
    return ExperimentSummary(
        experiment_id=experiment_id,
        rows=rows,
        results_json_path=json_path,
        results_csv_path=csv_path,
    )
