"""Deterministic imaging and reporting services used by tools and the UI."""

from __future__ import annotations

import json
import os
import struct
import time
import zlib
from pathlib import Path

import numpy as np
import torch

from src.vascutrace.ml.infer import (
    FROZEN_CHECKPOINT_PATH,
    load_inference_model,
    predict_mask,
    predict_abnormality_score,
    resolve_operating_point,
)
from vascutrace.contracts import (
    Finding,
    ModelOutput,
    QualityControl,
    QuantitativeMeasurements,
    ResearchReport,
)

DEMO_CASE_ID = "synthetic_iliac_seed_182"

# Real-model product bridge to the p6-cache validation split.
# ``create_siamese_research_case`` materializes one locally generated sample.
SIAMESE_VAL_CACHE_DIR = Path("data/processed/p6_cache_big/val")


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data))
    )


def _write_rgb_png(path: Path, image: np.ndarray) -> None:
    """Write an uint8 RGB array without adding a plotting dependency."""

    height, width, channels = image.shape
    if image.dtype != np.uint8 or channels != 3:
        raise ValueError("PNG input must be an uint8 RGB array")
    scanlines = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(scanlines, level=9))
        + _png_chunk(b"IEND", b"")
    )


def _to_gray(image: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image, (1, 99))
    return np.clip(255 * (image - low) / max(float(high - low), 1e-6), 0, 255).astype(
        np.uint8
    )


def create_demo_case(output_root: Path | str = "outputs/demo") -> Path:
    """Create a reproducible research-only bilateral PET/CT demo case."""

    case_dir = Path(output_root) / DEMO_CASE_ID
    case_dir.mkdir(parents=True, exist_ok=True)
    shape = (16, 64, 64)
    z, y, x = np.indices(shape)
    anatomy = np.exp(-(((x - 32) / 23) ** 2 + ((y - 32) / 27) ** 2))
    background = 0.72 + 0.18 * anatomy + 0.015 * np.sin(x / 5) * np.cos(y / 7)
    lesion = ((x - 42) ** 2 + (y - 35) ** 2 <= 5**2) & (z >= 4) & (z <= 11)
    pet = background.astype(np.float32)
    pet[lesion] += np.float32(0.76)
    ct = (-850 + 900 * anatomy).astype(np.float32)

    np.save(case_dir / "pet.npy", pet)
    np.save(case_dir / "ct.npy", ct)
    np.save(case_dir / "ground_truth_mask.npy", lesion.astype(np.uint8))
    _write_json(
        case_dir / "simulation_parameters.json",
        {
            "case_id": DEMO_CASE_ID,
            "case_type": "synthetic",
            "side": "right",
            "radius_mm": 5.0,
            "length_mm": 16.0,
            "uptake_multiplier": 1.6,
            "seed": 182,
            "research_only": True,
        },
    )
    return case_dir


def calculate_demo_metrics(case_dir: Path | str) -> QuantitativeMeasurements:
    """Calculate measurements from arrays; values never originate from prose."""

    case_dir = Path(case_dir)
    pet = np.load(case_dir / "pet.npy")
    mask = np.load(case_dir / "ground_truth_mask.npy").astype(bool)
    opposite = np.flip(mask, axis=2)
    target, control = pet[mask], pet[opposite]
    target_mean, control_mean = float(target.mean()), float(control.mean())
    return QuantitativeMeasurements(
        target_suvmax=float(target.max()),
        target_suvmean=target_mean,
        contralateral_suvmax=float(control.max()),
        contralateral_suvmean=control_mean,
        asymmetry_index=(target_mean - control_mean) / control_mean,
        metabolic_volume_ml=float(mask.sum()) / 1000.0,
        longitudinal_extent_mm=float(np.count_nonzero(mask.any(axis=(1, 2))) * 2),
        quality_flags=["partial_volume_risk"],
    )


def create_demo_overlay(case_dir: Path | str) -> Path:
    return create_demo_views(case_dir)["overlay_path"]


def create_demo_views(case_dir: Path | str) -> dict[str, Path]:
    """Create deterministic PET, CT, fusion, overlay, and bilateral PNG views."""

    case_dir = Path(case_dir)
    pet = np.load(case_dir / "pet.npy")[8]
    ct = np.load(case_dir / "ct.npy")[8]
    mask = np.load(case_dir / "ground_truth_mask.npy")[8].astype(bool)
    pet_gray, ct_gray = _to_gray(pet), _to_gray(ct)
    pet_rgb = np.repeat(pet_gray[..., None], 3, axis=2)
    ct_rgb = np.repeat(ct_gray[..., None], 3, axis=2)
    fused = np.stack(
        [pet_gray, ((pet_gray.astype(float) + ct_gray) / 2).astype(np.uint8), ct_gray],
        axis=2,
    )
    overlay = pet_rgb.copy()
    overlay[mask] = np.array([255, 40, 40], dtype=np.uint8)
    bilateral = np.concatenate(
        [
            np.flip(pet_rgb[:, :32], axis=1),
            np.full((64, 2, 3), 255, np.uint8),
            pet_rgb[:, 32:],
        ],
        axis=1,
    )
    views = {
        "pet_path": pet_rgb,
        "ct_path": ct_rgb,
        "fused_path": fused,
        "overlay_path": overlay,
        "bilateral_path": bilateral,
    }
    paths: dict[str, Path] = {}
    for name, image in views.items():
        path = case_dir / f"{name.removesuffix('_path')}.png"
        _write_rgb_png(path, image)
        paths[name] = path
    return paths


def run_demo_detection(case_dir: Path | str) -> ModelOutput:
    started = time.perf_counter()
    case_dir = Path(case_dir)
    metrics = calculate_demo_metrics(case_dir)
    metrics_path = case_dir / "metrics.json"
    _write_json(metrics_path, metrics.model_dump(mode="json"))
    overlay_path = create_demo_overlay(case_dir)
    result = ModelOutput(
        case_id=DEMO_CASE_ID,
        model_name="deterministic-synthetic-reference",
        model_version="1.0.0",
        laterality="right",
        abnormality_score=0.91,
        mask_path=case_dir / "ground_truth_mask.npy",
        metrics_path=metrics_path,
        overlay_path=overlay_path,
        runtime_seconds=time.perf_counter() - started,
    )
    _write_json(case_dir / "model_output.json", result.model_dump(mode="json"))
    return result


_LATERALITY_INTERPRETATION_PHRASE = {
    "left": "simulated left-sided asymmetric uptake",
    "right": "simulated right-sided asymmetric uptake",
    "bilateral": "simulated bilateral asymmetric uptake",
    "none": "no significant simulated asymmetric uptake detected",
}


def generate_demo_report(output: ModelOutput) -> ResearchReport:
    metrics = QuantitativeMeasurements.model_validate_json(
        output.metrics_path.read_text()
    )
    return ResearchReport(
        case_id=output.case_id,
        case_type="synthetic_research_case",
        finding=Finding(
            laterality=output.laterality,
            target_region="iliac_proximal_femoral_corridor",
            abnormality_score=output.abnormality_score,
        ),
        quantitative_measurements=metrics,
        quality_control=QualityControl(
            partial_volume_risk="partial_volume_risk" in metrics.quality_flags,
            misregistration_risk="misregistration_risk" in metrics.quality_flags,
            flags=metrics.quality_flags,
        ),
        interpretation=(
            "Synthetic research case with "
            f"{_LATERALITY_INTERPRETATION_PHRASE[output.laterality]}; "
            "this output is not for clinical use."
        ),
        limitations=[
            "Synthetic demonstration rather than patient pathology.",
            "Research-only output; not for clinical use.",
        ],
    )


# ---------------------------------------------------------------------------
# Real Siamese banked-B2 product bridge.
#
# ``run_demo_detection`` above is a deterministic GROUND-TRUTH ORACLE dressed
# up as "detection": it measures/returns ``ground_truth_mask.npy`` and a
# hardcoded ``abnormality_score``. The functions below are the REAL,
# checkpoint-driven alternative -- they never read the ground-truth mask for
# anything other than an optional labelled comparison reference, and they
# FAIL LOUD (propagate ``CheckpointError``/``FileNotFoundError``) rather
# than ever silently falling back to the reference oracle.
# ---------------------------------------------------------------------------


#: Default p6-cache validation sample for the Siamese research case.
#:
#: SELECTION RULE (auditable, deliberately NOT cherry-picked): this is the
#: MEDIAN-performance positive sample under the banked B2 checkpoint at
#: threshold 0.5 -- per-sample IoU 0.6552 against a val-set median of 0.6519
#: (78 positives, mean 0.6149). It is representative of typical behaviour.
#:
#: It is explicitly NOT the best-scoring sample (that would be cherry-picking)
#: and explicitly NOT ``sample_0000``, the previous default, which is one of
#: only 3 complete misses in the whole validation set (IoU 0.0 on a 6-pixel
#: lesion) and therefore just as unrepresentative in the opposite direction.
#: The full outcome distribution -- including every miss and the size/blur/
#: uptake strata where delineation degrades -- is reported in the
#: detectability atlas, which is the honest headline, not this single case.
SIAMESE_REPRESENTATIVE_SAMPLE_INDEX = 27


def create_siamese_research_case(
    output_root: Path | str = "outputs/siamese_demo",
    sample_index: int = SIAMESE_REPRESENTATIVE_SAMPLE_INDEX,
    *,
    cache_dir: Path | str = SIAMESE_VAL_CACHE_DIR,
) -> Path:
    """Materialize ONE real p6-cache validation sample into a research case
    directory the real Siamese backend (:func:`run_siamese_detection`) can
    run on: the Siamese input tensors (``left_view``/``right_view``/
    ``pet_diff``/``raw_pet``) plus ``simulation_parameters.json`` and a
    LABELLED REFERENCE mask.

    The reference mask is written as ``reference_lesion_mask.npy``. It is a
    labelled comparison artifact ONLY. It is deliberately *not* named
    ``ground_truth_mask.npy``: that filename is what the honest
    ``deterministic-synthetic-reference`` backend actually SERVES as its
    output, and reusing it here would invite exactly the
    ground-truth-mistaken-for-a-prediction confusion this backend exists to
    remove. :func:`run_siamese_detection` never reads it; the predicted mask
    it writes is a separate file (``predicted_mask.npy``).

    FAILS LOUD: raises :class:`FileNotFoundError` if the requested sample
    (or the cache directory) does not exist -- never fabricates a sample.
    """
    cache_dir = Path(cache_dir)
    sample_path = cache_dir / f"sample_{sample_index:04d}.npz"
    if not sample_path.is_file():
        raise FileNotFoundError(
            f"p6-cache validation sample not found: {sample_path} "
            "(the siamese research case export uses a real, locally "
            "-generated val cache -- it is never fabricated)"
        )

    with np.load(sample_path, allow_pickle=False) as npz:
        left_view = npz["left_view"]
        right_view = npz["right_view"]
        pet_diff = npz["pet_diff"]
        raw_pet = npz["raw_pet"]
        target_mask = npz["target_mask"]
        meta = json.loads(str(npz["meta_json"]))

    case_id = f"siamese_val_sample_{sample_index:04d}"
    case_dir = Path(output_root) / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    np.save(case_dir / "left_view.npy", left_view)
    np.save(case_dir / "right_view.npy", right_view)
    np.save(case_dir / "pet_diff.npy", pet_diff)
    np.save(case_dir / "raw_pet.npy", raw_pet)
    np.save(
        case_dir / "reference_lesion_mask.npy", (target_mask >= 0.5).astype(np.uint8)
    )

    _write_json(
        case_dir / "simulation_parameters.json",
        {
            "case_id": case_id,
            "case_type": "synthetic",
            "source": "p6_cache_val_sample",
            "sample_index": sample_index,
            "sample_npz": str(sample_path),
            "side": meta.get("side"),
            "positive": meta.get("positive"),
            "subject": meta.get("subject"),
            "session": meta.get("session"),
            "research_only": True,
        },
    )
    return case_dir


def _resolve_siamese_checkpoint() -> Path:
    """``VASCUTRACE_CHECKPOINT`` env var, default the banked B2 checkpoint
    (:data:`~src.vascutrace.ml.infer.FROZEN_CHECKPOINT_PATH`). Resolution
    only -- :func:`~src.vascutrace.ml.infer.load_inference_model` is what
    actually fails loud on a missing/corrupt file.
    """
    return Path(os.environ.get("VASCUTRACE_CHECKPOINT", FROZEN_CHECKPOINT_PATH))


def _laterality_from_predicted_mask(mask: np.ndarray) -> str:
    """Left/right/bilateral/none purely from WHERE the predicted mask sits
    in the crop's own (X, Y) = (H, W) pixel grid -- never from the
    ground-truth side label. Axis 0 (H) is the crop's R/L axis
    (``tensor_schema.py``: ``IN_PLANE_HW = (FIXED_CROP_SHAPE[0],
    FIXED_CROP_SHAPE[1])  # (X R/L, Y A/P)``; ``src.vascutrace.geometry``'s
    RAS+ convention: increasing index toward Right). The crop's own
    geometric H-midpoint stands in for the subject's fitted mid-sagittal
    reflection plane, which is not exported into the p6-cache sample --
    a documented APPROXIMATION, not an exact per-sample determination.
    """
    rows, _cols = np.nonzero(mask)
    if rows.size == 0:
        return "none"
    midline = mask.shape[0] / 2.0
    right_present = bool(np.any(rows > midline))
    left_present = bool(np.any(rows < midline))
    if left_present and right_present:
        return "bilateral"
    if right_present:
        return "right"
    if left_present:
        return "left"
    return "bilateral"  # every predicted pixel sits exactly on the midline


def _siamese_metrics_from_prediction(
    raw_pet_slice: np.ndarray,
    predicted_mask: np.ndarray,
    right_view: np.ndarray,
) -> QuantitativeMeasurements:
    """Deterministic quantification from ``raw_pet`` + the PREDICTED mask
    for one 2D center-slice p6-cache sample. Never touches the
    ground-truth mask.

    Real-engine decision (documented, not silent)
    --------------------------------------------------------------------
    ``src.vascutrace.quantification.measure.quantify_target`` (the real 3D
    engine) requires a validated ``GridGeometry`` (shape + affine/voxel
    spacing) and a genuine physical ``reflection_affine`` for a full 3D PET
    volume. The p6-cache validation sample this function is fed is a
    single 2D center slice, and its ``meta_json`` carries no affine or
    voxel-spacing field at all (only subject/session/center_z/positive/
    side/sim_params/schema versions -- confirmed by inspection). Building a
    ``GridGeometry`` here would require fabricating an in-plane (X, Y)
    voxel spacing this function has no basis for, which the same
    never-invent-a-number policy that protects SUV values also forbids.
    This function therefore computes measurements directly from the raw
    SUVbw array and the predicted mask and documents the limitation via
    ``quality_flags`` ("two_dimensional_center_slice_only" and
    "partial_volume_risk") instead.

    Contralateral reference
    --------------------------------------------------------------------
    ``right_view`` is the same crop physically reflected through the
    bundle's fitted mid-sagittal ``reflection_affine`` at cache-build time.
    Its PET channel is network-normalized with clipping at SUV 10. The raw
    contralateral SUV cannot be reconstructed after clipping, so the
    contralateral fields and derived asymmetry are structured nulls.
    """
    predicted_bool = predicted_mask.astype(bool)
    voxel_count = int(predicted_bool.sum())

    quality_flags = ["two_dimensional_center_slice_only", "partial_volume_risk"]

    if voxel_count == 0:
        quality_flags.append("empty_predicted_mask")
        reason = "empty_predicted_mask"
        return QuantitativeMeasurements(
            quality_flags=quality_flags,
            null_reasons={
                "target_suvmax": reason,
                "target_suvmean": reason,
                "contralateral_suvmax": reason,
                "contralateral_suvmean": reason,
                "asymmetry_index": reason,
                "metabolic_volume_ml": reason,
                "longitudinal_extent_mm": reason,
            },
        )

    target_values = raw_pet_slice[predicted_bool]
    null_reasons: dict[str, str] = {}
    if np.all(np.isfinite(target_values)):
        target_suvmax = float(target_values.max())
        target_suvmean = float(target_values.mean())
    else:
        target_suvmax = None
        target_suvmean = None
        quality_flags.append("nonfinite_target_suv")
        null_reasons.update(
            {
                "target_suvmax": "nonfinite_target_suv",
                "target_suvmean": "nonfinite_target_suv",
            }
        )

    # ``right_view`` is retained in the function signature because it records
    # the available bilateral input. Its PET channel is clipped at SUV 10, so
    # it cannot support a raw contralateral SUV measurement.
    _ = right_view

    # The cached reflected view is clipped at SUV 10 and lacks the affine and
    # in-plane voxel spacing needed by the 3D quantifier. Returning a number
    # here would mislabel a clipped or pixel-count proxy as a physical value.
    quality_flags.extend(
        [
            "raw_contralateral_suv_unavailable_from_2d_cache",
            "physical_volume_unavailable_missing_in_plane_spacing",
            "longitudinal_extent_unavailable_2d_slice",
        ]
    )
    null_reasons.update(
        {
            "contralateral_suvmax": "raw_contralateral_suv_unavailable_from_2d_cache",
            "contralateral_suvmean": "raw_contralateral_suv_unavailable_from_2d_cache",
            "asymmetry_index": "raw_contralateral_suv_unavailable_from_2d_cache",
            "metabolic_volume_ml": "physical_volume_unavailable_missing_in_plane_spacing",
            "longitudinal_extent_mm": "longitudinal_extent_unavailable_2d_slice",
        }
    )

    return QuantitativeMeasurements(
        target_suvmax=target_suvmax,
        target_suvmean=target_suvmean,
        quality_flags=quality_flags,
        null_reasons=null_reasons,
    )


def _write_siamese_overlay(
    case_dir: Path, raw_pet_slice: np.ndarray, predicted_mask: np.ndarray
) -> Path:
    """PET-only overlay PNG for one 2D siamese-case center slice, reusing
    the same ``_to_gray``/``_write_rgb_png`` helpers ``create_demo_views``
    uses -- no second PNG encoder.
    """
    pet_gray = _to_gray(raw_pet_slice)
    overlay = np.repeat(pet_gray[..., None], 3, axis=2)
    overlay[predicted_mask.astype(bool)] = np.array([255, 40, 40], dtype=np.uint8)
    path = case_dir / "overlay.png"
    _write_rgb_png(path, overlay)
    return path


def run_siamese_detection(case_dir: Path | str) -> ModelOutput:
    """Run the banked B2 Siamese checkpoint through the frozen inference
    package on one materialized
    :func:`create_siamese_research_case` case directory.

    FAILS LOUD: a missing/corrupt ``VASCUTRACE_CHECKPOINT`` (or the frozen
    default) raises :class:`~src.vascutrace.ml.checkpoint.CheckpointError`
    (propagated, unmodified, from
    :func:`~src.vascutrace.ml.infer.load_inference_model`); a missing case
    input array raises :class:`FileNotFoundError`. Never falls back to
    :func:`run_demo_detection`'s ground-truth oracle.

    Operating point, exploratory only: the hard-mask score threshold and
    minimum predicted-component size are resolved via
    :func:`~src.vascutrace.ml.infer.resolve_operating_point`, which reads
    the optional ``VASCUTRACE_SCORE_THRESHOLD``/
    ``VASCUTRACE_MIN_COMPONENT_SIZE`` env vars and otherwise returns the
    frozen banked ``(0.5, 0)`` operating point unchanged -- see that
    function's docstring for the full fail-loud-on-invalid-env contract.
    This is the ONLY thing threading those env vars affects: ``model_name``
    always stays the checkpoint's own real identity
    (``"siamese_p4b2_deepsup"``), never suffixed or polluted with the
    resolved threshold/size (module docstring's frozen contract; never a
    "recommended_low_fp"/"strict" preset name). The resolved operating
    point is recorded, unmodified, in ``case_dir / "operating_point.json"``
    for provenance.
    """
    started = time.perf_counter()
    case_dir = Path(case_dir)

    checkpoint_path = _resolve_siamese_checkpoint()
    model, metadata = load_inference_model(checkpoint_path, device="cpu")

    def _load(name: str) -> np.ndarray:
        path = case_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"siamese case input not found: {path}")
        return np.load(path)

    left_view = _load("left_view.npy")
    right_view = _load("right_view.npy")
    pet_diff = _load("pet_diff.npy")
    raw_pet = _load("raw_pet.npy")  # [1, H, W], unnormalized SUVbw

    score = predict_abnormality_score(
        model,
        torch.from_numpy(left_view).float(),
        torch.from_numpy(right_view).float(),
        torch.from_numpy(pet_diff).float(),
        "cpu",
    )
    score_threshold, min_component_size = resolve_operating_point()
    mask = predict_mask(
        score, threshold=score_threshold, min_component_size=min_component_size
    )
    _write_json(
        case_dir / "operating_point.json",
        {
            "score_threshold": score_threshold,
            "min_component_size": min_component_size,
            "exploratory": bool(score_threshold != 0.5 or min_component_size != 0),
        },
    )

    mask_path = case_dir / "predicted_mask.npy"
    np.save(mask_path, mask)

    metrics = _siamese_metrics_from_prediction(raw_pet[0], mask, right_view)
    metrics_path = case_dir / "metrics.json"
    _write_json(metrics_path, metrics.model_dump(mode="json"))

    overlay_path = _write_siamese_overlay(case_dir, raw_pet[0], mask)

    result = ModelOutput(
        case_id=case_dir.name,
        model_name=metadata.model_name,
        model_version=f"cfg-{metadata.config_hash}+ep{metadata.epoch}",
        laterality=_laterality_from_predicted_mask(mask),
        abnormality_score=float(score.max()) if score.size else 0.0,
        mask_path=mask_path,
        metrics_path=metrics_path,
        overlay_path=overlay_path,
        runtime_seconds=time.perf_counter() - started,
    )
    _write_json(case_dir / "model_output.json", result.model_dump(mode="json"))
    return result
