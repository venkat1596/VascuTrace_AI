"""Thin argparse CLI for the P6 training loop: ``doctor`` / ``dry-run`` /
``train`` / ``resume``.

RESEARCH_PROTOTYPE_WARNING
---------------------------------------------------------------------------
Research prototype. Trained and evaluated using simulated vascular-like
abnormalities, not confirmed human post-angioplasty lesions.
---------------------------------------------------------------------------

This module intentionally contains no training-loop logic of its own --
every subcommand parses arguments/config, resolves bundle directories, and
delegates to :mod:`src.vascutrace.ml.train` (:func:`~src.vascutrace.ml.
train.train` / :func:`~src.vascutrace.ml.train.resume`). See ``train.py``
and ``checkpoint.py`` for the algorithmic/design rationale. Run via
``uv run python -m src.vascutrace.ml.cli <command> ...``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from src.vascutrace.data.contract import (
    CROP_SCHEMA_VERSION,
    FIXED_CROP_SHAPE,
    ILIAC_LABEL_LEFT,
    ILIAC_LABEL_RIGHT,
    make_crop_bundle,
    save_crop_bundle,
)
from src.vascutrace.data.crops import build_reflection_affine
from src.vascutrace.data.split import (
    SPLIT_SEED,
    load_subject_sex_table,
    load_subject_split,
)
from src.vascutrace.data.split import (
    stratified_subject_split as _stratified_subject_split,
)
from src.vascutrace.geometry import RESEARCH_PROTOTYPE_WARNING, GeometrySidecar
from src.vascutrace.ml.cache import CachePrepError, precompute_synthetic_cache
from src.vascutrace.ml.checkpoint import CHECKPOINT_SCHEMA_VERSION, load_checkpoint
from src.vascutrace.ml.dataset import DatasetConfig
from src.vascutrace.ml.model import ModelConfig
from src.vascutrace.ml.tensor_schema import TENSOR_SCHEMA_VERSION
from src.vascutrace.ml.train import (
    CheckpointCompatibilityError,
    CudaOutOfMemoryError,
    CudaUnavailableError,
    NonFiniteLossError,
    TrainConfig,
    TrainConfigError,
    discover_bundle_dirs,
    resume as resume_run,
    train as train_run,
)

__all__ = ["build_parser", "main"]

_DEFAULT_DATA_ROOT = Path("data/processed/p2/crops/p2-crop-v2")

_TRAIN_CONFIG_SPECIAL_KEYS = {
    "data_root",
    "split_path",
    "train_bundle_dirs",
    "val_bundle_dirs",
    "model_config",
    "dataset_config",
    "run_root",
}


# ---------------------------------------------------------------------------
# Config-file -> TrainConfig
# ---------------------------------------------------------------------------


def _load_config_file(path: Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() in (".yaml", ".yml"):
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise TrainConfigError(f"config file {path} must contain a mapping/object")
    return payload


def _model_config_from_dict(payload: dict[str, Any]) -> ModelConfig:
    payload = dict(payload)
    if "channel_mult" in payload:
        payload["channel_mult"] = tuple(payload["channel_mult"])
    return ModelConfig(**payload)


def _dataset_config_from_dict(payload: dict[str, Any]) -> DatasetConfig:
    payload = dict(payload)
    for key in ("radius_mm_range", "uptake_multiplier_range", "blur_fwhm_mm_range"):
        if key in payload:
            payload[key] = tuple(payload[key])
    return DatasetConfig(**payload)


def _bundle_dirs_for_subjects(
    data_root: Path, subjects: Sequence[str]
) -> tuple[Path, ...]:
    wanted = set(subjects)
    return tuple(d for d in discover_bundle_dirs(data_root) if d.parent.name in wanted)


def _resolve_bundle_dirs(
    payload: dict[str, Any],
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Either an explicit ``train_bundle_dirs``/``val_bundle_dirs`` pair, or
    ``data_root`` + ``split_path`` (a :func:`~src.vascutrace.data.split.
    save_subject_split` JSON file) resolved to bundle directories via
    :func:`~src.vascutrace.ml.train.discover_bundle_dirs`. A cache-only
    config (``train_cache_dir``/``val_cache_dir`` set, no on-the-fly
    bundle-dir source given) resolves to two empty tuples -- valid per
    ``TrainConfig.__post_init__``'s cache-mode relaxation.
    """
    if "train_bundle_dirs" in payload or "val_bundle_dirs" in payload:
        train_dirs = tuple(Path(p) for p in payload.get("train_bundle_dirs", []))
        val_dirs = tuple(Path(p) for p in payload.get("val_bundle_dirs", []))
        return train_dirs, val_dirs

    if "split_path" not in payload:
        if "train_cache_dir" in payload and "val_cache_dir" in payload:
            return (), ()
        raise TrainConfigError(
            "config must supply either explicit train_bundle_dirs/"
            "val_bundle_dirs, data_root + split_path, or "
            "train_cache_dir + val_cache_dir (cache-only mode)"
        )
    data_root = Path(payload.get("data_root", _DEFAULT_DATA_ROOT))
    split = load_subject_split(Path(payload["split_path"]))
    train_dirs = _bundle_dirs_for_subjects(data_root, split.train)
    val_dirs = _bundle_dirs_for_subjects(data_root, split.val)
    return train_dirs, val_dirs


def _train_config_from_dict(payload: dict[str, Any], *, run_root: Path) -> TrainConfig:
    train_dirs, val_dirs = _resolve_bundle_dirs(payload)
    model_cfg = _model_config_from_dict(payload.get("model_config", {}))
    dataset_cfg = _dataset_config_from_dict(payload.get("dataset_config", {}))
    extra = {
        key: value
        for key, value in payload.items()
        if key not in _TRAIN_CONFIG_SPECIAL_KEYS
    }
    return TrainConfig(
        train_bundle_dirs=train_dirs,
        val_bundle_dirs=val_dirs,
        run_root=run_root,
        model_config=model_cfg,
        dataset_config=dataset_cfg,
        **extra,
    )


# ---------------------------------------------------------------------------
# ``doctor``
# ---------------------------------------------------------------------------


def cmd_doctor(args: argparse.Namespace) -> int:
    lines: list[str] = []
    lines.append(f"torch version: {torch.__version__}")

    cuda_available = torch.cuda.is_available()
    lines.append(f"CUDA available: {cuda_available}")
    if cuda_available:
        try:
            idx = torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            total_gib = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
            lines.append(f"CUDA device: {name}")
            lines.append(f"CUDA VRAM: {total_gib:.1f} GiB")
        except RuntimeError as exc:
            lines.append(f"CUDA device probe failed: {exc}")
            cuda_available = False
    else:
        lines.append("CUDA device: none")

    data_root = Path(args.data_root)
    bundle_dirs = discover_bundle_dirs(data_root)
    lines.append(f"data root: {data_root}")
    lines.append(f"crop bundles found: {len(bundle_dirs)}")
    lines.append(f"tensor_schema_version: {TENSOR_SCHEMA_VERSION}")
    lines.append(f"crop_schema_version: {CROP_SCHEMA_VERSION}")
    lines.append(f"checkpoint_schema_version: {CHECKPOINT_SCHEMA_VERSION}")
    lines.append(RESEARCH_PROTOTYPE_WARNING)

    trainable = True
    if args.require_cuda and not cuda_available:
        lines.append("FAIL: --require-cuda was set but CUDA is unavailable")
        trainable = False
    if args.require_data and len(bundle_dirs) == 0:
        lines.append(
            f"FAIL: --require-data was set but no crop bundles were found under {data_root}"
        )
        trainable = False

    lines.append(f"trainable: {trainable}")
    print("\n".join(lines))
    return 0 if trainable else 1


# ---------------------------------------------------------------------------
# ``dry-run``
# ---------------------------------------------------------------------------


def _build_synthetic_bundle_dir(output_root: Path, *, subject: str, seed: int) -> Path:
    """One tiny synthetic :class:`~src.vascutrace.data.contract.CropBundle`
    for ``dry-run`` when no real crop bundles are available under
    ``--data-root``. Mirrors the fixture pattern already established in
    ``tests/test_ml_dataset.py``'s ``_standard_bundle`` (a small, compact
    iliac-label mask so the local-simulation-window stays fast -- see
    ``dataset.py``'s module docstring, item 5). ``subject``/``seed`` let the
    caller build two *distinct* bundles (train vs val) -- reusing one
    bundle directory for both would fail :class:`TrainConfig`'s own
    leakage-safety check.
    """
    rng = np.random.default_rng(seed)
    pet = (rng.random(FIXED_CROP_SHAPE, dtype=np.float32) * 8.0 + 1.0).astype(
        np.float32
    )
    ct = (rng.random(FIXED_CROP_SHAPE, dtype=np.float32) * 200.0 - 100.0).astype(
        np.float32
    )
    valid = np.ones(FIXED_CROP_SHAPE, dtype=np.uint8)
    iliac = np.zeros(FIXED_CROP_SHAPE, dtype=np.uint8)
    iliac[60:64, 35:45, 50:90] = ILIAC_LABEL_LEFT
    iliac[80:84, 35:45, 50:90] = ILIAC_LABEL_RIGHT

    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    reflection = build_reflection_affine(
        np.array([1.0, 0.0, 0.0]), np.array([144.0, 0.0, 0.0])
    )
    sidecar = GeometrySidecar(
        canonical_shape=(440, 440, 531),
        original_shape=(440, 440, 531),
        canonical_affine_sha256="a" * 64,
        original_affine_sha256="b" * 64,
        original_voxel_from_canonical_voxel_sha256="c" * 64,
    )
    bundle = make_crop_bundle(
        subject=subject,
        session="Test",
        pet_suvbw=pet,
        ct_hu=ct,
        valid_pet_mask=valid,
        iliac_label_mask=iliac,
        reflection_affine=reflection,
        crop_to_pet_canonical_affine=affine,
        crop_origin_voxel=(0, 0, 0),
        original_voxel_from_pet_canonical_voxel=np.eye(4),
        geometry_sidecar=sidecar,
        reflection_residual_mm=1.0,
        reflection_qc_flag=False,
        bbox_exceeds_fixed_crop=(False, False, False),
        crop_margin_mm=15.0,
        pet_spacing_mm=(2.0, 2.0, 2.0),
        paired_point_count=1,
    )
    return save_crop_bundle(bundle, output_root)


def cmd_dry_run(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Train and val must be two DISTINCT bundle directories -- TrainConfig's
    # own leakage-safety check (see train.py's module docstring, item 8)
    # rejects any overlap, so a dry-run reusing one bundle for both would
    # fail exactly the same guard a real run would.
    data_root = Path(args.data_root)
    real_bundle_dirs = discover_bundle_dirs(data_root)
    if len(real_bundle_dirs) >= 2:
        train_dirs = real_bundle_dirs[:1]
        val_dirs = real_bundle_dirs[1:2]
        source = (
            f"2 distinct real crop bundles under {data_root} "
            f"({len(real_bundle_dirs)} available)"
        )
    else:
        synth_root = output_root / "synthetic_bundles"
        train_dirs = (
            _build_synthetic_bundle_dir(
                synth_root, subject="SYNTHETIC_DRYRUN_TRAIN", seed=0
            ),
        )
        val_dirs = (
            _build_synthetic_bundle_dir(
                synth_root, subject="SYNTHETIC_DRYRUN_VAL", seed=1
            ),
        )
        source = "2 distinct synthetic in-memory bundles (fewer than 2 real crop bundles found)"

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    config = TrainConfig(
        train_bundle_dirs=train_dirs,
        val_bundle_dirs=val_dirs,
        run_root=output_root / "dry_run",
        seed=0,
        val_seed=0,
        batch_size=1,
        max_epochs=1,
        lr=1e-3,
        weight_decay=0.0,
        early_stop_patience=5,
        device=device,
        amp=False,
        num_workers=0,
        model_config=ModelConfig(base_channels=4, seed=0),
        dataset_config=DatasetConfig(samples_per_bundle=4),
        limit_train_batches=4,
        limit_val_batches=1,
        log_every_n_steps=1,
    )

    print(f"dry-run: device={device}, data source={source}")
    result = train_run(config)
    print(
        f"dry-run: ran epoch={result.final_epoch} global_step={result.global_step} "
        f"val_dice={result.best_val_metric}"
    )
    print(f"dry-run: wrote checkpoint {result.last_checkpoint_path}")

    reloaded = load_checkpoint(result.last_checkpoint_path)
    print(
        f"dry-run: reloaded checkpoint OK (epoch={reloaded.epoch}, "
        f"global_step={reloaded.global_step}, calibration_status="
        f"{reloaded.calibration_status!r})"
    )
    print(f"dry-run: {RESEARCH_PROTOTYPE_WARNING}")
    return 0


# ---------------------------------------------------------------------------
# ``train`` / ``resume``
# ---------------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> int:
    payload = _load_config_file(Path(args.config))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    config = _train_config_from_dict(payload, run_root=output_root)

    result = train_run(config)
    print(
        f"train: final_epoch={result.final_epoch} global_step={result.global_step} "
        f"best_val_metric={result.best_val_metric} stopped_early={result.stopped_early}"
    )
    print(f"train: last checkpoint = {result.last_checkpoint_path}")
    if result.best_checkpoint_path is not None:
        print(f"train: best checkpoint = {result.best_checkpoint_path}")
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    run_root = Path(args.run)
    if args.config is None:
        raise TrainConfigError(
            "resume requires --config (the same config file the original "
            "train run used) -- a checkpoint never stores its raw bundle "
            "directory list, only a hash of it; see train.py's resume() "
            "docstring."
        )
    payload = _load_config_file(Path(args.config))
    config = _train_config_from_dict(payload, run_root=run_root)

    result = resume_run(run_root, config)
    print(
        f"resume: final_epoch={result.final_epoch} global_step={result.global_step} "
        f"best_val_metric={result.best_val_metric} stopped_early={result.stopped_early}"
    )
    return 0


# ---------------------------------------------------------------------------
# ``prepare-synthetic``
# ---------------------------------------------------------------------------

_DEMOGRAPHICS_FILENAME = "Demographics (All).xlsx"  # matches scripts/run_p2_pipeline.py


def cmd_prepare_synthetic(args: argparse.Namespace) -> int:
    """Precompute a synthetic-sample cache for one split ("train" or
    "val") -- see ``cache.py``'s module docstring for the full design.
    Only that split's subjects are ever touched (leakage-safe against both
    the other split AND the held-out "test" split, which this subcommand
    never reads). Prints AGGREGATE counts only -- never a subject/bundle
    identity string.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    data_root = Path(args.data_root)
    bundles_root = Path(args.bundles_root)
    output_root = Path(args.output_root)

    demographics_path = data_root / _DEMOGRAPHICS_FILENAME
    subject_sex = load_subject_sex_table(demographics_path)
    split = _stratified_subject_split(subject_sex, seed=args.seed)

    subjects = split.train if args.split == "train" else split.val
    bundle_dirs = _bundle_dirs_for_subjects(bundles_root, subjects)
    if not bundle_dirs:
        print(
            f"error: no bundle directories found for split={args.split!r} "
            f"under {bundles_root}",
            file=sys.stderr,
        )
        return 1

    manifest = precompute_synthetic_cache(
        bundle_dirs,
        output_root,
        n_positive_per_bundle=args.n_positive,
        n_negative_per_bundle=args.n_negative,
        seed=args.seed,
        num_workers=args.num_workers,
        exclude_qc_flagged=args.exclude_qc_flagged,
    )

    print(f"prepare-synthetic: split={args.split!r}")
    print(f"prepare-synthetic: subjects_in_split={len(subjects)}")
    print(f"prepare-synthetic: bundles_used={len(bundle_dirs)}")
    print(
        f"prepare-synthetic: bundles_excluded_qc_flagged="
        f"{len(manifest.excluded_qc_bundle_identities)}"
    )
    print(
        f"prepare-synthetic: total_samples={manifest.total_samples} "
        f"(positive={manifest.total_positive}, negative={manifest.total_negative})"
    )
    print(f"prepare-synthetic: wrote cache to {output_root}")
    print(f"prepare-synthetic: {RESEARCH_PROTOTYPE_WARNING}")
    return 0


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vascutrace-ml-train",
        description="VascuTrace P6 training loop CLI (doctor/dry-run/train/resume).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser(
        "doctor", help="Report torch/CUDA/data readiness for training"
    )
    doctor_parser.add_argument("--data-root", default=str(_DEFAULT_DATA_ROOT))
    doctor_parser.add_argument("--require-cuda", action="store_true")
    doctor_parser.add_argument("--require-data", action="store_true")
    doctor_parser.set_defaults(func=cmd_doctor)

    dry_run_parser = subparsers.add_parser(
        "dry-run", help="Tiny CPU-or-GPU end-to-end training smoke test"
    )
    dry_run_parser.add_argument("--output-root", required=True)
    dry_run_parser.add_argument("--data-root", default=str(_DEFAULT_DATA_ROOT))
    dry_run_parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda", "auto"]
    )
    dry_run_parser.set_defaults(func=cmd_dry_run)

    train_parser = subparsers.add_parser(
        "train", help="Full training run from a YAML/JSON config file"
    )
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--output-root", required=True)
    train_parser.set_defaults(func=cmd_train)

    resume_parser = subparsers.add_parser("resume", help="Resume a training run")
    resume_parser.add_argument("--run", required=True)
    resume_parser.add_argument(
        "--config", default=None, help="Same config file the original run used"
    )
    resume_parser.set_defaults(func=cmd_resume)

    prepare_synthetic_parser = subparsers.add_parser(
        "prepare-synthetic",
        help="Precompute a synthetic-sample cache for one split (train/val)",
    )
    prepare_synthetic_parser.add_argument(
        "--split", required=True, choices=["train", "val"]
    )
    prepare_synthetic_parser.add_argument(
        "--data-root",
        default="Data/QUADRA_HC",
        help="Root containing the demographics workbook (for split reconstruction)",
    )
    prepare_synthetic_parser.add_argument(
        "--bundles-root", default=str(_DEFAULT_DATA_ROOT)
    )
    prepare_synthetic_parser.add_argument("--output-root", required=True)
    prepare_synthetic_parser.add_argument("--n-positive", type=int, required=True)
    prepare_synthetic_parser.add_argument("--n-negative", type=int, required=True)
    prepare_synthetic_parser.add_argument("--num-workers", type=int, default=1)
    prepare_synthetic_parser.add_argument("--exclude-qc-flagged", action="store_true")
    prepare_synthetic_parser.add_argument(
        "--seed",
        type=int,
        default=SPLIT_SEED,
        help="Both the subject-split seed and the cache's own sample seed",
    )
    prepare_synthetic_parser.set_defaults(func=cmd_prepare_synthetic)

    return parser


_TYPED_ERRORS = (
    TrainConfigError,
    CudaUnavailableError,
    NonFiniteLossError,
    CudaOutOfMemoryError,
    CheckpointCompatibilityError,
    CachePrepError,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except _TYPED_ERRORS as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
