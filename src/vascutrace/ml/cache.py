"""Precomputed synthetic-sample cache for the P6 training loop.

RESEARCH_PROTOTYPE_WARNING
---------------------------------------------------------------------------
Research prototype. Trained and evaluated using simulated vascular-like
abnormalities, not confirmed human post-angioplasty lesions.
---------------------------------------------------------------------------

Implementation notes
============================================================================
This module exists because on-the-fly training became infeasible once
``dataset.py``'s own ``DatasetConfig.supersample`` was corrected from ``1``
to the accuracy-mandated ``5`` (dataset.py's own module docstring, item 6/
``supersample`` field comment): a positive sample now costs P3's own
lesion-simulation time directly (measured tens of seconds per sample on a
real bundle's ~73-point centerline, dominated by P3's occupancy computation,
not by anything in this module or ``dataset.py``'s own reflect/slice/
normalize path). ``precompute_synthetic_cache`` pays that cost ONCE, ahead
of time, in parallel across many CPU cores, and writes each resulting
:class:`~src.vascutrace.ml.dataset.Sample` to its own ``.npz`` file;
:class:`CachedSampleDataset` then loads a sample back in low-single-digit
milliseconds -- a plain array read, no reflection/simulation work at all --
so a training loop pointed at a cache directory runs at ordinary
data-loading speed regardless of the underlying synthetic-lesion cost.

1. "Do NOT re-implement placement" -- how index resolution is guaranteed
   drift-proof against ``dataset.py``
   ------------------------------------------------------------------------
   ``dataset.py``'s own module docstring, item 6, documents a real bug it
   found and fixed in its own positive-``center_z`` candidate selection (an
   over-estimated core span let some "positive" samples land in a genuinely
   empty-target overshoot band, ~13-17% of the time, before the fix). This
   module must reuse ``SiameseCropDataset``'s *exact* current
   index -> ``(bundle_dir, center_z, seed, positive, side)`` resolution --
   not a hand-copied re-derivation of the same arithmetic, which would
   silently drift out of sync with any future fix there (exactly the class
   of bug ``dataset.py``'s own item 6 describes). :func:`_resolve_index`
   therefore imports and calls ``dataset.py``'s own private
   ``_combine_seeds`` (the per-sample seed derivation) and
   ``_positive_center_z_candidates`` (the fixed candidate-selection
   function) directly -- ``dataset.py`` itself is never modified, only
   imported from, matching this implementation's "own NEW file `cache.py`... import
   from it read-only" instruction taken at its strictest: even the private
   resolution helpers are reused rather than re-derived. ``build_sample``
   itself -- the function that actually performs lesion placement -- is
   also called completely unmodified, with no private ``_precompute``
   override (that parameter is dataset.py's own internal optimization, not
   part of its public contract; see its docstring). ``TestCacheDatasetEquivalence``
   proves, directly, that a cached sample and a freshly-built
   ``SiameseCropDataset[i]`` sample for the same index are bit-identical.

2. Exact per-bundle counts via two extreme-``positive_fraction`` "virtual"
   datasets, not the probabilistic single-stream enumeration
   ------------------------------------------------------------------------
   ``SiameseCropDataset``'s own ``positive_fraction`` parameter makes
   whether a given index is positive a probabilistic draw
   (``rng.random() < positive_fraction``), appropriate for on-the-fly
   training but not for this module's contract of an *exact*
   ``n_positive_per_bundle``/``n_negative_per_bundle`` count. This module
   resolves that by constructing two ``SiameseCropDataset`` instances over
   the SAME ``bundle_dirs`` -- one at ``positive_fraction=1.0`` (every
   index is positive, since ``rng.random()`` is strictly ``< 1.0`` by
   construction: ``numpy.random.Generator.random`` draws from
   ``[0.0, 1.0)``) with ``samples_per_bundle=n_positive_per_bundle``, one
   at ``positive_fraction=0.0`` with ``samples_per_bundle=
   n_negative_per_bundle`` -- rather than inventing a new selection
   policy. Both are legitimate, already-supported uses of
   ``SiameseCropDataset``'s existing public constructor (no new API
   surface added to that class), so "reuse the dataset's own enumeration"
   is satisfied literally rather than approximately.

3. One task = one sample, picklable args only
   ------------------------------------------------------------------------
   :func:`precompute_synthetic_cache` resolves every sample's
   ``(bundle_dir, center_z, seed, positive, side)`` in the *main* process
   (cheap: no lesion simulation happens during resolution, only
   ``_positive_center_z_candidates``'s own centerline-derivation work),
   then dispatches one ``concurrent.futures.ProcessPoolExecutor`` task per
   sample, each calling ``build_sample`` with exactly those five
   arguments plus the shared ``DatasetConfig`` -- five plain values (a
   ``Path``, an ``int``, an ``int``, a ``bool``, an ``str | None``) plus one
   frozen dataclass of plain fields, all standard-picklable via the
   default ``pickle`` protocol
   (https://docs.python.org/3/library/pickle.html#pickle-picklable, "types
   ... that ... can be pickled": functions/classes accessible by
   qualified name, and objects whose ``__dict__``/fields are themselves
   picklable) -- no ``CropBundle`` (large NumPy arrays; reloaded fresh
   inside the worker via ``load_crop_bundle`` instead) and no
   ``SiameseCropDataset`` instance crosses the process boundary. This
   keeps every task an independent, retriable, individually-loggable unit
   of work -- a worker crash on one sample never corrupts another's
   already-written ``.npz`` file (see item 4). A small worker-local
   ``functools.lru_cache`` around ``load_crop_bundle`` (module-private,
   :func:`_load_bundle_cached`) is a legitimate additive throughput
   optimization beyond the literal per-task contract: it does not change
   which arguments cross the process boundary, only avoids re-reading the
   same bundle's ``.npz`` from disk when one worker process happens to be
   assigned several samples from the same bundle across its lifetime
   (``ProcessPoolExecutor``'s worker processes are long-lived, each
   handling many submitted tasks, not one process per task).

4. Atomic-ish per-sample writes
   ------------------------------------------------------------------------
   :func:`_save_sample_npz` follows the exact write-temp/fsync/
   ``os.replace`` pattern this project's ``checkpoint.py`` already
   establishes (see that module's docstring, item 1, for the full
   POSIX-atomicity argument) -- a temp file in the SAME directory as the
   final ``sample_XXXX.npz``, ``os.fsync``ed, then atomically renamed. A
   crash mid-write during a many-hour, many-worker precompute run can
   therefore never leave a truncated/corrupt ``sample_XXXX.npz`` for
   :class:`CachedSampleDataset` to load; at worst, a killed worker leaves
   that one sample's final file simply absent (re-running
   ``precompute_synthetic_cache`` with identical arguments regenerates it
   deterministically -- see item 5).

5. Determinism
   ------------------------------------------------------------------------
   Every random draw this module performs is either delegated entirely to
   ``dataset.py``'s own deterministic machinery (index resolution, lesion
   parameters, P3's own internal seed) or is itself a pure function of
   plain integers (this module introduces no new source of randomness of
   its own) -- so ``precompute_synthetic_cache(bundle_dirs, out_dir,
   n_positive_per_bundle=P, n_negative_per_bundle=N, seed=S, config=C)``
   called twice with identical arguments (against the same on-disk
   bundles) writes bit-identical ``.npz`` files both times, matching this
   implementation's "same args => same cache" requirement (``TestCacheDeterminism``).
   Task EXECUTION order across workers is not deterministic (OS
   scheduling), but each task's OWN output file name (``sample_XXXX.npz``,
   ``XXXX`` = a resolution-time-assigned, order-independent index) and
   content are -- so the finished cache directory's contents are
   deterministic even though the wall-clock order samples are written in
   is not.

6. The manifest's leakage-safety role
   ------------------------------------------------------------------------
   :class:`CacheManifest`'s ``bundle_identities`` (the sorted set of
   ``"<subject>/<session>"`` strings the cache was built from -- the same
   local-artifact convention already established by
   ``src.vascutrace.data.split.save_subject_split``'s own docstring:
   "Contains subject identifiers by design (a local, gitignored artifact)"
   -- ``out_dir`` is caller-supplied and MUST be a gitignored root, exactly
   like a checkpoint run-root) is what
   ``train.py``'s ``TrainConfig`` reads to prove a train cache and a val
   cache were built from disjoint bundles BEFORE any training allocation
   happens -- see ``train.py``'s module docstring for the wiring. This
   module never prints or logs a bundle identity string itself (only
   aggregate counts); the CLI's ``prepare-synthetic`` subcommand likewise
   only prints aggregate counts.

7. ``source_fraction`` propagation -- backward-compatible by construction,
   NOT a schema-version bump (soft-target experiment)
   ------------------------------------------------------------------------
   ``dataset.py``'s own module docstring, item 9, adds a new
   ``Sample.source_fraction`` field. This module writes it into every
   NEWLY built cache's ``.npz`` files (:func:`_save_sample_npz`) and sets
   the new ``CacheManifest.has_source_fraction = True`` on every cache it
   writes from this point forward. Deliberately does **not** bump
   :data:`CACHE_SCHEMA_VERSION`: reproducibility requires
   ``data/processed/p6_cache_big`` --
   and therefore the already-running/already-completed
   v4_big/v5exp/v6exp training runs that read it -- to remain completely
   undisturbed. Bumping ``CACHE_SCHEMA_VERSION`` would make
   :func:`read_cache_manifest`'s own strict-equality check reject
   ``p6_cache_big``'s still-``"p6-cache-v1"`` manifest outright, breaking
   that reproducibility guarantee for no benefit (the ``.npz`` ARRAY
   layout itself is additive/backward-compatible: an old reader that
   never asks for the ``source_fraction`` key is unaffected by its
   presence, and this module's own :func:`_load_sample_npz` tolerates its
   ABSENCE too -- see below). :func:`_load_sample_npz` therefore reads
   ``source_fraction`` if the key is present and falls back to an
   all-zero array (same shape/dtype as ``target_mask``) if it is absent
   (a pre-item-7 cache, e.g. ``p6_cache``/``p6_cache_big`` as they exist
   today). This fallback is safe -- NOT the "silently zero-fill a missing
   field" anti-pattern this project otherwise avoids -- only because
   ``train.py``'s ``TrainConfig.__post_init__`` refuses to start any
   ``soft_target=True`` run whose ``train_cache_dir`` manifest does not
   itself declare ``has_source_fraction=True`` (see :func:`cache_has_
   source_fraction`, and ``train.py``'s own module docstring): a
   ``soft_target=True`` run can therefore never silently receive these
   zero-filled placeholders, while every existing ``soft_target=False``
   (the default) training path -- which never reads ``source_fraction``
   at all -- is completely unaffected by whether the field is present,
   absent, or zero-filled.
============================================================================
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.vascutrace.data.contract import (
    CROP_SCHEMA_VERSION,
    CropBundle,
    load_crop_bundle,
)
from src.vascutrace.geometry import RESEARCH_PROTOTYPE_WARNING
from src.vascutrace.ml.dataset import (
    DATASET_BUILDER_VERSION,
    DatasetConfig,
    Sample,
    SiameseCropDataset,
    build_sample,
)
from src.vascutrace.ml.dataset import (
    _combine_seeds,  # noqa: PLC2701 -- deliberate, documented read-only reuse; see module docstring item 1
)
from src.vascutrace.ml.dataset import (
    _positive_center_z_candidates,  # noqa: PLC2701 -- same rationale
)
from src.vascutrace.ml.tensor_schema import (
    FIRST_CENTER_Z,
    LEFT_VIEW_SHAPE,
    LAST_CENTER_Z,
    PET_DIFF_SHAPE,
    RIGHT_VIEW_SHAPE,
    TARGET_SHAPE,
    TENSOR_SCHEMA_VERSION,
)

__all__ = [
    "RESEARCH_PROTOTYPE_WARNING",
    "CACHE_SCHEMA_VERSION",
    "CacheManifest",
    "CachePrepError",
    "CacheSchemaError",
    "precompute_synthetic_cache",
    "CachedSampleDataset",
    "read_cache_manifest",
    "cache_bundle_identities",
    "cache_has_source_fraction",
]

CACHE_SCHEMA_VERSION = "p6-cache-v1"

logger = logging.getLogger(__name__)

_SEED_UPPER_BOUND = (
    2**31 - 1
)  # matches dataset.py's own constant (small shared primitive)


class CachePrepError(ValueError):
    """Raised for invalid :func:`precompute_synthetic_cache` arguments."""


class CacheSchemaError(RuntimeError):
    """Raised when a cache directory's ``manifest.json`` is missing,
    unreadable, or schema/version-incompatible with the code reading it.
    """


# ---------------------------------------------------------------------------
# Small shared primitives (deliberately duplicated, not imported from
# train.py -- train.py imports FROM this module, so the reverse would be
# circular; matches this project's small-shared-primitive-duplication
# convention, e.g. dataset.py's own ``_apply_affine_points`` docstring).
# ---------------------------------------------------------------------------


def _bundle_identity(bundle_dir: Path) -> str:
    """The trailing ``<subject>/<session>`` path components only (matches
    ``data.contract.bundle_directory``'s own layout) -- NOT the full
    filesystem path, which could vary across machines.
    """
    parts = Path(bundle_dir).parts
    return "/".join(parts[-2:]) if len(parts) >= 2 else str(bundle_dir)


def _hash_repr(payload: Any) -> str:
    return sha256(repr(payload).encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Index resolution -- see module docstring, items 1-2.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ResolvedSpec:
    bundle_dir: Path
    center_z: int
    seed: int
    positive: bool
    side: str | None


def _resolve_index(
    dataset: SiameseCropDataset, index: int, bundle_cache: dict[Path, CropBundle]
) -> _ResolvedSpec:
    """Reproduce ``SiameseCropDataset.__getitem__``'s exact index
    resolution for ``index`` -- everything up to (but not including) the
    ``build_sample`` call, which this module defers to a worker process
    for parallelism. See module docstring, item 1.
    """
    length = len(dataset)
    if not (0 <= index < length):
        raise IndexError(index)
    bundle_index, local_index = divmod(index, dataset.config.samples_per_bundle)
    bundle_dir = dataset.bundle_dirs[bundle_index]

    spec_seed = _combine_seeds(dataset.seed, bundle_index, local_index)
    rng = np.random.default_rng(spec_seed)
    positive = bool(rng.random() < dataset.positive_fraction)

    side: str | None = None
    if positive:
        bundle = bundle_cache[bundle_dir]
        side = "left" if rng.random() < 0.5 else "right"
        candidates = _positive_center_z_candidates(bundle, side, dataset.config)
        center_z = int(rng.choice(candidates))
    else:
        center_z = int(rng.integers(FIRST_CENTER_Z, LAST_CENTER_Z + 1))

    sample_seed = int(rng.integers(0, _SEED_UPPER_BOUND))
    return _ResolvedSpec(
        bundle_dir=bundle_dir,
        center_z=center_z,
        seed=sample_seed,
        positive=positive,
        side=side,
    )


# ---------------------------------------------------------------------------
# Worker-process task -- see module docstring, item 3.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _SampleTask:
    """One ``ProcessPoolExecutor`` unit of work. Every field is a plain,
    standard-picklable value -- see module docstring, item 3.
    """

    sample_index: int
    bundle_dir: Path
    center_z: int
    seed: int
    positive: bool
    side: str | None


@lru_cache(maxsize=8)
def _load_bundle_cached(bundle_dir: Path) -> CropBundle:
    """Worker-process-local bundle cache (module docstring, item 3). Each
    ``ProcessPoolExecutor`` worker is a separate Python process with its
    own module-level state, so this cache is naturally per-worker with no
    cross-process synchronization -- the same "no shared mutable state, no
    locks needed" property ``dataset.py``'s own per-instance caches rely
    on (see ``dataset.py``'s module docstring, item 7).
    """
    return load_crop_bundle(bundle_dir)


def _build_and_save_sample(
    task: _SampleTask, out_dir: str, config: DatasetConfig
) -> str:
    """Runs in a worker process. Reloads the bundle (worker-local cache)
    and calls ``build_sample`` unmodified, with no ``_precompute``
    override -- every voxel of physics happens inside ``build_sample``,
    exactly as it would for an on-the-fly ``SiameseCropDataset`` sample
    (module docstring, item 1). Returns the sample's bundle identity
    (aggregate bookkeeping only, never printed on its own by the CLI).
    """
    bundle = _load_bundle_cached(task.bundle_dir)
    sample = build_sample(
        bundle,
        task.center_z,
        task.seed,
        task.positive,
        side=task.side,
        config=config,
    )
    _save_sample_npz(Path(out_dir), task.sample_index, sample)
    return _bundle_identity(task.bundle_dir)


# ---------------------------------------------------------------------------
# Atomic-ish per-sample write -- see module docstring, item 4.
# ---------------------------------------------------------------------------


def _sample_path(out_dir: Path, index: int) -> Path:
    return out_dir / f"sample_{index:04d}.npz"


def _save_sample_npz(out_dir: Path, index: int, sample: Sample) -> Path:
    final_path = _sample_path(out_dir, index)
    meta_json = json.dumps(dict(sample.meta))

    fd, tmp_name = tempfile.mkstemp(
        dir=str(out_dir), prefix=f".sample_{index:04d}.", suffix=".npz.tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            np.savez(
                fh,
                left_view=sample.left_view.numpy(),
                right_view=sample.right_view.numpy(),
                pet_diff=sample.pet_diff.numpy(),
                target_mask=sample.target_mask.numpy(),
                # module docstring, item 7 -- always written by this
                # (post-item-7) version of this module.
                source_fraction=sample.source_fraction.numpy(),
                valid_mask=sample.valid_mask.numpy(),
                raw_pet=sample.raw_pet.numpy(),
                meta_json=np.array(meta_json),
            )
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, final_path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp_path.unlink()
        raise
    return final_path


_BASE_SAMPLE_KEYS = frozenset(
    {
        "left_view",
        "right_view",
        "pet_diff",
        "target_mask",
        "valid_mask",
        "raw_pet",
        "meta_json",
    }
)


def _checked_float_array(
    *, path: Path, name: str, value: np.ndarray, shape: tuple[int, ...]
) -> np.ndarray:
    """Validate one cached tensor before constructing a ``Sample``.

    Cache files are persisted training inputs, so silently coercing an
    unexpected dtype/shape would make a damaged or mixed cache look valid.
    Newly written caches use float32 for every tensor; readers enforce that
    exact contract and finite values.
    """
    if value.shape != shape:
        raise CacheSchemaError(
            f"{path.name}: {name} shape {value.shape} != expected {shape}"
        )
    if value.dtype != np.float32:
        raise CacheSchemaError(
            f"{path.name}: {name} dtype {value.dtype} != expected float32"
        )
    if not bool(np.isfinite(value).all()):
        raise CacheSchemaError(f"{path.name}: {name} contains non-finite values")
    return value


def _load_sample_npz(path: Path, *, require_source_fraction: bool = False) -> Sample:
    try:
        archive = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise CacheSchemaError(f"{path.name}: unreadable NPZ sample") from exc

    with archive as npz:
        actual_keys = frozenset(npz.files)
        expected_keys = _BASE_SAMPLE_KEYS | (
            {"source_fraction"} if require_source_fraction else set()
        )
        missing = expected_keys - actual_keys
        unexpected = actual_keys - (_BASE_SAMPLE_KEYS | {"source_fraction"})
        if missing:
            raise CacheSchemaError(
                f"{path.name}: missing required sample field(s): "
                + ", ".join(sorted(missing))
            )
        if unexpected:
            raise CacheSchemaError(
                f"{path.name}: unexpected sample field(s): "
                + ", ".join(sorted(unexpected))
            )

        try:
            left_view = _checked_float_array(
                path=path,
                name="left_view",
                value=npz["left_view"],
                shape=LEFT_VIEW_SHAPE,
            )
            right_view = _checked_float_array(
                path=path,
                name="right_view",
                value=npz["right_view"],
                shape=RIGHT_VIEW_SHAPE,
            )
            pet_diff = _checked_float_array(
                path=path,
                name="pet_diff",
                value=npz["pet_diff"],
                shape=PET_DIFF_SHAPE,
            )
            target_mask = _checked_float_array(
                path=path,
                name="target_mask",
                value=npz["target_mask"],
                shape=TARGET_SHAPE,
            )
            valid_mask = _checked_float_array(
                path=path,
                name="valid_mask",
                value=npz["valid_mask"],
                shape=TARGET_SHAPE,
            )
            raw_pet = _checked_float_array(
                path=path,
                name="raw_pet",
                value=npz["raw_pet"],
                shape=TARGET_SHAPE,
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, CacheSchemaError):
                raise
            raise CacheSchemaError(f"{path.name}: invalid cached tensor") from exc

        for name, binary in (("target_mask", target_mask), ("valid_mask", valid_mask)):
            if not bool(np.logical_or(binary == 0.0, binary == 1.0).all()):
                raise CacheSchemaError(f"{path.name}: {name} must be binary float32")

        # Module docstring, item 7 -- absent on a pre-item-7 cache
        # (e.g. p6_cache/p6_cache_big as they exist today). A legacy cache
        # whose manifest does not claim soft-target support keeps the
        # all-zero compatibility fallback. Once a manifest declares
        # has_source_fraction=True, however, every sample must carry a
        # valid field; a stale/mixed/damaged cache fails closed instead of
        # silently training against zeros.
        if "source_fraction" in npz:
            source_fraction = _checked_float_array(
                path=path,
                name="source_fraction",
                value=npz["source_fraction"],
                shape=TARGET_SHAPE,
            )
            if bool((source_fraction < 0.0).any()) or bool(
                (source_fraction > 1.0).any()
            ):
                raise CacheSchemaError(
                    f"{path.name}: source_fraction values must lie in [0, 1]"
                )
            if not np.array_equal(source_fraction >= 0.5, target_mask >= 0.5):
                raise CacheSchemaError(
                    f"{path.name}: source_fraction hard-threshold invariant "
                    "does not match target_mask"
                )
        elif require_source_fraction:
            raise CacheSchemaError(
                f"{path.name}: manifest declares has_source_fraction=True "
                "but sample has no source_fraction field"
            )
        else:
            source_fraction = np.zeros_like(target_mask)
        try:
            meta = json.loads(str(npz["meta_json"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise CacheSchemaError(f"{path.name}: invalid meta_json") from exc
        if not isinstance(meta, dict):
            raise CacheSchemaError(f"{path.name}: meta_json must decode to an object")
        if (
            not isinstance(meta.get("subject"), str)
            or not meta["subject"]
            or not isinstance(meta.get("session"), str)
            or not meta["session"]
            or not isinstance(meta.get("positive"), bool)
        ):
            raise CacheSchemaError(
                f"{path.name}: meta_json requires non-empty string subject/session "
                "and boolean positive"
            )
    return Sample(
        left_view=torch.from_numpy(np.ascontiguousarray(left_view)),
        right_view=torch.from_numpy(np.ascontiguousarray(right_view)),
        pet_diff=torch.from_numpy(np.ascontiguousarray(pet_diff)),
        target_mask=torch.from_numpy(np.ascontiguousarray(target_mask)),
        source_fraction=torch.from_numpy(np.ascontiguousarray(source_fraction)),
        valid_mask=torch.from_numpy(np.ascontiguousarray(valid_mask)),
        raw_pet=torch.from_numpy(np.ascontiguousarray(raw_pet)),
        meta=meta,
    )


def _validate_sample_manifest_binding(
    sample: Sample,
    *,
    path: Path,
    index: int,
    total_positive: int,
    allowed_bundle_identities: frozenset[str],
    expected_bundle_identity: str | None = None,
    enforce_class_content: bool = True,
) -> str:
    """Bind one persisted sample to its manifest without emitting identities."""
    subject = sample.meta["subject"]
    session = sample.meta["session"]
    if any("/" in value or value in {".", ".."} for value in (subject, session)):
        raise CacheSchemaError(
            f"{path.name}: subject/session metadata must be single path components"
        )
    bundle_identity = f"{subject}/{session}"
    if bundle_identity not in allowed_bundle_identities:
        raise CacheSchemaError(
            f"{path.name}: sample bundle metadata is not present in the manifest"
        )
    if (
        expected_bundle_identity is not None
        and bundle_identity != expected_bundle_identity
    ):
        raise CacheSchemaError(
            f"{path.name}: sample bundle metadata changed after dataset construction"
        )

    expected_positive = index < total_positive
    if sample.meta["positive"] is not expected_positive:
        raise CacheSchemaError(
            f"{path.name}: sample class metadata disagrees with the manifest class order"
        )
    content_positive = bool(
        torch.logical_and(sample.target_mask >= 0.5, sample.valid_mask >= 0.5)
        .any()
        .item()
    )
    if enforce_class_content and content_positive is not expected_positive:
        raise CacheSchemaError(
            f"{path.name}: target content disagrees with the manifest sample class"
        )
    return bundle_identity


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CacheManifest:
    cache_schema_version: str
    tensor_schema_version: str
    crop_schema_version: str
    dataset_builder_version: str
    config_hash: str
    seed: int
    bundle_identities: tuple[str, ...]
    excluded_qc_bundle_identities: tuple[str, ...]
    per_bundle_counts: dict[str, dict[str, int]]
    total_positive: int
    total_negative: int
    total_samples: int
    # module docstring, item 7. False on any manifest written before this
    # item existed (json.load's own .get default below), so a reader can
    # tell "no source_fraction key" apart from "field simply absent from
    # an old manifest.json" without needing a CACHE_SCHEMA_VERSION bump.
    has_source_fraction: bool = False
    research_prototype_warning: str = RESEARCH_PROTOTYPE_WARNING
    created_at: str = ""


def _manifest_path(cache_dir: Path) -> Path:
    return Path(cache_dir) / "manifest.json"


def _write_manifest(out_dir: Path, manifest: CacheManifest) -> Path:
    payload = {
        "cache_schema_version": manifest.cache_schema_version,
        "tensor_schema_version": manifest.tensor_schema_version,
        "crop_schema_version": manifest.crop_schema_version,
        "dataset_builder_version": manifest.dataset_builder_version,
        "config_hash": manifest.config_hash,
        "seed": manifest.seed,
        "bundle_identities": list(manifest.bundle_identities),
        "excluded_qc_bundle_identities": list(manifest.excluded_qc_bundle_identities),
        "per_bundle_counts": manifest.per_bundle_counts,
        "total_positive": manifest.total_positive,
        "total_negative": manifest.total_negative,
        "total_samples": manifest.total_samples,
        "has_source_fraction": manifest.has_source_fraction,
        "research_prototype_warning": manifest.research_prototype_warning,
        "created_at": manifest.created_at,
    }
    path = _manifest_path(out_dir)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def read_cache_manifest(cache_dir: Path) -> dict[str, Any]:
    """Load and schema-verify a cache directory's
    ``manifest.json``. Typed :class:`CacheSchemaError` on any missing
    file, unreadable JSON, or schema-version mismatch.
    """
    path = _manifest_path(cache_dir)
    if not path.is_file():
        raise CacheSchemaError(f"no manifest.json found under {cache_dir}")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise CacheSchemaError(
            f"manifest.json under {cache_dir} is not valid JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise CacheSchemaError(f"manifest.json under {cache_dir} must be an object")

    required_keys = {
        "cache_schema_version",
        "tensor_schema_version",
        "crop_schema_version",
        "dataset_builder_version",
        "config_hash",
        "seed",
        "bundle_identities",
        "excluded_qc_bundle_identities",
        "per_bundle_counts",
        "total_positive",
        "total_negative",
        "total_samples",
    }
    missing = required_keys - payload.keys()
    if missing:
        raise CacheSchemaError(
            f"manifest.json under {cache_dir} is missing required field(s): "
            + ", ".join(sorted(missing))
        )

    if payload.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
        raise CacheSchemaError(
            f"cache manifest schema mismatch under {cache_dir}: "
            f"manifest={payload.get('cache_schema_version')!r} "
            f"current={CACHE_SCHEMA_VERSION!r}"
        )
    if payload.get("tensor_schema_version") != TENSOR_SCHEMA_VERSION:
        raise CacheSchemaError(
            f"cache manifest tensor_schema_version mismatch under {cache_dir}: "
            f"manifest={payload.get('tensor_schema_version')!r} "
            f"current={TENSOR_SCHEMA_VERSION!r}"
        )
    for name in ("crop_schema_version", "dataset_builder_version", "config_hash"):
        if not isinstance(payload[name], str) or not payload[name]:
            raise CacheSchemaError(
                f"manifest.json under {cache_dir}: {name} must be a non-empty string"
            )
    if not isinstance(payload["seed"], int) or isinstance(payload["seed"], bool):
        raise CacheSchemaError(
            f"manifest.json under {cache_dir}: seed must be an integer"
        )

    for name in ("bundle_identities", "excluded_qc_bundle_identities"):
        values = payload[name]
        if not isinstance(values, list) or not all(
            isinstance(value, str) and value for value in values
        ):
            raise CacheSchemaError(
                f"manifest.json under {cache_dir}: {name} must be a list of strings"
            )
        if len(values) != len(set(values)):
            raise CacheSchemaError(
                f"manifest.json under {cache_dir}: {name} contains duplicates"
            )

    totals: dict[str, int] = {}
    for name in ("total_positive", "total_negative", "total_samples"):
        value = payload[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise CacheSchemaError(
                f"manifest.json under {cache_dir}: {name} must be a non-negative integer"
            )
        totals[name] = value
    if totals["total_positive"] + totals["total_negative"] != totals["total_samples"]:
        raise CacheSchemaError(
            f"manifest.json under {cache_dir}: total_positive + total_negative "
            "must equal total_samples"
        )
    if totals["total_samples"] == 0:
        raise CacheSchemaError(
            f"manifest.json under {cache_dir}: total_samples must be positive"
        )

    has_source_fraction = payload.get("has_source_fraction", False)
    if not isinstance(has_source_fraction, bool):
        raise CacheSchemaError(
            f"manifest.json under {cache_dir}: has_source_fraction must be boolean"
        )

    per_bundle_counts = payload["per_bundle_counts"]
    if not isinstance(per_bundle_counts, dict):
        raise CacheSchemaError(
            f"manifest.json under {cache_dir}: per_bundle_counts must be an object"
        )
    if set(per_bundle_counts) != set(payload["bundle_identities"]):
        raise CacheSchemaError(
            f"manifest.json under {cache_dir}: per_bundle_counts keys must exactly "
            "match bundle_identities"
        )
    summed_positive = 0
    summed_negative = 0
    for identity, counts in per_bundle_counts.items():
        if not isinstance(counts, dict) or set(counts) != {"positive", "negative"}:
            raise CacheSchemaError(
                f"manifest.json under {cache_dir}: counts for {identity!r} must "
                "contain exactly positive and negative"
            )
        for name in ("positive", "negative"):
            value = counts[name]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise CacheSchemaError(
                    f"manifest.json under {cache_dir}: {identity!r} {name} count "
                    "must be a non-negative integer"
                )
        summed_positive += counts["positive"]
        summed_negative += counts["negative"]
    if (summed_positive, summed_negative) != (
        totals["total_positive"],
        totals["total_negative"],
    ):
        raise CacheSchemaError(
            f"manifest.json under {cache_dir}: per_bundle_counts totals disagree "
            "with total_positive/total_negative"
        )

    return payload


def cache_bundle_identities(cache_dir: Path) -> frozenset[str]:
    """The sorted set of ``"<subject>/<session>"`` bundle identities a
    cache was built from -- see module docstring, item 6. Used by
    ``train.py`` to prove a train cache and a val cache are leakage-safe
    (disjoint) before any training allocation happens.
    """
    manifest = read_cache_manifest(cache_dir)
    return frozenset(manifest["bundle_identities"])


def cache_has_source_fraction(cache_dir: Path) -> bool:
    """Whether every sample in this cache carries a real (non-fallback)
    ``source_fraction`` field -- see module docstring, item 7.
    ``payload.get(..., False)`` makes a pre-item-7 manifest.json (no
    ``has_source_fraction`` key at all) read as ``False``, exactly like a
    manifest that explicitly recorded ``False``. Used by ``train.py``'s
    ``TrainConfig.__post_init__`` to refuse a ``soft_target=True`` run
    whose ``train_cache_dir`` cannot actually supply real soft targets,
    rather than silently training on zero-filled fallback values.
    """
    manifest = read_cache_manifest(cache_dir)
    return bool(manifest.get("has_source_fraction", False))


# ---------------------------------------------------------------------------
# The precompute entry point
# ---------------------------------------------------------------------------


def precompute_synthetic_cache(
    bundle_dirs: Sequence[Path],
    out_dir: Path,
    *,
    n_positive_per_bundle: int,
    n_negative_per_bundle: int,
    seed: int,
    config: DatasetConfig | None = None,
    num_workers: int = 1,
    exclude_qc_flagged: bool = False,
    progress_every: int = 25,
) -> CacheManifest:
    """Precompute and write a synthetic-sample cache under ``out_dir``.
    See module docstring for the full design. ``out_dir`` is caller
    -supplied and MUST be a gitignored local root (derived data; never
    committed) -- this function does not enforce that, the caller does
    (matching ``checkpoint.py``'s ``save_checkpoint``/``train.py``'s
    ``run_root`` convention).
    """
    cfg = config or DatasetConfig()
    out_dir = Path(out_dir)

    sorted_dirs = tuple(sorted(Path(d) for d in bundle_dirs))
    if not sorted_dirs:
        raise CachePrepError("bundle_dirs must be non-empty")
    if n_positive_per_bundle < 0 or n_negative_per_bundle < 0:
        raise CachePrepError(
            "n_positive_per_bundle and n_negative_per_bundle must be >= 0; "
            f"got {n_positive_per_bundle}, {n_negative_per_bundle}"
        )
    if n_positive_per_bundle == 0 and n_negative_per_bundle == 0:
        raise CachePrepError(
            "at least one of n_positive_per_bundle/n_negative_per_bundle must be > 0"
        )
    if num_workers < 1:
        raise CachePrepError(f"num_workers must be >= 1, got {num_workers}")

    out_dir.mkdir(parents=True, exist_ok=True)

    bundle_cache: dict[Path, CropBundle] = {}
    included_dirs: list[Path] = []
    excluded_identities: list[str] = []
    for bundle_dir in sorted_dirs:
        bundle = load_crop_bundle(bundle_dir)
        bundle_cache[bundle_dir] = bundle
        if exclude_qc_flagged and bundle.reflection_qc_flag:
            excluded_identities.append(_bundle_identity(bundle_dir))
            continue
        included_dirs.append(bundle_dir)

    if not included_dirs:
        raise CachePrepError(
            "no bundle_dirs remained after exclude_qc_flagged filtering "
            f"({len(excluded_identities)} excluded, 0 remaining)"
        )

    per_bundle_counts: dict[str, dict[str, int]] = {
        _bundle_identity(d): {"positive": 0, "negative": 0} for d in included_dirs
    }
    tasks: list[_SampleTask] = []

    if n_positive_per_bundle > 0:
        pos_cfg = replace(cfg, samples_per_bundle=n_positive_per_bundle)
        pos_dataset = SiameseCropDataset(
            included_dirs, seed=seed, positive_fraction=1.0, config=pos_cfg
        )
        for i in range(len(pos_dataset)):
            spec = _resolve_index(pos_dataset, i, bundle_cache)
            tasks.append(
                _SampleTask(
                    sample_index=len(tasks),
                    bundle_dir=spec.bundle_dir,
                    center_z=spec.center_z,
                    seed=spec.seed,
                    positive=True,
                    side=spec.side,
                )
            )
            per_bundle_counts[_bundle_identity(spec.bundle_dir)]["positive"] += 1

    if n_negative_per_bundle > 0:
        neg_cfg = replace(cfg, samples_per_bundle=n_negative_per_bundle)
        neg_dataset = SiameseCropDataset(
            included_dirs, seed=seed, positive_fraction=0.0, config=neg_cfg
        )
        for i in range(len(neg_dataset)):
            spec = _resolve_index(neg_dataset, i, bundle_cache)
            tasks.append(
                _SampleTask(
                    sample_index=len(tasks),
                    bundle_dir=spec.bundle_dir,
                    center_z=spec.center_z,
                    seed=spec.seed,
                    positive=False,
                    side=None,
                )
            )
            per_bundle_counts[_bundle_identity(spec.bundle_dir)]["negative"] += 1

    total = len(tasks)
    completed = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_task = {
            executor.submit(_build_and_save_sample, task, str(out_dir), cfg): task
            for task in tasks
        }
        for future in as_completed(future_to_task):
            future.result()  # re-raises any worker exception immediately
            completed += 1
            if completed % progress_every == 0 or completed == total:
                logger.info(
                    "precompute_synthetic_cache: %d/%d samples written",
                    completed,
                    total,
                )

    total_positive = sum(v["positive"] for v in per_bundle_counts.values())
    total_negative = sum(v["negative"] for v in per_bundle_counts.values())
    config_hash = _hash_repr(
        (
            n_positive_per_bundle,
            n_negative_per_bundle,
            seed,
            exclude_qc_flagged,
            repr(cfg),
        )
    )

    manifest = CacheManifest(
        cache_schema_version=CACHE_SCHEMA_VERSION,
        tensor_schema_version=TENSOR_SCHEMA_VERSION,
        crop_schema_version=CROP_SCHEMA_VERSION,
        dataset_builder_version=DATASET_BUILDER_VERSION,
        config_hash=config_hash,
        seed=seed,
        bundle_identities=tuple(sorted(per_bundle_counts)),
        excluded_qc_bundle_identities=tuple(sorted(excluded_identities)),
        per_bundle_counts=per_bundle_counts,
        total_positive=total_positive,
        total_negative=total_negative,
        total_samples=total_positive + total_negative,
        # module docstring, item 7 -- this (post-item-7) version of this
        # module always writes a real source_fraction into every sample.
        has_source_fraction=True,
        created_at=datetime.now(UTC).isoformat(),
    )
    _write_manifest(out_dir, manifest)
    return manifest


# ---------------------------------------------------------------------------
# CachedSampleDataset
# ---------------------------------------------------------------------------


class CachedSampleDataset(torch.utils.data.Dataset):
    """torch ``Dataset`` over a directory of precomputed ``.npz`` samples
    written by :func:`precompute_synthetic_cache`. ``__getitem__`` is a
    plain array read (no reflection/simulation work) -- see module
    docstring for the measured throughput this unlocks.

    ``enforce_manifest_class_content=True`` is the fail-closed training
    default. Evaluation may set it to ``False`` only to load and report a
    generation-intent/actual-target mismatch as an explicit QA finding;
    manifest membership, class ordering, and aggregate counts remain strict.
    """

    def __init__(
        self, cache_dir: Path, *, enforce_manifest_class_content: bool = True
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self._enforce_manifest_class_content = enforce_manifest_class_content
        self.manifest = read_cache_manifest(self.cache_dir)
        self._sample_paths: tuple[Path, ...] = tuple(
            sorted(self.cache_dir.glob("sample_*.npz"))
        )
        if not self._sample_paths:
            raise CacheSchemaError(
                f"no sample_*.npz files found under {self.cache_dir}"
            )
        expected_paths = tuple(
            _sample_path(self.cache_dir, index)
            for index in range(self.manifest["total_samples"])
        )
        if self._sample_paths != expected_paths:
            raise CacheSchemaError(
                f"sample inventory under {self.cache_dir} must be exactly contiguous "
                f"sample_0000.npz..sample_{self.manifest['total_samples'] - 1:04d}.npz"
            )

        # Validate every persisted training input before a model/GPU run can
        # begin. __getitem__ still performs the same checks on each later read,
        # protecting against a cache modified after construction.
        require_source_fraction = bool(self.manifest.get("has_source_fraction", False))
        total_positive = self.manifest["total_positive"]
        allowed_bundle_identities = frozenset(self.manifest["bundle_identities"])
        observed_counts = {
            identity: {"positive": 0, "negative": 0}
            for identity in allowed_bundle_identities
        }
        sample_bundle_identities: list[str] = []
        for index, sample_path in enumerate(self._sample_paths):
            sample = _load_sample_npz(
                sample_path, require_source_fraction=require_source_fraction
            )
            identity = _validate_sample_manifest_binding(
                sample,
                path=sample_path,
                index=index,
                total_positive=total_positive,
                allowed_bundle_identities=allowed_bundle_identities,
                enforce_class_content=self._enforce_manifest_class_content,
            )
            sample_bundle_identities.append(identity)
            class_name = "positive" if index < total_positive else "negative"
            observed_counts[identity][class_name] += 1
        if observed_counts != self.manifest["per_bundle_counts"]:
            raise CacheSchemaError(
                f"sample metadata counts under {self.cache_dir} disagree with manifest"
            )
        self._sample_bundle_identities = tuple(sample_bundle_identities)

    def __len__(self) -> int:
        return len(self._sample_paths)

    def __getitem__(self, index: int) -> Sample:
        length = len(self)
        if index < 0:
            index += length
        if not (0 <= index < length):
            raise IndexError(index)
        sample = _load_sample_npz(
            self._sample_paths[index],
            require_source_fraction=bool(
                self.manifest.get("has_source_fraction", False)
            ),
        )
        _validate_sample_manifest_binding(
            sample,
            path=self._sample_paths[index],
            index=index,
            total_positive=self.manifest["total_positive"],
            allowed_bundle_identities=frozenset(self.manifest["bundle_identities"]),
            expected_bundle_identity=self._sample_bundle_identities[index],
            enforce_class_content=self._enforce_manifest_class_content,
        )
        return sample
