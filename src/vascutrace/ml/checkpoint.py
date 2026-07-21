"""Atomic, resumable checkpoint I/O for the P6 training loop.

RESEARCH_PROTOTYPE_WARNING
---------------------------------------------------------------------------
Research prototype. Trained and evaluated using simulated vascular-like
abnormalities, not confirmed human post-angioplasty lesions.
---------------------------------------------------------------------------

Implementation notes
============================================================================
This module answers two questions a paper reviewer asks about any resumable
training run: "can a crash mid-write ever corrupt my last good checkpoint?"
and "does 'resume' actually reproduce the exact optimization trajectory, or
just approximately continue from similar weights?"

1. Atomic writes (:func:`save_checkpoint`)
   ------------------------------------------------------------------------
   ``last.pt``/``best.pt`` are never opened for in-place writing. Every save
   writes a fresh temp file *in the same directory* (via ``tempfile.
   mkstemp`` -- guarantees a unique name, no collision with a concurrent
   writer), flushes and ``os.fsync``s it so the bytes are actually durable
   on disk (not just buffered in the OS page cache), then calls
   ``os.replace(tmp, final)``. POSIX guarantees ``rename()``/``replace()``
   within one filesystem is atomic: a reader always sees either the
   complete old file or the complete new file, never a partial one (Python
   docs, ``os.replace``: "If successful, the renaming will be an atomic
   operation"). If anything raises between opening the temp file and the
   final ``os.replace`` -- a serialization error, a disk-full condition, a
   killed process -- the ``except`` branch deletes the orphaned temp file
   and re-raises; ``os.replace`` is never reached, so the previous good
   ``last.pt``/``best.pt`` is untouched. This is the standard "write-temp,
   fsync, atomic-rename" pattern for crash-safe file updates; see
   ``tests/test_ml_train.py``'s ``TestAtomicCheckpointWrites`` for a direct
   simulated-interruption proof.

2. What one checkpoint binds together (:class:`CheckpointPayload`)
   ------------------------------------------------------------------------
   A checkpoint is useless for exact resume if it captures only the model
   weights: the optimizer's own per-parameter moment estimates (Adam-family
   optimizers are stateful), the AMP loss-scale state, and every RNG stream
   that influenced training so far (Python's ``random``, NumPy's legacy
   global generator, torch's CPU generator, torch's CUDA generator(s) if
   present, and the ``torch.Generator`` instance driving the training
   ``DataLoader``'s shuffle order) must all be captured, or "resume" merely
   *approximately* continues training rather than reproducing the same
   trajectory. :class:`CheckpointPayload` binds all of these plus the
   frozen contract versions it was trained against
   (``tensor_schema.TENSOR_SCHEMA_VERSION``, ``data.contract.
   CROP_SCHEMA_VERSION``, ``model.model_signature``) so a mismatched
   checkpoint fails loudly on resume (see ``train.py``'s
   ``_verify_resume_compatibility``) instead of silently loading
   incompatible weights into a differently-shaped network.

3. "Never store raw data or identifiers" -> hash, don't store
   ------------------------------------------------------------------------
   The implementation's own checkpoint contract is explicit here: "data/source
   hashes (hash the split subject lists + config)" -- not "store the split
   subject lists". This module therefore never persists a
   ``train_bundle_dirs``/``val_bundle_dirs`` list (which would embed
   subject codes like ``SYNTH_SUBJECT_001`` into the checkpoint payload) --
   only ``split_hash``, a one-way SHA-256 digest of the sorted bundle
   identities (see ``train.py``'s ``compute_split_hash``), and
   ``config_hash``, a digest of the architecture/training hyperparameters.
   A checkpoint is therefore safe to hand to a reviewer or back up
   off-machine without re-exposing which subjects were used, even though it
   already lives under this project's gitignored, never-committed run-root
   convention (plan-06, "Retained hard boundaries": "Raw volumes, crops,
   checkpoints, run outputs stay gitignored; never committed").
   :attr:`CheckpointPayload.calibration_status` is always
   ``"uncalibrated"`` and :attr:`CheckpointPayload.research_prototype_warning`
   always carries :data:`RESEARCH_PROTOTYPE_WARNING` -- neither field is
   ever settable to anything else by this module, so the scientific
   boundary travels with every checkpoint by construction.

4. ``resume(run_root)`` needs a caller-supplied config -- a resolved
   design decision, flagged for the reviewer
   ------------------------------------------------------------------------
   Because item 3 forbids storing the raw bundle-directory list, a truly
   zero-argument-beyond-``run_root`` resume is impossible without either
   (a) violating the "hash, don't store" instruction, or (b) resume being
   unable to reconstruct which data to train on. This module resolves that
   in favor of (a) staying safe: ``train.py``'s ``resume(run_root, config)``
   takes the caller's ``TrainConfig`` (the same one used to start the run,
   typically re-built by the CLI from the original config file) and
   verifies its ``split_hash``/``config_hash``/schema versions against the
   checkpoint's own before proceeding -- a mismatch is a loud, typed
   :class:`~src.vascutrace.ml.train.CheckpointCompatibilityError`, not a
   silent "close enough" resume.
============================================================================
"""

from __future__ import annotations

import contextlib
import os
import random
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.vascutrace.geometry import RESEARCH_PROTOTYPE_WARNING
from src.vascutrace.ml.dataset import DatasetConfig
from src.vascutrace.ml.model import ModelConfig

__all__ = [
    "RESEARCH_PROTOTYPE_WARNING",
    "CHECKPOINT_SCHEMA_VERSION",
    "RngState",
    "CheckpointPayload",
    "CheckpointError",
    "save_checkpoint",
    "load_checkpoint",
    "capture_rng_state",
    "restore_rng",
]

CHECKPOINT_SCHEMA_VERSION = "p6-checkpoint-v1"


class CheckpointError(RuntimeError):
    """Raised for a checkpoint I/O or format problem (missing file, wrong
    payload type). Distinct from
    :class:`~src.vascutrace.ml.train.CheckpointCompatibilityError`, which
    covers a *structurally valid* checkpoint that is incompatible with the
    caller's current config/schema versions.
    """


@dataclass
class RngState:
    """Every RNG stream that can influence a training run, captured
    together so :func:`restore_rng` re-seeds all of them atomically. See
    module docstring, item 2.
    """

    python_random: tuple[Any, ...]
    numpy_random: tuple[Any, ...]
    torch_cpu: torch.Tensor
    torch_cuda: list[torch.Tensor] | None
    dataloader_generator: torch.Tensor


@dataclass
class CheckpointPayload:
    """Everything one resumable checkpoint binds together. See module
    docstring, items 2-3, for what each field is and is deliberately not.
    """

    checkpoint_schema_version: str
    tensor_schema_version: str
    crop_schema_version: str
    model_signature: str
    model_config: ModelConfig
    dataset_config: DatasetConfig

    model_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any]
    scaler_state_dict: dict[str, Any]
    rng_state: RngState

    epoch: int
    global_step: int
    best_val_metric: float | None
    best_val_metric_name: str

    hyperparams: dict[str, Any] = field(default_factory=dict)
    split_hash: str = ""
    config_hash: str = ""

    calibration_status: str = "uncalibrated"
    research_prototype_warning: str = RESEARCH_PROTOTYPE_WARNING
    created_at: str = ""


# ---------------------------------------------------------------------------
# Atomic save / load
# ---------------------------------------------------------------------------


def save_checkpoint(path: Path, payload: CheckpointPayload) -> None:
    """Write ``payload`` to ``path`` atomically -- see module docstring,
    item 1. On any exception, the previous file at ``path`` (if any) is
    left byte-for-byte untouched.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            torch.save(payload, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)  # atomic on POSIX, same filesystem/dir
    except BaseException:
        with contextlib.suppress(OSError):
            tmp_path.unlink()
        raise


def load_checkpoint(path: Path) -> CheckpointPayload:
    """Load a :class:`CheckpointPayload` previously written by
    :func:`save_checkpoint`.

    ``weights_only=False`` is required and deliberate: this payload stores
    plain Python dataclasses (:class:`RngState`, :class:`ModelConfig`,
    :class:`DatasetConfig`), not only tensors, and torch >= 2.6's default
    ``weights_only=True`` restricted unpickler cannot load them (verified
    directly against this project's torch 2.9.1: the default raises
    ``UnpicklingError``; see
    https://docs.pytorch.org/docs/2.9/generated/torch.load.html). This is
    safe here because a VascuTrace checkpoint is always a trusted,
    locally-generated, gitignored file (plan-06, "Retained hard
    boundaries": checkpoints never committed, never leave the local
    machine) -- never a file loaded from an untrusted or network source.
    """
    path = Path(path)
    if not path.is_file():
        raise CheckpointError(f"checkpoint file not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, CheckpointPayload):
        raise CheckpointError(
            f"file at {path} did not deserialize to a CheckpointPayload "
            f"(got {type(payload).__name__})"
        )
    return payload


# ---------------------------------------------------------------------------
# RNG capture / restore -- see module docstring, item 2.
# ---------------------------------------------------------------------------


def capture_rng_state(dataloader_generator: torch.Generator) -> RngState:
    """Snapshot every RNG stream that can influence a training run into one
    :class:`RngState`. ``dataloader_generator`` is the ``torch.Generator``
    instance the training ``DataLoader``'s shuffling sampler was built
    with (a stream not covered by ``torch.get_rng_state()`` -- it is a
    separate, explicitly-owned generator object).
    """
    return RngState(
        python_random=random.getstate(),
        numpy_random=np.random.get_state(),
        torch_cpu=torch.get_rng_state(),
        torch_cuda=(
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        dataloader_generator=dataloader_generator.get_state(),
    )


def restore_rng(payload: CheckpointPayload) -> torch.Generator:
    """Re-seed every RNG stream in ``payload.rng_state`` for an exact
    resume, and return a fresh ``torch.Generator`` restored to the
    checkpointed DataLoader-shuffle stream state (the caller passes this
    generator to the resumed training ``DataLoader`` so its shuffle order
    continues the exact same pseudo-random sequence rather than restarting
    it).
    """
    state = payload.rng_state
    random.setstate(state.python_random)
    np.random.set_state(state.numpy_random)
    torch.set_rng_state(state.torch_cpu)
    if state.torch_cuda is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state.torch_cuda)

    generator = torch.Generator()
    generator.set_state(state.dataloader_generator)
    return generator
