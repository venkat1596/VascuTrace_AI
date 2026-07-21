"""Frozen inference wrapper for the banked B2 deep-supervision
Siamese PET/CT checkpoint.

RESEARCH_PROTOTYPE_WARNING
---------------------------------------------------------------------------
Research prototype. Trained and evaluated using simulated vascular-like
abnormalities, not confirmed human post-angioplasty lesions.
---------------------------------------------------------------------------

FROZEN CONTRACT -- do not retune, do not retrain, do not silently swap
============================================================================
  checkpoint : runs/siamese_p4b2_deepsup/best_constrained_iou.pt
  config_hash: 824d2a01df0af942 (must match the loaded
               ``CheckpointPayload.config_hash`` -- this module does not
               hardcode a single checkpoint path; callers pass their own)
  threshold  : 0.5 hard-mask cut on ``abnormality_score`` -- see
               ``metrics.py``'s ``DEFAULT_SCORE_THRESHOLD`` (do not retune;
               :data:`DEFAULT_INFERENCE_THRESHOLD` below)
  deep sup   : TRAIN-TIME ONLY. This checkpoint's ``ModelConfig.
               deep_supervision=True`` constructs the aux-head modules
               (``aux_head2``/``aux_head3``), but every call in this
               module leaves ``forward(..., return_aux=False)`` at its
               default -- the aux heads are never invoked here. Deep
               supervision regularizes the MAIN head during training; it
               has no separate role at inference (``model.py``'s own
               ``forward`` docstring: "every existing/default caller ...
               gets EXACTLY the pre-B2 return value -- a bare Tensor").
  calibration: "uncalibrated" -- :func:`~src.vascutrace.ml.model.
               abnormality_score` is a monotonic ``[0, 1]`` sigmoid
               transform, NOT a calibrated clinical probability
               (``model.py``'s own docstring; ``checkpoint.py`` module
               docstring, item 3). Never rename this "predict_proba" or
               present it as a diagnostic confidence.
============================================================================

Implementation notes
============================================================================
This module answers the reproducibility question raised by any "we deployed
our trained model" claim: does the deployed forward path actually match
the ONE path the checkpoint was validated under, or has a well-meaning
refactor silently changed it? ``evaluate.py``'s own ``_run_inference``
(lines ~275-296) is this project's already-reviewed, already-tested ground
truth for "how do you turn one ``Sample`` into an abnormality-score map with this
checkpoint" -- this module does not re-derive that logic, it copies its
exact sequence: unsqueeze a batch dim of 1, ``model.eval()`` +
``torch.no_grad()``, ``model(left, right, pet_diff)`` with ``return_aux``
left at its default ``False`` so only the main head's logits are ever
touched, then :func:`~src.vascutrace.ml.model.abnormality_score` for the
``[0, 1]`` transform. A checkpoint that passes ``evaluate.py``'s
``evaluate_checkpoint`` therefore receives the byte-identical forward
computation here.

This module deliberately does NOT import anything from ``train.py`` --
training carries an optimizer, AMP grad scaler, EMA shadow weights,
hard-negative mining, batch augmentation, and (when
``deep_supervision=True``) an aux-head loss term, none of which inference
needs or should touch. Inference is strictly a read-only subset of what
``evaluate.py`` already exercises: load frozen weights, forward the main
head once, threshold. :func:`predict_mask`'s optional small-component
filter reuses ``metrics.py``'s own tested
``_filter_min_component_size`` (module docstring item 8 there) rather
than a second, potentially-drifting reimplementation.

``InferenceMetadata.model_name`` -- guarding the reference-oracle confusion
----------------------------------------------------------------------
The product bridge can run either this
real Siamese checkpoint or a deterministic synthetic-reference backend
whose identity string is literally ``"deterministic-synthetic-reference"``.
:func:`load_inference_model` derives ``model_name`` from the
loaded checkpoint's own run-directory name (e.g.
``"siamese_p4b2_deepsup"`` for the frozen B2 checkpoint) -- never a
literal placeholder -- so a caller can assert
``metadata.model_name != "deterministic-synthetic-reference"`` as a cheap,
always-true-for-a-real-checkpoint guard against that specific confusion.
============================================================================

Operator-controlled operating point, exploratory only
----------------------------------------------------------------------
:func:`resolve_operating_point` lets a caller (currently only the E4
product siamese path, ``vascutrace.services.run_siamese_detection``)
override the hard-mask score threshold and/or the minimum predicted
connected-component size, for demo/exploration of the precision/clean-vs-
recall Pareto frontier. It reads two environment variables, both optional:

  ``VASCUTRACE_SCORE_THRESHOLD``      float, must be in ``(0, 1]``
  ``VASCUTRACE_MIN_COMPONENT_SIZE``  int, must be ``>= 0``

Unset (or an explicit ``None`` argument on both) resolves to the FROZEN
BANKED operating point this module's own header documents: ``threshold =
0.5``, ``min_component_size = 0`` -- i.e. behavior is byte-identical to
this module before Option A existed. An out-of-range or unparsable env
value FAILS LOUD (raises :class:`ValueError`) -- it is never silently
clamped or silently replaced by the default; a typo'd env var must not
masquerade as "using the banked science operating point."

Any resolved operating point other than ``(0.5, 0)`` is EXPLORATORY, not a
second banked, recommended, or production preset. A post-hoc threshold and
size sweep did not pass its preregistered gate. Nothing in this module recommends, defaults
to, or names a non-default operating point as a "low-FP" or "strict" mode.
============================================================================
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.vascutrace.geometry import RESEARCH_PROTOTYPE_WARNING
from src.vascutrace.ml.cache import (
    _load_sample_npz,  # noqa: PLC2701 -- deliberate, documented read-only reuse of cache.py's one tested npz->Sample reader; matches this project's own cache.py/train.py private-helper-reuse convention (see cache.py module docstring, item 1)
)
from src.vascutrace.ml.checkpoint import load_checkpoint
from src.vascutrace.ml.dataset import Sample
from src.vascutrace.ml.metrics import (
    DEFAULT_MIN_COMPONENT_SIZE,
    _filter_min_component_size,  # noqa: PLC2701 -- deliberate, documented reuse of metrics.py's one tested component-size filter (module docstring, item 8); never reimplemented here
)
from src.vascutrace.ml.model import abnormality_score, build_model, model_signature

__all__ = [
    "RESEARCH_PROTOTYPE_WARNING",
    "FROZEN_CHECKPOINT_PATH",
    "FROZEN_CONFIG_HASH",
    "DEFAULT_INFERENCE_THRESHOLD",
    "ENV_SCORE_THRESHOLD",
    "ENV_MIN_COMPONENT_SIZE",
    "InferenceMetadata",
    "InferenceResult",
    "load_inference_model",
    "predict_abnormality_score",
    "predict_mask",
    "resolve_operating_point",
    "run_sample_inference",
]

# Informational freeze of the banked incumbent. Callers
# still pass their own `checkpoint_path` explicitly (this module never
# reads these constants itself), but they are exported so a caller/test
# can assert against the frozen, reviewed values rather than a literal
# re-typed elsewhere.
FROZEN_CHECKPOINT_PATH = "runs/siamese_p4b2_deepsup/best_constrained_iou.pt"
FROZEN_CONFIG_HASH = "824d2a01df0af942"

# Matches metrics.py's own DEFAULT_SCORE_THRESHOLD -- the hard-mask cut this
# checkpoint was selected and evaluated under. Do not retune for a reported
# banked result.
DEFAULT_INFERENCE_THRESHOLD: float = 0.5

# Operator-controlled operating-point env var names (Option A, module
# docstring's "Operator-controlled operating point" section). Exported so a
# caller/test can reference the exact string rather than re-typing it.
ENV_SCORE_THRESHOLD = "VASCUTRACE_SCORE_THRESHOLD"
ENV_MIN_COMPONENT_SIZE = "VASCUTRACE_MIN_COMPONENT_SIZE"

# The product-side reference-oracle backend's identity string. This is never
# this module's own identity; see the module
# docstring's "guarding the GT-oracle confusion" section.
_GT_ORACLE_MODEL_NAME = "deterministic-synthetic-reference"


@dataclass(frozen=True, slots=True)
class InferenceMetadata:
    """Provenance a reviewer (or the E4 product bridge) needs in order to
    know EXACTLY which checkpoint produced a prediction, and under what
    scientific-boundary caveats -- see module docstring.
    """

    checkpoint_path: str
    config_hash: str
    epoch: int
    model_signature: str
    calibration_status: str
    model_name: str
    research_prototype_warning: str = RESEARCH_PROTOTYPE_WARNING


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """One deterministic forward pass's output. ``n_pred_px`` is
    ``int(mask.sum())`` -- redundant with ``mask`` but convenient for a
    caller that only needs the scalar foreground pixel count (e.g. an
    empty-prediction check) without re-touching the array.
    """

    abnormality_score_map: np.ndarray
    mask: np.ndarray
    threshold: float
    runtime_s: float
    n_pred_px: int
    metadata: InferenceMetadata


def load_inference_model(
    checkpoint_path: Path | str, device: str = "cpu"
) -> tuple[torch.nn.Module, InferenceMetadata]:
    """Load ``checkpoint_path`` and construct the eval-mode model it
    describes.

    FAILS LOUD: a missing or corrupt file raises
    :class:`~src.vascutrace.ml.checkpoint.CheckpointError` (propagated
    unmodified from :func:`~src.vascutrace.ml.checkpoint.load_checkpoint`
    -- see that module's own docstring for why ``path.is_file()`` is
    checked before any deserialization is attempted). This function never
    falls back to a different checkpoint, a freshly-initialized model, or
    ``None`` on a missing path -- the caller's ``except`` (or lack of one)
    decides what happens next, not this module.

    Mirrors ``evaluate.py``'s ``evaluate_checkpoint`` model-construction
    sequence exactly: ``build_model(payload.model_config)`` ->
    ``model.load_state_dict(payload.model_state_dict)`` (``strict=True``,
    torch's default -- a state_dict key/shape mismatch raises rather than
    silently loading a partial or renamed architecture) -> ``model.to(
    device)`` -> ``model.eval()``. A checkpoint this project's own
    evaluation harness accepts is therefore guaranteed loadable here too.

    The returned model additionally carries its own freshly-built
    :class:`InferenceMetadata` as the plain attribute ``model.
    inference_metadata`` (``torch.nn.Module.__setattr__`` only intercepts
    ``Parameter``/``Module``/``Tensor`` values -- a dataclass instance
    falls through to ordinary ``object`` attribute storage, so this
    neither registers a buffer/parameter nor changes ``state_dict()``'s
    keys) purely so :func:`run_sample_inference` can recover it from
    ``model`` alone when a caller does not pass ``metadata=`` explicitly.
    The same :class:`InferenceMetadata` is also returned directly, so a
    caller never needs to reach into the model for it.
    """
    checkpoint_path = Path(checkpoint_path)
    # CheckpointError propagates unmodified -- fail loud, never a silent
    # fallback (checkpoint.py's own load_checkpoint already raises before
    # any torch.load deserialization is attempted if the path is missing).
    payload = load_checkpoint(checkpoint_path)

    model = build_model(payload.model_config)
    model.load_state_dict(payload.model_state_dict)  # strict=True (torch default)
    model.to(device)
    model.eval()

    metadata = InferenceMetadata(
        checkpoint_path=str(checkpoint_path),
        config_hash=payload.config_hash,
        epoch=payload.epoch,
        model_signature=model_signature(payload.model_config),
        calibration_status=payload.calibration_status,
        # The checkpoint's own run-directory name (e.g.
        # "siamese_p4b2_deepsup") -- never a hardcoded literal, so this
        # always reflects which checkpoint was actually loaded. Falls back
        # to the file stem only for a checkpoint saved directly at a
        # filesystem root with no parent run directory (not this
        # project's convention, but keeps this total rather than raising).
        model_name=checkpoint_path.parent.name or checkpoint_path.stem,
    )
    assert metadata.model_name != _GT_ORACLE_MODEL_NAME, (
        "a real trained checkpoint's run-directory name collided with the "
        "product's GT-oracle reference-backend identity string -- refusing "
        "to return ambiguous InferenceMetadata"
    )
    model.inference_metadata = metadata  # type: ignore[attr-defined]
    return model, metadata


def predict_abnormality_score(
    model: torch.nn.Module,
    left: torch.Tensor,
    right: torch.Tensor,
    pet_diff: torch.Tensor,
    device: str,
) -> np.ndarray:
    """One sample's forward pass -> ``score`` ``[H, W]`` float32 numpy
    array, MAIN HEAD ONLY.

    Copies ``evaluate.py``'s own ``_run_inference`` (lines ~275-296)
    exactly: ``left``/``right``/``pet_diff`` are the UNBATCHED per-sample
    tensors (``Sample.left_view``/``right_view``/``pet_diff`` shapes,
    ``[2K, H, W]``/``[2K, H, W]``/``[K, H, W]`` -- ``tensor_schema.py``'s
    frozen contract), ``unsqueeze(0)``'d to a batch of 1 HERE (the caller
    passes the plain per-sample tensor, not a pre-batched one).
    ``model(left, right, pet_diff)`` is called with ``return_aux`` left at
    its default ``False`` -- the deep-supervision aux heads (constructed
    when this checkpoint's ``ModelConfig.deep_supervision=True``) are
    never invoked, matching the module docstring's frozen "deep sup is
    train-time only" contract. ``torch.no_grad()`` + ``model.eval()``
    (idempotent if the caller already called it, e.g. via
    :func:`load_inference_model`) keep this a pure, side-effect-free,
    deterministic read of the frozen weights.
    """
    model.eval()
    left_b = left.unsqueeze(0).to(device)
    right_b = right.unsqueeze(0).to(device)
    pet_diff_b = pet_diff.unsqueeze(0).to(device)
    with torch.no_grad():
        # return_aux=False (default) -- main head only, see module docstring.
        logits = model(left_b, right_b, pet_diff_b)
        score = abnormality_score(logits)
    return score.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)


def predict_mask(
    score: np.ndarray,
    threshold: float = DEFAULT_INFERENCE_THRESHOLD,
    min_component_size: int = DEFAULT_MIN_COMPONENT_SIZE,
) -> np.ndarray:
    """Threshold ``score`` at ``threshold`` (frozen default ``0.5`` -- do
    not retune, see module docstring) and optionally drop small predicted
    connected components via ``metrics.py``'s own tested
    ``_filter_min_component_size`` (module docstring item 8 there) --
    reused, not reimplemented, so the pixel-count-per-component semantics
    here are IDENTICAL to what ``evaluate.py``'s own
    ``min_component_size`` sweep already measures.
    ``min_component_size=0`` (default) is a true no-op. Returns a strictly
    binary ``{0, 1}`` ``uint8`` array (same ``[H, W]`` shape as ``score``).
    """
    binary = score >= threshold
    filtered = _filter_min_component_size(binary, min_component_size)
    return filtered.astype(np.uint8)


def resolve_operating_point(
    *,
    score_threshold: float | None = None,
    min_component_size: int | None = None,
) -> tuple[float, int]:
    """Resolve the ``(threshold, min_component_size)`` operating point a
    caller should pass to :func:`predict_mask`, per the module docstring's
    "Operator-controlled operating point" section. This is a PURE
    function apart from reading ``os.environ`` -- it never writes an env
    var, never mutates global state, and never touches the filesystem.

    Precedence (highest to lowest):
      1. an explicit, non-``None`` keyword argument here,
      2. the corresponding environment variable
         (:data:`ENV_SCORE_THRESHOLD` / :data:`ENV_MIN_COMPONENT_SIZE`),
      3. the frozen banked default (:data:`DEFAULT_INFERENCE_THRESHOLD`
         ``= 0.5`` / ``metrics.py``'s ``DEFAULT_MIN_COMPONENT_SIZE = 0``).

    FAILS LOUD, never silently clamps or falls back to the default: an
    explicit argument or an env value that does not parse as the expected
    type, or that parses but falls outside the valid range (``threshold``
    strictly in ``(0, 1]``; ``min_component_size >= 0``), raises
    :class:`ValueError` with a message naming the offending source (the
    literal env var name, or ``"score_threshold"``/``"min_component_size"``
    for a bad explicit argument) and the invalid value. A caller must never
    catch this and substitute the default -- that would silently mask a
    typo'd env var as "using the banked science operating point," exactly
    what this function exists to prevent.

    Returns ``(0.5, 0)`` -- BYTE-IDENTICAL to this module's pre-Option-A
    behavior -- whenever both env vars are unset and both arguments are
    ``None`` (this project's own default). Any other returned pair is
    EXPLORATORY (module docstring); this function itself does not label or
    write that flag -- the caller (``vascutrace.services.
    run_siamese_detection``) computes ``exploratory = (thr, m) != (0.5,
    0)`` and records it in ``operating_point.json``.
    """
    if score_threshold is None:
        raw_threshold = os.environ.get(ENV_SCORE_THRESHOLD)
        threshold_source = ENV_SCORE_THRESHOLD
        if raw_threshold is None:
            resolved_threshold = DEFAULT_INFERENCE_THRESHOLD
        else:
            resolved_threshold = _parse_score_threshold(raw_threshold, threshold_source)
    else:
        resolved_threshold = _validate_score_threshold(
            score_threshold, "score_threshold"
        )

    if min_component_size is None:
        raw_min_component = os.environ.get(ENV_MIN_COMPONENT_SIZE)
        min_component_source = ENV_MIN_COMPONENT_SIZE
        if raw_min_component is None:
            resolved_min_component = DEFAULT_MIN_COMPONENT_SIZE
        else:
            resolved_min_component = _parse_min_component_size(
                raw_min_component, min_component_source
            )
    else:
        resolved_min_component = _validate_min_component_size(
            min_component_size, "min_component_size"
        )

    return resolved_threshold, resolved_min_component


def _validate_score_threshold(value: float, source: str) -> float:
    """Range-check an already-numeric ``score_threshold``; raises
    :class:`ValueError` naming ``source`` (an env var name or argument
    name) if ``value`` is not strictly in ``(0, 1]``.
    """
    if not (0.0 < value <= 1.0):
        raise ValueError(
            f"{source}={value!r} is invalid: score threshold must be in (0, 1]"
        )
    return float(value)


def _parse_score_threshold(raw: str, source: str) -> float:
    """Parse and range-check an environment value for the score
    threshold. Raises :class:`ValueError` naming ``source`` on either a
    non-float string or an out-of-range float -- never silently falls
    back to the default (module docstring: "FAILS LOUD... never silently
    clamped").
    """
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"{source}={raw!r} is invalid: must parse as a float in (0, 1]"
        ) from exc
    return _validate_score_threshold(value, source)


def _validate_min_component_size(value: int, source: str) -> int:
    """Range-check an already-integral ``min_component_size``; raises
    :class:`ValueError` naming ``source`` if ``value`` is negative or not
    an ``int`` (a caller passing a ``bool`` is accepted -- ``bool`` is an
    ``int`` subclass in Python -- but a ``float`` such as ``1.5`` is
    rejected explicitly rather than silently truncated).
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{source}={value!r} is invalid: min_component_size must be an int >= 0"
        )
    if value < 0:
        raise ValueError(
            f"{source}={value!r} is invalid: min_component_size must be >= 0"
        )
    return value


def _parse_min_component_size(raw: str, source: str) -> int:
    """Parse + range-check an env-var string for the minimum
    connected-component size. ``int(raw)`` already rejects a non-integer
    string such as ``"1.5"`` or ``"abc"`` (raises :class:`ValueError`) --
    this never falls back to ``float`` truncation, matching the module
    docstring's "non-int ... => raise" contract exactly.
    """
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{source}={raw!r} is invalid: must parse as an int >= 0"
        ) from exc
    return _validate_min_component_size(value, source)


def _load_sample(sample_or_npz_path: Sample | Path | str) -> Sample:
    """Accept either an already-loaded :class:`~src.vascutrace.ml.
    dataset.Sample` or a path to a ``sample_*.npz`` file written by this
    project's own ``cache.py``. A path is read via ``cache.py``'s own
    tested ``_load_sample_npz`` (module docstring's reuse note) rather
    than a hand-rolled ``np.load`` parse, so field validation (shape,
    dtype, finiteness, binary target/valid masks) is IDENTICAL to the
    training/evaluation cache path. ``require_source_fraction=False``:
    inference only ever reads ``left_view``/``right_view``/``pet_diff``
    off the returned ``Sample`` -- ``source_fraction`` is training-input
    metadata only (``dataset.py``'s own ``Sample`` docstring) and is never
    touched here.
    """
    if isinstance(sample_or_npz_path, Sample):
        return sample_or_npz_path
    if isinstance(sample_or_npz_path, (str, Path)):
        path = Path(sample_or_npz_path)
        if not path.is_file():
            raise FileNotFoundError(f"sample .npz not found: {path}")
        return _load_sample_npz(path, require_source_fraction=False)
    raise TypeError(
        "sample_or_npz_path must be a Sample, str, or Path, got "
        f"{type(sample_or_npz_path).__name__}"
    )


def run_sample_inference(
    model: torch.nn.Module,
    sample_or_npz_path: Sample | Path | str,
    device: str,
    threshold: float = DEFAULT_INFERENCE_THRESHOLD,
    *,
    min_component_size: int = DEFAULT_MIN_COMPONENT_SIZE,
    metadata: InferenceMetadata | None = None,
) -> InferenceResult:
    """Run one deterministic forward pass end-to-end: accept/load a
    ``Sample`` (see :func:`_load_sample`) -> :func:`predict_abnormality_score`
    -> :func:`predict_mask`, wrapped in an :class:`InferenceResult`.

    ``metadata`` identifies which checkpoint produced this result. If not
    passed explicitly, this function reads ``model.inference_metadata``
    (the attribute :func:`load_inference_model` attaches). A caller that
    constructed ``model`` some other way (bypassing
    :func:`load_inference_model`) must pass ``metadata=`` explicitly, or
    this raises :class:`ValueError` -- it never silently returns an
    unattributed or ``None`` identity (this project's "never label an
    unidentified prediction as the real model" boundary; see the module
    docstring's GT-oracle-confusion note).

    Deterministic: identical ``(model, sample, device, threshold)`` inputs
    always produce a bitwise-identical ``mask``. ``model.eval()`` +
    ``torch.no_grad()`` + this architecture's GroupNorm (no BatchNorm
    running-stat state to drift) + ``dropout_p=0.0`` on the frozen B2
    checkpoint together mean the forward pass has no source of
    run-to-run randomness.
    """
    sample = _load_sample(sample_or_npz_path)
    resolved_metadata = metadata or getattr(model, "inference_metadata", None)
    if resolved_metadata is None:
        raise ValueError(
            "no InferenceMetadata available -- pass metadata= explicitly, "
            "or build `model` via load_inference_model() which attaches it "
            "as model.inference_metadata"
        )

    start = time.perf_counter()
    score = predict_abnormality_score(
        model, sample.left_view, sample.right_view, sample.pet_diff, device
    )
    mask = predict_mask(
        score, threshold=threshold, min_component_size=min_component_size
    )
    runtime_s = time.perf_counter() - start

    return InferenceResult(
        abnormality_score_map=score,
        mask=mask,
        threshold=threshold,
        runtime_s=runtime_s,
        n_pred_px=int(mask.sum()),
        metadata=resolved_metadata,
    )
