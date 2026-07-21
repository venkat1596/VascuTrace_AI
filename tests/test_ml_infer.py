"""Tests for the frozen B2 path in ``src.vascutrace.ml.infer``.

Most of this file exercises the module against the real, already-trained,
banked B2 checkpoint (``runs/siamese_p4b2_deepsup/best_constrained_iou.pt``)
and one real precomputed validation sample (``data/processed/p6_cache_big/
val/sample_0000.npz``) -- both gitignored, locally-generated-artifact paths
(``checkpoint.py``/``cache.py`` module docstrings: never committed). Per
this repo's existing convention for a real, already-completed checkpoint
artifact (``tests/test_ml_deep_supervision.py``'s ``_V6EXP_CHECKPOINT``
pattern), tests that need it runtime-``pytest.skip`` when the artifact is
absent rather than using a dedicated pytest marker -- this repo's own
``local_data``/``gpu`` markers scope raw ``Data/`` access and CUDA
respectively; a locally-generated checkpoint/cache directory is neither.

Only ``TestLoadInferenceModel.test_missing_checkpoint_fails_loud`` (T6) and
``test_load_sample_rejects_unknown_type`` need no local artifact and
always run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.vascutrace.geometry import RESEARCH_PROTOTYPE_WARNING
from src.vascutrace.ml.checkpoint import CheckpointError
from src.vascutrace.ml.infer import (
    DEFAULT_INFERENCE_THRESHOLD,
    FROZEN_CONFIG_HASH,
    InferenceMetadata,
    InferenceResult,
    load_inference_model,
    predict_mask,
    predict_abnormality_score,
    run_sample_inference,
)
from src.vascutrace.ml.infer import _load_sample  # noqa: PLC2701 -- direct unit test
from src.vascutrace.ml.model import build_model

_ROOT = Path(__file__).resolve().parents[1]
_CHECKPOINT = _ROOT / "runs" / "siamese_p4b2_deepsup" / "best_constrained_iou.pt"
_VAL_CACHE_DIR = _ROOT / "data" / "processed" / "p6_cache_big" / "val"
_SAMPLE_NPZ = _VAL_CACHE_DIR / "sample_0000.npz"

_EXPECTED_HW = (144, 80)

# Guard: this string must never equal InferenceMetadata.model_name for a
# real, checkpoint-backed model.
_GT_ORACLE_MODEL_NAME = "deterministic-synthetic-reference"


def _require_real_artifacts() -> None:
    if not _CHECKPOINT.exists():
        pytest.skip(f"real B2 checkpoint artifact missing: {_CHECKPOINT}")
    if not _SAMPLE_NPZ.exists():
        pytest.skip(f"real val cache sample missing: {_SAMPLE_NPZ}")


@pytest.fixture(scope="module")
def loaded_model() -> tuple[torch.nn.Module, InferenceMetadata]:
    _require_real_artifacts()
    return load_inference_model(_CHECKPOINT, device="cpu")


class TestLoadInferenceModel:
    def test_load_b2_does_not_raise(self, loaded_model) -> None:
        """T1: loading the real, banked B2 checkpoint does not raise."""
        model, metadata = loaded_model
        assert model is not None
        assert metadata.config_hash == FROZEN_CONFIG_HASH
        assert metadata.calibration_status == "uncalibrated"
        assert metadata.research_prototype_warning == RESEARCH_PROTOTYPE_WARNING
        assert metadata.epoch >= 0
        assert isinstance(metadata.model_signature, str) and metadata.model_signature

    def test_missing_checkpoint_fails_loud(self) -> None:
        """T6: a missing checkpoint path raises -- never returns GT or None."""
        missing = _ROOT / "runs" / "siamese_p4b2_deepsup" / "does_not_exist.pt"
        with pytest.raises(CheckpointError):
            load_inference_model(missing, device="cpu")

    def test_model_identity_never_equals_gt_oracle_reference(
        self, loaded_model
    ) -> None:
        """T5: metadata identity never collides with the product's
        deterministic-synthetic-reference (GT-oracle) backend name.
        """
        _, metadata = loaded_model
        assert metadata.model_name != _GT_ORACLE_MODEL_NAME
        assert metadata.checkpoint_path != _GT_ORACLE_MODEL_NAME
        assert metadata.model_signature != _GT_ORACLE_MODEL_NAME
        assert metadata.model_name == "siamese_p4b2_deepsup"


class TestPredictProbabilityAndMask:
    def test_shapes_match_sample_hw(self, loaded_model) -> None:
        """T2: predicted score/mask shapes match the sample's (H, W)."""
        model, _ = loaded_model
        sample = _load_sample(_SAMPLE_NPZ)
        score = predict_abnormality_score(
            model, sample.left_view, sample.right_view, sample.pet_diff, "cpu"
        )
        assert score.shape == _EXPECTED_HW
        assert score.dtype == np.float32
        assert bool(np.all(score >= 0.0)) and bool(np.all(score <= 1.0))

        mask = predict_mask(score, threshold=DEFAULT_INFERENCE_THRESHOLD)
        assert mask.shape == _EXPECTED_HW

    def test_threshold_yields_strictly_binary_mask(self, loaded_model) -> None:
        """T4: threshold 0.5 yields a strictly binary {0, 1} mask."""
        model, _ = loaded_model
        sample = _load_sample(_SAMPLE_NPZ)
        score = predict_abnormality_score(
            model, sample.left_view, sample.right_view, sample.pet_diff, "cpu"
        )
        mask = predict_mask(score, threshold=0.5)
        assert mask.dtype == np.uint8
        assert set(np.unique(mask).tolist()) <= {0, 1}

    def test_min_component_size_filter_is_reused_not_reimplemented(
        self, loaded_model
    ) -> None:
        """predict_mask's min_component_size filter matches metrics.py's
        own semantics: a component-size cutoff larger than every predicted
        component's size drives the mask to all-zero.
        """
        model, _ = loaded_model
        sample = _load_sample(_SAMPLE_NPZ)
        score = predict_abnormality_score(
            model, sample.left_view, sample.right_view, sample.pet_diff, "cpu"
        )
        huge_cutoff_mask = predict_mask(
            score, threshold=0.5, min_component_size=10_000_000
        )
        assert int(huge_cutoff_mask.sum()) == 0
        assert huge_cutoff_mask.dtype == np.uint8


class TestRunSampleInference:
    def test_determinism_bitwise_identical_mask(self, loaded_model) -> None:
        """T3: identical input twice gives a bitwise-identical mask."""
        model, _ = loaded_model
        result1 = run_sample_inference(model, _SAMPLE_NPZ, "cpu", threshold=0.5)
        result2 = run_sample_inference(model, _SAMPLE_NPZ, "cpu", threshold=0.5)
        assert np.array_equal(result1.mask, result2.mask)
        assert np.array_equal(
            result1.abnormality_score_map, result2.abnormality_score_map
        )

    def test_result_fields_and_metadata(self, loaded_model) -> None:
        model, metadata = loaded_model
        result = run_sample_inference(model, _SAMPLE_NPZ, "cpu", threshold=0.5)
        assert isinstance(result, InferenceResult)
        assert result.abnormality_score_map.shape == _EXPECTED_HW
        assert result.mask.shape == _EXPECTED_HW
        assert result.threshold == 0.5
        assert result.runtime_s >= 0.0
        assert result.n_pred_px == int(result.mask.sum())
        assert result.metadata == metadata

    def test_accepts_sample_object_directly(self, loaded_model) -> None:
        model, _ = loaded_model
        sample = _load_sample(_SAMPLE_NPZ)
        result = run_sample_inference(model, sample, "cpu", threshold=0.5)
        assert result.mask.shape == _EXPECTED_HW

    def test_missing_metadata_raises_without_model_attribute(self) -> None:
        """A model built WITHOUT load_inference_model (no
        model.inference_metadata attribute) and no explicit metadata=
        fails loud rather than returning an unattributed InferenceResult.
        """
        _require_real_artifacts()
        bare_model = build_model()
        sample = _load_sample(_SAMPLE_NPZ)
        with pytest.raises(ValueError):
            run_sample_inference(bare_model, sample, "cpu")

    def test_explicit_metadata_overrides_model_attribute(self, loaded_model) -> None:
        model, metadata = loaded_model
        override = InferenceMetadata(
            checkpoint_path="custom.pt",
            config_hash="deadbeef",
            epoch=0,
            model_signature="custom-sig",
            calibration_status="uncalibrated",
            model_name="custom-model",
        )
        result = run_sample_inference(
            model, _SAMPLE_NPZ, "cpu", threshold=0.5, metadata=override
        )
        assert result.metadata == override
        assert result.metadata != metadata


class TestLoadSampleHelper:
    def test_rejects_unknown_type(self) -> None:
        with pytest.raises(TypeError):
            _load_sample(12345)  # type: ignore[arg-type]

    def test_missing_npz_path_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            _load_sample(_ROOT / "data" / "processed" / "does_not_exist.npz")
