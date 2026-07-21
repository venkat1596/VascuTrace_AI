"""Unit tests for the product detection-backend switch.

Implemented from the public backend contract (env var
``VASCUTRACE_DETECTION_BACKEND`` in ``{"siamese", "reference"}``, default
``"reference"``; ``VASCUTRACE_CHECKPOINT`` override; the legacy reference
backend's ``model_name == "deterministic-synthetic-reference"``/hardcoded
``abnormality_score == 0.91``; the frozen banked checkpoint identity
``siamese_p4b2_deepsup`` / config hash ``824d2a01df0af942``) -- NOT by
reading ``vascutrace/services.py``'s or ``vascutrace/tools.py``'s
implementation to decide what to assert. Where an implementation-level
detail was needed to make an assertion well-formed (e.g. the real
``predicted_mask.npy`` vs ``ground_truth_mask.npy`` array shapes), it was
confirmed empirically by *running* the product path once, never by reading
the source.

These are the safety/fidelity invariants of the whole product:
  1. Default backend is "reference" (legacy GT-oracle behaviour preserved).
  2. Siamese backend reports its real checkpoint identity.
  3. The siamese backend NEVER returns the ground-truth mask as its
     prediction (the single most important test here).
  4. A missing/invalid checkpoint under the siamese backend FAILS LOUD --
     never silently falls back to the reference/GT path.
  5. An unrecognised backend value raises rather than silently defaulting.
  6. The Siamese backend's abnormality score is model-derived, not the legacy
     hardcoded ``0.91`` sentinel.
  7. The MCP tool entry points' signatures are unchanged (schema stability).

Checkpoint/cache-dependent tests (2, 3, 6) runtime-``pytest.skip`` when the
banked checkpoint or p6-cache validation sample is absent, following this
repo's own established convention for a locally-generated, gitignored
artifact (see ``tests/test_ml_infer.py`` module docstring: neither the
``local_data`` marker, which scopes raw ``Data/`` access, nor the ``gpu``
marker, which scopes CUDA, describes a locally-generated checkpoint/cache
directory -- and CI's plain ``uv run pytest -q`` applies no ``-m`` filter,
so a marker alone would not keep a clean CI runner green). Tests 1, 4, 5,
and 7 need no local artifact and always run.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from src.vascutrace.ml.checkpoint import CheckpointError
from vascutrace.services import _siamese_metrics_from_prediction
from vascutrace.tools import (
    _resolve_detection_backend,
    load_case,
    run_vascular_detection,
)

_ROOT = Path(__file__).resolve().parents[1]
_BANKED_CHECKPOINT = _ROOT / "runs" / "siamese_p4b2_deepsup" / "best_constrained_iou.pt"
_VAL_CACHE_SAMPLE = (
    _ROOT / "data" / "processed" / "p6_cache_big" / "val" / "sample_0000.npz"
)

_GT_ORACLE_MODEL_NAME = "deterministic-synthetic-reference"
_LEGACY_HARDCODED_SCORE = 0.91
_EXPECTED_SIAMESE_MODEL_NAME_FRAGMENT = "siamese_p4b2"
_EXPECTED_CONFIG_HASH = "824d2a01df0af942"


def test_empty_prediction_returns_structured_null_measurements() -> None:
    raw_pet = np.ones((4, 4), dtype=np.float32)
    predicted_mask = np.zeros((4, 4), dtype=np.uint8)
    right_view = np.zeros((10, 4, 4), dtype=np.float32)

    metrics = _siamese_metrics_from_prediction(raw_pet, predicted_mask, right_view)

    nullable_fields = {
        "target_suvmax",
        "target_suvmean",
        "contralateral_suvmax",
        "contralateral_suvmean",
        "asymmetry_index",
        "metabolic_volume_ml",
        "longitudinal_extent_mm",
    }
    assert all(getattr(metrics, name) is None for name in nullable_fields)
    assert set(metrics.null_reasons) == nullable_fields
    assert set(metrics.null_reasons.values()) == {"empty_predicted_mask"}


def test_2d_prediction_does_not_invent_physical_or_clipped_suv_values() -> None:
    raw_pet = np.full((4, 4), 2.5, dtype=np.float32)
    predicted_mask = np.zeros((4, 4), dtype=np.uint8)
    predicted_mask[1:3, 1:3] = 1
    right_view = np.ones((10, 4, 4), dtype=np.float32)

    metrics = _siamese_metrics_from_prediction(raw_pet, predicted_mask, right_view)

    assert metrics.target_suvmax == 2.5
    assert metrics.target_suvmean == 2.5
    assert metrics.contralateral_suvmax is None
    assert metrics.contralateral_suvmean is None
    assert metrics.asymmetry_index is None
    assert metrics.metabolic_volume_ml is None
    assert metrics.longitudinal_extent_mm is None
    assert set(metrics.null_reasons) == {
        "contralateral_suvmax",
        "contralateral_suvmean",
        "asymmetry_index",
        "metabolic_volume_ml",
        "longitudinal_extent_mm",
    }


def test_nonfinite_target_suv_returns_null_with_reason() -> None:
    raw_pet = np.ones((3, 3), dtype=np.float32)
    raw_pet[1, 1] = np.nan
    predicted_mask = np.zeros((3, 3), dtype=np.uint8)
    predicted_mask[1, 1] = 1

    metrics = _siamese_metrics_from_prediction(
        raw_pet,
        predicted_mask,
        np.zeros((10, 3, 3), dtype=np.float32),
    )

    assert metrics.target_suvmax is None
    assert metrics.target_suvmean is None
    assert metrics.null_reasons["target_suvmax"] == "nonfinite_target_suv"
    assert metrics.null_reasons["target_suvmean"] == "nonfinite_target_suv"


def _require_siamese_artifacts() -> None:
    """Skip a checkpoint-dependent test when the banked checkpoint or the
    real p6-cache validation sample is not present locally (both are
    gitignored, locally-generated artifacts -- see module docstring).
    """
    if not _BANKED_CHECKPOINT.is_file():
        pytest.skip(f"banked siamese checkpoint missing: {_BANKED_CHECKPOINT}")
    if not _VAL_CACHE_SAMPLE.is_file():
        pytest.skip(f"p6-cache validation sample missing: {_VAL_CACHE_SAMPLE}")


@pytest.fixture
def clean_backend_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Guarantee ``VASCUTRACE_DETECTION_BACKEND``/``VASCUTRACE_CHECKPOINT``
    start unset for every test in this module, and rely on ``monkeypatch``'s
    own automatic teardown restoration so neither var can ever leak into a
    later, unrelated test (a leaked backend selection could turn CI red
    elsewhere).
    """
    monkeypatch.delenv("VASCUTRACE_DETECTION_BACKEND", raising=False)
    monkeypatch.delenv("VASCUTRACE_CHECKPOINT", raising=False)
    return monkeypatch


# ---------------------------------------------------------------------------
# 1. Default backend is "reference" -- legacy behaviour preserved.
# ---------------------------------------------------------------------------


class TestDefaultBackendIsReference:
    def test_no_env_var_set_resolves_to_reference(
        self, clean_backend_env: pytest.MonkeyPatch
    ) -> None:
        assert _resolve_detection_backend() == "reference"

    def test_default_run_vascular_detection_returns_gt_oracle_model_name(
        self, tmp_path: Path, clean_backend_env: pytest.MonkeyPatch
    ) -> None:
        case = load_case(str(tmp_path))
        output = run_vascular_detection(case["case_dir"])
        assert output["model_name"] == _GT_ORACLE_MODEL_NAME


# ---------------------------------------------------------------------------
# 2. Siamese identity -- backend=siamese reports its real checkpoint.
# ---------------------------------------------------------------------------


class TestSiameseIdentity:
    def test_siamese_backend_reports_real_checkpoint_identity(
        self, tmp_path: Path, clean_backend_env: pytest.MonkeyPatch
    ) -> None:
        _require_siamese_artifacts()
        clean_backend_env.setenv("VASCUTRACE_DETECTION_BACKEND", "siamese")

        case = load_case(str(tmp_path))
        output = run_vascular_detection(case["case_dir"])

        assert _EXPECTED_SIAMESE_MODEL_NAME_FRAGMENT in output["model_name"]
        assert _EXPECTED_CONFIG_HASH in output["model_version"]


# ---------------------------------------------------------------------------
# 3. NEVER GT-as-siamese -- the single most important invariant here.
# ---------------------------------------------------------------------------


class TestNeverGroundTruthAsSiamese:
    def test_predicted_mask_provenance_and_content_differ_from_ground_truth(
        self, tmp_path: Path, clean_backend_env: pytest.MonkeyPatch
    ) -> None:
        _require_siamese_artifacts()
        clean_backend_env.setenv("VASCUTRACE_DETECTION_BACKEND", "siamese")

        case = load_case(str(tmp_path))
        case_dir = Path(case["case_dir"])
        output = run_vascular_detection(str(case_dir))

        # --- Provenance: the returned mask must be the PREDICTED mask. ---
        mask_path = Path(output["mask_path"])
        assert mask_path.name != "ground_truth_mask.npy"
        assert mask_path.name == "predicted_mask.npy"
        assert output["model_name"] != _GT_ORACLE_MODEL_NAME

        # The siamese case dir must NOT contain a file named
        # ``ground_truth_mask.npy`` at all: that name belongs to the honest
        # reference backend's SERVED output, and reusing it here would invite
        # exactly the GT-mistaken-for-prediction confusion this backend
        # exists to remove. The labelled comparison mask is named
        # ``reference_lesion_mask.npy`` instead.
        assert not (case_dir / "ground_truth_mask.npy").exists()

        # --- Content: predicted array must not equal the labelled reference. ---
        gt_path = case_dir / "reference_lesion_mask.npy"
        assert gt_path.is_file(), (
            "expected a labelled reference lesion mask to exist "
            "alongside the siamese case for this comparison"
        )
        predicted = np.load(mask_path)
        ground_truth = np.squeeze(np.load(gt_path))

        assert predicted.shape == ground_truth.shape, (
            f"predicted mask shape {predicted.shape} and squeezed "
            f"ground-truth shape {ground_truth.shape} disagree -- cannot "
            "compare content"
        )

        predicted_bool = predicted.astype(bool)
        ground_truth_bool = ground_truth.astype(bool)
        arrays_equal = bool(np.array_equal(predicted_bool, ground_truth_bool))

        if arrays_equal:
            # If this specific (deterministic) sample's real prediction
            # ever coincides pixel-for-pixel with its GT reference, the
            # safety invariant is that provenance still distinguishes the
            # two paths -- re-assert that explicitly rather than letting a
            # coincidental match masquerade as failure OR as silent pass.
            pytest.fail(
                "predicted siamese mask is bitwise-identical to its "
                "ground-truth reference on this sample; provenance still "
                f"holds (mask_path={mask_path.name!r}, "
                f"model_name={output['model_name']!r} != {_GT_ORACLE_MODEL_NAME!r}) "
                "but this coincidence should be reviewed by a human, not "
                "silently accepted"
            )
        assert not arrays_equal


# ---------------------------------------------------------------------------
# 4. Fail loud -- missing/invalid checkpoint never falls back to GT/reference.
# ---------------------------------------------------------------------------


class TestFailLoudOnMissingCheckpoint:
    def test_missing_checkpoint_raises_checkpoint_error_and_writes_nothing(
        self, tmp_path: Path, clean_backend_env: pytest.MonkeyPatch
    ) -> None:
        clean_backend_env.setenv("VASCUTRACE_DETECTION_BACKEND", "siamese")
        missing_checkpoint = tmp_path / "does_not_exist.pt"
        clean_backend_env.setenv("VASCUTRACE_CHECKPOINT", str(missing_checkpoint))

        # Deliberately not materialized: a fail-loud checkpoint load must
        # raise before this backend ever touches case inputs.
        fake_case_dir = tmp_path / "fake_case"

        with pytest.raises(CheckpointError):
            run_vascular_detection(str(fake_case_dir))

        # No ModelOutput/product artifact -- and in particular nothing
        # sourced from the GT/reference path -- was ever written.
        assert not fake_case_dir.exists()
        assert not (fake_case_dir / "model_output.json").exists()
        assert not (fake_case_dir / "predicted_mask.npy").exists()
        assert not (fake_case_dir / "ground_truth_mask.npy").exists()
        assert not (fake_case_dir / "reference_lesion_mask.npy").exists()


# ---------------------------------------------------------------------------
# 5. Invalid backend value raises rather than silently defaulting.
# ---------------------------------------------------------------------------


class TestInvalidBackendRaises:
    def test_unrecognised_backend_raises_value_error_directly(
        self, clean_backend_env: pytest.MonkeyPatch
    ) -> None:
        clean_backend_env.setenv("VASCUTRACE_DETECTION_BACKEND", "not_a_real_backend")
        with pytest.raises(ValueError):
            _resolve_detection_backend()

    def test_unrecognised_backend_raises_through_load_case(
        self, tmp_path: Path, clean_backend_env: pytest.MonkeyPatch
    ) -> None:
        clean_backend_env.setenv("VASCUTRACE_DETECTION_BACKEND", "not_a_real_backend")
        with pytest.raises(ValueError):
            load_case(str(tmp_path))

    def test_unrecognised_backend_raises_through_run_vascular_detection(
        self, tmp_path: Path, clean_backend_env: pytest.MonkeyPatch
    ) -> None:
        clean_backend_env.setenv("VASCUTRACE_DETECTION_BACKEND", "not_a_real_backend")
        with pytest.raises(ValueError):
            run_vascular_detection(str(tmp_path / "irrelevant_case"))


# ---------------------------------------------------------------------------
# 6. Model-derived abnormality score, bounded and not the legacy 0.91 sentinel.
# ---------------------------------------------------------------------------


class TestModelDerivedAbnormalityScore:
    def test_siamese_score_is_bounded_and_not_legacy_sentinel(
        self, tmp_path: Path, clean_backend_env: pytest.MonkeyPatch
    ) -> None:
        _require_siamese_artifacts()
        clean_backend_env.setenv("VASCUTRACE_DETECTION_BACKEND", "siamese")

        case = load_case(str(tmp_path))
        output = run_vascular_detection(case["case_dir"])

        score = output["abnormality_score"]
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert score != _LEGACY_HARDCODED_SCORE


# ---------------------------------------------------------------------------
# 7. MCP signature stability -- unchanged tool parameters/schema.
# ---------------------------------------------------------------------------


class TestMcpSignatureStability:
    def test_load_case_signature_unchanged(self) -> None:
        params = inspect.signature(load_case).parameters
        assert list(params) == ["output_root"]
        assert params["output_root"].default == "outputs/demo"
        assert params["output_root"].annotation is str

    def test_run_vascular_detection_signature_unchanged(self) -> None:
        params = inspect.signature(run_vascular_detection).parameters
        assert list(params) == ["case_dir"]
        assert params["case_dir"].default is inspect.Parameter.empty
        assert params["case_dir"].annotation is str
