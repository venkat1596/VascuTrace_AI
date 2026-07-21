"""Tests for the exploratory operator-controlled operating-point knobs:

  * ``src.vascutrace.ml.infer.resolve_operating_point`` -- the pure
    ``(score_threshold, min_component_size)`` resolution helper (explicit
    kwarg > ``VASCUTRACE_SCORE_THRESHOLD``/``VASCUTRACE_MIN_COMPONENT_SIZE``
    env > frozen default ``(0.5, 0)``; fail-loud on invalid input).
  * ``vascutrace.services.run_siamese_detection`` -- the only product call
    site that threads the resolved point into the real Siamese backend and
    records it, unmodified, in ``case_dir / "operating_point.json"``.
  * ``vascutrace.services.run_demo_detection`` -- the reference/GT-oracle
    backend, which must never read either env var for its own identity or
    output.

Implemented from the public environment contract and the module docstrings
of ``infer.py`` and ``services.py``, not by copying ``vascutrace/services.py``'s or
``src/vascutrace/ml/infer.py``'s control flow to decide what to assert. In
particular, T9's "predicted mask is identical to calling with explicit
(0.5, 0)" is checked by independently recomputing the mask through
``infer.py``'s own public ``predict_abnormality_score``/``predict_mask`` API
(this repo's already-reviewed frozen inference path, ``tests/
test_ml_infer.py``), not by re-deriving ``services.py``'s internal
sequence.

T1-T7 need no local artifact and always run. T8/T9 exercise the real
banked B2 checkpoint (``runs/siamese_p4b2_deepsup/best_constrained_iou.pt``)
and a real p6-cache validation sample -- both gitignored, locally-generated
artifacts -- and runtime-``pytest.skip`` when either is absent, following
this repo's own established convention (see ``tests/test_ml_infer.py``/
``tests/test_product_backends.py`` module docstrings: a locally-generated
checkpoint/cache directory is neither the ``local_data`` nor the ``gpu``
marker's scope, and CI's plain ``uv run pytest -q`` applies no ``-m``
filter, so a marker alone would not keep a clean CI runner green).

CRITICAL: every test that touches ``VASCUTRACE_SCORE_THRESHOLD``/
``VASCUTRACE_MIN_COMPONENT_SIZE`` uses the module-scoped ``clean_op_env``
fixture (``monkeypatch``), which deletes both vars before the test body
runs and relies on ``monkeypatch``'s own automatic teardown restoration --
a leaked ``VASCUTRACE_*`` env var could otherwise turn an unrelated later
CI test red.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.vascutrace.ml.infer import (
    ENV_MIN_COMPONENT_SIZE,
    ENV_SCORE_THRESHOLD,
    FROZEN_CHECKPOINT_PATH,
    load_inference_model,
    predict_mask,
    predict_abnormality_score,
    resolve_operating_point,
)
from vascutrace.services import (
    create_demo_case,
    create_siamese_research_case,
    run_demo_detection,
    run_siamese_detection,
)

_ROOT = Path(__file__).resolve().parents[1]
_CHECKPOINT_PATH = _ROOT / FROZEN_CHECKPOINT_PATH

_GT_ORACLE_MODEL_NAME = "deterministic-synthetic-reference"
_SIAMESE_MODEL_NAME = "siamese_p4b2_deepsup"

_EXPLORATORY_SCORE_THRESHOLD_ENV = "0.7"
_EXPLORATORY_MIN_COMPONENT_SIZE_ENV = "10"


@pytest.fixture
def clean_op_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Guarantee ``VASCUTRACE_SCORE_THRESHOLD``/``VASCUTRACE_MIN_COMPONENT_SIZE``
    start unset for every test in this module, and rely on ``monkeypatch``'s
    own automatic teardown restoration so neither var can ever leak into a
    later, unrelated test.
    """
    monkeypatch.delenv(ENV_SCORE_THRESHOLD, raising=False)
    monkeypatch.delenv(ENV_MIN_COMPONENT_SIZE, raising=False)
    return monkeypatch


def _require_siamese_case(tmp_path: Path) -> Path:
    """Materialize one real siamese research case, runtime-``pytest.skip``-
    ing when the banked checkpoint or the p6-cache validation sample it
    depends on is not available locally (this repo's own convention, see
    module docstring). ``create_siamese_research_case`` itself already
    FAILS LOUD with ``FileNotFoundError`` on a missing sample (its own
    docstring); that documented exception is caught here purely to convert
    "artifact absent on this machine" into a skip rather than a failure.
    """
    if not _CHECKPOINT_PATH.is_file():
        pytest.skip(f"banked siamese checkpoint missing: {_CHECKPOINT_PATH}")
    try:
        return create_siamese_research_case(output_root=tmp_path)
    except FileNotFoundError as exc:
        pytest.skip(f"p6-cache validation sample missing: {exc}")


# ---------------------------------------------------------------------------
# T1: unset env -> frozen default (0.5, 0).
# ---------------------------------------------------------------------------


class TestT1DefaultWhenEnvUnset:
    def test_unset_env_resolves_to_frozen_default(
        self, clean_op_env: pytest.MonkeyPatch
    ) -> None:
        assert resolve_operating_point() == (0.5, 0)


# ---------------------------------------------------------------------------
# T2/T3: a single env var honored, the other stays at its default.
# ---------------------------------------------------------------------------


class TestT2ProbThresholdEnvHonored:
    def test_score_threshold_env_only(self, clean_op_env: pytest.MonkeyPatch) -> None:
        clean_op_env.setenv(ENV_SCORE_THRESHOLD, "0.7")
        assert resolve_operating_point() == (0.7, 0)


class TestT3MinComponentSizeEnvHonored:
    def test_min_component_size_env_only(
        self, clean_op_env: pytest.MonkeyPatch
    ) -> None:
        clean_op_env.setenv(ENV_MIN_COMPONENT_SIZE, "10")
        assert resolve_operating_point() == (0.5, 10)


# ---------------------------------------------------------------------------
# T4: invalid score-threshold env values fail loud.
# ---------------------------------------------------------------------------


class TestT4InvalidProbThresholdEnvRaises:
    @pytest.mark.parametrize(
        "raw_value",
        ["1.5", "-0.1", "abc", "0"],
        ids=["above_range", "below_range", "unparsable", "zero_not_in_range"],
    )
    def test_invalid_score_threshold_env_raises_value_error(
        self, clean_op_env: pytest.MonkeyPatch, raw_value: str
    ) -> None:
        clean_op_env.setenv(ENV_SCORE_THRESHOLD, raw_value)
        with pytest.raises(ValueError):
            resolve_operating_point()


# ---------------------------------------------------------------------------
# T5: invalid min-component-size env values fail loud.
# ---------------------------------------------------------------------------


class TestT5InvalidMinComponentSizeEnvRaises:
    @pytest.mark.parametrize(
        "raw_value",
        ["-1", "1.5", "abc"],
        ids=["negative", "non_integer", "unparsable"],
    )
    def test_invalid_min_component_size_env_raises_value_error(
        self, clean_op_env: pytest.MonkeyPatch, raw_value: str
    ) -> None:
        clean_op_env.setenv(ENV_MIN_COMPONENT_SIZE, raw_value)
        with pytest.raises(ValueError):
            resolve_operating_point()


# ---------------------------------------------------------------------------
# T6: explicit kwargs win over env, including an invalid explicit kwarg.
# ---------------------------------------------------------------------------


class TestT6ExplicitKwargsOverrideEnv:
    def test_explicit_kwargs_override_env(
        self, clean_op_env: pytest.MonkeyPatch
    ) -> None:
        clean_op_env.setenv(ENV_SCORE_THRESHOLD, "0.7")
        clean_op_env.setenv(ENV_MIN_COMPONENT_SIZE, "10")
        assert resolve_operating_point(score_threshold=0.5, min_component_size=0) == (
            0.5,
            0,
        )

    def test_invalid_explicit_score_threshold_raises(
        self, clean_op_env: pytest.MonkeyPatch
    ) -> None:
        with pytest.raises(ValueError):
            resolve_operating_point(score_threshold=1.5)

    def test_invalid_explicit_min_component_size_raises(
        self, clean_op_env: pytest.MonkeyPatch
    ) -> None:
        with pytest.raises(ValueError):
            resolve_operating_point(min_component_size=-1)


# ---------------------------------------------------------------------------
# T7: reference backend ignores the env vars for its own identity/output.
# ---------------------------------------------------------------------------


class TestT7ReferenceBackendIgnoresEnv:
    def test_reference_backend_untouched_by_env(
        self, tmp_path: Path, clean_op_env: pytest.MonkeyPatch
    ) -> None:
        clean_op_env.setenv(ENV_SCORE_THRESHOLD, _EXPLORATORY_SCORE_THRESHOLD_ENV)
        clean_op_env.setenv(ENV_MIN_COMPONENT_SIZE, _EXPLORATORY_MIN_COMPONENT_SIZE_ENV)

        case_dir = create_demo_case(output_root=tmp_path)
        output = run_demo_detection(case_dir)

        assert output.model_name == _GT_ORACLE_MODEL_NAME
        assert not (case_dir / "operating_point.json").exists()


# ---------------------------------------------------------------------------
# T8: siamese identity is not polluted by thr/m; operating_point.json
# records the exploratory point when env is set.
# ---------------------------------------------------------------------------


class TestT8SiameseBackendHonorsEnvIdentityUnpolluted:
    def test_siamese_identity_stable_and_operating_point_exploratory(
        self, tmp_path: Path, clean_op_env: pytest.MonkeyPatch
    ) -> None:
        clean_op_env.setenv(ENV_SCORE_THRESHOLD, _EXPLORATORY_SCORE_THRESHOLD_ENV)
        clean_op_env.setenv(ENV_MIN_COMPONENT_SIZE, _EXPLORATORY_MIN_COMPONENT_SIZE_ENV)
        case_dir = _require_siamese_case(tmp_path)

        output = run_siamese_detection(case_dir)

        assert output.model_name == _SIAMESE_MODEL_NAME

        op_json = json.loads((case_dir / "operating_point.json").read_text())
        assert op_json == {
            "score_threshold": 0.7,
            "min_component_size": 10,
            "exploratory": True,
        }


# ---------------------------------------------------------------------------
# T9: env unset -> behavior-stable vs the pre-A default on a fixed sample.
# ---------------------------------------------------------------------------


class TestT9SiameseBehaviorStableAtDefault:
    def test_default_operating_point_json_and_mask_match_explicit_call(
        self, tmp_path: Path, clean_op_env: pytest.MonkeyPatch
    ) -> None:
        case_dir = _require_siamese_case(tmp_path)

        output = run_siamese_detection(case_dir)
        assert output.model_name == _SIAMESE_MODEL_NAME

        op_json = json.loads((case_dir / "operating_point.json").read_text())
        assert op_json == {
            "score_threshold": 0.5,
            "min_component_size": 0,
            "exploratory": False,
        }

        predicted_mask = np.load(case_dir / "predicted_mask.npy")

        # Independently recompute the mask through infer.py's own public,
        # already-reviewed API with an EXPLICIT (0.5, 0) operating point,
        # and assert bitwise identity with what the env-unset product path
        # wrote -- this is the "identical to calling with explicit (0.5,
        # 0)" requirement from A.5's T9.
        model, _ = load_inference_model(_CHECKPOINT_PATH, device="cpu")
        left_view = np.load(case_dir / "left_view.npy")
        right_view = np.load(case_dir / "right_view.npy")
        pet_diff = np.load(case_dir / "pet_diff.npy")
        score = predict_abnormality_score(
            model,
            torch.from_numpy(left_view).float(),
            torch.from_numpy(right_view).float(),
            torch.from_numpy(pet_diff).float(),
            "cpu",
        )
        explicit_mask = predict_mask(score, threshold=0.5, min_component_size=0)

        assert np.array_equal(predicted_mask, explicit_mask)
