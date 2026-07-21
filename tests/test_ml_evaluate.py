"""Generated/offline tests for evaluation operating-point plumbing.

No checkpoint, cache, ``Data/``, CUDA, or network resource is accessed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import src.vascutrace.ml.evaluate as evaluate_module
from src.vascutrace.geometry import RESEARCH_PROTOTYPE_WARNING
from src.vascutrace.ml.evaluate import (
    EvaluationError,
    EvaluationReport,
    NegativeMetrics,
    PositiveMetrics,
    build_parser,
    evaluate_checkpoint,
)
from src.vascutrace.ml.evaluate import _process_sample  # noqa: PLC2701


def _empty_report(min_component_size: int) -> EvaluationReport:
    positives = PositiveMetrics(
        n_positive_samples=0,
        mean_dice=float("nan"),
        median_dice=float("nan"),
        std_dice=float("nan"),
        pct_dice_above_0_5=float("nan"),
        mean_iou=float("nan"),
        median_iou=float("nan"),
        n_lesions_total=0,
        n_lesions_detected=0,
        n_lesions_missed=0,
        positive_detection_fraction=float("nan"),
        lesion_precision=float("nan"),
        lesion_f1=float("nan"),
        lesion_f2=float("nan"),
        n_found_for_surface_metrics=0,
        mean_hd95_found_mm=None,
        median_hd95_found_mm=None,
        mean_assd_found_mm=None,
        median_assd_found_mm=None,
    )
    negatives = NegativeMetrics(
        n_negative_samples=0,
        pct_fully_clean=float("nan"),
        mean_fp_components_per_case=float("nan"),
        median_fp_components_per_case=float("nan"),
        total_fp_components=0,
        mean_fp_voxels_per_case=float("nan"),
    )
    return EvaluationReport(
        checkpoint_path="generated.pt",
        cache_dir="generated-cache",
        device="cpu",
        score_threshold=0.5,
        iou_threshold=0.1,
        spacing_mm=(1.0, 1.0),
        n_bootstrap=0,
        min_component_size=min_component_size,
        n_samples=0,
        n_positive=0,
        n_negative=0,
        n_meta_positive_actual_empty=0,
        n_meta_negative_actual_nonempty=0,
        positives=positives,
        negatives=negatives,
        bootstrap={},
        selection_metric_name="dice_x_clean",
        selection_metric_value=float("nan"),
        selection_metric_formula="generated fixture",
        model_signature="generated",
        checkpoint_tensor_schema_version="generated",
        checkpoint_best_val_metric=None,
        checkpoint_best_val_metric_name="dice_x_clean",
        warnings=(),
        research_prototype_warning=RESEARCH_PROTOTYPE_WARNING,
        generated_at="2026-07-19T00:00:00+00:00",
    )


class TestMinComponentPlumbing:
    def test_cutoff_is_consistent_across_overlap_detection_and_surface_metrics(
        self,
    ) -> None:
        target = np.zeros((12, 12), dtype=np.float32)
        target[1:5, 1:5] = 1.0  # 16-pixel simulated-source component
        score_map = target.copy()
        score_map[8:10, 8:10] = 1.0  # 4-pixel spurious component
        valid = np.ones_like(target)
        meta = {"subject": "generated-subject", "positive": True}

        unfiltered = _process_sample(
            score_map,
            target,
            valid,
            meta,
            score_threshold=0.5,
            iou_threshold=0.1,
            spacing_mm=(1.0, 1.0),
            min_component_size=0,
        )
        filtered = _process_sample(
            score_map,
            target,
            valid,
            meta,
            score_threshold=0.5,
            iou_threshold=0.1,
            spacing_mm=(1.0, 1.0),
            min_component_size=5,
        )

        assert unfiltered.dice is not None and unfiltered.dice < 1.0
        assert unfiltered.iou is not None and unfiltered.iou < 1.0
        assert unfiltered.lesion_tp == 1
        assert unfiltered.lesion_fp == 1
        assert filtered.dice == pytest.approx(1.0)
        assert filtered.iou == pytest.approx(1.0)
        assert filtered.detected is True
        assert filtered.lesion_tp == 1
        assert filtered.lesion_fp == 0
        assert filtered.lesion_fn == 0
        assert filtered.hd95_mm == pytest.approx(0.0)
        assert filtered.assd_mm == pytest.approx(0.0)

    def test_negative_component_count_and_voxels_use_same_filtered_mask(self) -> None:
        score_map = np.zeros((12, 12), dtype=np.float32)
        score_map[1:5, 1:5] = 1.0  # 16 pixels
        score_map[8:10, 8:10] = 1.0  # 4 pixels
        target = np.zeros_like(score_map)
        valid = np.ones_like(score_map)
        meta = {"subject": "generated-subject", "positive": False}

        unfiltered = _process_sample(
            score_map,
            target,
            valid,
            meta,
            score_threshold=0.5,
            iou_threshold=0.1,
            spacing_mm=(1.0, 1.0),
            min_component_size=0,
        )
        filtered = _process_sample(
            score_map,
            target,
            valid,
            meta,
            score_threshold=0.5,
            iou_threshold=0.1,
            spacing_mm=(1.0, 1.0),
            min_component_size=5,
        )

        assert (unfiltered.fp_components, unfiltered.fp_voxels) == (2, 20)
        assert (filtered.fp_components, filtered.fp_voxels) == (1, 16)


class TestEvaluateFilterValidation:
    def test_evaluator_explicitly_enables_intent_mismatch_audit_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        observed: dict[str, object] = {}

        class StopAfterDatasetConstruction(Exception):
            pass

        class CompatiblePayload:
            tensor_schema_version = evaluate_module.TENSOR_SCHEMA_VERSION

        def dataset_spy(cache_dir: Path, **kwargs: object) -> None:
            observed["cache_dir"] = cache_dir
            observed.update(kwargs)
            raise StopAfterDatasetConstruction

        monkeypatch.setattr(
            evaluate_module, "load_checkpoint", lambda path: CompatiblePayload()
        )
        monkeypatch.setattr(evaluate_module, "CachedSampleDataset", dataset_spy)

        with pytest.raises(StopAfterDatasetConstruction):
            evaluate_checkpoint(tmp_path / "generated.pt", tmp_path / "generated-cache")

        assert observed == {
            "cache_dir": tmp_path / "generated-cache",
            "enforce_manifest_class_content": False,
        }

    @pytest.mark.parametrize("value", [-1, -0.5, 1.5, True])
    def test_invalid_cutoff_fails_before_checkpoint_or_cache_access(
        self, tmp_path: Path, value: object
    ) -> None:
        with pytest.raises(EvaluationError, match="min_component_size"):
            evaluate_checkpoint(
                tmp_path / "missing.pt",
                tmp_path / "missing-cache",
                min_component_size=value,  # type: ignore[arg-type]
            )

    def test_report_and_parser_preserve_accepted_cutoff(self) -> None:
        report = _empty_report(7)
        payload = report.to_dict()
        markdown = report.to_markdown()
        assert payload["min_component_size"] == 7
        assert "simulated-source component detection" in payload["metric_scope"]
        assert "min_component_size=7" in markdown
        assert "Positive simulated-source samples" in markdown
        assert "Synthetic-source component detection" in markdown
        assert "Per-lesion" not in markdown
        assert "sensitivity=" not in markdown
        args = build_parser().parse_args(
            [
                "--checkpoint",
                "generated.pt",
                "--cache",
                "generated",
                "--min-component-size",
                "7",
            ]
        )
        assert args.min_component_size == 7
        with pytest.raises(SystemExit):
            build_parser().parse_args(
                [
                    "--checkpoint",
                    "generated.pt",
                    "--cache",
                    "generated",
                    "--min-component-size",
                    "1.5",
                ]
            )

    def test_cli_forwards_cutoff_unchanged(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        observed: dict[str, int] = {}

        def fake_evaluate(
            checkpoint_path: Path,
            cache_dir: Path,
            **kwargs: object,
        ) -> EvaluationReport:
            observed["cutoff"] = int(kwargs["min_component_size"])
            return _empty_report(observed["cutoff"])

        monkeypatch.setattr(evaluate_module, "evaluate_checkpoint", fake_evaluate)
        assert (
            evaluate_module.main(
                [
                    "--checkpoint",
                    "generated.pt",
                    "--cache",
                    "generated-cache",
                    "--min-component-size",
                    "9",
                    "--n-bootstrap",
                    "0",
                ]
            )
            == 0
        )
        assert observed["cutoff"] == 9
        assert "min_component_size=9" in capsys.readouterr().out
