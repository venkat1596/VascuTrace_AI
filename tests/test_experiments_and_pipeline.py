import json
from pathlib import Path

from vascutrace.experiments import run_detectability_experiment
from vascutrace.orchestrator import run_experiment_request
from vascutrace.pipeline import run_complete_case


def test_detectability_sweep_exports_reference_results(tmp_path: Path) -> None:
    summary = run_detectability_experiment(
        tmp_path,
        radii_mm=[2.0, 6.0],
        uptake_multipliers=[1.2, 2.0],
        blur_fwhm_mm=[5.0],
        source_offset_mm=[0.0, 4.0],
        seeds=[182],
    )
    assert len(summary.rows) == 8
    assert summary.results_csv_path.is_file()
    assert summary.results_json_path.is_file()
    assert all(0 <= row.pet_only.dice <= 1 for row in summary.rows)
    assert all(
        0 <= row.pet_ct_non_siamese.positive_voxel_recall <= 1 for row in summary.rows
    )


def test_experiment_orchestrator_is_traced(tmp_path: Path) -> None:
    result = run_experiment_request(str(tmp_path))
    assert result.route == "experiment"
    assert result.trace == ["run_detectability_experiment"]
    assert len(result.payload["rows"]) == 144


def test_one_command_pipeline_writes_verified_manifest(tmp_path: Path) -> None:
    manifest_path = run_complete_case(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    case_dir = manifest_path.parent

    assert manifest["verification"]["accepted"]
    assert manifest["research_only"] is True
    names = {artifact["name"] for artifact in manifest["artifacts"]}
    assert {
        "model_output.json",
        "metrics.json",
        "report.json",
        "report.md",
        "trace.json",
    } <= names
    assert all(len(artifact["sha256"]) == 64 for artifact in manifest["artifacts"])
    assert (case_dir / "report.md").is_file()
