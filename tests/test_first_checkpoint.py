import json
from pathlib import Path

from vascutrace.orchestrator import route_request, run_first_checkpoint
from vascutrace.tools import TOOL_REGISTRY, create_overlay


def test_first_checkpoint_preserves_metrics_and_safety(tmp_path: Path) -> None:
    result = run_first_checkpoint(str(tmp_path))
    payload = result.payload

    assert payload["verification"] == {"accepted": True, "issues": []}
    assert payload["metrics"] == payload["report"]["quantitative_measurements"]
    assert payload["report"]["case_type"] == "synthetic_research_case"
    assert payload["report"]["research_only_warning"] is True
    assert result.trace == [
        "load_case",
        "run_vascular_detection",
        "calculate_pet_metrics",
        "generate_research_report",
        "verify_report",
    ]

    for key in ("mask_path", "metrics_path", "overlay_path"):
        assert Path(payload["model_output"][key]).is_file()
    json.loads(Path(payload["model_output"]["metrics_path"]).read_text())

    views = create_overlay(payload["case"]["case_dir"])
    assert set(views) == {
        "pet_path",
        "ct_path",
        "fused_path",
        "overlay_path",
        "bilateral_path",
    }
    assert all(
        Path(path).read_bytes().startswith(b"\x89PNG") for path in views.values()
    )


def test_tools_are_thin_and_discoverable() -> None:
    assert set(TOOL_REGISTRY) == {
        "load_case",
        "run_vascular_detection",
        "calculate_pet_metrics",
        "create_overlay",
        "generate_research_report",
        "assess_report_grounding",
        "verify_report",
        "retrieve_research_evidence",
        "run_detectability_experiment",
    }


def test_explicit_routes() -> None:
    assert route_request("create the report") == "report"
    assert route_request("find supporting literature") == "evidence"
    assert route_request("run an ablation experiment") == "experiment"
    assert route_request("detect vascular uptake") == "imaging"
