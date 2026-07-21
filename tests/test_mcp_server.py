from pathlib import Path

import pytest

from vascutrace.mcp_server import _safe_artifact_path, mcp


async def call_structured(name: str, arguments: dict) -> dict:
    _, structured = await mcp.call_tool(name, arguments)
    assert structured is not None
    return structured


@pytest.mark.anyio
async def test_mcp_exposes_expected_tool_schemas() -> None:
    tools = await mcp.list_tools()
    by_name = {tool.name: tool for tool in tools}

    assert set(by_name) == {
        "load_case",
        "run_vascular_detection",
        "calculate_pet_metrics",
        "create_overlay",
        "generate_research_report",
        "verify_report",
        "retrieve_research_evidence",
        "run_detectability_experiment",
    }
    assert by_name["load_case"].inputSchema["properties"] == {}
    assert by_name["run_vascular_detection"].inputSchema["required"] == ["case_dir"]
    assert by_name["verify_report"].inputSchema["required"] == [
        "report",
        "metrics",
        "expected_laterality",
    ]


@pytest.mark.anyio
async def test_mcp_tool_call_runs_checkpoint(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VASCUTRACE_OUTPUT_ROOT", str(tmp_path))

    loaded = await call_structured("load_case", {})
    output = await call_structured(
        "run_vascular_detection", {"case_dir": loaded["case_dir"]}
    )
    metrics = await call_structured(
        "calculate_pet_metrics", {"metrics_path": output["metrics_path"]}
    )
    report = await call_structured("generate_research_report", {"model_output": output})
    verification = await call_structured(
        "verify_report",
        {
            "report": report,
            "metrics": metrics,
            "expected_laterality": output["laterality"],
        },
    )

    assert verification == {"accepted": True, "issues": []}
    assert report["quantitative_measurements"] == metrics


def test_mcp_rejects_paths_outside_artifact_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VASCUTRACE_OUTPUT_ROOT", str(tmp_path / "allowed"))

    with pytest.raises(ValueError, match="outside VASCUTRACE_OUTPUT_ROOT"):
        _safe_artifact_path(str(tmp_path / "other" / "metrics.json"))
