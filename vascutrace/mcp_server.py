"""Official MCP transport for VascuTrace deterministic research tools."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from vascutrace import tools as services

mcp = FastMCP(
    "VascuTrace Imaging",
    instructions=(
        "Synthetic and research-only vascular PET/CT tools. Outputs are not for "
        "clinical use. Quantitative values come from deterministic services."
    ),
)


def _output_root() -> Path:
    return Path(os.environ.get("VASCUTRACE_OUTPUT_ROOT", "outputs/demo")).resolve()


def _safe_artifact_path(value: str) -> str:
    """Allow MCP access only to artifacts within the configured output root."""

    path = Path(value).resolve()
    if not path.is_relative_to(_output_root()):
        raise ValueError("Artifact path is outside VASCUTRACE_OUTPUT_ROOT")
    return str(path)


@mcp.tool()
def load_case() -> dict[str, Any]:
    """Create and load the deterministic synthetic research demonstration case."""

    return services.load_case(str(_output_root()))


@mcp.tool()
def run_vascular_detection(case_dir: str) -> dict[str, Any]:
    """Run deterministic reference detection for a loaded synthetic case."""

    return services.run_vascular_detection(_safe_artifact_path(case_dir))


@mcp.tool()
def calculate_pet_metrics(metrics_path: str) -> dict[str, Any]:
    """Read schema-validated deterministic PET measurements from an artifact."""

    return services.calculate_pet_metrics(_safe_artifact_path(metrics_path))


@mcp.tool()
def create_overlay(case_dir: str) -> dict[str, str]:
    """Create a PNG overlay for a loaded synthetic case."""

    return services.create_overlay(_safe_artifact_path(case_dir))


@mcp.tool()
def generate_research_report(model_output: dict[str, Any]) -> dict[str, Any]:
    """Generate a structured research-only report from model output."""

    model_output = dict(model_output)
    for field in ("mask_path", "metrics_path", "overlay_path"):
        model_output[field] = _safe_artifact_path(model_output[field])
    return services.generate_research_report(model_output)


@mcp.tool()
def verify_report(
    report: dict[str, Any], metrics: dict[str, Any], expected_laterality: str
) -> dict[str, Any]:
    """Verify numeric fidelity, laterality, synthetic status, and claim safety."""

    return services.verify_research_report(report, metrics, expected_laterality)


@mcp.tool()
def retrieve_research_evidence(query: str, top_k: int = 3) -> dict[str, Any]:
    """Retrieve research evidence with provenance and case-safe semantic caching."""

    return services.retrieve_research_evidence(query, top_k)


@mcp.tool()
def run_detectability_experiment() -> dict[str, Any]:
    """Run the deterministic synthetic detectability reference sweep."""

    return services.run_detectability_experiment(str(_output_root() / "experiments"))


def main() -> None:
    """Run the local MCP server over standard input/output."""

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
