"""Explicit single-orchestrator MVP with auditable routing and tool traces."""

from dataclasses import dataclass, field
from typing import Any, Literal

from vascutrace.tools import TOOL_REGISTRY

Route = Literal["imaging", "evidence", "experiment", "report"]


@dataclass
class OrchestrationResult:
    route: Route
    payload: dict[str, Any]
    trace: list[str] = field(default_factory=list)


def route_request(request: str) -> Route:
    normalized = request.lower()
    if any(word in normalized for word in ("report", "summary", "verify")):
        return "report"
    if any(word in normalized for word in ("paper", "evidence", "literature")):
        return "evidence"
    if any(word in normalized for word in ("experiment", "sweep", "ablation")):
        return "experiment"
    return "imaging"


def run_first_checkpoint(output_root: str = "outputs/demo") -> OrchestrationResult:
    trace: list[str] = []
    loaded = TOOL_REGISTRY["load_case"](output_root=output_root)
    trace.append("load_case")
    output = TOOL_REGISTRY["run_vascular_detection"](case_dir=loaded["case_dir"])
    trace.append("run_vascular_detection")
    metrics = TOOL_REGISTRY["calculate_pet_metrics"](
        metrics_path=output["metrics_path"]
    )
    trace.append("calculate_pet_metrics")
    report = TOOL_REGISTRY["generate_research_report"](model_output=output)
    trace.append("generate_research_report")
    verification = TOOL_REGISTRY["verify_report"](
        report=report, metrics=metrics, expected_laterality=output["laterality"]
    )
    trace.append("verify_report")
    payload: dict[str, Any] = {
        "case": loaded,
        "model_output": output,
        "metrics": metrics,
        "report": report,
        "verification": verification,
        "research_only": True,
    }
    # Grounding metric (spec sec. 12) is surfaced only for the LLM report backend;
    # the default template backend never loads the Qwen embedder (offline/CI-safe).
    import os

    if os.environ.get("VASCUTRACE_REPORT_BACKEND") == "llm" and report.get("evidence"):
        grounding = TOOL_REGISTRY["assess_report_grounding"](report=report)
        payload["grounding_score"] = grounding["grounding_score"]
        trace.append("assess_report_grounding")
    return OrchestrationResult(route="report", payload=payload, trace=trace)


def run_evidence_request(query: str, top_k: int = 3) -> OrchestrationResult:
    evidence = TOOL_REGISTRY["retrieve_research_evidence"](query=query, top_k=top_k)
    return OrchestrationResult(
        route="evidence", payload=evidence, trace=["retrieve_research_evidence"]
    )


def run_experiment_request(
    output_root: str = "outputs/experiments",
) -> OrchestrationResult:
    experiment = TOOL_REGISTRY["run_detectability_experiment"](output_root=output_root)
    return OrchestrationResult(
        route="experiment", payload=experiment, trace=["run_detectability_experiment"]
    )
