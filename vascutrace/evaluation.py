"""Executable evaluation suite for agentic/product integration guarantees."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import Field

from vascutrace.contracts import ModelOutput, ResearchReport, StrictModel
from vascutrace.evidence import EvidenceStore
from vascutrace.orchestrator import run_first_checkpoint
from vascutrace.tools import calculate_pet_metrics, verify_research_report


class EvaluationCaseResult(StrictModel):
    name: str
    category: Literal[
        "schema",
        "numeric_fidelity",
        "claim_safety",
        "cache_isolation",
        "service_failure",
        "trace",
    ]
    passed: bool
    detail: str


class EvaluationSummary(StrictModel):
    suite_version: Literal["1.0"] = "1.0"
    cases: list[EvaluationCaseResult]
    passed: int = Field(ge=0)
    failed: int = Field(ge=0)
    all_passed: bool


def _case(name: str, category: str, check: Callable[[], str]) -> EvaluationCaseResult:
    try:
        detail = check()
        return EvaluationCaseResult(
            name=name, category=category, passed=True, detail=detail
        )
    except (
        Exception
    ) as error:  # Evaluation records failures instead of aborting the suite.
        return EvaluationCaseResult(
            name=name,
            category=category,
            passed=False,
            detail=f"{type(error).__name__}: {error}",
        )


def run_evaluation_suite(
    output_root: Path | str = "outputs/evaluation",
) -> EvaluationSummary:
    """Run deterministic positive and negative-path product evaluations."""

    checkpoint = run_first_checkpoint(str(Path(output_root) / "artifacts"))
    payload = checkpoint.payload

    def schema_check() -> str:
        ModelOutput.model_validate(payload["model_output"])
        ResearchReport.model_validate(payload["report"])
        return "Model output and report satisfy strict versioned schemas."

    def numeric_check() -> str:
        if not payload["verification"]["accepted"]:
            raise AssertionError(payload["verification"]["issues"])
        if payload["metrics"] != payload["report"]["quantitative_measurements"]:
            raise AssertionError("Report measurements changed from source metrics.")
        return "All report measurements exactly match deterministic source metrics."

    def unsafe_claim_check() -> str:
        changed = dict(payload["report"])
        changed["interpretation"] = "This scan confirms diagnosis of disease."
        result = verify_research_report(
            changed, payload["metrics"], payload["model_output"]["laterality"]
        )
        if result["accepted"] or not any(
            issue["code"] == "diagnosis" for issue in result["issues"]
        ):
            raise AssertionError("Unsafe diagnostic claim was not rejected.")
        return "Verifier rejects an injected diagnostic claim."

    def cache_check() -> str:
        store = EvidenceStore()
        query = "What is SUVmax for subject 006 in this scan?"
        first, second = store.search(query), store.search(query)
        if first.cache_eligible or second.cache_hit or store.cache_size:
            raise AssertionError("Case-specific query entered the semantic cache.")
        return "Repeated case-specific measurement queries bypass the cache."

    def service_failure_check() -> str:
        missing = Path(output_root) / "missing" / "metrics.json"
        try:
            calculate_pet_metrics(str(missing))
        except FileNotFoundError:
            return "Missing metrics artifact produces an explicit service failure."
        raise AssertionError("Missing metrics artifact was silently accepted.")

    def trace_check() -> str:
        expected = [
            "load_case",
            "run_vascular_detection",
            "calculate_pet_metrics",
            "generate_research_report",
            "verify_report",
        ]
        if checkpoint.trace != expected:
            raise AssertionError(f"Expected {expected!r}, got {checkpoint.trace!r}.")
        return "Trace contains every required checkpoint tool in order."

    cases = [
        _case("strict_contracts", "schema", schema_check),
        _case("measurement_preservation", "numeric_fidelity", numeric_check),
        _case("unsafe_claim_rejection", "claim_safety", unsafe_claim_check),
        _case("case_cache_bypass", "cache_isolation", cache_check),
        _case("missing_artifact_failure", "service_failure", service_failure_check),
        _case("complete_tool_trace", "trace", trace_check),
    ]
    passed = sum(case.passed for case in cases)
    return EvaluationSummary(
        cases=cases,
        passed=passed,
        failed=len(cases) - passed,
        all_passed=passed == len(cases),
    )


def write_evaluation_summary(summary: EvaluationSummary, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    return path
