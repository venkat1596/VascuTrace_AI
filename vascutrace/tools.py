"""Thin, serializable tool wrappers around deterministic services."""

import os
from pathlib import Path
from typing import Any, Callable

from vascutrace.contracts import QuantitativeMeasurements, ResearchReport
from vascutrace.evidence import evidence_store
from vascutrace.experiments import run_detectability_experiment as run_experiment
from vascutrace.report_verifier import verify_report
from vascutrace.services import (
    create_demo_case,
    create_demo_views,
    create_siamese_research_case,
    generate_demo_report,
    run_demo_detection,
    run_siamese_detection,
)

# Product backend switch. "reference" is the default so existing callers
# keep the deterministic synthetic-reference behavior (``model_name ==
# "deterministic-synthetic-reference"``) unless a caller opts in. "siamese"
# runs the REAL banked B2 checkpoint via ``vascutrace.services.
# run_siamese_detection`` and FAILS LOUD rather than ever silently falling
# back to "reference" -- see ``_resolve_detection_backend``.
_DETECTION_BACKENDS = frozenset({"siamese", "reference"})
_DEFAULT_DETECTION_BACKEND = "reference"


def _resolve_detection_backend() -> str:
    """Read ``VASCUTRACE_DETECTION_BACKEND`` (default ``"reference"``).

    An unrecognized value raises :class:`ValueError` and never silently
    defaults to ``"reference"``, so a mistyped value cannot masquerade as the
    ground-truth path.
    """
    backend = os.environ.get("VASCUTRACE_DETECTION_BACKEND", _DEFAULT_DETECTION_BACKEND)
    if backend not in _DETECTION_BACKENDS:
        raise ValueError(
            f"VASCUTRACE_DETECTION_BACKEND={backend!r} is not one of "
            f"{sorted(_DETECTION_BACKENDS)!r}"
        )
    return backend


def load_case(output_root: str = "outputs/demo") -> dict[str, Any]:
    """Backend-aware case loader. The "siamese" backend
    materializes a real p6-cache research case
    (:func:`~vascutrace.services.create_siamese_research_case`); the
    default "reference" backend keeps the legacy synthetic demo case
    (:func:`~vascutrace.services.create_demo_case`) unchanged.
    """
    backend = _resolve_detection_backend()
    if backend == "siamese":
        case_dir = create_siamese_research_case(output_root)
    else:
        case_dir = create_demo_case(output_root)
    return {"case_id": case_dir.name, "case_dir": str(case_dir), "synthetic": True}


def run_vascular_detection(case_dir: str) -> dict[str, Any]:
    """Keep the tool name and signature stable for the MCP schema.
    Routes to the real Siamese backend or the legacy GT-oracle reference
    backend per ``VASCUTRACE_DETECTION_BACKEND`` -- never silently mixes
    the two (the "siamese" branch never falls back to
    ``run_demo_detection`` on failure; it raises).
    """
    backend = _resolve_detection_backend()
    if backend == "siamese":
        return run_siamese_detection(case_dir).model_dump(mode="json")
    return run_demo_detection(case_dir).model_dump(mode="json")


def calculate_pet_metrics(metrics_path: str) -> dict[str, Any]:
    return QuantitativeMeasurements.model_validate_json(
        Path(metrics_path).read_text(encoding="utf-8")
    ).model_dump(mode="json")


def create_overlay(case_dir: str) -> dict[str, str]:
    return {name: str(path) for name, path in create_demo_views(case_dir).items()}


# GenAI report backend switch (spec sec. 8-9). "template" is the DEFAULT so
# every existing test/caller keeps the deterministic templated report and CI
# needs no network. "llm" uses gpt-5-mini to write the interpretation grounded
# in Qwen-RAG evidence, with all NUMBERS copied verbatim from the deterministic
# measurements; if the LLM/Qwen stack is unavailable it degrades to "template"
# (never fails the pipeline). An unrecognized value raises (no silent default).
_REPORT_BACKENDS = frozenset({"template", "llm"})
_DEFAULT_REPORT_BACKEND = "template"
_EVIDENCE_BACKENDS = frozenset({"keyword", "rag"})
_DEFAULT_EVIDENCE_BACKEND = "keyword"


def _resolve_report_backend() -> str:
    backend = os.environ.get("VASCUTRACE_REPORT_BACKEND", _DEFAULT_REPORT_BACKEND)
    if backend not in _REPORT_BACKENDS:
        raise ValueError(
            f"VASCUTRACE_REPORT_BACKEND={backend!r} is not one of {sorted(_REPORT_BACKENDS)!r}"
        )
    return backend


def _resolve_evidence_backend() -> str:
    backend = os.environ.get("VASCUTRACE_EVIDENCE_BACKEND", _DEFAULT_EVIDENCE_BACKEND)
    if backend not in _EVIDENCE_BACKENDS:
        raise ValueError(
            f"VASCUTRACE_EVIDENCE_BACKEND={backend!r} is not one of "
            f"{sorted(_EVIDENCE_BACKENDS)!r}"
        )
    return backend


def generate_research_report(model_output: dict[str, Any]) -> dict[str, Any]:
    from vascutrace.contracts import ModelOutput

    output = ModelOutput.model_validate(model_output)
    if _resolve_report_backend() == "llm":
        # lazy import: only pull the heavy genai stack when llm mode is selected
        from src.vascutrace.genai.report_agent import safe_generate_report

        report, _grounding, _backend = safe_generate_report(output)
        # returns a clean ResearchReport dict (grounding is surfaced by the
        # orchestrator payload, never mixed into the schema-strict report)
        return report.model_dump(mode="json")
    return generate_demo_report(output).model_dump(mode="json")


def assess_report_grounding(report: dict[str, Any]) -> dict[str, Any]:
    """Grounding metric (spec sec. 12): supported claims / total, from the report's
    interpretation vs its cited evidence. Loads Qwen only when called (llm path)."""
    from src.vascutrace.genai.report_agent import compute_grounding
    from vascutrace.contracts import EvidenceReference

    rep = ResearchReport.model_validate(report)
    evidence = [EvidenceReference.model_validate(e.model_dump()) for e in rep.evidence]
    return {"grounding_score": compute_grounding(rep.interpretation, evidence)}


def verify_research_report(
    report: dict[str, Any], metrics: dict[str, Any], expected_laterality: str
) -> dict[str, Any]:
    result = verify_report(
        ResearchReport.model_validate(report),
        QuantitativeMeasurements.model_validate(metrics),
        expected_laterality=expected_laterality,
    )
    return {
        "accepted": result.accepted,
        "issues": [issue.__dict__ for issue in result.issues],
    }


def retrieve_research_evidence(query: str, top_k: int = 3) -> dict[str, Any]:
    """Evidence Agent. Default "keyword" backend = deterministic in-repo store
    (offline, CI-safe). "rag" backend = Qwen3-Embedding retrieval + Qwen3-Reranker
    over the chunked literature/research corpus (spec sec. 9).
    """
    if _resolve_evidence_backend() == "rag":
        # lazy import: only pull the heavy genai stack when rag mode is selected
        from src.vascutrace.genai.rag import RagIndex, RagRetriever
        from vascutrace.contracts import EvidenceReference
        from vascutrace.evidence import EvidenceResponse, is_cache_eligible

        retriever = RagRetriever(RagIndex.load())
        hits = retriever.retrieve_and_rerank(
            query, pool=max(top_k * 4, 20), top_m=top_k
        )
        # Conform to the SAME EvidenceResponse contract as the keyword backend so
        # every consumer (orchestrator, dashboard, MCP server) is backend-agnostic.
        references = [
            EvidenceReference(
                citation_id=h.chunk.chunk_id,
                title=h.chunk.title,
                source_url=h.chunk.source,
                supporting_text=h.chunk.text[:500],
            )
            for h in hits
        ]
        return EvidenceResponse(
            query=query,
            evidence=references,
            cache_eligible=is_cache_eligible(query),
            cache_hit=False,
        ).model_dump(mode="json")
    return evidence_store.search(query, top_k).model_dump(mode="json")


def run_detectability_experiment(
    output_root: str = "outputs/experiments",
) -> dict[str, Any]:
    return run_experiment(output_root).model_dump(mode="json")


TOOL_REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {
    "load_case": load_case,
    "run_vascular_detection": run_vascular_detection,
    "calculate_pet_metrics": calculate_pet_metrics,
    "create_overlay": create_overlay,
    "generate_research_report": generate_research_report,
    "assess_report_grounding": assess_report_grounding,
    "verify_report": verify_research_report,
    "retrieve_research_evidence": retrieve_research_evidence,
    "run_detectability_experiment": run_detectability_experiment,
}
