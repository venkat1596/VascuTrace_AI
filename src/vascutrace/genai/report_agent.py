"""GenAI report agent: gpt-5-mini writes grounded interpretation over verified numbers.

Research prototype. Implements the spec's separation of responsibility (sec. 8-9):
deterministic code measures; the Evidence Agent retrieves cited passages (Qwen RAG);
the reasoning LLM (gpt-5-mini) writes ONLY the interpretation/limitations prose,
grounded in the retrieved evidence. Every numeric field in the report is copied
verbatim from the deterministic measurements -- the LLM never sets a number. The
result is checked by ``report_verifier`` and scored for grounding.

Offline-safe: if the LLM key or Qwen models are unavailable, callers fall back to
the deterministic template report, so CI never needs the network.
"""

from __future__ import annotations

import json
import re

import numpy as np

from src.vascutrace.genai.llm import LLMUnavailableError, VascuTraceLLM
from src.vascutrace.genai.rag import RagIndex, RagRetriever, get_embedder
from vascutrace.contracts import (
    EvidenceReference,
    Finding,
    QualityControl,
    QuantitativeMeasurements,
    ResearchReport,
)

_SENT = re.compile(r"[^.!?]+[.!?]")
_TARGET_REGION = "iliac_proximal_femoral_corridor"

_LATERALITY_PHRASE = {
    "left": "simulated left-sided asymmetric uptake",
    "right": "simulated right-sided asymmetric uptake",
    "bilateral": "simulated bilateral asymmetric uptake",
    "none": "no significant simulated asymmetric uptake",
}


def retrieve_evidence(
    retriever: RagRetriever, query: str, *, pool: int = 20, top_m: int = 4
) -> list[EvidenceReference]:
    """Evidence Agent: Qwen retrieve + rerank -> cited EvidenceReference list."""
    hits = retriever.retrieve_and_rerank(query, pool=pool, top_m=top_m)
    refs: list[EvidenceReference] = []
    for h in hits:
        refs.append(
            EvidenceReference(
                citation_id=h.chunk.chunk_id,
                title=h.chunk.title,
                source_url=h.chunk.source
                if h.chunk.source.startswith("http")
                else None,
                supporting_text=h.chunk.text[:500],
            )
        )
    return refs


def compute_grounding(interpretation: str, evidence: list[EvidenceReference]) -> float:
    """Grounding metric (spec sec. 12): supported claims / total claims.

    A sentence is "supported" if its Qwen-embedding cosine similarity to some
    evidence passage is >= 0.45. Returns 1.0 for an empty interpretation.
    """
    sentences = [
        s.strip() for s in _SENT.findall(interpretation) if len(s.split()) >= 4
    ]
    if not sentences:
        return 1.0
    if not evidence:
        return 0.0
    emb = get_embedder()
    sv = emb.encode_queries(sentences)
    ev = emb.encode_documents([e.supporting_text for e in evidence])
    sims = sv @ ev.T
    return float(np.mean(sims.max(axis=1) >= 0.45))


def _has_prohibited(text: str) -> bool:
    """True if the text trips any of the verifier's prohibited-claim patterns."""
    from vascutrace.report_verifier import _PROHIBITED_CLAIMS

    return any(p.search(text) for p in _PROHIBITED_CLAIMS.values())


def _drop_prohibited_sentences(text: str) -> str:
    """Remove any sentence that trips a prohibited-claim pattern (e.g. the
    LLM writing 'diagnostic', 'restenosis', 'prognosis'). This guarantees the
    verified-agent report never emits a prohibited claim -- the LLM explains,
    but the deterministic filter has the final say (spec sec. 15 mitigation)."""
    kept = [
        s.strip() for s in _SENT.findall(text) if s.strip() and not _has_prohibited(s)
    ]
    return " ".join(kept).strip()


def _load_metrics(output) -> QuantitativeMeasurements:
    return QuantitativeMeasurements.model_validate_json(output.metrics_path.read_text())


def generate_grounded_report(
    output,
    *,
    retriever: RagRetriever | None = None,
    llm: VascuTraceLLM | None = None,
) -> tuple[ResearchReport, float]:
    """gpt-5-mini grounded report + grounding score.

    Numbers come verbatim from the deterministic ``metrics.json``; only the
    interpretation/limitations prose is written by the model, grounded in RAG
    evidence. Raises :class:`LLMUnavailableError` if the LLM cannot be built
    (caller falls back to the deterministic template).
    """
    llm = llm or VascuTraceLLM()
    if retriever is None:
        retriever = RagRetriever(RagIndex.load())
    metrics = _load_metrics(output)

    query = (
        f"{_LATERALITY_PHRASE[output.laterality]} in the {_TARGET_REGION}; "
        "FDG PET/CT vascular detectability, partial-volume effects, contralateral "
        "asymmetry, and false positives on healthy scans"
    )
    evidence = retrieve_evidence(retriever, query)

    ev_block = "\n".join(
        f"[{e.citation_id}] {e.title}: {e.supporting_text}" for e in evidence
    )
    m = metrics.model_dump()
    asymmetry_sign = (
        "unavailable"
        if m["asymmetry_index"] is None
        else "positive"
        if m["asymmetry_index"] >= 0
        else "negative"
    )
    prompt = (
        "You are the reporting agent for VascuTrace, a research prototype that "
        "detects SIMULATED vascular-like FDG abnormalities in real healthy PET/CT. "
        "Write a JSON object with keys 'interpretation' (2-4 sentences) and "
        "'limitations' (a list of 2-4 short strings).\n\n"
        "HARD RULES:\n"
        "- This is a SYNTHETIC research case: the word 'synthetic' MUST appear in the "
        "interpretation.\n"
        "- Do NOT state or invent any numeric value; the numbers below are fixed and "
        "reported separately by deterministic code.\n"
        "- Ground every claim in the EVIDENCE passages; cite them by their [id].\n"
        "- FORBIDDEN: diagnosis, restenosis/atherosclerosis/vasculitis claims, patient "
        "outcome/prognosis/treatment, or claiming FDG identifies a specific cell type.\n"
        "- End with an explicit research-only, not-for-clinical-use statement.\n\n"
        f"DETERMINISTIC FINDINGS (do not restate the raw numbers): laterality="
        f"{output.laterality}, region={_TARGET_REGION}, "
        f"asymmetry_index sign={asymmetry_sign}, "
        f"quality_flags={metrics.quality_flags}.\n\n"
        f"EVIDENCE:\n{ev_block}\n"
    )
    raw = llm.chat(
        [{"role": "user", "content": prompt}],
        json_mode=True,
        reasoning_effort="low",
        max_completion_tokens=900,
    )
    try:
        parsed = json.loads(raw)
        interpretation = str(parsed.get("interpretation", "")).strip()
        limitations = [str(x) for x in parsed.get("limitations", []) if str(x).strip()]
    except (json.JSONDecodeError, AttributeError):
        interpretation, limitations = "", []
    # safety net: guarantee EVERY verifier invariant regardless of LLM output.
    interpretation = _drop_prohibited_sentences(interpretation)
    limitations = [x for x in limitations if not _has_prohibited(x)]
    if "synthetic" not in interpretation.lower():
        interpretation = (
            f"Synthetic research case with {_LATERALITY_PHRASE[output.laterality]}. "
            + interpretation
        ).strip()
    if not interpretation:
        interpretation = (
            f"Synthetic research case with {_LATERALITY_PHRASE[output.laterality]}; "
            "grounded in the retrieved evidence. Not for clinical use."
        )
    if not any("clinical" in x.lower() or "research" in x.lower() for x in limitations):
        limitations.append("Research-only output; not for clinical use.")

    report = ResearchReport(
        case_id=output.case_id,
        case_type="synthetic_research_case",
        finding=Finding(
            laterality=output.laterality,
            target_region=_TARGET_REGION,
            abnormality_score=output.abnormality_score,
        ),
        quantitative_measurements=metrics,  # verbatim, deterministic
        quality_control=QualityControl(
            partial_volume_risk="partial_volume_risk" in metrics.quality_flags,
            misregistration_risk="misregistration_risk" in metrics.quality_flags,
            flags=metrics.quality_flags,
        ),
        interpretation=interpretation,
        evidence=evidence,
        limitations=limitations,
    )
    grounding = compute_grounding(interpretation, evidence)
    return report, grounding


def safe_generate_report(output):
    """Grounded report if the GenAI stack is available, else the template report.

    Returns ``(report, grounding_or_None, backend)``. Never raises for a missing
    key / model -- degrades to the deterministic template so the pipeline always
    produces a valid, verified report.
    """
    try:
        report, grounding = generate_grounded_report(output)
        return report, grounding, "llm"
    except (LLMUnavailableError, Exception):  # noqa: BLE001 - offline-safe fallback
        from vascutrace.services import generate_demo_report

        return generate_demo_report(output), None, "template"
