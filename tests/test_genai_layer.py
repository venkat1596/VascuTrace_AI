"""Tests for the GenAI/RAG layer (``src/vascutrace/genai/{llm,rag,report_agent}.py``
and the ``vascutrace/tools.py`` backend switches that front them).

Offline-safe by design: CI runs plain ``uv run pytest -q`` with NO network and
NO downloaded Qwen models. Every test in this file either (a) needs no network
at all (pure functions, mocked OpenAI client, mocked embedder), or (b)
runtime-``pytest.skip``s when the cached RAG index / Qwen models are not
locally available, following this repo's own established convention for a
locally-generated, gitignored artifact (see ``tests/test_ml_infer.py`` /
``tests/test_product_backends.py`` module docstrings: neither the
``local_data`` marker, which scopes raw ``Data/`` access, nor the ``gpu``
marker, which scopes CUDA, describes a locally-cached model/index directory --
and CI's plain ``uv run pytest -q`` applies no ``-m`` filter, so a marker
alone would not keep a clean CI runner green).

Every test that touches ``OPENAI_API_KEY`` / ``OPEN_AI_KEY`` / the module's
``_ENV_PATH`` global does so through ``monkeypatch`` so nothing leaks into
later, unrelated tests -- this repo's real ``.env`` file at the project root
holds a live key; tests that must simulate "no key available" always patch
``_ENV_PATH`` to a nonexistent path rather than reading or touching the real
file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from src.vascutrace.genai import llm as llm_mod
from src.vascutrace.genai import report_agent
from src.vascutrace.genai.llm import LLMUnavailableError, VascuTraceLLM, load_openai_key
from src.vascutrace.genai.rag import (
    DEFAULT_INDEX_DIR,
    RagChunk,
    RagIndex,
    RagRetriever,
    Retrieved,
    build_corpus,
    chunk_text,
)
from src.vascutrace.genai.report_agent import (
    _drop_prohibited_sentences,
    _has_prohibited,
    generate_grounded_report,
    safe_generate_report,
)
from vascutrace.contracts import ModelOutput, QuantitativeMeasurements, ResearchReport
from vascutrace.report_verifier import verify_report
from vascutrace.tools import (
    _resolve_evidence_backend,
    _resolve_report_backend,
    generate_research_report,
    retrieve_research_evidence,
)

_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def _write_metrics(tmp_path: Path) -> tuple[Path, QuantitativeMeasurements]:
    """Write a synthetic ``metrics.json`` and return (path, the parsed model)."""
    metrics = QuantitativeMeasurements(
        target_suvmax=4.2,
        target_suvmean=2.1,
        contralateral_suvmax=1.5,
        contralateral_suvmean=0.9,
        asymmetry_index=0.42,
        metabolic_volume_ml=3.7,
        longitudinal_extent_mm=16.0,
        quality_flags=["partial_volume_risk"],
    )
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(metrics.model_dump_json())
    return metrics_path, metrics


def _make_output(tmp_path: Path, *, laterality: str = "left") -> ModelOutput:
    metrics_path, _ = _write_metrics(tmp_path)
    return ModelOutput(
        case_id="synthetic_test_case",
        model_name="deterministic-synthetic-reference",
        model_version="1.0.0",
        laterality=laterality,
        abnormality_score=0.87,
        mask_path=tmp_path / "ground_truth_mask.npy",
        metrics_path=metrics_path,
        overlay_path=tmp_path / "overlay.png",
        runtime_seconds=0.01,
    )


# =========================================================================== #
# 1. LLM key resolution
# =========================================================================== #
class TestLoadOpenAiKey:
    def test_openai_api_key_env_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPEN_AI_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        assert load_openai_key() == "sk-test"

    def test_open_ai_key_env_used_when_openai_api_key_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("OPEN_AI_KEY", "sk-other")
        assert load_openai_key() == "sk-other"

    def test_both_unset_and_no_env_file_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPEN_AI_KEY", raising=False)
        monkeypatch.setattr(llm_mod, "_ENV_PATH", tmp_path / "does_not_exist.env")
        assert load_openai_key() is None


# =========================================================================== #
# 2. LLM fail-loud
# =========================================================================== #
class TestLlmFailLoud:
    def test_no_key_raises_llm_unavailable_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPEN_AI_KEY", raising=False)
        monkeypatch.setattr(llm_mod, "_ENV_PATH", tmp_path / "does_not_exist.env")
        with pytest.raises(LLMUnavailableError):
            VascuTraceLLM(api_key=None)


# =========================================================================== #
# 3. LLM chat/embed against a fake OpenAI client (no network)
# =========================================================================== #
class _FakeMessage:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str | None) -> None:
        self.message = _FakeMessage(content)


class _FakeChatResponse:
    def __init__(self, content: str | None) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeChatCompletions:
    def __init__(self, content: str | None) -> None:
        self._content = content
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeChatResponse(self._content)


class _FakeChat:
    def __init__(self, content: str | None) -> None:
        self.completions = _FakeChatCompletions(content)


class _FakeEmbeddingDatum:
    def __init__(self, embedding: list[float]) -> None:
        self.embedding = embedding


class _FakeEmbeddingsResponse:
    def __init__(self, embeddings: list[list[float]]) -> None:
        self.data = [_FakeEmbeddingDatum(e) for e in embeddings]


class _FakeEmbeddings:
    def __init__(self, vector: list[float]) -> None:
        self._vector = vector
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeEmbeddingsResponse([self._vector for _ in kwargs["input"]])


class _FakeOpenAIClient:
    def __init__(
        self, content: str | None = "OK", embed_vector: list[float] | None = None
    ) -> None:
        self.chat = _FakeChat(content)
        self.embeddings = _FakeEmbeddings(embed_vector or [0.1, 0.2])


class TestLlmChatAndEmbed:
    def _llm(self, monkeypatch: pytest.MonkeyPatch) -> VascuTraceLLM:
        # Building VascuTraceLLM makes no network call (the OpenAI() client
        # constructor is lazy); we then swap the client for a fake before any
        # request-shaped call is made.
        llm = VascuTraceLLM(api_key="sk-test")
        return llm

    def test_chat_returns_content_and_default_kwargs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        llm = self._llm(monkeypatch)
        fake_client = _FakeOpenAIClient(content="OK")
        monkeypatch.setattr(llm, "_client", fake_client)

        result = llm.chat([{"role": "user", "content": "hi"}])

        assert result == "OK"
        kwargs = fake_client.chat.completions.calls[0]
        assert kwargs["model"] == "gpt-5-mini"
        assert kwargs["reasoning_effort"] == "low"
        assert kwargs["max_completion_tokens"] == 1200
        assert "response_format" not in kwargs

    def test_chat_json_mode_sets_response_format(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        llm = self._llm(monkeypatch)
        fake_client = _FakeOpenAIClient(content="{}")
        monkeypatch.setattr(llm, "_client", fake_client)

        llm.chat([{"role": "user", "content": "hi"}], json_mode=True)

        kwargs = fake_client.chat.completions.calls[-1]
        assert kwargs["response_format"] == {"type": "json_object"}

    def test_chat_none_content_returns_empty_string(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        llm = self._llm(monkeypatch)
        fake_client = _FakeOpenAIClient(content=None)
        monkeypatch.setattr(llm, "_client", fake_client)

        assert llm.chat([{"role": "user", "content": "hi"}]) == ""

    def test_embed_returns_vectors_and_uses_embed_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        llm = self._llm(monkeypatch)
        fake_client = _FakeOpenAIClient(embed_vector=[0.1, 0.2])
        monkeypatch.setattr(llm, "_client", fake_client)

        out = llm.embed(["a", "b"])

        assert out == [[0.1, 0.2], [0.1, 0.2]]
        kwargs = fake_client.embeddings.calls[0]
        assert kwargs["model"] == "text-embedding-3-small"
        assert kwargs["input"] == ["a", "b"]


# =========================================================================== #
# 4. RAG pure functions / RagIndex.retrieve (numpy only, no models)
# =========================================================================== #
class TestChunkText:
    def test_chunks_meet_minimum_word_count(self) -> None:
        paragraphs = [" ".join(f"word{i}" for i in range(30)) for _ in range(3)]
        text = "\n\n".join(paragraphs)
        chunks = chunk_text(text, target_words=50, overlap_words=5)
        assert chunks
        for c in chunks:
            assert len(c.split()) >= 15

    def test_paragraph_over_target_is_hard_split(self) -> None:
        long_paragraph = " ".join(f"word{i}" for i in range(250))
        chunks = chunk_text(long_paragraph, target_words=100, overlap_words=10)
        # 250 words / 100-word target -> 3 hard-split pieces (100, 100, 50);
        # all pieces are >= 15 words so none is filtered out.
        assert len(chunks) == 3
        assert len(chunks[0].split()) == 100
        assert len(chunks[1].split()) == 100
        assert len(chunks[2].split()) == 50


class TestBuildCorpus:
    def test_build_corpus_nonempty_with_unique_chunk_ids(self) -> None:
        chunks = build_corpus()
        assert len(chunks) > 0
        assert all(isinstance(c, RagChunk) for c in chunks)
        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))


class TestRagIndexRetrieve:
    def test_retrieve_returns_k_sorted_by_descending_score(self) -> None:
        chunks = [
            RagChunk(chunk_id="c0", doc_id="d0", title="T0", source="s0", text="text0"),
            RagChunk(chunk_id="c1", doc_id="d1", title="T1", source="s1", text="text1"),
            RagChunk(chunk_id="c2", doc_id="d2", title="T2", source="s2", text="text2"),
        ]
        embeddings = np.eye(3, dtype=np.float32)
        index = RagIndex(chunks, embeddings)
        query_vec = np.array([0.9, 0.1, 0.0], dtype=np.float32)

        results = index.retrieve(query_vec, 2)

        assert len(results) == 2
        assert all(isinstance(r, Retrieved) for r in results)
        assert results[0].score >= results[1].score
        assert [r.rank for r in results] == [0, 1]
        assert results[0].chunk.chunk_id == "c0"
        assert results[1].chunk.chunk_id == "c1"


# =========================================================================== #
# 5. Tool backend switch (template/keyword default; llm/rag opt-in)
# =========================================================================== #
class TestBackendSwitch:
    def test_default_report_backend_is_template(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("VASCUTRACE_REPORT_BACKEND", raising=False)
        assert _resolve_report_backend() == "template"

    def test_default_evidence_backend_is_keyword(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("VASCUTRACE_EVIDENCE_BACKEND", raising=False)
        assert _resolve_evidence_backend() == "keyword"

    def test_report_backend_llm_selectable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VASCUTRACE_REPORT_BACKEND", "llm")
        assert _resolve_report_backend() == "llm"

    def test_evidence_backend_rag_selectable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VASCUTRACE_EVIDENCE_BACKEND", "rag")
        assert _resolve_evidence_backend() == "rag"

    def test_rag_evidence_backend_matches_keyword_contract(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The rag backend must return the SAME payload contract as the keyword
        backend (an ``EvidenceResponse``) so every consumer -- orchestrator, MCP
        server, dashboard -- stays backend-agnostic. Regression: the rag branch
        once returned a divergent ``{backend, results}`` shape lacking
        ``cache_eligible``, which crashed the dashboard's evidence panel the
        moment the rag backend was enabled. Uses fakes (no Qwen, offline)."""
        import src.vascutrace.genai.rag as rag_mod
        from vascutrace.evidence import evidence_store

        query = "Why can partial-volume effects bias PET uptake?"
        monkeypatch.setenv("VASCUTRACE_EVIDENCE_BACKEND", "rag")
        monkeypatch.setattr(rag_mod.RagIndex, "load", classmethod(lambda cls: None))
        monkeypatch.setattr(
            rag_mod, "RagRetriever", lambda index: _FakeRetriever(_canned_evidence())
        )

        payload = retrieve_research_evidence(query, top_k=2)
        keyword_payload = evidence_store.search(query, 2).model_dump(mode="json")

        # identical top-level contract as the keyword backend
        assert set(payload) == set(keyword_payload)
        assert payload["cache_eligible"] == keyword_payload["cache_eligible"]
        assert payload["cache_hit"] is False
        # evidence items carry exactly the EvidenceReference fields consumers read
        assert len(payload["evidence"]) == 2
        first = payload["evidence"][0]
        assert set(first) == set(keyword_payload["evidence"][0])
        assert first["citation_id"] == "c1"
        assert first["source_url"] == "docs/example.md"

    def test_invalid_report_backend_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VASCUTRACE_REPORT_BACKEND", "not_a_backend")
        with pytest.raises(ValueError):
            _resolve_report_backend()

    def test_invalid_evidence_backend_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VASCUTRACE_EVIDENCE_BACKEND", "not_a_backend")
        with pytest.raises(ValueError):
            _resolve_evidence_backend()

    def test_default_backend_returns_valid_report_without_genai_stack(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("VASCUTRACE_REPORT_BACKEND", raising=False)
        # Force any attempted import of the genai report-agent module to
        # fail loudly: proves the template backend never touches the genai
        # stack, regardless of whether some *other* test already imported
        # (and cached) it earlier in the session.
        monkeypatch.setitem(sys.modules, "src.vascutrace.genai.report_agent", None)

        output = _make_output(tmp_path)
        result = generate_research_report(output.model_dump(mode="json"))

        # Validates as a real ResearchReport.
        report = ResearchReport.model_validate(result)
        assert report.case_id == output.case_id
        assert report.finding.laterality == output.laterality


# =========================================================================== #
# 6. Report sanitizer + verbatim measurement preservation (mock LLM + fake
#    retriever, no Qwen model loaded).
# =========================================================================== #
class _FakeLlm:
    def __init__(self, raw_json: str) -> None:
        self._raw_json = raw_json
        self.calls: list[dict] = []

    def chat(
        self,
        messages,
        *,
        json_mode=False,
        reasoning_effort=None,
        max_completion_tokens=None,
    ):
        self.calls.append(
            {
                "messages": messages,
                "json_mode": json_mode,
                "reasoning_effort": reasoning_effort,
                "max_completion_tokens": max_completion_tokens,
            }
        )
        return self._raw_json


class _FakeRetriever:
    def __init__(self, hits: list[Retrieved]) -> None:
        self._hits = hits

    def retrieve_and_rerank(self, query, *, pool=20, top_m=4):
        return self._hits


class _FakeEmbedder:
    """Deterministic embedder: every text maps to the same unit vector, so
    ``compute_grounding`` runs (and always finds max cosine similarity 1.0)
    without ever loading a real Qwen model."""

    def encode_queries(self, texts):
        return np.ones((len(texts), 4), dtype=np.float32)

    def encode_documents(self, texts):
        return np.ones((len(texts), 4), dtype=np.float32)


def _canned_evidence() -> list[Retrieved]:
    chunk1 = RagChunk(
        chunk_id="c1",
        doc_id="doc1",
        title="Evidence Doc 1",
        source="docs/example.md",
        text="Contralateral asymmetry can arise from partial-volume effects in FDG PET/CT.",
    )
    chunk2 = RagChunk(
        chunk_id="c2",
        doc_id="doc2",
        title="Evidence Doc 2",
        source="docs/example2.md",
        text="False positives on healthy scans are a known detectability confound.",
    )
    return [Retrieved(chunk1, 0.9, 0), Retrieved(chunk2, 0.7, 1)]


class TestGeneratedGroundedReportSanitizer:
    def test_measurement_preservation_and_sanitizer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        output = _make_output(tmp_path, laterality="left")
        source_metrics = QuantitativeMeasurements.model_validate_json(
            output.metrics_path.read_text()
        )

        raw_json = (
            '{"interpretation":"This shows diagnostic restenosis in the patient. '
            'The synthetic case flags asymmetry [c1].","limitations":["x"]}'
        )
        fake_llm = _FakeLlm(raw_json)
        fake_retriever = _FakeRetriever(_canned_evidence())

        # compute_grounding() calls get_embedder(); patch it to a fake so no
        # Qwen model ever loads.
        monkeypatch.setattr(report_agent, "get_embedder", lambda: _FakeEmbedder())

        report, grounding = generate_grounded_report(
            output, retriever=fake_retriever, llm=fake_llm
        )

        # (a) quantitative_measurements EXACTLY equal the source metrics.
        assert report.quantitative_measurements == source_metrics

        # (b) laterality matches the source ModelOutput.
        assert report.finding.laterality == output.laterality

        # (c) interpretation contains "synthetic" and no prohibited term --
        # the diagnostic/restenosis sentence was dropped.
        assert "synthetic" in report.interpretation.lower()
        assert "diagnostic" not in report.interpretation.lower()
        assert "restenosis" not in report.interpretation.lower()
        assert not _has_prohibited(report.interpretation)
        for limitation in report.limitations:
            assert not _has_prohibited(limitation)

        # (d) the verifier accepts the sanitized report.
        result = verify_report(
            report, source_metrics, expected_laterality=output.laterality
        )
        assert result.accepted is True, result.issues

        assert isinstance(grounding, float)


# =========================================================================== #
# 7. _has_prohibited / _drop_prohibited_sentences unit tests
# =========================================================================== #
class TestProhibitedSentenceFiltering:
    def test_diagnostic_sentence_flagged(self) -> None:
        assert _has_prohibited("This is diagnostic of the condition.") is True

    def test_restenosis_claim_sentence_flagged(self) -> None:
        assert (
            _has_prohibited("The scan demonstrates restenosis in this vessel.") is True
        )

    def test_prognosis_sentence_flagged(self) -> None:
        assert _has_prohibited("The patient's prognosis is guarded.") is True

    def test_clean_sentence_not_flagged(self) -> None:
        assert _has_prohibited("The synthetic case shows mild asymmetry.") is False

    def test_drop_prohibited_sentences_keeps_only_clean_sentence(self) -> None:
        text = (
            "This is diagnostic of the condition. "
            "The scan demonstrates restenosis in this vessel. "
            "The patient's prognosis is guarded. "
            "The synthetic case shows mild asymmetry."
        )
        result = _drop_prohibited_sentences(text)
        assert result == "The synthetic case shows mild asymmetry."


# =========================================================================== #
# 8. safe_generate_report fallback to the deterministic template
# =========================================================================== #
class TestSafeGenerateReportFallback:
    def test_llm_unavailable_falls_back_to_template(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raise(*args, **kwargs):
            raise LLMUnavailableError("no key for this test")

        monkeypatch.setattr(report_agent, "VascuTraceLLM", _raise)

        output = _make_output(tmp_path)
        report, grounding, backend = safe_generate_report(output)

        assert isinstance(report, ResearchReport)
        assert grounding is None
        assert backend == "template"
        assert report.case_id == output.case_id


# =========================================================================== #
# 9. Qwen retrieve+rerank end-to-end -- RUNTIME-SKIP when the cached RAG
#    index or the Qwen models are not locally available (no network in CI).
# =========================================================================== #
class TestQwenRetrieveAndRerankEndToEnd:
    def test_retrieve_and_rerank_returns_descending_scores(self) -> None:
        index_dir = DEFAULT_INDEX_DIR
        if (
            not (index_dir / "embeddings.npy").is_file()
            or not (index_dir / "corpus.jsonl").is_file()
        ):
            pytest.skip(f"cached RAG index not present at {index_dir}")
        try:
            index = RagIndex.load(index_dir)
            retriever = RagRetriever(index)
            hits = retriever.retrieve_and_rerank(
                "PET false positives under noise", pool=10, top_m=3
            )
        except Exception as exc:  # noqa: BLE001 - offline/model-unavailable skip
            pytest.skip(f"Qwen RAG stack unavailable: {exc}")

        assert len(hits) <= 3
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)
