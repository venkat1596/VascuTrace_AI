"""Provenance-aware deterministic retrieval and case-safe semantic caching."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from pydantic import Field

from vascutrace.contracts import EvidenceReference, StrictModel

DEFAULT_CORPUS_PATH = Path(__file__).parents[1] / "knowledge" / "research_corpus.json"
_TOKENS = re.compile(r"[a-z0-9]+")
_STOP_WORDS = {"a", "an", "explain", "how", "is", "the", "what", "why"}
_CASE_CONTEXT = re.compile(
    r"\b(subject|patient|case|scan|current|this|left|right)\b|\b(?:subject|case)[-_ ]?\d+\b",
    re.I,
)
_CASE_MEASUREMENT = re.compile(
    r"\b(suv(?:max|mean)?|asymmetry|volume|laterality|abnormal side|which side)\b",
    re.I,
)


class EvidenceDocument(StrictModel):
    document_id: str
    title: str
    source_url: str
    published_year: int = Field(ge=1900, le=2100)
    license: str
    text: str

    def reference(self) -> EvidenceReference:
        return EvidenceReference(
            citation_id=self.document_id,
            title=self.title,
            source_url=self.source_url,
            supporting_text=self.text,
        )


class EvidenceResponse(StrictModel):
    query: str
    evidence: list[EvidenceReference]
    cache_eligible: bool
    cache_hit: bool


def _terms(text: str) -> Counter[str]:
    return Counter(
        token for token in _TOKENS.findall(text.lower()) if token not in _STOP_WORDS
    )


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    numerator = sum(count * right[token] for token, count in left.items())
    left_norm = math.sqrt(sum(count * count for count in left.values()))
    right_norm = math.sqrt(sum(count * count for count in right.values()))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def is_cache_eligible(query: str) -> bool:
    """Reject scan-specific measurement queries from the semantic cache."""

    return not (_CASE_CONTEXT.search(query) and _CASE_MEASUREMENT.search(query))


class EvidenceStore:
    """Small local corpus with deterministic retrieval and an in-memory semantic cache."""

    def __init__(
        self,
        corpus_path: Path | str = DEFAULT_CORPUS_PATH,
        similarity_threshold: float = 0.82,
    ) -> None:
        raw = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
        self.documents = [EvidenceDocument.model_validate(item) for item in raw]
        self.similarity_threshold = similarity_threshold
        self._cache: list[tuple[str, Counter[str], list[EvidenceReference]]] = []

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def _retrieve(self, query: str, top_k: int) -> list[EvidenceReference]:
        query_terms = _terms(query)
        ranked = sorted(
            self.documents,
            key=lambda document: _cosine(
                query_terms, _terms(f"{document.title} {document.text}")
            ),
            reverse=True,
        )
        return [document.reference() for document in ranked[:top_k]]

    def search(self, query: str, top_k: int = 3) -> EvidenceResponse:
        if not query.strip():
            raise ValueError("Evidence query must not be empty")
        if not 1 <= top_k <= 5:
            raise ValueError("top_k must be between 1 and 5")

        eligible = is_cache_eligible(query)
        query_terms = _terms(query)
        if eligible:
            for _, cached_terms, references in self._cache:
                if _cosine(query_terms, cached_terms) >= self.similarity_threshold:
                    return EvidenceResponse(
                        query=query,
                        evidence=references[:top_k],
                        cache_eligible=True,
                        cache_hit=True,
                    )

        references = self._retrieve(query, top_k)
        if eligible:
            self._cache.append((query, query_terms, references))
        return EvidenceResponse(
            query=query,
            evidence=references,
            cache_eligible=eligible,
            cache_hit=False,
        )


evidence_store = EvidenceStore()
