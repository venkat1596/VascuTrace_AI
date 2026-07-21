"""RAG evidence pipeline: chunked corpus + Qwen dense retrieval + Qwen rerank.

Research prototype. Implements the Evidence Agent (spec sec. 9): retrieve grounded
passages from the source paper + selected literature so the reasoning LLM
(gpt-5-mini) only ever explains verified measurements over *cited* text.

Model roles (kept separate on purpose):
  * EMBEDDING  -> Qwen3-Embedding-0.6B  (local, 1024-d, purpose-built retriever)
  * RERANK     -> Qwen3-Reranker-0.6B   (local cross-encoder)
  * REASONING  -> gpt-5-mini            (agent/report only; NOT embed/rerank)

Two-stage retrieval matching the evaluation protocol:
  1. dense retrieve top-K by cosine over Qwen embeddings (fast recall)
  2. rerank the shortlist with the Qwen cross-encoder (precision at the top)

Retrieval is fully local; the OpenAI reasoning model is never in this path.
"""

from __future__ import annotations

import functools
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INDEX_DIR = _REPO_ROOT / "knowledge" / "rag_index"
_WORD = re.compile(r"\S+")

EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
RERANK_MODEL = "Qwen/Qwen3-Reranker-0.6B"
# Qwen3-Embedding recommends an instruction on the QUERY side only; documents get none.
QUERY_INSTRUCTION = (
    "Instruct: Given a research question about PET/CT vascular-imaging "
    "detectability, retrieve passages that answer it.\nQuery: "
)


@dataclass(frozen=True)
class RagChunk:
    chunk_id: str
    doc_id: str
    title: str
    source: str
    text: str


@dataclass(frozen=True)
class Retrieved:
    chunk: RagChunk
    score: float
    rank: int


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# local Qwen models (lazy, process-cached)
# --------------------------------------------------------------------------- #
class QwenEmbedder:
    """Local Qwen3-Embedding-0.6B. Query/document asymmetry per Qwen recipe."""

    def __init__(
        self, model_name: str = EMBED_MODEL, device: str | None = None
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device or _device())

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        # documents: no instruction prompt
        return np.asarray(
            self.model.encode(
                [t if t.strip() else " " for t in texts],
                normalize_embeddings=True,
                prompt="",
                batch_size=64,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(
                [QUERY_INSTRUCTION + t for t in texts],
                normalize_embeddings=True,
                prompt="",
                batch_size=64,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )


class QwenReranker:
    """Local Qwen3-Reranker-0.6B cross-encoder. Higher score = more relevant."""

    def __init__(
        self, model_name: str = RERANK_MODEL, device: str | None = None
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(
            model_name, device=device or _device(), trust_remote_code=True
        )

    def score(self, query: str, docs: list[str]) -> list[float]:
        if not docs:
            return []
        return [float(s) for s in self.model.predict([(query, d) for d in docs])]


@functools.lru_cache(maxsize=1)
def get_embedder() -> QwenEmbedder:
    return QwenEmbedder()


@functools.lru_cache(maxsize=1)
def get_reranker() -> QwenReranker:
    return QwenReranker()


# --------------------------------------------------------------------------- #
# corpus construction (chunking)
# --------------------------------------------------------------------------- #
def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def chunk_text(
    text: str, *, target_words: int = 200, overlap_words: int = 30
) -> list[str]:
    """Greedy paragraph-packing chunker with a small word overlap."""
    chunks: list[str] = []
    buf: list[str] = []
    buf_n = 0
    for para in _split_paragraphs(text):
        words = _WORD.findall(para)
        if not words:
            continue
        if len(words) > target_words:
            for i in range(0, len(words), target_words):
                chunks.append(" ".join(words[i : i + target_words]))
            continue
        if buf_n + len(words) > target_words and buf:
            chunks.append(" ".join(buf))
            tail = " ".join(buf).split()[-overlap_words:] if overlap_words else []
            buf = list(tail)
            buf_n = len(buf)
        buf.extend(words)
        buf_n += len(words)
    if buf:
        chunks.append(" ".join(buf))
    return [c for c in chunks if len(_WORD.findall(c)) >= 15]


def _iter_source_texts(
    *, research_corpus: Path, doc_globs: list[Path]
) -> list[tuple[str, str, str, str]]:
    out: list[tuple[str, str, str, str]] = []
    if research_corpus.is_file():
        for d in json.loads(research_corpus.read_text()):
            out.append(
                (
                    d["document_id"],
                    d.get("title", d["document_id"]),
                    d.get("source_url", str(research_corpus)),
                    d["text"],
                )
            )
    for path in doc_globs:
        if not path.is_file() or path.name.endswith(".provenance.md"):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        text = re.sub(r"^---\n.*?\n---\n", "", text, count=1, flags=re.S)
        title = next(
            (ln.lstrip("# ").strip() for ln in text.splitlines() if ln.startswith("#")),
            path.stem,
        )
        out.append((path.stem, title[:160], str(path.relative_to(_REPO_ROOT)), text))
    return out


def build_corpus(
    *,
    research_corpus: Path | None = None,
    doc_dirs: list[Path] | None = None,
    target_words: int = 200,
) -> list[RagChunk]:
    """Build chunks from the public literature JSON and explicit doc folders.

    The default intentionally reads no repository documentation directory.
    Callers may pass audited public folders through ``doc_dirs``.
    """
    research_corpus = research_corpus or (
        _REPO_ROOT / "knowledge" / "research_corpus.json"
    )
    if doc_dirs is None:
        doc_dirs = []
    doc_globs: list[Path] = []
    for d in doc_dirs:
        if d.is_dir():
            doc_globs.extend(sorted(d.glob("*.md")))
    chunks: list[RagChunk] = []
    for doc_id, title, source, text in _iter_source_texts(
        research_corpus=research_corpus, doc_globs=doc_globs
    ):
        for i, ctext in enumerate(chunk_text(text, target_words=target_words)):
            chunks.append(
                RagChunk(
                    chunk_id=f"{doc_id}::{i:03d}",
                    doc_id=doc_id,
                    title=title,
                    source=source,
                    text=ctext,
                )
            )
    return chunks


# --------------------------------------------------------------------------- #
# index (Qwen embeddings)
# --------------------------------------------------------------------------- #
class RagIndex:
    """Cached Qwen-embedding index over a chunked corpus."""

    def __init__(self, chunks: list[RagChunk], embeddings: np.ndarray) -> None:
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("chunks and embeddings length mismatch")
        self.chunks = chunks
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.embeddings = embeddings / np.clip(norms, 1e-8, None)

    @classmethod
    def build(
        cls, chunks: list[RagChunk], embedder: QwenEmbedder | None = None
    ) -> RagIndex:
        embedder = embedder or get_embedder()
        vecs = embedder.encode_documents([c.text for c in chunks])
        return cls(chunks, vecs)

    def save(self, index_dir: Path = DEFAULT_INDEX_DIR) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / "embeddings.npy", self.embeddings.astype(np.float32))
        (index_dir / "corpus.jsonl").write_text(
            "\n".join(json.dumps(asdict(c)) for c in self.chunks) + "\n"
        )
        (index_dir / "index_meta.json").write_text(
            json.dumps(
                {
                    "embed_model": EMBED_MODEL,
                    "dim": int(self.embeddings.shape[1]),
                    "n_chunks": len(self.chunks),
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, index_dir: Path = DEFAULT_INDEX_DIR) -> RagIndex:
        emb = np.load(index_dir / "embeddings.npy")
        chunks = [
            RagChunk(**json.loads(ln))
            for ln in (index_dir / "corpus.jsonl").read_text().splitlines()
            if ln.strip()
        ]
        return cls(chunks, emb)

    def retrieve(self, query_vec: np.ndarray, k: int) -> list[Retrieved]:
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        q = q / max(float(np.linalg.norm(q)), 1e-8)
        scores = self.embeddings @ q
        k = min(k, len(self.chunks))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [
            Retrieved(self.chunks[i], float(scores[i]), r) for r, i in enumerate(top)
        ]


class RagRetriever:
    """Query-time retriever: Qwen-embed the query, dense top-K, Qwen cross-encoder rerank."""

    def __init__(
        self,
        index: RagIndex,
        embedder: QwenEmbedder | None = None,
        reranker: QwenReranker | None = None,
    ) -> None:
        self.index = index
        self.embedder = embedder or get_embedder()
        self._reranker = reranker

    @property
    def reranker(self) -> QwenReranker:
        if self._reranker is None:
            self._reranker = get_reranker()
        return self._reranker

    def retrieve(self, query: str, k: int) -> list[Retrieved]:
        qv = self.embedder.encode_queries([query])[0]
        return self.index.retrieve(qv, k)

    def rerank(
        self, query: str, candidates: list[Retrieved], top_m: int
    ) -> list[Retrieved]:
        """Qwen cross-encoder rerank: reorder candidates by relevance, return top_m."""
        if not candidates:
            return []
        scores = self.reranker.score(query, [c.chunk.text for c in candidates])
        order = sorted(range(len(candidates)), key=lambda i: -scores[i])
        return [
            Retrieved(candidates[i].chunk, scores[i], new_rank)
            for new_rank, i in enumerate(order[:top_m])
        ]

    def retrieve_and_rerank(
        self, query: str, *, pool: int = 20, top_m: int = 5
    ) -> list[Retrieved]:
        return self.rerank(query, self.retrieve(query, pool), top_m)
