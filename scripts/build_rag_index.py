"""Build + cache the Qwen RAG index for the VascuTrace Evidence Agent.

Research prototype. Chunks the literature/research corpus and embeds it with
Qwen3-Embedding-0.6B (local), caching to knowledge/rag_index/ so retrieval never
needs the network. Run once (re-run when the corpus changes).

    uv run python -m scripts.build_rag_index
"""

from __future__ import annotations

import time

from src.vascutrace.genai.rag import RagIndex, build_corpus, get_embedder


def main() -> None:
    chunks = build_corpus()
    n_docs = len({c.doc_id for c in chunks})
    print(f"corpus: {len(chunks)} chunks from {n_docs} docs")
    t = time.time()
    index = RagIndex.build(chunks, get_embedder())
    index.save()
    print(
        f"embedded {index.embeddings.shape} with Qwen3-Embedding-0.6B "
        f"in {time.time() - t:.1f}s -> knowledge/rag_index/"
    )


if __name__ == "__main__":
    main()
