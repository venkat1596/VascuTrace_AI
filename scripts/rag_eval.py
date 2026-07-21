"""RAG retrieval evaluation: R@K sweep -> Qwen rerank -> nDCG@M/Hit@M/P@M.

Research prototype. Executes the pre-registered retrieval protocol:
  1. dense retrieve top-500 per query (Qwen3-Embedding-0.6B cosine)
  2. R@K for K in {5,10,20,50,100,500} -> find where recall is high
  3. rerank the top-`pool` shortlist (>10 candidates) with Qwen3-Reranker-0.6B
  4. nDCG@M / Hit@M / P@M for M in {2,5,10}, dense vs reranked (rerank value)

Model roles: embedding + rerank are LOCAL Qwen models; gpt-5-mini (reasoning LLM)
is used ONLY to write the synthetic eval questions, not for retrieval.

Eval set = synthetic single-gold: gpt-5-mini writes one specific question each
sampled chunk answers; that chunk is the gold. Cached to eval_set.jsonl.
Single-gold => P@M is structurally capped at 1/M (reported honestly); recall@K /
Hit@M / nDCG@M are the load-bearing scores.

Outputs -> runs/diagnostics/rag_eval_2026-07-20/{eval_set.jsonl, scoreboard.md, curve.png}
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np

from src.vascutrace.genai.llm import VascuTraceLLM
from src.vascutrace.genai.rag import RagIndex, RagRetriever, get_embedder, get_reranker

OUT = Path("runs/diagnostics/rag_eval_2026-07-20")
K_SWEEP = [5, 10, 20, 50, 100, 500]
M_LIST = [2, 5, 10]
RERANK_POOL = 20
N_QUERIES = 60
SEED = 0


def gen_eval_set(index: RagIndex, llm: VascuTraceLLM, n: int) -> list[dict]:
    rng = random.Random(SEED)
    by_doc: dict[str, list[int]] = {}
    for i, c in enumerate(index.chunks):
        if len(c.text.split()) >= 40:
            by_doc.setdefault(c.doc_id, []).append(i)
    docs = sorted(by_doc)
    rng.shuffle(docs)
    picks: list[int] = []
    di = 0
    while len(picks) < n and any(by_doc.values()):
        d = docs[di % len(docs)]
        if by_doc[d]:
            picks.append(by_doc[d].pop(rng.randrange(len(by_doc[d]))))
        di += 1
    evalset: list[dict] = []
    for i in picks:
        c = index.chunks[i]
        prompt = (
            "You write evaluation questions for a PET/CT vascular-imaging research "
            "copilot's retrieval system. Given the PASSAGE, write ONE specific, "
            "self-contained research question that THIS passage directly answers "
            "(a researcher's question, not a generic one; do not quote the passage "
            'verbatim). Return JSON {"question": "..."}.\n\nPASSAGE:\n' + c.text[:1200]
        )
        try:
            q = json.loads(
                llm.chat(
                    [{"role": "user", "content": prompt}],
                    json_mode=True,
                    reasoning_effort="low",
                    max_completion_tokens=400,
                )
            )["question"].strip()
        except Exception:
            continue
        if q:
            evalset.append(
                {"question": q, "gold_chunk_id": c.chunk_id, "gold_doc_id": c.doc_id}
            )
    return evalset


def dcg(rels: list[int]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at(ranked_ids: list[str], gold: str, m: int) -> float:
    rels = [1 if cid == gold else 0 for cid in ranked_ids[:m]]
    ideal = dcg(sorted(rels, reverse=True) or [0])
    return dcg(rels) / ideal if ideal > 0 else 0.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    index = RagIndex.load()
    embedder = get_embedder()
    reranker = get_reranker()
    retriever = RagRetriever(index, embedder=embedder, reranker=reranker)

    eval_path = OUT / "eval_set.jsonl"
    if eval_path.is_file():
        evalset = [
            json.loads(ln) for ln in eval_path.read_text().splitlines() if ln.strip()
        ]
    else:
        evalset = gen_eval_set(index, VascuTraceLLM(), N_QUERIES)
        eval_path.write_text("\n".join(json.dumps(e) for e in evalset) + "\n")
    print(f"eval set: {len(evalset)} single-gold queries")

    # dense retrieve top-500 per query with Qwen query embeddings
    qvecs = embedder.encode_queries([e["question"] for e in evalset])
    dense = [[r.chunk.chunk_id for r in index.retrieve(qv, 500)] for qv in qvecs]

    recall = {
        k: float(np.mean([e["gold_chunk_id"] in d[:k] for e, d in zip(evalset, dense)]))
        for k in K_SWEEP
    }
    best_k = (
        next((k for k in K_SWEEP if k > 10 and recall[k] >= 0.90), None) or RERANK_POOL
    )

    pool = max(RERANK_POOL, 20)
    metrics = {
        "dense": {m: {"ndcg": [], "hit": [], "p": []} for m in M_LIST},
        "rerank": {m: {"ndcg": [], "hit": [], "p": []} for m in M_LIST},
    }
    for e, qv in zip(evalset, qvecs):
        gold = e["gold_chunk_id"]
        cands = index.retrieve(qv, pool)
        dense_ids = [r.chunk.chunk_id for r in cands]
        rr_ids = [
            r.chunk.chunk_id for r in retriever.rerank(e["question"], cands, top_m=pool)
        ]
        for tag, ids in (("dense", dense_ids), ("rerank", rr_ids)):
            for m in M_LIST:
                top = ids[:m]
                metrics[tag][m]["hit"].append(1.0 if gold in top else 0.0)
                metrics[tag][m]["p"].append(sum(c == gold for c in top) / m)
                metrics[tag][m]["ndcg"].append(ndcg_at(ids, gold, m))

    lines = [
        "# RAG retrieval evaluation (Qwen3-Embedding-0.6B + Qwen3-Reranker-0.6B)",
        "",
        "> Research prototype. Retrieval quality on a synthetic single-gold eval set.",
        f"Corpus: {len(index.chunks)} chunks / {len({c.doc_id for c in index.chunks})} docs. "
        f"Eval: {len(evalset)} queries (gpt-5-mini-generated, one gold chunk each). "
        "Embedding+rerank = local Qwen; gpt-5-mini writes questions only.",
        "",
        "## Stage 1 - Recall@K (dense retrieval; single gold => recall==hit)",
        "| K | " + " | ".join(str(k) for k in K_SWEEP) + " |",
        "|---|" + "|".join("--:" for _ in K_SWEEP) + "|",
        "| Recall@K | " + " | ".join(f"{recall[k]:.3f}" for k in K_SWEEP) + " |",
        "",
        f"Chosen rerank shortlist: top-{pool} (recall@{best_k} = "
        f"{recall.get(best_k, recall[max(K_SWEEP)]):.3f}).",
        "",
        "## Stage 2 - Rerank quality: dense vs Qwen cross-encoder rerank",
        "| M | metric | dense | reranked | Δ |",
        "|--:|---|--:|--:|--:|",
    ]
    for m in M_LIST:
        for name, key in (("nDCG@M", "ndcg"), ("Hit@M", "hit"), ("P@M", "p")):
            d = float(np.mean(metrics["dense"][m][key]))
            r = float(np.mean(metrics["rerank"][m][key]))
            cap = f" (max {1 / m:.2f})" if key == "p" else ""
            lines.append(f"| {m} | {name}{cap} | {d:.3f} | {r:.3f} | {r - d:+.3f} |")
    (OUT / "scoreboard.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[5:]))
    _plot(recall)
    print(f"\nwrote {OUT}/scoreboard.md, eval_set.jsonl, curve.png")


def _plot(recall: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(list(recall), list(recall.values()), "o-", color="#2c7fb8", lw=2)
    ax.set_xscale("log")
    ax.set_xticks(K_SWEEP)
    ax.set_xticklabels([str(k) for k in K_SWEEP])
    ax.set_xlabel("K (retrieved)")
    ax.set_ylabel("Recall@K")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.set_title("RAG dense Recall@K (Qwen3-Embedding-0.6B)")
    fig.tight_layout()
    fig.savefig(OUT / "curve.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
