import json
import time
from pathlib import Path

import numpy as np

from arxiv_agent.evals.retrieval import (
    first_hit_rank,
    gold_indices,
    mean_reciprocal_rank,
    recall_at_k,
)
from arxiv_agent.ingestion.chunker import MAX_TOKENS, chunk_paper
from arxiv_agent.ingestion.embedder import BGE_QUERY_PREFIX, Embedder
from arxiv_agent.ingestion.html_parser import parse_arxiv_html
from arxiv_agent.ingestion.models import ParsedPaper

PAGE = Path("data/html/2411.15594v6.html")
EVAL_SET = Path("evals/retrieval_survey.jsonl")
HEADER_ALLOWANCE = 64
QWEN_QUERY_PROMPT = (
    "Instruct: Given a web search query, retrieve relevant passages "
    "that answer the query\nQuery:"
)
MODELS = [
    ("MiniLM-L6", "sentence-transformers/all-MiniLM-L6-v2", ""),
    ("bge-small", "BAAI/bge-small-en-v1.5", BGE_QUERY_PREFIX),
    ("bge-base", "BAAI/bge-base-en-v1.5", BGE_QUERY_PREFIX),
    ("bge-large", "BAAI/bge-large-en-v1.5", BGE_QUERY_PREFIX),
    ("qwen3-0.6B", "Qwen/Qwen3-Embedding-0.6B", QWEN_QUERY_PROMPT),
]


def evaluate(
    model_id: str, query_prefix: str, paper: ParsedPaper, questions: list[dict]
) -> dict:
    embedder = Embedder(model_id, query_prefix=query_prefix)
    budget = min(MAX_TOKENS, embedder.max_tokens - HEADER_ALLOWANCE)
    chunks = chunk_paper(
        paper, "2411.15594", 6, max_tokens=budget, count_tokens=embedder.count_tokens
    )
    texts = [c.text for c in chunks]

    start = time.perf_counter()
    vectors = embedder.embed_passages([c.embed_text for c in chunks])
    seconds = time.perf_counter() - start

    ranks = []
    for question in questions:
        scores = vectors @ embedder.embed_query(question["query"])
        ranking = np.argsort(-scores).tolist()
        gold = gold_indices(texts, question["evidence"])
        ranks.append(first_hit_rank(ranking, gold))
    return {
        "chunks": len(chunks),
        "dim": embedder.dimension,
        "seconds": seconds,
        "ranks": ranks,
    }


def main() -> None:
    paper = parse_arxiv_html(PAGE.read_text(encoding="utf-8"))
    lines = EVAL_SET.read_text(encoding="utf-8").splitlines()
    questions = [json.loads(line) for line in lines if line.strip()]

    results = {}
    for name, model_id, prefix in MODELS:
        print(f"running {name} ...")
        results[name] = evaluate(model_id, prefix, paper, questions)

    print(
        f"\n{'model':11} {'chunks':>6} {'dim':>5} {'embed s':>7}  "
        f"{'R@1':>4} {'R@3':>4} {'R@5':>4} {'MRR':>4}"
    )
    for name, r in results.items():
        ranks = r["ranks"]
        print(
            f"{name:11} {r['chunks']:6} {r['dim']:5} {r['seconds']:7.1f}  "
            f"{recall_at_k(ranks, 1):4.2f} {recall_at_k(ranks, 3):4.2f} "
            f"{recall_at_k(ranks, 5):4.2f} {mean_reciprocal_rank(ranks):4.2f}"
        )

    for kind in ("paraphrase", "exact_term"):
        picked = [i for i, q in enumerate(questions) if q["type"] == kind]
        scores = "  ".join(
            f"{name} {recall_at_k([r['ranks'][i] for i in picked], 5):.2f}"
            for name, r in results.items()
        )
        print(f"\nrecall@5 on {kind} ({len(picked)} questions):  {scores}")

    print("\nRank of the first right chunk per question ('-' = not found):")
    print("  Q  " + " ".join(f"{name:>10}" for name in results))
    for i, question in enumerate(questions):
        cells = " ".join(f"{r['ranks'][i] or '-':>10}" for r in results.values())
        print(f"{i + 1:3}  {cells}   {question['query'][:48]}")


if __name__ == "__main__":
    main()
