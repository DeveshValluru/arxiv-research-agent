from pathlib import Path
from statistics import median

from transformers import AutoTokenizer

from arxiv_agent.ingestion.chunker import _approx_tokens, chunk_paper
from arxiv_agent.ingestion.html_parser import parse_arxiv_html

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")


def count_real_tokens(text: str) -> int:
    return len(tokenizer(text, verbose=False)["input_ids"])


print("How text becomes tokens:")
for text in [
    "LLM judges prefer the first answer.",
    "$\\mathcal{P}_{\\mathcal{LLM}}$",
    "arXiv:2306.05685",
]:
    pieces = tokenizer.tokenize(text)
    print(f"  {len(text):3} chars -> {len(pieces):3} tokens  {pieces}")

paper = parse_arxiv_html(
    Path("data/html/2411.15594v6.html").read_text(encoding="utf-8")
)

for label, counter in [
    ("estimate", _approx_tokens),
    ("real tokenizer", count_real_tokens),
]:
    chunks = chunk_paper(paper, "2411.15594", 6, count_tokens=counter)
    rows = [
        (count_real_tokens(c.embed_text), _approx_tokens(c.embed_text), c)
        for c in chunks
    ]
    ratios = sorted(real / estimate for real, estimate, _ in rows)
    over = sum(real > 512 for real, _, _ in rows)

    print(f"\n=== Chunked using the {label} ({len(chunks)} chunks)")
    print(
        f"real / estimate: min {ratios[0]:.2f}, "
        f"median {median(ratios):.2f}, max {ratios[-1]:.2f}"
    )
    print(f"chunks over 512 real tokens: {over}")
    print("Largest chunks by real tokens:")
    for real, estimate, chunk in sorted(rows, key=lambda r: r[0], reverse=True)[:3]:
        path = " > ".join(chunk.section_path)[-45:]
        print(f"  {chunk.chunk_id}  real {real:3}  estimated {estimate:3}  ...{path}")
