from pathlib import Path
from statistics import median

from arxiv_agent.ingestion.chunker import MAX_TOKENS, _approx_tokens, chunk_paper
from arxiv_agent.ingestion.html_parser import parse_arxiv_html

PAGE = Path("data/html/2411.15594v6.html")

paper = parse_arxiv_html(PAGE.read_text(encoding="utf-8"))
chunks = chunk_paper(paper, "2411.15594", 6)

kinds = {
    kind: sum(c.kind == kind for c in chunks) for kind in ("abstract", "text", "table")
}
sizes = sorted(c.token_count for c in chunks)
over_text = sum(_approx_tokens(c.text) > MAX_TOKENS for c in chunks)
print(f"{len(chunks)} chunks: {kinds}")
print(f"embed_text tokens: min {sizes[0]}, median {median(sizes)}, max {sizes[-1]}")
print(f"chunks whose text is over {MAX_TOKENS} tokens: {over_text}")
print(
    f"chunks over the 512-token embedder limit: {sum(c.token_count > 512 for c in chunks)}"
)

print("\nLargest chunks:")
for c in sorted(chunks, key=lambda c: c.token_count, reverse=True)[:3]:
    path = " > ".join(c.section_path)[:60]
    print(f"  {c.chunk_id}  {c.kind:8} {c.token_count:4} tok  {path}")

sections_text = " ".join(s.text for s in paper.sections)
words_in = f"{paper.abstract} {sections_text}".split()
words_out = " ".join(c.text for c in chunks if c.kind != "table").split()
same = words_in == words_out
print(f"\nwords in: {len(words_in)} | words out: {len(words_out)} | identical: {same}")

print("\nOne chunk, as the embedder will see it:\n")
print(chunks[5].embed_text[:500])
