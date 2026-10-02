# arXiv Research Agent — Architecture v2

Scope: **A** literature review · **B** paper Q&A.
(The research radar, mode C, was dropped on 2026-09-24. **D**, a reproducibility agent that runs paper code in a sandbox, may be added later.)

This is the reference map. We'll walk through it one piece at a time (see "Walkthrough order" at the bottom). You don't need to absorb it all at once.

Cross-cutting decisions carried over from the financial agent (versioning, auth, secrets, cost control, API + SSE) are in `../financial_agent/FINANCIAL_AGENT_ARCHITECTURE.md`.

---

## Diagram 1 — System overview

```
                        ┌──────────────┐
   You (browser) ─────▶ │ Web UI       │
                        │ (Next.js)    │
                        └──────┬───────┘
                               │ HTTP + SSE
                               ▼
        ┌──────────────────────────────────────────────────────┐
        │ API layer (FastAPI)                                  │
        │ auth · rate limits · input guardrails ·              │
        │ trace tagging · SSE · background jobs (deep review)  │
        └───────────┬──────────────────────────┬───────────────┘
                    │                          │
                    ▼                          ▼
          ┌──────────────────┐       ┌──────────────────┐
          │ Graph A          │       │ Graph B          │
          │ Literature review│       │ Paper Q&A        │
          └─────────┬────────┘       └─────────┬────────┘
                    └────────────┬─────────────┘
                                 │ tool calls (allowlisted per agent)
                                 ▼
        ┌──────────────────────── TOOLS ───────────────────────┐
        │ MCP: arXiv server        MCP: Semantic Scholar server│
        │ (you build)              (you build)                 │
        │                                                      │
        │ Internal: library_search (hybrid + rerank)           │
        │           ensure_ingested(id)                        │
        └───────────┬────────────────────────────────┬─────────┘
                    │                                │
                    ▼                                ▼
        ┌─────────────────────────┐   ┌──────────────────────────────┐
        │ On-demand ingestion     │──▶│ Data stores                  │
        │ fetch → parse →         │   │ Postgres: papers, users,     │
        │ chunk → embed → index   │   │   reading lists, jobs        │
        └─────────────────────────┘   │ Vector + BM25 chunk index    │
                                      │ PDF cache · eval sets (git)  │
                                      └──────────────────────────────┘

   Every box emits traces ──▶ Langfuse ──▶ flywheel (Diagram 8)
   Output guardrails (citation verifier, claim-support check) run before any answer leaves a graph.
```

**Mode selection is a button, not an LLM.** The UI calls a different endpoint per mode (`/api/review`, `/api/ask`), so there's no LLM "router". Don't pay an LLM to make a decision the user already made.

### What the user provides

| Mode | You give | The system finds |
|---|---|---|
| A. Literature review | A question. Optional: year range, quick/deep, seed papers you already like | The papers, the evidence, the written review |
| B. Paper Q&A | A paper (arXiv ID, link, or title) + a question | The passages that answer it (and cited papers if needed) |

### Depth: quick vs deep (mode A)

- **Quick** (~20–40 s): abstracts only, no full-text ingestion. For "what's out there on X?"
- **Deep** (~2–5 min first time): the full funnel with full-text reading. For "write my related-work section."

---

## Build map: layers and which phase builds them

Each layer only uses the layer below it. We build bottom-up, so every layer can be tested on its own before anything depends on it.

```
 LAYER 5  UI + API         Web UI (Next.js) · FastAPI · SSE · background jobs      Phase 8 (UI); API from Phase 2
              │
 LAYER 4  AGENTS           Graph B: Paper Q&A                                      Phase 2 (first end-to-end!)
                           Graph A: Literature review                              Phase 5
              │ call
 LAYER 3  TOOLS            MCP servers (thin wrappers over clients)                Phase 4
                           library_search · ensure_ingested                        Phase 1–3
              │ use
 LAYER 2  BUILDING BLOCKS  Ingestion: fetch → parse → chunk → embed → index        Phase 1.2–1.5
                           Retrieval: hybrid search + rerank                        Phase 2–3
              │ use
 LAYER 1  CLIENTS          clients/arxiv.py                                        Phase 1.1
                           clients/semantic_scholar.py                             Phase 4
              │ HTTP
 OUTSIDE                   arXiv API · Semantic Scholar API · HF models · Postgres

 CROSS-CUTTING             Langfuse traces (from Phase 2) · guardrails (Phase 6) · evals + CI (seeded Phase 2, gated Phase 7)
```

### Who uses `clients/arxiv.py`

| Caller | Function | Why |
|---|---|---|
| Graph A Searcher (through the MCP tool) | `search_papers` | Top of the funnel: find candidate papers |
| Quick mode (A) | `search_papers` | The abstracts *are* the material |
| Graph B "Resolve paper" | `search_papers` | Title → arXiv ID |
| `ensure_ingested` | `get_metadata` | Latest version, title, authors, which source to fetch |
| Citation Critic | `get_metadata` | Does this ID exist? Does the title match? |

### Each layer has one job

| Layer | Its job | Not its job |
|---|---|---|
| Client | Talk to arXiv correctly: build queries, pace requests, retry, parse XML, normalize fields, return a **complete, clean, typed record** | Deciding what to search; trimming for an LLM; storing anything |
| Tool (MCP) | Shape the record for the model: **trim to ~5 fields**, cap limits, readable errors | Talking HTTP; parsing XML |
| Agent | Judgment: which query, which paper, what's relevant | Knowing how arXiv works |

### Component spec: `clients/arxiv.py` (Phase 1.1)

```
 search_papers(query, max_results, sort_by)        get_metadata(ids)
            │ builds query params                        │ builds id_list params (batches of up to 50)
            └───────────────────┬────────────────────────┘
                                ▼
                     _request(params)    ← the ONLY place that does HTTP:
                                │          3 s pacing · timeout · retries · User-Agent
                                │ XML text
                                ▼
                     _parse_feed(xml)    ← pure function, no network
                                │          "Error" entry → ArxivQueryError
                                │          per entry: _split_id, _clean, dates
                                ▼
                     list[PaperSummary]
```

| Piece | Input | Output | Why it exists |
|---|---|---|---|
| `PaperSummary` | — | Record: `arxiv_id`, `version`, `title`, `authors`, `abstract`, `published`, `updated`, `primary_category`, `categories` | One clean, typed shape for every caller. No URLs: code builds them from the ID |
| `search_papers` | query string (field prefixes like `abs:`, `ti:`), `max_results` (hard cap 100), `sort_by` | `list[PaperSummary]` (empty list = no results, not an error) | Funnel top, title → ID, quick mode |
| `get_metadata` | list of IDs, with or without version (`2401.12345`, `2401.12345v2`, old-style `hep-th/9901001`) | `dict[id → PaperSummary]`; a missing key = not found | Citation Critic, `ensure_ingested`. Batched: 14 citations = 1 request, not 14 |
| `_request` | query params | raw XML text | Politeness and retries enforced in one place nobody can bypass. Retries 429 (rate limited, honoring `Retry-After`), 503, and timeouts with backoff (max 3); never retries 400. Per arXiv's API Terms of Use: at most 1 request every 3 s and **one connection at a time** (so parallel ingestion shares one paced client); never circumvent a block (no switching machines or IPs) |
| `_parse_feed` | XML text | `list[PaperSummary]` | Pure, so it's testable with saved XML and no network |
| `_split_id` | `http://arxiv.org/abs/2401.12345v2` | `("2401.12345", 2)` | Versions drive supersession |
| `_clean` | text with newlines and double spaces | single-spaced text | Defensive. The live API returned single-line titles and abstracts (2026-10-01), but older records and other sources may contain hard line breaks |
| `ArxivClient` (decided in 1.1d) | optional `http: httpx.Client`, `sleep` | an object with `search_papers` / `get_metadata` methods | Holds **state**: one shared HTTP connection and the time of the last request (pacing). Tests inject a fake `http` (MockTransport) and a fake `sleep`, so retry and pacing logic is tested offline |
| `ArxivError` | — | base class of both errors below | Callers can catch "any arXiv problem" with one `except`, or tell the two apart |
| `ArxivQueryError` | — | raised for a bad query | Caller's fault: the tool tells the LLM to fix the query |
| `ArxivUnavailableError` | — | raised when arXiv is down after retries | Not the caller's fault. **The Critic must not mark a citation fake just because arXiv was down.** |

---

## Diagram 2 — Corpus strategy: the funnel

You can't pre-ingest arXiv: 2M+ papers, ~100M+ chunks, months of PDF parsing, and tens of thousands of new papers a month. Instead, do what a human researcher does: **search, skim abstracts, then read only a few papers in full.**

```
 arXiv: 2M+ papers
      │
      │  arXiv API search: metadata + abstracts only        ← cheap, no PDFs
      ▼
 ~50–100 candidate papers (title + abstract)
      │
      │  Screener: rank abstracts against the question      ← small model
      ▼
 ~5–15 papers worth reading
      │
      │  ensure_ingested(): fetch, parse, chunk,            ← expensive, but only here
      │  embed, index
      ▼
 Full-text chunks in the library index
      │
      │  hybrid search + rerank, scoped to these papers
      ▼
 Top chunks ──▶ Synthesizer / Answerer
```

- **Warm cache.** Once a paper is ingested it stays. The library grows with use, and papers you save are always there.
- **Versions.** arXiv papers have versions (`2401.12345v1`, `v2`). A new version supersedes the old one, the same lesson as amended 10-Ks.

---

## Diagram 3 — Graph A: Literature review

```
 question: "related work on X"
   │
   ▼
 Planner ────────── splits the topic into 3–6 sub-queries + inclusion criteria
   │
   ▼
 Searcher ───────── arXiv search per sub-query
   │                + Semantic Scholar: follow references of the best hits ("snowballing")
   ▼
 Screener ───────── scores abstracts, keeps top N, records why each was kept or dropped
   │
   ▼
 PAUSE: your review  (deep mode) graph pauses via LangGraph interrupt; you see the
   │                 kept list + reasons, remove papers or add one by arXiv ID, then continue
   ▼
 Reader ─────────── ensure_ingested() per paper, then extracts key claims:
   │                {claim, arxiv_id, chunk_id, quote}
   ▼
 Synthesizer ────── writes the section from the claims, cites [arXiv:ID]
   │
   ▼
 Citation Critic ── deterministic: every arXiv ID exists and the title matches
   │                model-based: is each claim supported by its cited chunk?
   │
   ├── fails + budget left ──▶ back to Searcher (coverage gap) or Synthesizer (rewrite)
   ▼
 answer + bibliography + evidence map (claim → quote → paper)
```

- The graph state carries a **budget** (max iterations, tokens, deadline), the same as the financial agent.
- **Human-in-the-loop checkpoint (deep mode).** The pause sits right before the expensive step, so a bad paper list costs you seconds instead of minutes of reading. While paused, the graph state is saved to Postgres by a LangGraph checkpointer. Nothing is held open, and the job shows `awaiting_review` until you continue. It expires after 24 h.
- **Your edits are labels.** A paper you remove is a Screener false positive. A paper you add is something search or screening missed. Both go into the eval set.
- **Deep reviews run as background jobs.** A deep review takes minutes. If it ran inside one HTTP request, closing the browser tab would kill it. So it runs as a job with a `job_id`, and the SSE stream just *watches* the job. You can close the tab, reconnect, and pick up the progress.

---

## Diagram 4 — Graph B: Paper Q&A

```
 question + paper (arXiv ID, URL, or title)
   │
   ▼
 Resolve paper ──── title → arXiv ID via search; asks you if it's ambiguous
   │
   ▼
 ensure_ingested(paper)  (+ optionally its references via Semantic Scholar)
   │
   ▼
 Retrieve ───────── hybrid search + rerank, scoped to this paper (and its refs)
   │
   ▼
 Answerer ───────── answers with section/page citations
   │
   ▼
 Output guardrails ─ citation check; says "the paper doesn't say" when unsupported
   │
   ▼
 answer
```

Graph B is the **Phase 2 baseline**: the simplest complete RAG system in the project.

---

## Diagram 5 — Tools and MCP

```
        ┌──────────── your app = MCP client ────────────┐
        │ LangGraph agents call tools through it        │
        └───────────┬───────────────────────┬───────────┘
                    │                       │
                    ▼                       ▼
            arXiv MCP server        Semantic Scholar MCP server
            (you build)             (you build)
            ─────────────           ─────────────
            search_papers           get_paper
            get_metadata            get_references
            get_source_url          get_citations
```

**Why these are MCP servers and `library_search` isn't:** MCP is for tool boundaries you'd reuse from other clients. Your arXiv server works in Claude Desktop or any other MCP client too, which makes it a demo on its own. `library_search` and `ensure_ingested` are tied to your database, so they stay as plain internal functions.

**A tool is plain code, not an LLM call.** One tool use = LLM call → your code runs the function → next LLM call reads the result. The tool costs no tokens; its result does, which is why results are trimmed. Our tools contain no LLMs, so they stay fast, cheap, and deterministic.

### Tool allowlist per agent (least privilege)

**Code decides** when the next step is predictable. **The LLM decides** when the step needs judgment.

| Agent | Tools it may call | Who decides the calls |
|---|---|---|
| Planner | none | plain LLM call |
| Searcher | `search_papers`, `get_metadata`, `get_references`, `get_citations` | **LLM** (agentic loop: search, look at results, refine the query or follow references; max ~8 calls) |
| Screener | none (reads abstracts from state) | plain LLM call |
| Reader | `ensure_ingested`, `library_search` | **Code** (fixed steps); the LLM only extracts claims from what the code retrieved |
| Synthesizer | none | plain LLM call |
| Citation Critic | `get_metadata`, `library_search` | **Code** (one lookup per citation); an LLM judges claim support without tools |
| Answerer (B) | `library_search`, `get_references`, `ensure_ingested` (max 2 cited papers per question) | **LLM** (decides whether to search again or look into a cited paper; max ~4 calls) |

Allowlists matter most for LLM-decided calls. For code-decided calls, the code itself is the allowlist.

The Reader reads untrusted PDFs, so it only gets read-only tools. Even if a hidden instruction fools it, it has nothing dangerous to call. **Permissions are a guardrail.**

### Tool design rules

1. **The description is a prompt.** Name, docstring, and argument names are all the model sees. Say what the tool does, when to use it, what it returns, its limits, and which tool to use instead when two overlap.
2. **Return only what the model needs.** Trim raw API output (e.g. arXiv XML, ~600 tokens per paper) to the useful fields (~250 tokens).
3. **Handle transient errors inside the tool** (rate limits, 503s: pace and retry). **Return actionable errors to the model** as structured data with a hint (`no_results`, `not_found`). Never return stack traces.
4. **Few tools, one job each,** with names that can't be confused. No "mode" arguments.
5. **Strict arguments:** enums over free text, sensible defaults, limits and validation enforced in code, not trusted from the docstring.
6. **Test and trace tools like code:** unit tests without an LLM, one Langfuse span per call, and a tool-choice eval (right tool, right arguments).
7. **Tools take identifiers, never URLs.** `ensure_ingested` takes an arXiv ID; our code builds the arxiv.org address. No model-written text can make the system visit an arbitrary website.

If D (reproducibility) is added later, it brings the first truly dangerous tool: running code. That's when sandboxing and human approval come back into the design.

---

## Diagram 6 — Guardrails: four layers

```
 your input ──▶ [1 INPUT] ──▶ agents ──▶ [3 TOOL-CALL] ──▶ tools
                                ▲                          │
                                │                          ▼
                          [2 CONTENT] ◀────────── PDFs, abstracts, API results
                                │
 draft answer ──▶ [4 OUTPUT] ──▶ you
```

| Layer | Threat | Deterministic checks (code) | Model-based checks | When it fires |
|---|---|---|---|---|
| **1 Input** | Using the agent as a free general chatbot, user-typed injection, huge inputs | Length limit, rate limit | Small topic classifier; injection classifier (e.g. Prompt Guard) | Reject with a friendly message |
| **2 Content** | Hidden instructions in PDFs, instruction-like text in abstracts, ranking manipulation ("this paper is essential reading") aimed at the Screener | Hidden-text detection at parse time (white/tiny/off-page text); retrieved text wrapped in `<paper_content>` delimiters ("spotlighting") | Injection classifier on chunks | Strip or flag the chunk; trace it |
| **3 Tool-call** | Wrong tool, bad arguments, runaway loops, hammering arXiv | Per-agent allowlist, schema validation, argument caps, call-count caps, budget, arXiv pacing. Future write tools get **argument pinning** | none needed | Refuse the call, return an error the model can read |
| **4 Output** | Fake citations, unsupported claims, citing papers outside `kept`, unknown links, system prompt leaks | Citation verifier (ID exists, title matches, ID in `kept`); URL allowlist (arxiv.org, doi.org, semanticscholar.org); prompt-leak string match; output schema validation | Claim-support check (NLI model or LLM) | Revise loop, or drop the sentence / say "the paper doesn't say" |

### Guardrail principles

1. **Assume the model will sometimes be fooled.** No detector catches every injection. Design so that a fooled model *can't do much*: least privilege (layer 3) is the real safety net, and detection (layers 1, 2) is a bonus.
2. **Deterministic checks first.** They're free, instant, and never wrong in the same way twice. Use model-based checks only for judgment calls like "does this quote support this claim?"
3. **Every guardrail can misfire.** A strict topic classifier blocks real research questions. Measure both directions: the red-team block rate (should be high) and the false-block rate on normal questions (should be low).
4. **Every firing is traced.** Guardrail hits are free flywheel signals (Diagram 8).

---

## Diagram 7 — On-demand ingestion

```
 ensure_ingested(arxiv_id)
   │
   ├── already in the library at this version? ──▶ return (cache hit)
   ▼
 fetch source ────── source ladder, best format first (paced for arXiv):
   │                   1. arXiv HTML  (real tables, headings, math markup)
   │                   2. PDF → Docling / Marker (tables) + GROBID (references)
   │                   (later: LaTeX source, the most exact but messiest)
   ▼
 parse ───────────── sections, tables, equations (kept as LaTeX),
   │                 figure captions, references → structured list
   │                 + hidden-text detection: white / tiny text     ← guardrail layer 2
   │                   (PyMuPDF font color + size; style checks for HTML)
   ▼
 chunk by section ── Abstract / Intro / Method / Results / ...
   │                 • one table = one chunk, never split
   │                   (embed a description, return the raw markdown table)
   │                 • never split mid-equation
   │                 • reference list is NOT embedded (keyword pollution)
   ▼
 embed ───────────── specter2 vs bge-large (Phase 1 benchmark)
   ▼
 index ───────────── vector + BM25, metadata:
   │                 {arxiv_id, version, section, page, title, authors, year}
   ▼
 supersede older versions (v1 → v2)
```

Invariants (from the financial agent): **idempotent** on `(arxiv_id, version, chunk_no)`, **incremental**, **reversible**.

---

## Diagram 8 — Flywheel

```
 Langfuse traces
   │
   │  automatic flags: citation-verifier failures, claim-support failures,
   │  budget_exceeded, your 👎 on answers, online judge scores
   ▼
 Review queue ──▶ you confirm ──▶ eval sets (A / B / red-team)
   │
   ▼
 Eval runner (offline judge, different model family from the Synthesizer)
   │
   ▼
 CI gate (small subset on every PR, full set on main) ──▶ deploy ──▶ new traces
```

Guardrail failures are **free flywheel signals**: they're deterministic, so no judge is needed to flag them.

### Evals per mode

| Mode | Eval set | Metrics |
|---|---|---|
| A | Survey papers: their reference lists are the gold set | Recall of expert-cited papers (per funnel stage), citation validity %, claim-support rate |
| B | QASPER + your own questions | Answer correctness (judge), evidence retrieval recall |
| Both | Red-team: PDFs with planted hidden instructions | Injection success rate (target: 0) |
| Both | Context hygiene: dropped paper IDs in later prompts | Leak count (target: 0) |

---

## Data stores

| Store | Holds |
|---|---|
| Postgres | Paper metadata, users, reading lists, background jobs |
| Chunk index | Vectors + BM25. Leaning Postgres + pgvector + full-text search, so there's one database to run (decide in 1.5) |
| PDF cache | Downloaded sources + parsed text, so we can re-chunk without re-downloading |
| Langfuse | Traces, tagged with user, system version, models, prompt versions, index name |
| Git | Prompts, configs, eval sets |

---

## Models per role (size class; exact picks happen just-in-time)

Names are as of mid-2026. We'll confirm the latest on Hugging Face when we pick.

| Role | Size class | Candidates to benchmark |
|---|---|---|
| Embedder | small | `allenai/specter2`, `BAAI/bge-large-en-v1.5`, optionally `Qwen/Qwen3-Embedding-0.6B` |
| Reranker | small | `BAAI/bge-reranker-v2-m3`, `Qwen/Qwen3-Reranker-0.6B` |
| Planner, Screener | small–medium | `Qwen/Qwen3-8B`, `meta-llama/Llama-3.1-8B-Instruct` |
| Reader (claim extraction) | medium | `Qwen/Qwen3-14B`, `google/gemma-3-12b-it` |
| Synthesizer, Answerer | largest affordable | `Qwen/Qwen3-32B`, `meta-llama/Llama-3.3-70B-Instruct` |
| Claim-support check | tiny NLI model vs medium LLM | a DeBERTa NLI model vs an 8B LLM (a good benchmark) |
| Injection / safety classifier | tiny–small | Llama Prompt Guard, Llama Guard |
| Offline judge | large, **different family** from the Synthesizer | If the Synthesizer is Qwen, the judge is Llama or Gemma |

---

## Worked example: one deep literature review, query to output

Paper IDs and titles are placeholders (P1, P2, ...), not real papers. Timings are rough.

**0. You type** (mode A, deep):
> "Write a related-work section on using LLMs as judges to evaluate other LLMs."

**1. API layer (0 s).** Checks your session → `user_id`. Rate limit OK. Input guardrail: length fine, topic classifier says "research question". Opens a Langfuse trace with version tags, starts Graph A as a background job, opens the SSE stream to watch it.

**2. Planner (~2 s).** Small LLM returns:
```json
{
  "sub_queries": [
    "LLM-as-a-judge evaluation methods",
    "biases of LLM evaluators (position bias, self-preference)",
    "agreement between LLM judges and human raters",
    "reference-free evaluation of generated text with LLMs"
  ],
  "inclusion": "papers that propose or analyze LLM-based evaluators, 2022-2026",
  "exclusion": "papers that only use human evaluation"
}
```
UI shows: *Planning: 4 sub-questions*

**3. Searcher (~5 s).** Four tool calls, e.g. `search_papers(query="LLM-as-a-judge evaluation methods", max_results=25)` → 100 results → 73 after removing duplicates. Snowballing: `get_references` on the top 3 hits adds 40 → **96 candidates** after duplicates.
UI shows: *Found 96 candidate papers*

**4. Screener (~20 s).** Embedding similarity pre-ranks to the top 30, then a small LLM checks each abstract against the inclusion criteria:
```
KEEP  P1   proposes pairwise LLM judging; measures agreement with humans
KEEP  P2   documents position bias in LLM judges
KEEP  P5   shows judges prefer outputs from their own model family
DROP  P31  uses an LLM to generate training data, not to evaluate
...   9 kept, 21 dropped, every reason logged in the trace
```
UI shows the 9-paper list. This is already useful on its own.

**4b. Your review (deep mode pauses here).** You remove P7 (off-topic for you) and add a paper you already know by its arXiv ID. You click Continue, and the graph resumes with 9 papers. Your two edits are saved as Screener labels.

**5. Reader (~2–3 min).** For each of the 9 papers, `ensure_ingested()`:
- 2 are cache hits (you asked about them last week)
- 7 are new: fetch → parse → chunk → embed → index
- **Guardrail fires:** P6's PDF contains hidden white text telling AI systems to ignore their instructions. The parser detects it, strips it, and flags the trace.

Then, per paper, it searches the paper's own chunks and extracts claims with evidence:
```json
{
  "claim": "LLM judges tend to prefer whichever answer is shown first in pairwise comparisons",
  "arxiv_id": "P2",
  "chunk_id": "P2-sec4-c3",
  "quote": "<exact sentence from section 4 of P2>"
}
```
About 35 claims in total. UI shows: *Reading 3/9... 7/9...*

**6. Synthesizer (~30 s, streamed).** A large LLM groups the claims into themes (judge designs, known biases, agreement with humans) and writes the section, citing each sentence as `[arXiv:ID]`. Words stream into the UI as they're generated.

**7. Citation Critic (~10 s).**
- Deterministic check: 14 citations, each looked up with `get_metadata`. 13 match. **1 ID doesn't exist** (the model garbled an ID).
- Claim-support check: 12 of 13 sentences are supported by their cited chunk. **1 overstates it**: it says "all LLM judges" when the paper tested one model.
- Verdict: **revise** (iteration 1 of max 3). Both issues go back to the Synthesizer as a fix list.

**8. Synthesizer, second pass (~20 s).** Fixes both issues. The Critic re-checks: **pass**.

**9. Final output (~4 min total).** The `done` event delivers:
```
## Related Work
LLM-based evaluators have been proposed as a scalable alternative to
human rating [P1, P3]. Several studies document systematic biases: judges
tend to prefer the first answer shown in pairwise comparisons [P2], and
rate outputs from their own model family more highly [P5]. ...

References (all verified against arXiv)
[P1] <title>, <authors>, <year>  arXiv:<id>  ✓
...

Evidence map: every sentence → quote → paper section (click to open)
```

**10. After the response (in the background).**
- The trace is complete: every LLM call, tool call, token count, and timing
- The online judge scores the answer
- The garbled citation the Critic caught becomes an **automatic flag** in the review queue (flywheel)
- Your 👍 or 👎 on any sentence is saved to the trace

### Mode B, shorter

"What datasets did P2 evaluate on?" → resolve P2 → ingest it (~15–30 s the first time, instant after) → search its chunks + rerank → answer with section citations → citation check → done. Follow-up questions take ~3–8 s.

---

## Context engineering: what each LLM call sees

The graph **state** holds everything. Each node builds its **own prompt** from only the state fields it needs. Agents do not share one growing chat history.

| LLM call | Sees | Does NOT see | Rough prompt size |
|---|---|---|---|
| Planner | your question | anything else | ~300 tokens |
| Screener | the criteria + **one batch of ~10 abstracts** (many small calls in parallel) | full text, other batches | ~3k |
| Reader | **one paper's** top ~8 chunks: always abstract + contributions + conclusion, the rest matched to the sub-questions (one call per paper, in parallel) | other papers, the rest of this paper | ~4k |
| Synthesizer | your question + **the claims list only** (~35 claims with quotes) | any paper text, dropped papers | ~5k |
| Citation Critic (claim support) | **one sentence + the chunk it cites** (one call per sentence) | everything else | ~600 |
| Citation Critic (ID check) | no LLM, deterministic code | — | 0 |

The largest prompt is about 5k tokens. That fits easily in the context window of an 8B model.

### Five rules that keep context clean

1. **Typed state fields, not a shared message list.** `candidates`, `kept`, `dropped`, `claims`, and `draft` are separate fields. The Synthesizer's prompt template reads only `claims`. Dropped papers aren't in its input because the code never puts them there. We don't rely on the model to ignore them.
2. **Scoped retrieval.** The library is shared across all requests, so it holds papers from last week's unrelated questions too. Every `library_search` in a review filters on `arxiv_id IN kept`.
3. **Distill between stages.** Papers become chunks, chunks become claims, claims become the draft. Each stage passes on something smaller and cleaner.
4. **Allowed-citation check.** The Critic confirms every cited ID is in `kept`. This also catches papers the model remembers from pretraining and cites from memory.
5. **Context-hygiene test.** Automated check on traces: no dropped paper ID appears in any prompt after the Screener.

If a topic grows to 30+ papers and the claims no longer fit, synthesize per theme first, then combine the theme summaries. That's map-reduce again.

---

## Walkthrough order (Phase 0)

1. ~~**The funnel** (Diagram 2)~~ done, plus the worked example, context engineering, and parsing
2. ~~**Long requests as background jobs**~~ done
3. **Tools and MCP** (Diagram 5): what MCP is and why least privilege matters ← next
4. **Guardrails** (Diagram 6): the four layers
5. **Minimal project layout**, then Phase 1
