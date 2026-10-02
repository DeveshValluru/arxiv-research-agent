# Curriculum — arXiv Research Agent

Our shared map. Claude follows it so context isn't lost; you use it to see where we are.

**Rules of engagement:**
- Claude teaches, you code. Claude does not fill code files for you (the frontend is the exception; Claude writes it).
- One concept at a time. When you say "got it" we move on.
- Every decision gets a one-liner in the Decisions Log.

**What we're building (scope A + B):**
- **A. Literature review.** "Write a related-work section on X": plan, search, read, synthesize, verify citations.
- **B. Paper Q&A.** Chat with a paper and the papers it cites.

**Maybe later:** **D. Reproducibility agent.** Reads a paper, runs its code in a sandbox, and compares results to the reported numbers. (The research radar, C, was dropped.)

The architecture is in `ARXIV_AGENT_ARCHITECTURE.md`. The financial agent (now the active project) lives in `../financial_agent/`, and its Phase 0 decisions are reused here.

---

## Current position
> **ACTIVE (resumed 2026-10-01).** The finance agent runs in a separate chat. **Phase 1 · Step 1.1c**: typed `PaperSummary` + parsing, built against the real fixture `tests/fixtures/arxiv_search_llm_judge.xml`. 1.1b is committed (`f926f28`, `d5ffe9b`).
>
> *History:* 2026-09-30 to 10-01, arXiv returned 429 to this machine (home and mobile networks) while its status page showed the API up; likely a shared-IP block. Access returned on 10-01. Lessons: respect the Terms of Use, never circumvent; develop on fixtures; fallback sources (OpenAlex works; Semantic Scholar needs a key) if it happens again. Runs need `uv run --env-file .env ...` so the contact User-Agent (name + email) is sent. Real titles and abstracts come back single-line, so `_clean` is defensive; the synthetic fixture stays as an edge case.

---

## Phase plan

### Phase 0 · Architecture ✅
- [x] 0.5 Diagram → `ARXIV_AGENT_ARCHITECTURE.md` (v2: radar removed)
- [x] 0.1 Walkthrough: the funnel (Diagram 2). Also covered: worked example, context engineering, table parsing
- [x] 0.2 Walkthrough: long requests run as background jobs; "gate actions, not agents"
- [x] 0.3 Walkthrough: tools and MCP. What a tool call is, MCP roles, tool design rules, workflow vs agent (Diagram 5)
- [x] 0.4 Walkthrough: guardrails, the four layers (Diagram 6). Also covered: tools take IDs not URLs, ranking manipulation
- [x] 0.6 Minimal project layout: uv project, own git repo, first commit `5708026`

### Phase 1 · Paper ingestion (we are here)
- [ ] 1.1 arXiv API client (`src/arxiv_agent/clients/arxiv.py`, branch `phase-1/arxiv-client`)
  - [ ] 1.1a Look at a raw API response in the browser
  - [x] 1.1b First call: search, print id + title (httpx + feedparser). Blocked by arXiv 429s on 2026-09-30; **verified live on 2026-10-01**; real fixture saved (`tests/fixtures/arxiv_search_llm_judge.xml`)
  - [x] 1.1c Parse into a typed `PaperSummary` (clean whitespace, split id and version). Pydantic model with `extra="forbid"`; `_split_id` (regex, raises on bad IDs), `_clean`, pure `_parse_feed`; verified on real + synthetic fixtures (2026-10-02)
  - [ ] 1.1d Politeness + errors (built and tested OFFLINE with a fake arXiv; checked live later):
    - [x] 1.1d-1 Error classes: `ArxivError` → `ArxivQueryError`, `ArxivUnavailableError`
    - [x] 1.1d-2 `ArxivClient` class: holds one shared `httpx.Client` (injectable for tests); `search_papers` becomes a method; first pytest test with `httpx.MockTransport` (`tests/test_arxiv_client.py`)
    - [x] 1.1d-3 `_request` method (6 offline tests passing): retries 429/503/5xx/timeouts with backoff (honor `Retry-After`), raises `ArxivQueryError` on other 4xx (no retry), `ArxivUnavailableError` after the last attempt
    - [x] 1.1d-4 Pacing: at least 3 s between requests (injectable `sleep` + `clock`; tests use a `FakeClock` whose sleep advances its own time). 9 offline tests passing
    - [ ] Later, when ingestion runs in parallel: a lock so only one request is in flight (arXiv: "one connection at a time")
  - [ ] 1.1e Tests without the network (saved XML samples)
  - [ ] 1.1f `get_metadata(ids)` via `id_list`
- [ ] 1.2 Paper parsing: arXiv HTML first, PDF fallback (benchmark Docling / Marker / GROBID); hidden-text detection
- [ ] 1.3 Chunking by section; tables as single chunks (embed description, return raw table); references as a structured list, not embedded
- [ ] 1.4 Embedder benchmark: `allenai/specter2` (scientific) vs `bge-large` (general)
- [ ] 1.5 Vector DB + keyword index + paper cache

### Phase 2 · Baseline RAG = Paper Q&A (B)
- [ ] 2.1 Single-paper Q&A with one generator
- [ ] 2.2 Langfuse instrumentation from the first LLM call
- [ ] 2.3 Seed eval set (QASPER + your own questions); baseline metrics

### Phase 3 · Hybrid search + reranking
- [ ] 3.1 BM25 + dense fusion
- [ ] 3.2 Cross-encoder reranker
- [ ] 3.3 Measure delta vs Phase 2

### Phase 4 · MCP servers + tools
- [ ] 4.1 Build an arXiv MCP server (`search_papers`, `get_metadata`, `get_source_url`)
- [ ] 4.2 Build a Semantic Scholar MCP server (`get_paper`, `get_references`, `get_citations`)
- [ ] 4.3 Connect as a client; tool-call tracing

### Phase 5 · Multi-agent literature review (A)
- [ ] 5.1 LangGraph: Planner → Searcher → Screener → Reader → Synthesizer → Citation Critic
- [ ] 5.2 State schema, budget, loop control; deep reviews as background jobs
- [ ] 5.3 Human-in-the-loop: pause after the Screener (interrupt + Postgres checkpointer); remove/add papers; edits become Screener labels
- [ ] 5.4 Survey-paper eval: did it find the papers the experts cited? (recall per funnel stage)

### Phase 6 · Guardrails
- [ ] 6.1 Citation verifier (every arXiv ID real, title matches, ID in the kept set)
- [ ] 6.2 Injection defense for instructions hidden in PDFs
- [ ] 6.3 Claim-support check (is the claim actually in the cited paper?)
- [ ] 6.4 Hand-built checks vs a library (e.g. Llama Guard) — compare

### Phase 7 · Evals in CI + flywheel
- [ ] 7.1 Eval runner + LLM-judge (different model family from the generator)
- [ ] 7.2 CI eval gate (subset on PR, full on main)
- [ ] 7.3 Flagged failures → eval set loop

### Phase 8 · Frontend + deploy
- [ ] 8.1 UI: chat, paper cards, reading list, citation graph (Claude writes)
- [ ] 8.2 Deploy

### Later (maybe) · D. Reproducibility agent
- Sandboxed code execution, human approval before running, comparing reported vs reproduced numbers

---

## Decisions log
| Date | Decision | Reason |
|---|---|---|
| 2026-09-23 | Project = arXiv Research Agent | Covers the AI-engineering pillars; user is a real user and can write gold answers |
| 2026-09-23 | Carry over from financial Phase 0: LangGraph, Langfuse, free HF models, flywheel + CI gate, git-SHA versioning, API-layer auth, secrets rules, 4-layer cost control, FastAPI + SSE | Cross-cutting decisions don't depend on the domain |
| 2026-09-23 | Build order: Paper Q&A (B) first as the baseline, then tools, then multi-agent (A), then guardrails | Each phase builds on the one before; B is the simplest complete system |
| 2026-09-23 | Instrument with Langfuse from the first LLM call (Phase 2), not at the end | Traces are most useful while you're still building and debugging |
| 2026-09-23 | Corpus strategy = funnel: arXiv search (abstracts) → screen → ingest only the few papers worth reading; ingested papers stay cached | Pre-ingesting 2M+ papers is infeasible, and users only ever need a tiny slice |
| 2026-09-23 | Mode chosen by endpoint (`/api/review`, `/api/ask`), not an LLM router | Don't pay an LLM for a decision the user already made |
| 2026-09-23 | MCP servers for reusable external boundaries (arXiv, Semantic Scholar); internal functions for DB-bound tools | MCP is for tools you'd reuse from other clients |
| 2026-09-23 | Per-agent tool allowlists; agents that read untrusted content get read-only tools | A fooled agent has nothing dangerous to misuse |
| 2026-09-23 | Mode A has quick (abstracts only, ~30 s) and deep (full text, minutes) depth | Many questions don't need full text; let the user choose the wait |
| 2026-09-23 | Context engineering: typed state fields (no shared message list), each node builds its own prompt, retrieval scoped to kept papers, distill between stages, allowed-citation check, context-hygiene test on traces | Keeps every prompt around 5k tokens or less, so small models work and dropped papers never leak |
| 2026-09-23 | Parsing uses a source ladder: arXiv HTML first, PDF parsers as fallback; tables, equations, and references handled by parsers, not LLMs | Deterministic tools don't change numbers; HTML already has real table structure |
| 2026-09-23 | Tables = one chunk each; embed a description, return the raw table. Reference lists not embedded | Numeric tables embed poorly; reference lists pollute retrieval |
| 2026-09-23 | Reader always includes abstract, contributions, and conclusion chunks | Papers put key claims in predictable places |
| 2026-09-24 | Deep reviews run as background jobs; SSE watches the job and can reconnect | A multi-minute run shouldn't die when a tab closes |
| 2026-09-24 | **Scope = A + B. Research radar (C) dropped**; its scheduling, inbox, and approval-gate designs removed. D (reproducibility) maybe later | User choice |
| 2026-09-24 | Principle kept for any future write tool: gate actions, not agents; pin risky arguments (e.g. a fixed email recipient) in code | Makes an action safe by construction, sometimes without needing approval |
| 2026-09-24 | Human-in-the-loop checkpoint in deep mode: pause after the Screener, you edit the paper list, then continue. Edits become Screener eval labels | Puts the human right before the expensive step; shows the HITL pattern in the project |
| 2026-09-28 | Tool design rules: description is a prompt; trimmed outputs; transient errors handled in the tool, actionable errors returned with hints; few single-purpose tools; strict args enforced in code; tools unit-tested, traced, and evaluated for tool choice | The model only sees what the tool shows it; bad tools cause wrong calls and bloated context |
| 2026-09-28 | Tools are plain code (no LLM inside). Searcher and Answerer are LLM-driven (agentic tool calling, capped at ~8 and ~4 calls); Reader and Citation Critic call tools from code | Code for predictable steps, LLM for judgment; caps stop agent loops |
| 2026-09-28 | Guardrail principles: assume the model gets fooled (least privilege is the real safety net); deterministic checks first; measure false blocks as well as catches; trace every firing. Output layer adds a URL allowlist | Detection can't be perfect, so limit what a fooled model can do |
| 2026-09-29 | Tools take identifiers (arXiv IDs), never URLs; code builds every address. No agent gets a browse/fetch-URL tool | A fooled model can't point the system at an arbitrary website |
| 2026-09-29 | Tooling: uv (src layout, package `arxiv-agent`, Python 3.12), pytest + ruff; folders mirror architecture boxes and are created just-in-time | Lockfile makes dependencies reproducible; code layout matches the diagram |
| 2026-09-29 | Git workflow: skeleton commit on main, then one short-lived branch + PR per step; semver tags per phase milestone (v0.x); always commit uv.lock; `CHUNKER_VERSION` constant in index names; eval scores stored with git SHA + eval-set version | A version = everything needed to reproduce a behavior |
| 2026-10-02 | The arXiv client becomes an `ArxivClient` class (shared `httpx.Client`, last-request time, injectable `http` + `sleep`); retry/pacing logic is tested offline with `httpx.MockTransport` | Pacing needs state, which justifies a class; you can't make real arXiv return a 429 on demand, but a fake can |
| 2026-10-02 | Records are pydantic models with `extra="forbid"`; parsing helpers are private module-level functions (`_` prefix), not class methods | Strict contract catches typos on both sides; records stay source-agnostic so an OpenAlex client could build the same `PaperSummary` |
| 2026-10-01 | `.gitattributes` with `* text=auto eol=lf` in every repo (do it before the first commit in new projects) | Consistent LF line endings across Windows and Linux CI; byte-exact fixtures; no noisy diffs |
| 2026-09-30 | One git repo per project: the arXiv agent's repo is its own folder | Focused GitHub repo, README, and CI; one clean link for the CV |
| 2026-09-30 | External API logic lives in plain clients (`clients/arxiv.py`, later `clients/semantic_scholar.py`); MCP servers are thin wrappers around them (Phase 4) | Logic testable without MCP; shared by Searcher, ingestion, and Critic |
| 2026-09-30 | Layered build, bottom-up: clients → building blocks → tools → agents → API/UI. Clients return complete clean records; the tool layer trims for the LLM | One job per layer; different callers need different fields |
| 2026-09-30 | Follow arXiv's API Terms of Use: ≤ 1 request / 3 s, one connection at a time, never circumvent a block. Develop and test against saved fixtures, not the live API. If a block persists, wait, then contact arXiv support | Hit persistent 429s with minimal traffic; circumvention is prohibited |
| 2026-09-30 | Commit what can't be regenerated (code, prompts, configs, evals, uv.lock); ignore rebuildable caches (`data/`, `.venv`) and secrets (`.env`). Eval cases from real traces are scrubbed of personal info before committing | Git is for irreplaceable, small files; some papers' licenses don't allow redistribution; user questions are private |

---

## Open questions
- Vector DB choice
- Generator / reranker / judge models (picked just-in-time per phase)
- Deploy target

---

## Glossary
- **MCP (Model Context Protocol)**: a standard way to expose tools and data to LLM apps. Covered in 0.3.
- **Indirect prompt injection**: malicious instructions hidden in content the agent *reads* (a PDF, a web page), not typed by the user.
- **Citation hallucination**: the model cites a paper that doesn't exist, or one that doesn't say what's claimed.
- **Funnel**: search many abstracts cheaply, read few papers in full. The expensive step (ingestion) only runs at the narrow end.
- **Least privilege**: each agent can call only the tools it needs, so a fooled agent can't do damage.
- **Argument pinning**: fixing a risky tool argument (like an email recipient) in code, so the model can't choose it.
- **Context engineering**: deciding exactly what goes into each LLM call's prompt; state is not context.
- **Human-in-the-loop (HITL)**: the graph pauses, saves its state, and waits for a person's decision before continuing.
- **Tool call**: the model outputs a structured request; your code runs a normal function and feeds the result back. The model never executes anything.
- **Workflow vs agent**: in a workflow, code decides the steps; in an agent, the LLM decides which tools to call and when.
- **Commit / branch / PR / tag**: a snapshot / a parallel line of work / a request to merge (where CI runs) / a permanent name for a milestone commit.
- **Semver**: MAJOR.MINOR.PATCH version numbers. The git SHA is the precise version; semver is the human-friendly label.
- **Lockfile (`uv.lock`)**: exact versions of every dependency, so any checkout installs the same packages.
- *(Terms from the financial agent, such as chunk, embedder, reranker, and flywheel, are in `../financial_agent/FINANCIAL_AGENT_CURRICULUM.md`.)*
