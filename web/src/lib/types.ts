// The API's shapes (src/arxiv_agent/api/app.py and the models it returns).

export type Paper = {
  arxiv_id: string;
  version: number;
  title: string;
  authors: string[];
  abstract: string;
  published: string;
  categories: string[];
  indexed: boolean;
  saved: boolean;
};

export type SavedPaper = {
  arxiv_id: string;
  version: number;
  title: string;
  authors: string[];
  published: string;
  note: string;
  added_at?: string | null;
};

export type Source = {
  label: string;
  chunk_id: string;
  section_path: string[];
  score: number;
  text: string;
};

export type Verdict =
  | "supported"
  | "overstated"
  | "unsupported"
  | "unchecked"
  | "uncited";

export type SentenceSupport = {
  sentence: string;
  sources: number[];
  verdict: Verdict;
  reason: string;
};

export type QAResult = {
  question: string;
  arxiv_id: string;
  version: number;
  answer: {
    text: string;
    status: "answered" | "refused" | "invalid";
    cited: number[];
    problems: string[];
  };
  sources: Source[];
  model: string;
  provider: string;
  retrieval_ms: number;
  generation_ms: number;
  support_ms: number;
  support: SentenceSupport[];
  repair: {
    outcome: "repaired" | "fallback";
    reason: string | null;
  } | null;
  trace_id: string | null;
};

export type AskStep =
  | { kind: "retrieved"; sources: { label: string; section: string }[] }
  | { kind: "answered"; status: string }
  | { kind: "checked"; cited: number; failed: number; repair: string | null };

export type JobStatus =
  | "queued"
  | "running"
  | "awaiting_review"
  | "done"
  | "failed"
  | "expired";

export type Job = {
  job_id: string;
  question: string;
  pause_for_review: boolean;
  status: JobStatus;
  attempts: number;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  trace_id: string | null;
  error: string | null;
};

export type ScreenedPaper = {
  arxiv_id: string;
  version: number;
  title: string;
  authors: string[];
  published: string;
  via: "search" | "snowball" | "reviewer";
  found_by: string[];
  score: number;
  reason: string;
};

export type ReviewResult = {
  question: string;
  review: string;
  criteria: string[];
  sub_queries: string[];
  kept: ScreenedPaper[];
  dropped: ScreenedPaper[];
  references: ScreenedPaper[];
  removed: string[];
  read: { arxiv_id: string; claims: number; source: string }[];
  spent: {
    llm_calls: number;
    prompt_tokens: number;
    completion_tokens: number;
    cost_usd: number | null;
  };
  stopped?: string | null;
  edits?: { label: string; title: string; action: string; arxiv_id: string }[];
};

// What the pause shows per paper (review/human.py _card): no authors.
export type ReviewCard = {
  arxiv_id: string;
  title: string;
  score: number;
  reason: string;
  via: ScreenedPaper["via"];
  flags?: string[]; // the content guard caught it trying to steer the Screener
};

export type ReviewRequest = {
  question: string;
  kept: ReviewCard[];
  dropped: ReviewCard[];
};

export type ReviewView = {
  job: Job;
  result: ReviewResult | null;
  review_request: ReviewRequest | null;
};

export type ReviewEvent = {
  seq: number;
  kind: string;
  text: string;
  at: string;
  data: Record<string, unknown>;
};

export type GraphNode = {
  arxiv_id: string;
  title: string;
  via: string;
  kept: boolean;
  cited: boolean;
  score: number | null;
};

export type Graph = {
  nodes: GraphNode[];
  edges: { source: string; target: string }[];
};
