"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useRef, useState } from "react";

import { AnswerView, Steps } from "@/components/Answer";
import { PaperCard, SaveButton } from "@/components/PaperCard";
import { Button, ErrorLine, Spinner, errorMessage } from "@/components/ui";
import { api, postStream } from "@/lib/api";
import type { AskStep, Paper, QAResult } from "@/lib/types";

type Turn = {
  question: string;
  steps: AskStep[];
  result: QAResult | null;
  error: string | null;
};

const EXAMPLE = { id: "2411.15594", version: 6 };

export default function AskPage() {
  return (
    <Suspense>
      <Ask />
    </Suspense>
  );
}

function Ask() {
  const params = useSearchParams();
  const router = useRouter();
  const arxivId = params.get("paper");
  const version = Number(params.get("v")) || undefined;

  const [paper, setPaper] = useState<Paper | null>(null);
  const [loading, setLoading] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  const [idInput, setIdInput] = useState("");

  useEffect(() => {
    if (!arxivId) return;
    setLoading(true);
    setProblem(null);
    setPaper(null);
    api
      .get<Paper>(`/api/papers/${arxivId}${version ? `?version=${version}` : ""}`)
      .then(setPaper)
      .catch((error) => setProblem(errorMessage(error)))
      .finally(() => setLoading(false));
  }, [arxivId, version]);

  function open(event: React.FormEvent) {
    event.preventDefault();
    const match = idInput.trim().match(/^(.+?)(?:v(\d+))?$/);
    if (!match) return;
    router.push(`/?paper=${match[1]}${match[2] ? `&v=${match[2]}` : ""}`);
  }

  return (
    <div className="space-y-6">
      <section>
        <h1 className="text-2xl font-semibold tracking-tight">Ask a paper</h1>
        <p className="mt-1 text-zinc-600 dark:text-zinc-400">
          Answers come only from the paper&apos;s own text. Every sentence cites
          its passage and is checked against it before you see it.
        </p>
      </section>

      <form onSubmit={open} className="flex flex-wrap gap-2">
        <input
          value={idInput}
          onChange={(e) => setIdInput(e.target.value)}
          placeholder="arXiv id, e.g. 2411.15594v6"
          className="min-w-64 flex-1 rounded-md border border-zinc-300 bg-white px-3 py-2 text-sm dark:border-zinc-700 dark:bg-zinc-900"
        />
        <Button type="submit" variant="secondary">
          Open
        </Button>
        <Link
          href="/papers"
          className="self-center text-sm text-indigo-600 hover:underline"
        >
          or search arXiv
        </Link>
      </form>

      {!arxivId && (
        <p className="text-sm text-zinc-500">
          Try{" "}
          <Link
            href={`/?paper=${EXAMPLE.id}&v=${EXAMPLE.version}`}
            className="text-indigo-600 hover:underline"
          >
            A Survey on LLM-as-a-Judge ({EXAMPLE.id}v{EXAMPLE.version})
          </Link>
          .
        </p>
      )}
      {loading && <Spinner className="text-indigo-500" />}
      <ErrorLine message={problem} />
      {paper && <PaperSession key={`${paper.arxiv_id}v${paper.version}`} initial={paper} />}
    </div>
  );
}

function PaperSession({ initial }: { initial: Paper }) {
  const [paper, setPaper] = useState(initial);
  const [ingesting, setIngesting] = useState(false);
  const [note, setNote] = useState<string | null>(null);
  const [question, setQuestion] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [asking, setAsking] = useState(false);
  const abort = useRef<AbortController | null>(null);

  useEffect(() => () => abort.current?.abort(), []);

  async function ingest() {
    setIngesting(true);
    setNote(null);
    try {
      const body = await api.post<{ source: string; paper: Paper }>(
        `/api/papers/${paper.arxiv_id}/ingest?version=${paper.version}`,
      );
      setPaper(body.paper);
      if (body.source === "abstract_only") {
        setNote("This paper has no HTML version on arXiv, so only its abstract is stored. Questions need the full text.");
      } else if (body.source === "unavailable") {
        setNote("arXiv couldn't provide this paper right now. Try again in a minute.");
      }
    } catch (error) {
      setNote(errorMessage(error));
    } finally {
      setIngesting(false);
    }
  }

  async function ask(event: React.FormEvent) {
    event.preventDefault();
    const text = question.trim();
    if (text.length < 3 || asking) return;
    setQuestion("");
    setAsking(true);
    const index = turns.length;
    setTurns((all) => [...all, { question: text, steps: [], result: null, error: null }]);
    const update = (change: Partial<Turn> | ((turn: Turn) => Partial<Turn>)) =>
      setTurns((all) =>
        all.map((turn, i) =>
          i === index ? { ...turn, ...(typeof change === "function" ? change(turn) : change) } : turn,
        ),
      );

    abort.current = new AbortController();
    try {
      await postStream(
        "/api/ask",
        { question: text, arxiv_id: paper.arxiv_id, version: paper.version },
        ({ event, data }) => {
          if (event === "result") update({ result: data as QAResult });
          else if (event === "error") update({ error: (data as { message: string }).message });
          else update((turn) => ({ steps: [...turn.steps, { kind: event, ...(data as object) } as AskStep] }));
        },
        abort.current.signal,
      );
    } catch (error) {
      update({ error: errorMessage(error) });
    } finally {
      setAsking(false);
    }
  }

  return (
    <div className="space-y-6">
      <PaperCard
        paper={paper}
        actions={
          <>
            {!paper.indexed && (
              <Button onClick={ingest} disabled={ingesting}>
                {ingesting && <Spinner />}
                {ingesting ? "Fetching and indexing…" : "Index this paper"}
              </Button>
            )}
            <SaveButton
              paper={paper}
              saved={paper.saved}
              onChange={(saved) => setPaper({ ...paper, saved })}
            />
          </>
        }
      />
      {ingesting && (
        <p className="text-sm text-zinc-500">
          Downloading the paper from arXiv (politely: one request every 3 s),
          splitting it into passages and embedding them. Usually 10–30 s.
        </p>
      )}
      <ErrorLine message={note} />

      {turns.map((turn, i) => (
        <section key={i} className="space-y-3">
          <p className="font-medium">
            <span className="text-zinc-400">Q.</span> {turn.question}
          </p>
          {!turn.result && <Steps steps={turn.steps} finished={Boolean(turn.error)} />}
          <ErrorLine message={turn.error} />
          {turn.result && <AnswerView result={turn.result} />}
        </section>
      ))}

      <form onSubmit={ask} className="flex gap-2">
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          disabled={!paper.indexed || asking}
          maxLength={500}
          placeholder={
            paper.indexed
              ? "Ask a question about this paper"
              : "Index the paper first"
          }
          className="flex-1 rounded-md border border-zinc-300 bg-white px-3 py-2 dark:border-zinc-700 dark:bg-zinc-900"
        />
        <Button type="submit" disabled={!paper.indexed || asking || question.trim().length < 3}>
          {asking && <Spinner />}
          Ask
        </Button>
      </form>
    </div>
  );
}
