"use client";

import { useParams } from "next/navigation";
import { useCallback, useEffect, useState } from "react";

import { CitationGraph } from "@/components/CitationGraph";
import { PaperCard, SaveButton } from "@/components/PaperCard";
import { PauseEditor, ReviewText, StatusBadge } from "@/components/Review";
import { Button, ErrorLine, Spinner, errorMessage } from "@/components/ui";
import { API_URL, api } from "@/lib/api";
import type { Graph, ReviewEvent, ReviewView } from "@/lib/types";

const KINDS = ["status", "step", "progress"] as const;

export default function ReviewPage() {
  const { id } = useParams<{ id: string }>();
  const [view, setView] = useState<ReviewView | null>(null);
  const [events, setEvents] = useState<ReviewEvent[]>([]);
  const [graph, setGraph] = useState<Graph | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [saved, setSaved] = useState<Set<string>>(new Set());

  const refresh = useCallback(
    () =>
      api
        .get<ReviewView>(`/api/reviews/${id}`)
        .then(setView)
        .catch((error) => setProblem(errorMessage(error))),
    [id],
  );

  // Progress: the job's event log over SSE. EventSource reconnects by itself
  // and sends the last id it saw, so the API resumes where it left off.
  useEffect(() => {
    refresh();
    const source = new EventSource(`${API_URL}/api/reviews/${id}/events`);
    for (const kind of KINDS) {
      source.addEventListener(kind, (message) => {
        const data = JSON.parse(message.data);
        const seq = Number(message.lastEventId);
        setEvents((all) =>
          all.some((e) => e.seq === seq)
            ? all
            : [...all, { seq, kind, text: data.text, at: data.at, data }],
        );
        if (kind === "status") refresh();
      });
    }
    source.addEventListener("end", () => {
      source.close();
      refresh();
    });
    return () => source.close();
  }, [id, refresh]);

  const status = view?.job.status;
  useEffect(() => {
    if (status !== "done") return;
    api.get<Graph>(`/api/reviews/${id}/graph`).then(setGraph).catch(() => setGraph(null));
    api
      .get<{ arxiv_id: string }[]>("/api/reading-list")
      .then((list) => setSaved(new Set(list.map((p) => p.arxiv_id))))
      .catch(() => {});
  }, [id, status]);

  async function decide(decision: { remove: string[]; add: string[] }) {
    try {
      await api.post(`/api/reviews/${id}/decision`, decision);
      await refresh();
    } catch (error) {
      setProblem(errorMessage(error));
    }
  }

  async function retry() {
    try {
      await api.post(`/api/reviews/${id}/retry`);
      await refresh();
    } catch (error) {
      setProblem(errorMessage(error));
    }
  }

  if (!view) {
    return problem ? <ErrorLine message={problem} /> : <Spinner className="text-indigo-500" />;
  }
  const { job, result, review_request: request } = view;
  const reading = [...events].reverse().find((e) => e.kind === "progress");
  const active = job.status === "queued" || job.status === "running";

  return (
    <div className="space-y-8">
      <section className="space-y-2">
        <div className="flex flex-wrap items-center gap-3">
          <StatusBadge status={job.status} />
          {job.pause_for_review && <span className="text-xs text-zinc-400">deep review</span>}
          <span className="text-xs text-zinc-400">{new Date(job.created_at).toLocaleString()}</span>
        </div>
        <h1 className="text-2xl font-semibold tracking-tight">{job.question}</h1>
        {result && (
          <p className="text-sm text-zinc-500">
            {result.references.length} papers cited of {result.kept.length} kept ·{" "}
            {result.spent.llm_calls} LLM calls ·{" "}
            {(result.spent.prompt_tokens + result.spent.completion_tokens).toLocaleString()} tokens
            {result.spent.cost_usd != null && ` · $${result.spent.cost_usd.toFixed(4)}`}
            {result.stopped && ` · stopped early: ${result.stopped}`}
          </p>
        )}
      </section>

      <ErrorLine message={problem} />

      {job.status === "queued" && events.length <= 1 && (
        <p className="text-sm text-zinc-500">
          Waiting for the review worker to pick it up. If the badge at the top
          says no worker is running, start it: <code>python scripts/review_worker.py</code>
        </p>
      )}

      {job.status === "awaiting_review" && request && (
        <PauseEditor request={request} onDecide={decide} />
      )}

      {job.status === "failed" && (
        <div className="space-y-2">
          <ErrorLine message={job.error ?? "The review failed."} />
          <Button variant="secondary" onClick={retry}>
            Retry from the last finished step
          </Button>
        </div>
      )}

      {result && (
        <section className="space-y-3">
          <h2 className="text-lg font-semibold">Review</h2>
          <ReviewText text={result.review} papers={result.references} />
          {result.removed.length > 0 && (
            <details className="text-sm text-zinc-500">
              <summary className="cursor-pointer">
                {result.removed.length} sentence{result.removed.length === 1 ? "" : "s"} removed: their sources didn&apos;t support them
              </summary>
              <ul className="mt-1 list-disc pl-5">
                {result.removed.map((s) => (
                  <li key={s}>{s}</li>
                ))}
              </ul>
            </details>
          )}
        </section>
      )}

      {result && graph && (
        <section className="space-y-3">
          <h2 className="text-lg font-semibold">How the papers cite each other</h2>
          <CitationGraph graph={graph} />
        </section>
      )}

      {result && (
        <section className="space-y-3">
          <h2 className="text-lg font-semibold">References</h2>
          {result.references.map((paper) => (
            <PaperCard
              key={paper.arxiv_id}
              paper={paper}
              badges={<span className="text-xs text-zinc-400">{paper.via}</span>}
              actions={
                <SaveButton
                  paper={paper}
                  saved={saved.has(paper.arxiv_id)}
                  onChange={(on) =>
                    setSaved((all) => {
                      const next = new Set(all);
                      if (on) next.add(paper.arxiv_id);
                      else next.delete(paper.arxiv_id);
                      return next;
                    })
                  }
                />
              }
              footer={<span className="text-sm text-zinc-500">{paper.reason}</span>}
            />
          ))}
        </section>
      )}

      <section className="space-y-2">
        <h2 className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-zinc-500">
          Progress {active && <Spinner className="h-3 w-3 text-indigo-500" />}
        </h2>
        {active && reading && (
          <div className="h-1.5 w-full overflow-hidden rounded bg-zinc-200 dark:bg-zinc-800">
            <div
              className="h-full bg-indigo-500 transition-all"
              style={{
                width: `${(100 * Number(reading.data.done)) / Number(reading.data.total)}%`,
              }}
            />
          </div>
        )}
        <ol className="space-y-1 font-mono text-xs text-zinc-600 dark:text-zinc-400">
          {events.map((e) => (
            <li key={e.seq} className="flex gap-3">
              <span className="text-zinc-400">{new Date(e.at).toLocaleTimeString()}</span>
              <span className={e.kind === "status" ? "font-semibold text-zinc-800 dark:text-zinc-200" : ""}>
                {e.text}
              </span>
            </li>
          ))}
        </ol>
      </section>
    </div>
  );
}
