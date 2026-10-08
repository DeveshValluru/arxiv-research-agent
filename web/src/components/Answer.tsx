"use client";

import { Fragment, useState } from "react";

import type { AskStep, QAResult, SentenceSupport } from "@/lib/types";

import { Badge, Spinner } from "./ui";

const STEP_LABELS: { kind: AskStep["kind"]; label: string }[] = [
  { kind: "retrieved", label: "Finding the passages" },
  { kind: "answered", label: "Writing the answer" },
  { kind: "checked", label: "Checking each sentence against its sources" },
];

export function Steps({
  steps,
  finished,
}: {
  steps: AskStep[];
  finished: boolean;
}) {
  const done = new Set(steps.map((s) => s.kind));
  const checked = steps.find((s) => s.kind === "checked");
  const answered = steps.find((s) => s.kind === "answered");
  return (
    <ol className="space-y-1 text-sm">
      {STEP_LABELS.map(({ kind, label }, i) => {
        // The check only runs on an answer (not a refusal).
        if (kind === "checked" && answered?.kind === "answered" && answered.status !== "answered") {
          return null;
        }
        const isDone = done.has(kind);
        const isCurrent =
          !finished && !isDone && STEP_LABELS.slice(0, i).every((s) => done.has(s.kind));
        if (!isDone && !isCurrent) return null;
        return (
          <li key={kind} className="flex items-center gap-2 text-zinc-600 dark:text-zinc-400">
            {isDone ? <span className="text-emerald-600">✓</span> : <Spinner className="text-indigo-500" />}
            {label}
            {kind === "retrieved" && isDone && (
              <span className="text-zinc-400">
                ({(steps[0] as Extract<AskStep, { kind: "retrieved" }>).sources.length} passages)
              </span>
            )}
            {kind === "checked" && checked?.kind === "checked" && (
              <span className="text-zinc-400">
                ({checked.cited} checked
                {checked.failed ? `, ${checked.failed} failed${checked.repair ? `, ${checked.repair}` : ""}` : ", all supported"})
              </span>
            )}
          </li>
        );
      })}
    </ol>
  );
}

const CITATION = /\[S(\d+)\]/g;

export function AnswerView({ result }: { result: QAResult }) {
  const [active, setActive] = useState<number | null>(null);
  const { answer } = result;
  const failed = result.support.filter(
    (s) => s.verdict === "overstated" || s.verdict === "unsupported",
  );

  return (
    <div className="space-y-4">
      <div
        className={`rounded-lg p-4 leading-relaxed ${
          answer.status === "refused"
            ? "border border-amber-200 bg-amber-50 text-amber-900 dark:border-amber-900 dark:bg-amber-950 dark:text-amber-200"
            : "bg-zinc-50 dark:bg-zinc-900"
        }`}
      >
        {answer.text.split("\n").map((paragraph, i) => (
          <p key={i} className="mb-2 last:mb-0">
            {withCitations(paragraph, active, setActive)}
          </p>
        ))}
        {answer.status === "invalid" && (
          <p className="mt-2 text-sm text-red-700">
            This answer failed the format checks: {answer.problems.join("; ")}
          </p>
        )}
      </div>

      {(failed.length > 0 || result.repair) && <SupportNote failed={failed} result={result} />}

      <div className="space-y-2">
        <h4 className="text-xs font-semibold uppercase tracking-wide text-zinc-500">
          Sources
        </h4>
        {result.sources.map((source, i) => {
          const n = i + 1;
          const cited = answer.cited.includes(n);
          return (
            <details
              key={source.chunk_id}
              open={active === n}
              className={`rounded-md border p-3 text-sm ${
                active === n
                  ? "border-indigo-400 bg-indigo-50 dark:border-indigo-700 dark:bg-indigo-950"
                  : "border-zinc-200 dark:border-zinc-800"
              }`}
            >
              <summary className="flex cursor-pointer items-center gap-2" onClick={(e) => { e.preventDefault(); setActive(active === n ? null : n); }}>
                <Badge color={cited ? "blue" : "gray"}>S{n}</Badge>
                <span className="truncate text-zinc-600 dark:text-zinc-400">
                  {source.section_path.join(" › ")}
                </span>
                {!cited && <span className="ml-auto text-xs text-zinc-400">not cited</span>}
              </summary>
              <p className="mt-2 whitespace-pre-line text-zinc-700 dark:text-zinc-300">
                {source.text}
              </p>
            </details>
          );
        })}
      </div>

      <p className="text-xs text-zinc-400">
        {result.model.split("/").pop()} via {result.provider} · retrieval{" "}
        {Math.round(result.retrieval_ms)} ms · answer {Math.round(result.generation_ms)} ms
        {result.support_ms ? ` · check ${Math.round(result.support_ms)} ms` : ""}
      </p>
    </div>
  );
}

function withCitations(
  text: string,
  active: number | null,
  setActive: (n: number | null) => void,
) {
  const parts: React.ReactNode[] = [];
  let last = 0;
  for (const match of text.matchAll(CITATION)) {
    parts.push(text.slice(last, match.index));
    const n = Number(match[1]);
    parts.push(
      <button
        key={`${match.index}-${n}`}
        onClick={() => setActive(active === n ? null : n)}
        className={`mx-0.5 rounded px-1 align-baseline text-xs font-semibold ${
          active === n
            ? "bg-indigo-600 text-white"
            : "bg-indigo-100 text-indigo-700 hover:bg-indigo-200 dark:bg-indigo-950 dark:text-indigo-300"
        }`}
      >
        S{n}
      </button>,
    );
    last = (match.index ?? 0) + match[0].length;
  }
  parts.push(text.slice(last));
  return parts.map((part, i) => <Fragment key={i}>{part}</Fragment>);
}

function SupportNote({
  failed,
  result,
}: {
  failed: SentenceSupport[];
  result: QAResult;
}) {
  const outcome = result.repair?.outcome;
  return (
    <details className="rounded-md border border-amber-200 bg-amber-50 p-3 text-sm dark:border-amber-900 dark:bg-amber-950">
      <summary className="cursor-pointer text-amber-900 dark:text-amber-200">
        The support check failed {failed.length} sentence{failed.length === 1 ? "" : "s"}
        {outcome === "repaired" && " and the answer was rewritten to fix them"}
        {outcome === "fallback" && " and removed them"}
      </summary>
      <ul className="mt-2 space-y-2">
        {failed.map((s) => (
          <li key={s.sentence} className="text-zinc-700 dark:text-zinc-300">
            <span className="line-through decoration-amber-500">{s.sentence}</span>
            <br />
            <span className="text-xs text-zinc-500">
              {s.verdict}: {s.reason}
            </span>
          </li>
        ))}
      </ul>
    </details>
  );
}
