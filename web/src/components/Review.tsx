"use client";

import { Fragment, useMemo, useState } from "react";

import type { JobStatus, ReviewCard, ReviewRequest, ScreenedPaper } from "@/lib/types";

import { Badge, Button, Spinner, arxivUrl } from "./ui";

const STATUS: Record<JobStatus, [string, "gray" | "green" | "amber" | "red" | "blue"]> = {
  queued: ["queued", "gray"],
  running: ["running", "blue"],
  awaiting_review: ["waiting for you", "amber"],
  done: ["done", "green"],
  failed: ["failed", "red"],
  expired: ["expired", "gray"],
};

export function StatusBadge({ status }: { status: JobStatus }) {
  const [label, color] = STATUS[status];
  return (
    <Badge color={color}>
      {status === "running" && <Spinner className="mr-1 h-3 w-3" />}
      {label}
    </Badge>
  );
}

// "[arXiv:2406.12624, arXiv:2306.05685]" becomes links with the papers' titles.
const CITATION_RUN = /\[(arXiv:[^\]]+)\]/g;

export function ReviewText({
  text,
  papers,
}: {
  text: string;
  papers: ScreenedPaper[];
}) {
  const titles = useMemo(
    () => Object.fromEntries(papers.map((p) => [p.arxiv_id, p.title])),
    [papers],
  );
  return (
    <div className="space-y-3 leading-relaxed">
      {text.split(/\n\s*\n/).map((paragraph, i) => {
        const parts: React.ReactNode[] = [];
        let last = 0;
        for (const match of paragraph.matchAll(CITATION_RUN)) {
          parts.push(paragraph.slice(last, match.index));
          const ids = match[1].split(",").map((s) => s.trim().replace(/^arXiv:/, ""));
          parts.push(
            <span key={match.index} className="text-sm">
              [
              {ids.map((id, j) => (
                <Fragment key={id}>
                  {j > 0 && ", "}
                  <a
                    href={arxivUrl(id)}
                    target="_blank"
                    rel="noreferrer"
                    title={titles[id] ?? id}
                    className="text-indigo-600 hover:underline dark:text-indigo-400"
                  >
                    {id}
                  </a>
                </Fragment>
              ))}
              ]
            </span>,
          );
          last = (match.index ?? 0) + match[0].length;
        }
        parts.push(paragraph.slice(last));
        return <p key={i}>{parts}</p>;
      })}
    </div>
  );
}

// The pause in a deep review: the person checks the screened paper list.
export function PauseEditor({
  request,
  onDecide,
}: {
  request: ReviewRequest;
  onDecide: (decision: { remove: string[]; add: string[] }) => Promise<void>;
}) {
  const [removed, setRemoved] = useState<Set<string>>(new Set());
  const [added, setAdded] = useState<Set<string>>(new Set());
  const [extra, setExtra] = useState("");
  const [busy, setBusy] = useState(false);

  const toggle = (set: Set<string>, id: string) => {
    const next = new Set(set);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    return next;
  };

  async function submit() {
    setBusy(true);
    const typed = extra
      .split(/[\s,]+/)
      .map((s) => s.replace(/v\d+$/, ""))
      .filter(Boolean);
    try {
      await onDecide({ remove: [...removed], add: [...added, ...typed] });
    } finally {
      setBusy(false);
    }
  }

  const row = (paper: ReviewCard, checked: boolean, onChange: () => void) => (
    <label
      key={paper.arxiv_id}
      className="flex cursor-pointer gap-3 rounded-md p-2 hover:bg-zinc-50 dark:hover:bg-zinc-900"
    >
      <input type="checkbox" checked={checked} onChange={onChange} className="mt-1" />
      <span className="min-w-0 flex-1 text-sm">
        <a
          href={arxivUrl(paper.arxiv_id)}
          target="_blank"
          rel="noreferrer"
          className="font-medium hover:underline"
        >
          {paper.title}
        </a>{" "}
        <span className="text-zinc-500">
          · {paper.arxiv_id} · {paper.via}
        </span>
        <span className="block text-zinc-600 dark:text-zinc-400">
          <Badge color={paper.score >= 7 ? "green" : "gray"}>{paper.score}/10</Badge>{" "}
          {paper.reason}
        </span>
        {paper.flags?.map((flag) => (
          <span key={flag} className="mt-1 block text-xs text-red-700 dark:text-red-400">
            ⚠ This paper tried to steer the screener ({flag})
          </span>
        ))}
      </span>
    </label>
  );

  return (
    <section className="space-y-4 rounded-lg border border-amber-300 bg-amber-50/50 p-4 dark:border-amber-800 dark:bg-amber-950/30">
      <div>
        <h2 className="font-semibold">Check the papers before the review is written</h2>
        <p className="text-sm text-zinc-600 dark:text-zinc-400">
          Untick papers that don&apos;t belong, tick dropped ones that do, or add
          ids search missed. Your edits are saved as eval labels.
        </p>
      </div>
      <div>
        <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-zinc-500">
          Kept ({request.kept.length - removed.size})
        </h3>
        {request.kept.map((p) =>
          row(p, !removed.has(p.arxiv_id), () => setRemoved(toggle(removed, p.arxiv_id))),
        )}
      </div>
      <details>
        <summary className="cursor-pointer text-xs font-semibold uppercase tracking-wide text-zinc-500">
          Dropped by the screener ({request.dropped.length})
        </summary>
        {request.dropped.map((p) =>
          row(p, added.has(p.arxiv_id), () => setAdded(toggle(added, p.arxiv_id))),
        )}
      </details>
      <div className="flex flex-wrap items-center gap-2">
        <input
          value={extra}
          onChange={(e) => setExtra(e.target.value)}
          placeholder="Add arXiv ids, e.g. 2306.05685"
          className="min-w-64 flex-1 rounded-md border border-zinc-300 bg-white px-3 py-1.5 text-sm dark:border-zinc-700 dark:bg-zinc-900"
        />
        <Button onClick={submit} disabled={busy}>
          {busy && <Spinner />}
          Continue the review
        </Button>
      </div>
    </section>
  );
}
