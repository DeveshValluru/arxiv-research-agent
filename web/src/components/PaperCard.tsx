"use client";

import Link from "next/link";
import { useState, type ReactNode } from "react";

import { savePaper, unsavePaper } from "@/lib/api";
import type { SavedPaper } from "@/lib/types";

import { Badge, Button, arxivUrl, authorsLine, errorMessage } from "./ui";

export type CardPaper = {
  arxiv_id: string;
  version: number;
  title: string;
  authors: string[];
  published: string;
  abstract?: string;
  indexed?: boolean;
  saved?: boolean;
};

export function PaperCard({
  paper,
  badges,
  actions,
  footer,
}: {
  paper: CardPaper;
  badges?: ReactNode;
  actions?: ReactNode;
  footer?: ReactNode;
}) {
  const [open, setOpen] = useState(false);
  return (
    <article className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="min-w-0 flex-1">
          <h3 className="font-medium leading-snug">
            <a
              href={arxivUrl(paper.arxiv_id, paper.version)}
              target="_blank"
              rel="noreferrer"
              className="hover:underline"
            >
              {paper.title}
            </a>
          </h3>
          <p className="mt-1 text-sm text-zinc-500">
            {authorsLine(paper.authors)} · {paper.published.slice(0, 10)} ·{" "}
            <span className="font-mono">
              {paper.arxiv_id}v{paper.version}
            </span>
          </p>
        </div>
        <div className="flex flex-wrap gap-1">
          {paper.indexed !== undefined && (
            <Badge color={paper.indexed ? "green" : "gray"}>
              {paper.indexed ? "indexed" : "not indexed"}
            </Badge>
          )}
          {badges}
        </div>
      </div>
      {paper.abstract && (
        <p
          className={`mt-2 text-sm text-zinc-700 dark:text-zinc-300 ${open ? "" : "line-clamp-3"}`}
          onClick={() => setOpen(!open)}
        >
          {paper.abstract}
        </p>
      )}
      {(actions || footer) && (
        <div className="mt-3 flex flex-wrap items-center gap-2">
          {actions}
          {footer}
        </div>
      )}
    </article>
  );
}

export function AskLink({ paper }: { paper: CardPaper }) {
  return (
    <Link
      href={`/?paper=${paper.arxiv_id}&v=${paper.version}`}
      className="rounded-md bg-indigo-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-indigo-500"
    >
      Ask about it
    </Link>
  );
}

export function SaveButton({
  paper,
  saved,
  onChange,
}: {
  paper: CardPaper;
  saved: boolean;
  onChange?: (saved: boolean) => void;
}) {
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);

  async function toggle() {
    setBusy(true);
    setProblem(null);
    try {
      if (saved) {
        await unsavePaper(paper.arxiv_id);
      } else {
        const entry: SavedPaper = {
          arxiv_id: paper.arxiv_id,
          version: paper.version,
          title: paper.title,
          authors: paper.authors,
          published: paper.published.slice(0, 10),
          note: "",
        };
        await savePaper(entry);
      }
      onChange?.(!saved);
    } catch (error) {
      setProblem(errorMessage(error));
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <Button variant="secondary" onClick={toggle} disabled={busy}>
        {saved ? "★ Saved" : "☆ Save"}
      </Button>
      {problem && <span className="text-xs text-red-600">{problem}</span>}
    </>
  );
}
