"use client";

import { useState } from "react";

import { AskLink, PaperCard, SaveButton } from "@/components/PaperCard";
import { Button, ErrorLine, Spinner, errorMessage } from "@/components/ui";
import { api } from "@/lib/api";
import type { Paper } from "@/lib/types";

export default function PapersPage() {
  const [query, setQuery] = useState("");
  const [papers, setPapers] = useState<Paper[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);

  async function search(event: React.FormEvent) {
    event.preventDefault();
    if (!query.trim()) return;
    setBusy(true);
    setProblem(null);
    try {
      setPapers(
        await api.get<Paper[]>(
          `/api/papers/search?q=${encodeURIComponent(query)}&limit=15`,
        ),
      );
    } catch (error) {
      setProblem(errorMessage(error));
    } finally {
      setBusy(false);
    }
  }

  function setSaved(arxivId: string, saved: boolean) {
    setPapers((all) =>
      all?.map((p) => (p.arxiv_id === arxivId ? { ...p, saved } : p)) ?? null,
    );
  }

  return (
    <div className="space-y-6">
      <section>
        <h1 className="text-2xl font-semibold tracking-tight">Search arXiv</h1>
        <p className="mt-1 text-zinc-600 dark:text-zinc-400">
          Plain words or an arXiv id. Save papers to your reading list, or ask
          one a question.
        </p>
      </section>
      <form onSubmit={search} className="flex gap-2">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="position bias LLM judges"
          className="flex-1 rounded-md border border-zinc-300 bg-white px-3 py-2 dark:border-zinc-700 dark:bg-zinc-900"
        />
        <Button type="submit" disabled={busy || !query.trim()}>
          {busy && <Spinner />}
          Search
        </Button>
      </form>
      <ErrorLine message={problem} />
      {papers?.length === 0 && (
        <p className="text-sm text-zinc-500">No papers matched.</p>
      )}
      <div className="space-y-3">
        {papers?.map((paper) => (
          <PaperCard
            key={paper.arxiv_id}
            paper={paper}
            actions={
              <>
                <AskLink paper={paper} />
                <SaveButton
                  paper={paper}
                  saved={paper.saved}
                  onChange={(saved) => setSaved(paper.arxiv_id, saved)}
                />
              </>
            }
          />
        ))}
      </div>
    </div>
  );
}
