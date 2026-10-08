"use client";

import { useEffect, useState } from "react";

import { AskLink, PaperCard, SaveButton } from "@/components/PaperCard";
import { ErrorLine, Spinner, errorMessage } from "@/components/ui";
import { api } from "@/lib/api";
import type { SavedPaper } from "@/lib/types";

export default function ReadingListPage() {
  const [papers, setPapers] = useState<SavedPaper[] | null>(null);
  const [problem, setProblem] = useState<string | null>(null);

  useEffect(() => {
    api
      .get<SavedPaper[]>("/api/reading-list")
      .then(setPapers)
      .catch((error) => setProblem(errorMessage(error)));
  }, []);

  return (
    <div className="space-y-6">
      <section>
        <h1 className="text-2xl font-semibold tracking-tight">Reading list</h1>
        <p className="mt-1 text-zinc-600 dark:text-zinc-400">
          Papers you saved from searches and reviews.
        </p>
      </section>
      <ErrorLine message={problem} />
      {!papers && !problem && <Spinner className="text-indigo-500" />}
      {papers?.length === 0 && (
        <p className="text-sm text-zinc-500">
          Nothing saved yet: use ☆ Save on a paper card.
        </p>
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
                  saved
                  onChange={() =>
                    setPapers((all) =>
                      all?.filter((p) => p.arxiv_id !== paper.arxiv_id) ?? null,
                    )
                  }
                />
              </>
            }
            footer={
              paper.added_at && (
                <span className="ml-auto text-xs text-zinc-400">
                  saved {new Date(paper.added_at).toLocaleDateString()}
                </span>
              )
            }
          />
        ))}
      </div>
    </div>
  );
}
