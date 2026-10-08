"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { StatusBadge } from "@/components/Review";
import { Button, ErrorLine, Spinner, errorMessage } from "@/components/ui";
import { api } from "@/lib/api";
import type { Job } from "@/lib/types";

export default function ReviewsPage() {
  const router = useRouter();
  const [question, setQuestion] = useState("");
  const [deep, setDeep] = useState(true);
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  const [jobs, setJobs] = useState<Job[] | null>(null);

  useEffect(() => {
    let alive = true;
    const load = () =>
      api
        .get<Job[]>("/api/reviews?limit=20")
        .then((j) => alive && setJobs(j))
        .catch((error) => alive && setProblem(errorMessage(error)));
    load();
    const timer = setInterval(load, 5_000);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, []);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setProblem(null);
    try {
      const { job_id } = await api.post<{ job_id: string }>("/api/reviews", {
        question: question.trim(),
        deep,
      });
      router.push(`/reviews/${job_id}`);
    } catch (error) {
      setProblem(errorMessage(error));
      setBusy(false);
    }
  }

  return (
    <div className="space-y-8">
      <section className="space-y-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Literature reviews</h1>
          <p className="mt-1 text-zinc-600 dark:text-zinc-400">
            A team of agents searches arXiv, screens the papers, follows their
            citations, reads the best ones and writes a short review in which
            every sentence is checked against the papers it cites. A few minutes.
          </p>
        </div>
        <form onSubmit={submit} className="space-y-3">
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            rows={2}
            maxLength={300}
            placeholder="How biased are LLMs used as judges?"
            className="w-full rounded-md border border-zinc-300 bg-white px-3 py-2 dark:border-zinc-700 dark:bg-zinc-900"
          />
          <div className="flex flex-wrap items-center gap-4">
            <label className="flex items-center gap-2 text-sm">
              <input type="checkbox" checked={deep} onChange={(e) => setDeep(e.target.checked)} />
              Let me check the paper list before it&apos;s written (deep review)
            </label>
            <Button type="submit" disabled={busy || question.trim().length < 10} className="ml-auto">
              {busy && <Spinner />}
              Start the review
            </Button>
          </div>
        </form>
        <ErrorLine message={problem} />
      </section>

      <section className="space-y-2">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-zinc-500">Recent</h2>
        {!jobs && !problem && <Spinner className="text-indigo-500" />}
        {jobs?.length === 0 && <p className="text-sm text-zinc-500">No reviews yet.</p>}
        <ul className="divide-y divide-zinc-200 rounded-lg border border-zinc-200 dark:divide-zinc-800 dark:border-zinc-800">
          {jobs?.map((job) => (
            <li key={job.job_id}>
              <Link
                href={`/reviews/${job.job_id}`}
                className="flex flex-wrap items-center gap-3 px-4 py-3 hover:bg-zinc-50 dark:hover:bg-zinc-900"
              >
                <span className="min-w-0 flex-1 truncate">{job.question}</span>
                {job.pause_for_review && <span className="text-xs text-zinc-400">deep</span>}
                <span className="text-xs text-zinc-400">
                  {new Date(job.created_at).toLocaleString()}
                </span>
                <StatusBadge status={job.status} />
              </Link>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
