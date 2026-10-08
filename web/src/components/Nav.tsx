"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

import { api } from "@/lib/api";

const LINKS = [
  { href: "/", label: "Ask a paper" },
  { href: "/papers", label: "Search" },
  { href: "/reviews", label: "Reviews" },
  { href: "/reading-list", label: "Reading list" },
];

type Health = { ok: boolean; worker: boolean } | null;

export function Nav() {
  const path = usePathname();
  const [health, setHealth] = useState<Health | "down">(null);

  useEffect(() => {
    let alive = true;
    const check = () =>
      api
        .get<{ ok: boolean; worker: boolean }>("/api/health")
        .then((h) => alive && setHealth(h))
        .catch(() => alive && setHealth("down"));
    check();
    const timer = setInterval(check, 10_000);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, []);

  return (
    <header className="border-b border-zinc-200 bg-white/80 backdrop-blur dark:border-zinc-800 dark:bg-zinc-950/80">
      <nav className="mx-auto flex max-w-5xl flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3">
        <Link href="/" className="font-semibold tracking-tight">
          arXiv Research Agent
        </Link>
        <div className="flex flex-wrap gap-1 text-sm">
          {LINKS.map((link) => {
            const active =
              link.href === "/" ? path === "/" : path.startsWith(link.href);
            return (
              <Link
                key={link.href}
                href={link.href}
                className={`rounded-md px-3 py-1.5 ${
                  active
                    ? "bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-900"
                    : "text-zinc-600 hover:bg-zinc-100 dark:text-zinc-400 dark:hover:bg-zinc-900"
                }`}
              >
                {link.label}
              </Link>
            );
          })}
        </div>
        <HealthBadge health={health} />
      </nav>
    </header>
  );
}

function HealthBadge({ health }: { health: Health | "down" }) {
  if (health === null) return null;
  const [color, text, title] =
    health === "down"
      ? ["bg-red-500", "API offline", "Start it: python -m uvicorn arxiv_agent.api.app:app --port 8000"]
      : health.worker
        ? ["bg-emerald-500", "API + worker", "Reviews will run"]
        : ["bg-amber-500", "No review worker", "Reviews wait in the queue until you run: python scripts/review_worker.py"];
  return (
    <span
      className="ml-auto flex items-center gap-2 text-xs text-zinc-500"
      title={title}
    >
      <span className={`h-2 w-2 rounded-full ${color}`} />
      {text}
    </span>
  );
}
