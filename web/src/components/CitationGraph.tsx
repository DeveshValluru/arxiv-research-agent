"use client";

import {
  forceCenter,
  forceCollide,
  forceLink,
  forceManyBody,
  forceSimulation,
  forceX,
  forceY,
  type SimulationLinkDatum,
  type SimulationNodeDatum,
} from "d3-force";
import { useMemo, useState } from "react";

import type { Graph, GraphNode } from "@/lib/types";

import { arxivUrl } from "./ui";

type Node = GraphNode & SimulationNodeDatum;
type Link = SimulationLinkDatum<Node>;

const WIDTH = 720;
const HEIGHT = 440;

// The review's papers and who cites whom, from the snowball step's links.
// Laid out once (300 ticks of a force simulation), then drawn as plain SVG.
export function CitationGraph({ graph }: { graph: Graph }) {
  const [hover, setHover] = useState<Node | null>(null);

  const { nodes, links } = useMemo(() => {
    const nodes: Node[] = graph.nodes.map((n) => ({ ...n }));
    const links: Link[] = graph.edges.map((e) => ({ ...e }));
    forceSimulation(nodes)
      .force("link", forceLink<Node, Link>(links).id((n) => n.arxiv_id).distance(70))
      .force("charge", forceManyBody().strength(-160))
      .force("center", forceCenter(WIDTH / 2, HEIGHT / 2))
      .force("collide", forceCollide(14))
      // A weak pull to the middle: papers with no links would otherwise be
      // pushed to the edges by the repulsion and nothing would bring them back.
      .force("x", forceX(WIDTH / 2).strength(0.06))
      .force("y", forceY(HEIGHT / 2).strength(0.08))
      .stop()
      .tick(300);
    // Keep everything on the canvas.
    for (const n of nodes) {
      n.x = Math.max(16, Math.min(WIDTH - 16, n.x ?? 0));
      n.y = Math.max(16, Math.min(HEIGHT - 16, n.y ?? 0));
    }
    return { nodes, links };
  }, [graph]);

  if (!graph.nodes.length) {
    return <p className="text-sm text-zinc-500">No citation links recorded for this review.</p>;
  }

  return (
    <div className="space-y-2">
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className="w-full rounded-lg border border-zinc-200 bg-white dark:border-zinc-800 dark:bg-zinc-900"
        role="img"
        aria-label="Citation graph of the review's papers"
      >
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="17" refY="5" markerWidth="5" markerHeight="5" orient="auto">
            <path d="M0,0 L10,5 L0,10 z" className="fill-zinc-400" />
          </marker>
        </defs>
        {links.map((link, i) => {
          const s = link.source as Node;
          const t = link.target as Node;
          const lit = hover && (s === hover || t === hover);
          return (
            <line
              key={i}
              x1={s.x}
              y1={s.y}
              x2={t.x}
              y2={t.y}
              markerEnd="url(#arrow)"
              className={lit ? "stroke-indigo-500" : "stroke-zinc-300 dark:stroke-zinc-700"}
              strokeWidth={lit ? 1.6 : 1}
            />
          );
        })}
        {nodes.map((n) => (
          <a key={n.arxiv_id} href={arxivUrl(n.arxiv_id)} target="_blank" rel="noreferrer">
            <circle
              cx={n.x}
              cy={n.y}
              r={n.kept ? 9 : 6}
              onMouseEnter={() => setHover(n)}
              onMouseLeave={() => setHover(null)}
              className={`${
                n.cited
                  ? "fill-indigo-600"
                  : n.kept
                    ? "fill-indigo-300 dark:fill-indigo-800"
                    : "fill-zinc-300 dark:fill-zinc-600"
              } stroke-white dark:stroke-zinc-900`}
              strokeWidth={2}
            >
              <title>{`${n.title || n.arxiv_id} (${n.arxiv_id})`}</title>
            </circle>
          </a>
        ))}
      </svg>
      <div className="flex flex-wrap gap-4 text-xs text-zinc-500">
        <Legend className="fill-indigo-600" label="cited in the review" />
        <Legend className="fill-indigo-300" label="kept, not cited" />
        <Legend className="fill-zinc-300" label="linked, dropped" />
        <span>arrow: cites</span>
        {hover && (
          <span className="ml-auto truncate text-zinc-700 dark:text-zinc-300">
            {hover.title} ({hover.arxiv_id})
          </span>
        )}
      </div>
    </div>
  );
}

function Legend({ className, label }: { className: string; label: string }) {
  return (
    <span className="flex items-center gap-1">
      <svg width="10" height="10">
        <circle cx="5" cy="5" r="5" className={className} />
      </svg>
      {label}
    </span>
  );
}
