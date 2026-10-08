# Web UI

Next.js 15 (App Router, TypeScript, Tailwind). Every page is a client
component that talks to the API in the browser (`NEXT_PUBLIC_API_URL`, default
`http://localhost:8000`); nothing runs server-side.

- `/` ask a paper: index it if needed, then questions stream their steps
  (passages found, answer written, sentences checked) and the answer arrives
  with `[S#]` citations linked to its sources
- `/papers` search arXiv, save papers, ask about one
- `/reviews` start a literature review; `/reviews/[id]` follows it live
  (EventSource on the job's event log, resumable), pauses for your paper
  check in a deep review, then shows the review, its references and how the
  papers cite each other
- `/reading-list` saved papers

## Run it

Needs the API (and, for reviews, the worker) running; see the main README.
Node runs in Docker here, so nothing is installed on Windows (Smart App
Control blocks some of Next.js's native binaries). From the repo root, in
PowerShell:

```powershell
docker run --rm -v "${PWD}\web:/app" -v arxiv_web_node_modules:/app/node_modules -w /app node:22-bookworm-slim npm ci
docker run --rm -it -v "${PWD}\web:/app" -v arxiv_web_node_modules:/app/node_modules -w /app -p 127.0.0.1:3000:3000 -e WATCHPACK_POLLING=true node:22-bookworm-slim npm run dev -- -H 0.0.0.0
```

Then open http://localhost:3000. With Node installed locally instead:
`npm ci` and `npm run dev` in this folder.

Checks: `npm run typecheck`, `npm run lint`, `npm run build`.
