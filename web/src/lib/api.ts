// Talking to the API. The browser calls it directly (the API allows this
// origin), so every page is a client component and nothing runs server-side.

import type { SavedPaper } from "./types";

export const API_URL = (
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
).replace(/\/$/, "");

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
    });
  } catch {
    throw new ApiError(0, "The API isn't reachable. Is it running?");
  }
  if (!response.ok) {
    throw new ApiError(response.status, await errorText(response));
  }
  return response.status === 204 ? (undefined as T) : response.json();
}

async function errorText(response: Response): Promise<string> {
  // FastAPI sends {"detail": "..."} or, for invalid input, a list of problems.
  try {
    const body = await response.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) {
      return body.detail.map((d: { msg: string }) => d.msg).join("; ");
    }
  } catch {
    // not JSON
  }
  return `Request failed (HTTP ${response.status})`;
}

export const api = {
  get: <T>(path: string) => request<T>(path),
  post: <T>(path: string, body?: unknown) =>
    request<T>(path, { method: "POST", body: JSON.stringify(body ?? {}) }),
  put: <T>(path: string, body: unknown) =>
    request<T>(path, { method: "PUT", body: JSON.stringify(body) }),
  delete: (path: string) => request<void>(path, { method: "DELETE" }),
};

export function savePaper(paper: SavedPaper) {
  return api.put<SavedPaper>(`/api/reading-list/${paper.arxiv_id}`, paper);
}

export function unsavePaper(arxivId: string) {
  return api.delete(`/api/reading-list/${arxivId}`);
}

export type StreamEvent = { event: string; data: unknown; id?: string };

// POST with a body and read the reply as server-sent events. EventSource
// can't send a body, so Q&A reads the stream itself.
export async function postStream(
  path: string,
  body: unknown,
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    });
  } catch (error) {
    if (signal?.aborted) return;
    throw error instanceof Error
      ? new ApiError(0, "The API isn't reachable. Is it running?")
      : error;
  }
  if (!response.ok || !response.body) {
    throw new ApiError(response.status, await errorText(response));
  }
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value.replace(/\r\n/g, "\n");
    // Events end with a blank line; the last piece may be incomplete.
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() ?? "";
    for (const block of blocks) {
      const event = parseEvent(block);
      if (event) onEvent(event);
    }
  }
}

function parseEvent(block: string): StreamEvent | null {
  let event = "message";
  let id: string | undefined;
  const data: string[] = [];
  for (const line of block.split("\n")) {
    if (line.startsWith(":")) continue; // a ping
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    const value = colon === -1 ? "" : line.slice(colon + 1).replace(/^ /, "");
    if (field === "event") event = value;
    else if (field === "data") data.push(value);
    else if (field === "id") id = value;
  }
  if (!data.length) return null;
  return { event, data: JSON.parse(data.join("\n")), id };
}
