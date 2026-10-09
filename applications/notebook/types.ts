import type { Untrusted } from '@natlang/node';

export type Cell = { id: string, needs: string[], description: string, engine: "sqlite" | "javascript", revision: number };
export type CellResult = { id: string, status: "ok" | "failed" | "stale", revision: number, output_sha256: string, sample: string, detail: string };
export type NotebookRun = { goal: string, cells: Cell[], order: string[], results: CellResult[], blocked: string[],
  status: "running" | "done" | "blocked" | "invalid" | "failed" | "stale", detail: string, answer: string };
/**
 * What the explanation reads of one cell: facts the host computed from the result (NULLs, emptiness, truncation) and the
 * bounded sample. The output hash is not part of it.
 */
export type CellEvidence = { id: string, revision: number, status: "ok" | "failed" | "stale", sample: Untrusted<string>,
  has_null: boolean, empty: boolean, truncated: boolean, detail: string };
