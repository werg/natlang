export type Cell = { id: string, needs: string[], description: string, engine: "sqlite" | "javascript", revision: number };
export type CellResult = { id: string, status: "ok" | "failed" | "stale", revision: number, output_sha256: string, sample: string, detail: string };
export type NotebookRun = { goal: string, cells: Cell[], order: string[], results: CellResult[], blocked: string[],
  status: "running" | "done" | "blocked" | "invalid" | "failed" | "stale", detail: string, answer: string };
