export type Cell = { id: string, needs: string[], description: string, engine: string, revision: number };
export type File = { kind: "text", text: string, bytes: number } | { kind: "binary", bytes: number };
export type CellResult = { id: string, status: string, revision: number, output_sha256: string, sample: string, detail: string };
export type NotebookState = { goal: string, cells: Cell[], order: string[], results: CellResult[], blocked: string[], status: string, detail: string, answer: string };
