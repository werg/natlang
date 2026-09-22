export type Cell = { id: Text, needs: Text[], description: Text, engine: Text, revision: Num };
export type File = { kind: "text", text: Text, bytes: Num } | { kind: "binary", bytes: Num };
export type CellResult = { id: Text, status: Text, revision: Num, output_sha256: Text, sample: Text, detail: Text };
export type NotebookState = { goal: Text, cells: Cell[], order: Text[], results: CellResult[], blocked: Text[], status: Text, detail: Text, answer: Text };
