export type Cell = { id: Text, needs: Text[], description: Text, engine: Text, revision: Num };
export type CellResult = { id: Text, status: Text, revision: Num, output_sha256: Text, sample: Text, detail: Text };
export type NotebookState = { goal: Text, cells: Cell[], order: Text[], results: CellResult[], blocked: Text[], status: Text, detail: Text, answer: Text };
export type ConsoleEvent = { id: Text, kind: Text, value: Text };
export type ConsoleState = { requests: Text[], runs: NotebookState[], status: Text };
export type ViewBlock = { kind: Text, text?: Text, tone?: Text, items?: Text[], ordered?: Bool, columns?: Text[], rows?: Text[][] };
export type TerminalView = { title?: Text, subtitle?: Text, blocks: ViewBlock[], prompt?: Text, busy?: Bool, help?: Text[] };
