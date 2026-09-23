export type Cell = { id: string; engine: string; source: string; needs: string[]; result: string; result_id: string; status: string }; export type State = { revision: number; notice: string; cells: Cell[] };
export type Step = { state: State; ok: boolean; detail: string };
export type Decision = { action: "add"; target: string; secondary: string; text: string; ids?: string[]; } | { action: "save"; target: string; text: string; } | { action: "remove"; target: string; } | { action: "execute"; target: string; };
export type UiEvent = { id: string; kind: string; value?: string };
export type View = { heading: string; summary: string; focus: string[]; suggestions: string[] };
