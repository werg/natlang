export type Cell = { id: Text; engine: Text; source: Text; needs: Text[]; result: Text; result_id: Text; status: Text }; export type State = { revision: Num; notice: Text; cells: Cell[] };
export type Step = { state: State; ok: Bool; detail: Text };
export type Decision = { action: "add"; target: Text; secondary: Text; text: Text; ids?: Text[]; } | { action: "save"; target: Text; text: Text; } | { action: "remove"; target: Text; } | { action: "execute"; target: Text; };
export type UiEvent = { id: Text; kind: Text; value?: Text };
export type View = { heading: Text; summary: Text; focus: Text[]; suggestions: Text[] };
