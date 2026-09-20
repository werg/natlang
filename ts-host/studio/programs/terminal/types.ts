export type Receipt = { id: Text; operation: Text; status: Text; output: Text }; export type Line = { id: Text; command: Text; output: Text; status: Text }; export type State = { revision: Num; notice: Text; lines: Line[]; receipts: Receipt[] };
export type Step = { state: State; ok: Bool; detail: Text };
export type Decision = { action: "execute"; text: Text; } | { action: "recipe"; target: Text; };
export type UiEvent = { id: Text; kind: Text; value?: Text };
export type View = { heading: Text; summary: Text; focus: Text[]; suggestions: Text[] };
