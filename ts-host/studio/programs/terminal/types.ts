export type Receipt = { id: string; operation: string; status: string; output: string }; export type Line = { id: string; command: string; output: string; status: string }; export type State = { revision: number; notice: string; lines: Line[]; receipts: Receipt[] };
export type Step = { state: State; ok: boolean; detail: string };
export type Decision = { action: "execute"; text: string; } | { action: "recipe"; target: string; };
export type UiEvent = { id: string; kind: string; value?: string };
export type View = { heading: string; summary: string; focus: string[]; suggestions: string[] };
