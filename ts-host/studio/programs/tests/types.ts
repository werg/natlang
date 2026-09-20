export type Case = { id: Text; input: Text; expected: Text; actual: Text; status: Text }; export type State = { revision: Num; notice: Text; source: Text; contract: Text; cases: Case[] };
export type Step = { state: State; ok: Bool; detail: Text };
export type Decision = { action: "save"; text: Text; secondary: Text; } | { action: "add"; target: Text; text: Text; secondary: Text; } | { action: "run_case"; target: Text; };
export type UiEvent = { id: Text; kind: Text; value?: Text };
export type View = { heading: Text; summary: Text; focus: Text[]; suggestions: Text[] };
