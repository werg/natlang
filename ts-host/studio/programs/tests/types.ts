export type Case = { id: string; input: string; expected: string; actual: string; status: string }; export type State = { revision: number; notice: string; source: string; contract: string; cases: Case[] };
export type Step = { state: State; ok: boolean; detail: string };
export type Decision = { action: "save"; text: string; secondary: string; } | { action: "add"; target: string; text: string; secondary: string; } | { action: "run_case"; target: string; };
export type UiEvent = { id: string; kind: string; value?: string };
export type View = { heading: string; summary: string; focus: string[]; suggestions: string[] };
