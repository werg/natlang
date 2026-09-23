export type File = { id: string; source: string }; export type Scenario = { id: string; input: string; expected: string; actual: string; status: string }; export type State = { revision: number; notice: string; files: File[]; selected: string; result: string; trace_id: string; trace_count: number; trace_cursor: number; trace_frame: string; scenarios: Scenario[] };
export type Step = { state: State; ok: boolean; detail: string };
export type Decision = { action: "save"; target: string; text: string; } | { action: "select"; target: string; } | { action: "add"; target: string; text: string; } | { action: "run"; text?: string; } | { action: "inspect"; amount: number; } | { action: "scenario"; text: string; secondary: string; } | { action: "evaluate_case"; target: string; };
export type UiEvent = { id: string; kind: string; value?: string };
export type View = { heading: string; summary: string; focus: string[]; suggestions: string[] };
