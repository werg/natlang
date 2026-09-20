export type File = { id: Text; source: Text }; export type Scenario = { id: Text; input: Text; expected: Text; actual: Text; status: Text }; export type State = { revision: Num; notice: Text; files: File[]; selected: Text; result: Text; trace_id: Text; trace_count: Num; trace_cursor: Num; trace_frame: Text; scenarios: Scenario[] };
export type Step = { state: State; ok: Bool; detail: Text };
export type Decision = { action: "save"; target: Text; text: Text; } | { action: "select"; target: Text; } | { action: "add"; target: Text; text: Text; } | { action: "run"; text?: Text; } | { action: "inspect"; amount: Num; } | { action: "scenario"; text: Text; secondary: Text; } | { action: "evaluate_case"; target: Text; };
export type UiEvent = { id: Text; kind: Text; value?: Text };
export type View = { heading: Text; summary: Text; focus: Text[]; suggestions: Text[] };
