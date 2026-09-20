export type Contract = { id: Text; signature: Text; reason: Text }; export type Diagnostic = { id: Text; location: Text; detail: Text }; export type State = { revision: Num; notice: Text; source: Text; contracts: Contract[]; diagnostics: Diagnostic[] };
export type Step = { state: State; ok: Bool; detail: Text };
export type Decision = { action: "save"; text: Text; } | { action: "infer"; target: Text; text: Text; secondary: Text; } | { action: "diagnostic"; target: Text; text: Text; } | { action: "clear";  };
export type UiEvent = { id: Text; kind: Text; value?: Text };
export type View = { heading: Text; summary: Text; focus: Text[]; suggestions: Text[] };
