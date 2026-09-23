export type Contract = { id: string; signature: string; reason: string }; export type Diagnostic = { id: string; location: string; detail: string }; export type State = { revision: number; notice: string; source: string; contracts: Contract[]; diagnostics: Diagnostic[] };
export type Step = { state: State; ok: boolean; detail: string };
export type Decision = { action: "save"; text: string; } | { action: "infer"; target: string; text: string; secondary: string; } | { action: "diagnostic"; target: string; text: string; } | { action: "clear";  };
export type UiEvent = { id: string; kind: string; value?: string };
export type View = { heading: string; summary: string; focus: string[]; suggestions: string[] };
