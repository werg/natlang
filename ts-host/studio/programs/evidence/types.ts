export type Passage = { id: string; title: string; text: string }; export type Claim = { id: string; text: string; passage: string; quote: string }; export type State = { revision: number; notice: string; passages: Passage[]; query: string; hits: string[]; claims: Claim[] };
export type Step = { state: State; ok: boolean; detail: string };
export type Decision = { action: "add"; target: string; secondary: string; text: string; } | { action: "search"; text: string; } | { action: "claim"; target: string; text: string; secondary: string; } | { action: "remove"; target: string; };
export type UiEvent = { id: string; kind: string; value?: string };
export type View = { heading: string; summary: string; focus: string[]; suggestions: string[] };
