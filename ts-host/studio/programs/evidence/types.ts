export type Passage = { id: Text; title: Text; text: Text }; export type Claim = { id: Text; text: Text; passage: Text; quote: Text }; export type State = { revision: Num; notice: Text; passages: Passage[]; query: Text; hits: Text[]; claims: Claim[] };
export type Step = { state: State; ok: Bool; detail: Text };
export type Decision = { action: "add"; target: Text; secondary: Text; text: Text; } | { action: "search"; text: Text; } | { action: "claim"; target: Text; text: Text; secondary: Text; } | { action: "remove"; target: Text; };
export type UiEvent = { id: Text; kind: Text; value?: Text };
export type View = { heading: Text; summary: Text; focus: Text[]; suggestions: Text[] };
