export type Log = { id: Text; time: Text; level: Text; message: Text }; export type Incident = { id: Text; title: Text; evidence: Text[]; severity: Text; status: Text }; export type State = { revision: Num; notice: Text; logs: Log[]; incidents: Incident[]; receipts: Text[] };
export type Step = { state: State; ok: Bool; detail: Text };
export type Decision = { action: "ingest"; text: Text; } | { action: "incident"; text: Text; target: Text; ids: Text[]; } | { action: "resolve"; target: Text; } | { action: "escalate"; target: Text; };
export type UiEvent = { id: Text; kind: Text; value?: Text };
export type View = { heading: Text; summary: Text; focus: Text[]; suggestions: Text[] };
