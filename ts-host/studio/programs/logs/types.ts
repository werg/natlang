export type Log = { id: string; time: string; level: string; message: string }; export type Incident = { id: string; title: string; evidence: string[]; severity: string; status: string }; export type State = { revision: number; notice: string; logs: Log[]; incidents: Incident[]; receipts: string[] };
export type Step = { state: State; ok: boolean; detail: string };
export type Decision = { action: "ingest"; text: string; } | { action: "incident"; text: string; target: string; ids: string[]; } | { action: "resolve"; target: string; } | { action: "escalate"; target: string; };
export type UiEvent = { id: string; kind: string; value?: string };
export type View = { heading: string; summary: string; focus: string[]; suggestions: string[] };
