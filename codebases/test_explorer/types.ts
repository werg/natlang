export type Observation = { id: string, source_revision: string, status: string, violations: string[], minimized: string[], trace_sha256: string, detail: string };
export type Assessment = { confirmed_ids: string[], unknown_ids: string[], findings: string[], unknowns: string[], followups: string[] };
export type Case = { id: string, description: string, group: string };
export type Selection = { ids: string[], reason: string };
