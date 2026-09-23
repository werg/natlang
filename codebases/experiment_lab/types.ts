export type Case = { id: string, group: string, description: string, expected: string };
export type Plan = { selected: string[], reason: string };
export type Candidate = { id: string, source_revision: string, model_id: string, model_seed: number };
export type Trial = { candidate: string, case_id: string, attempt: number, status: "done" | "quiesced" | "exception" | "missing", label: string, provenance: boolean, quality: "pass" | "fail" | "pending", value_digest: string, trace_id: string };
export type Metric = { candidate: string, planned: number, done: number, failed: number, missing: number, provenance_ok: number, reviewed_pass: number, reviewed_fail: number, review_pending: number, repeats_compared: number, repeats_agree: number };
export type Analysis = { interpretation: string, unknowns: string[], followups: string[] };
export type Report = { plan: Plan, metrics: Metric[], trials: Trial[], interpretation: string, unknowns: string[], followups: string[] };
