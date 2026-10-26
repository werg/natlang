export type PreparedPostingReceipt = { candidate: PayableDisposition; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsPostingReceipt = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditPostingReceipt = { verdict: GateVerdict; findings: string[]; repairTarget: string };
