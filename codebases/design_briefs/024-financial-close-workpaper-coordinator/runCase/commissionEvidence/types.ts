export type PreparedCloseTask = { candidate: ReconciliationBreak; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsCloseTask = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditCloseTask = { verdict: GateVerdict; findings: string[]; repairTarget: string };
