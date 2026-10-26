export type PreparedReconciliationBreak = { candidate: ReconciliationScope; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsReconciliationBreak = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditReconciliationBreak = { verdict: GateVerdict; findings: string[]; repairTarget: string };
