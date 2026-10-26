export type PreparedReconciliationScope = { candidate: CloseWorkpaper; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsReconciliationScope = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditReconciliationScope = { verdict: GateVerdict; findings: string[]; repairTarget: string };
