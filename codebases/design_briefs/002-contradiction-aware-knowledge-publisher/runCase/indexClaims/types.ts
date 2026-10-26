export type PreparedScopedClaim = { candidate: KnowledgeRevision; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsScopedClaim = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditScopedClaim = { verdict: GateVerdict; findings: string[]; repairTarget: string };
