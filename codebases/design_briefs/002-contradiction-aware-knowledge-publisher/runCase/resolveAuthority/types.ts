export type PreparedAuthorityDecision = { candidate: ClaimConflict; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsAuthorityDecision = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditAuthorityDecision = { verdict: GateVerdict; findings: string[]; repairTarget: string };
