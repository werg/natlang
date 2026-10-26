export type PreparedClaimConflict = { candidate: ScopedClaim; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsClaimConflict = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditClaimConflict = { verdict: GateVerdict; findings: string[]; repairTarget: string };
