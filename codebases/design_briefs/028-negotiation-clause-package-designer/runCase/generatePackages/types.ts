export type PreparedClausePackage = { candidate: ClauseDependency; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsClausePackage = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditClausePackage = { verdict: GateVerdict; findings: string[]; repairTarget: string };
