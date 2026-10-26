export type PreparedClauseDependency = { candidate: NegotiationPosition; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsClauseDependency = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditClauseDependency = { verdict: GateVerdict; findings: string[]; repairTarget: string };
