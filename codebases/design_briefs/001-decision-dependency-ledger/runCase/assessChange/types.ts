export type PreparedChangeImpact = { candidate: AssumptionBinding; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsChangeImpact = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditChangeImpact = { verdict: GateVerdict; findings: string[]; repairTarget: string };
