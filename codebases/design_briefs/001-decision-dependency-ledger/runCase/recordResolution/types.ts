export type PreparedResolutionDelta = { candidate: ReviewPlan; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsResolutionDelta = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditResolutionDelta = { verdict: GateVerdict; findings: string[]; repairTarget: string };
