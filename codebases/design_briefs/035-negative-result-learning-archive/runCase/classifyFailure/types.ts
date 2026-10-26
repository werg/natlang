export type PreparedNegativeFinding = { candidate: ResearchAttempt; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsNegativeFinding = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditNegativeFinding = { verdict: GateVerdict; findings: string[]; repairTarget: string };
