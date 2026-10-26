export type PreparedResearchAttempt = { candidate: NegativeResultCase; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsResearchAttempt = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditResearchAttempt = { verdict: GateVerdict; findings: string[]; repairTarget: string };
