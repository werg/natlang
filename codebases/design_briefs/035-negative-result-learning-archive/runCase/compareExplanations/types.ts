export type PreparedFailureExplanation = { candidate: NegativeFinding; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsFailureExplanation = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditFailureExplanation = { verdict: GateVerdict; findings: string[]; repairTarget: string };
