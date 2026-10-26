export type PreparedDecisionAssertion = { candidate: DecisionCase; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsDecisionAssertion = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditDecisionAssertion = { verdict: GateVerdict; findings: string[]; repairTarget: string };
