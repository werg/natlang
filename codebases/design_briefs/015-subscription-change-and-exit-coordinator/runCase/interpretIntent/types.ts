export type PreparedChangeIntent = { candidate: SubscriptionCase; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsChangeIntent = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditChangeIntent = { verdict: GateVerdict; findings: string[]; repairTarget: string };
