export type PreparedSubscriptionOperation = { candidate: ChangeOption; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsSubscriptionOperation = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditSubscriptionOperation = { verdict: GateVerdict; findings: string[]; repairTarget: string };
