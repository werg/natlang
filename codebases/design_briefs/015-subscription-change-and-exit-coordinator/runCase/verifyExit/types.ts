export type PreparedExitReceipt = { candidate: SubscriptionOperation; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsExitReceipt = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditExitReceipt = { verdict: GateVerdict; findings: string[]; repairTarget: string };
