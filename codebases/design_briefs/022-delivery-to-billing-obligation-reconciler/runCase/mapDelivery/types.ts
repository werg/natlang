export type PreparedDeliveryEvidence = { candidate: BillingCondition; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsDeliveryEvidence = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditDeliveryEvidence = { verdict: GateVerdict; findings: string[]; repairTarget: string };
