export type PreparedBillingCondition = { candidate: BillingCase; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsBillingCondition = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditBillingCondition = { verdict: GateVerdict; findings: string[]; repairTarget: string };
