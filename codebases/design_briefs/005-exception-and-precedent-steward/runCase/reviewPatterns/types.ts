export type PreparedPolicyReviewSignal = { candidate: ExceptionDisposition; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsPolicyReviewSignal = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditPolicyReviewSignal = { verdict: GateVerdict; findings: string[]; repairTarget: string };
