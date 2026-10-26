export type PreparedPolicyCaseFacts = { candidate: PolicyDelta; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsPolicyCaseFacts = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditPolicyCaseFacts = { verdict: GateVerdict; findings: string[]; repairTarget: string };
