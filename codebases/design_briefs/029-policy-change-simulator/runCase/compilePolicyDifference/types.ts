export type PreparedPolicyDelta = { candidate: PolicySimulation; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsPolicyDelta = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditPolicyDelta = { verdict: GateVerdict; findings: string[]; repairTarget: string };
