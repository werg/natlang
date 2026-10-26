export type PreparedPolicyOption = { candidate: BoundaryScenario; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsPolicyOption = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditPolicyOption = { verdict: GateVerdict; findings: string[]; repairTarget: string };
