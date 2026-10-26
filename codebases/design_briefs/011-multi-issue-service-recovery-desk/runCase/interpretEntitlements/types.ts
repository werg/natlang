export type PreparedEntitlementFinding = { candidate: ServiceIssue; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsEntitlementFinding = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditEntitlementFinding = { verdict: GateVerdict; findings: string[]; repairTarget: string };
