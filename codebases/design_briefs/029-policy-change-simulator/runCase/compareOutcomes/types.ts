export type PreparedOutcomeDifference = { candidate: PolicyCaseFacts; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsOutcomeDifference = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditOutcomeDifference = { verdict: GateVerdict; findings: string[]; repairTarget: string };
