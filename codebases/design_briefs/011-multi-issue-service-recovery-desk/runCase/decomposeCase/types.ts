export type PreparedServiceIssue = { candidate: RecoveryCase; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsServiceIssue = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditServiceIssue = { verdict: GateVerdict; findings: string[]; repairTarget: string };
