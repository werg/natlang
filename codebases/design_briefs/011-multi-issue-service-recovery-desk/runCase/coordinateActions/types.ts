export type PreparedRecoveryAction = { candidate: RemedyBundle; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsRecoveryAction = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditRecoveryAction = { verdict: GateVerdict; findings: string[]; repairTarget: string };
