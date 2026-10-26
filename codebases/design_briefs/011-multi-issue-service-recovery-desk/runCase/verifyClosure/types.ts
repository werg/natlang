export type PreparedRecoveryClosure = { candidate: RecoveryAction; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsRecoveryClosure = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditRecoveryClosure = { verdict: GateVerdict; findings: string[]; repairTarget: string };
