export type PreparedExceptionRecord = { candidate: ExceptionCase; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsExceptionRecord = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditExceptionRecord = { verdict: GateVerdict; findings: string[]; repairTarget: string };
