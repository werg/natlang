export type PreparedPrecedentCandidate = { candidate: ExceptionRecord; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsPrecedentCandidate = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditPrecedentCandidate = { verdict: GateVerdict; findings: string[]; repairTarget: string };
