export type PreparedExceptionDisposition = { candidate: ApplicabilityFinding; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsExceptionDisposition = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditExceptionDisposition = { verdict: GateVerdict; findings: string[]; repairTarget: string };
