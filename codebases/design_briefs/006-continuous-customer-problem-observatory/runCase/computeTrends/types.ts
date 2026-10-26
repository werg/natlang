export type PreparedTrendSlice = { candidate: ProblemConcept; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsTrendSlice = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditTrendSlice = { verdict: GateVerdict; findings: string[]; repairTarget: string };
