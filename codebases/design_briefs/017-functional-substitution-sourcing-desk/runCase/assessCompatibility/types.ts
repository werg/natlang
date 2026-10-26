export type PreparedCompatibilityFinding = { candidate: SubstituteCandidate; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsCompatibilityFinding = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditCompatibilityFinding = { verdict: GateVerdict; findings: string[]; repairTarget: string };
