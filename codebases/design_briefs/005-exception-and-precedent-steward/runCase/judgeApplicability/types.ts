export type PreparedApplicabilityFinding = { candidate: PrecedentCandidate; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsApplicabilityFinding = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditApplicabilityFinding = { verdict: GateVerdict; findings: string[]; repairTarget: string };
