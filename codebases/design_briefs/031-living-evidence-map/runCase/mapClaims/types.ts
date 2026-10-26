export type PreparedEvidenceClaim = { candidate: StudyObservation; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsEvidenceClaim = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditEvidenceClaim = { verdict: GateVerdict; findings: string[]; repairTarget: string };
