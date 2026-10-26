export type PreparedStudyObservation = { candidate: StudyIdentity; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsStudyObservation = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditStudyObservation = { verdict: GateVerdict; findings: string[]; repairTarget: string };
