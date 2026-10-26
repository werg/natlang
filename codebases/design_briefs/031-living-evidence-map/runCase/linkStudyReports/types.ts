export type PreparedStudyIdentity = { candidate: EligibilityCriterion; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsStudyIdentity = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditStudyIdentity = { verdict: GateVerdict; findings: string[]; repairTarget: string };
