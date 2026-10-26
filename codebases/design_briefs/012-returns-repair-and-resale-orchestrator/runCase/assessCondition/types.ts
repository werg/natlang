export type PreparedConditionAssessment = { candidate: ReturnCase; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsConditionAssessment = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditConditionAssessment = { verdict: GateVerdict; findings: string[]; repairTarget: string };
