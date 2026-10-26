export type PreparedReturnDisposition = { candidate: ConditionAssessment; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsReturnDisposition = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditReturnDisposition = { verdict: GateVerdict; findings: string[]; repairTarget: string };
