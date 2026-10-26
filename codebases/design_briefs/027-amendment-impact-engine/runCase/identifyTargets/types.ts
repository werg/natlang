export type PreparedAmendmentTarget = { candidate: AmendmentCase; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsAmendmentTarget = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditAmendmentTarget = { verdict: GateVerdict; findings: string[]; repairTarget: string };
