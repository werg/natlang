export type PreparedAmendmentScope = { candidate: AmendmentTarget; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsAmendmentScope = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditAmendmentScope = { verdict: GateVerdict; findings: string[]; repairTarget: string };
