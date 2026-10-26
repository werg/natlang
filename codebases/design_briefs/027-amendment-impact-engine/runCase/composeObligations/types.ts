export type PreparedAmendedObligation = { candidate: AmendmentScope; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsAmendedObligation = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditAmendedObligation = { verdict: GateVerdict; findings: string[]; repairTarget: string };
