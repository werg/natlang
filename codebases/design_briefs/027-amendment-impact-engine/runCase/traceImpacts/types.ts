export type PreparedAmendmentImpact = { candidate: AmendedObligation; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsAmendmentImpact = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditAmendmentImpact = { verdict: GateVerdict; findings: string[]; repairTarget: string };
