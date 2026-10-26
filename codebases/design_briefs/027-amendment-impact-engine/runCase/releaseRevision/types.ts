export type PreparedAmendmentRelease = { candidate: AmendmentImpact; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsAmendmentRelease = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditAmendmentRelease = { verdict: GateVerdict; findings: string[]; repairTarget: string };
