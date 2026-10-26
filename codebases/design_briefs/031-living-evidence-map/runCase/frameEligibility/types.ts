export type PreparedEligibilityCriterion = { candidate: EvidenceReview; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsEligibilityCriterion = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditEligibilityCriterion = { verdict: GateVerdict; findings: string[]; repairTarget: string };
