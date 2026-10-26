export type PreparedReviewPlan = { candidate: ChangeImpact; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsReviewPlan = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditReviewPlan = { verdict: GateVerdict; findings: string[]; repairTarget: string };
