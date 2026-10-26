export type PreparedReviewDossier = { candidate: DisputeRecommendation; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsReviewDossier = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditReviewDossier = { verdict: GateVerdict; findings: string[]; repairTarget: string };
