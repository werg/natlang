export type PreparedDisputeRecommendation = { candidate: EvidenceFinding; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsDisputeRecommendation = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditDisputeRecommendation = { verdict: GateVerdict; findings: string[]; repairTarget: string };
