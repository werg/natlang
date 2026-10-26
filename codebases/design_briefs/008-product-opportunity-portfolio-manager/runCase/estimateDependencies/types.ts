export type PreparedOpportunityEstimate = { candidate: Intervention; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsOpportunityEstimate = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditOpportunityEstimate = { verdict: GateVerdict; findings: string[]; repairTarget: string };
