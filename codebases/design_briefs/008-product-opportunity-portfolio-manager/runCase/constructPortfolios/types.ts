export type PreparedPortfolioOption = { candidate: OpportunityEstimate; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsPortfolioOption = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditPortfolioOption = { verdict: GateVerdict; findings: string[]; repairTarget: string };
