export type PreparedOpportunityExperiment = { candidate: PortfolioOption; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsOpportunityExperiment = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditOpportunityExperiment = { verdict: GateVerdict; findings: string[]; repairTarget: string };
