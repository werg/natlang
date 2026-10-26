export type PreparedCustomerNeed = { candidate: OpportunityPortfolio; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsCustomerNeed = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditCustomerNeed = { verdict: GateVerdict; findings: string[]; repairTarget: string };
