export type PreparedExpenseLine = { candidate: ExpenseCase; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsExpenseLine = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditExpenseLine = { verdict: GateVerdict; findings: string[]; repairTarget: string };
