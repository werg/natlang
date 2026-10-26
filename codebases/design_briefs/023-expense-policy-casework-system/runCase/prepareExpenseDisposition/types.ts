export type PreparedExpenseDisposition = { candidate: ExpenseQuestion; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsExpenseDisposition = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditExpenseDisposition = { verdict: GateVerdict; findings: string[]; repairTarget: string };
