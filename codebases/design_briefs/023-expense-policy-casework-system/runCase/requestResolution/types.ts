export type PreparedExpenseQuestion = { candidate: ExpenseFinding; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsExpenseQuestion = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditExpenseQuestion = { verdict: GateVerdict; findings: string[]; repairTarget: string };
