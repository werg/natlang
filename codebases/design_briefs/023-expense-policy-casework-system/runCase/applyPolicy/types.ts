export type PreparedExpenseFinding = { candidate: PurposeAllocation; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsExpenseFinding = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditExpenseFinding = { verdict: GateVerdict; findings: string[]; repairTarget: string };
