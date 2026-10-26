export type PreparedPurposeAllocation = { candidate: ExpenseLine; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsPurposeAllocation = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditPurposeAllocation = { verdict: GateVerdict; findings: string[]; repairTarget: string };
