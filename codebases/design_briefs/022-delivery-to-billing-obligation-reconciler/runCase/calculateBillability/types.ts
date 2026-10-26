export type PreparedBillableAllocation = { candidate: AcceptanceFinding; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsBillableAllocation = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditBillableAllocation = { verdict: GateVerdict; findings: string[]; repairTarget: string };
