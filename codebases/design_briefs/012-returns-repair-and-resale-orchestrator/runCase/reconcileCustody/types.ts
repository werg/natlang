export type PreparedCustodyEvent = { candidate: RepairPlan; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsCustodyEvent = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditCustodyEvent = { verdict: GateVerdict; findings: string[]; repairTarget: string };
