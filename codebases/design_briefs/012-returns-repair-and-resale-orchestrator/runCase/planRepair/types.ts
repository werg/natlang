export type PreparedRepairPlan = { candidate: ReturnDisposition; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsRepairPlan = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditRepairPlan = { verdict: GateVerdict; findings: string[]; repairTarget: string };
