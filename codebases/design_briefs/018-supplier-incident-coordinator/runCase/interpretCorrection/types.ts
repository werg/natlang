export type PreparedCorrectiveAction = { candidate: ContainmentPlan; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsCorrectiveAction = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditCorrectiveAction = { verdict: GateVerdict; findings: string[]; repairTarget: string };
