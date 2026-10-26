export type PreparedContainmentPlan = { candidate: ExposureFinding; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsContainmentPlan = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditContainmentPlan = { verdict: GateVerdict; findings: string[]; repairTarget: string };
