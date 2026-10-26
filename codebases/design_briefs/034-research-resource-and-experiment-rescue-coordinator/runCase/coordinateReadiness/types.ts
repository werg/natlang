export type PreparedReadinessCheck = { candidate: RescueOption; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsReadinessCheck = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditReadinessCheck = { verdict: GateVerdict; findings: string[]; repairTarget: string };
