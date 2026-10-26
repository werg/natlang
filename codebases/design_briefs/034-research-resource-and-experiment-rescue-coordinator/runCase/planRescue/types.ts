export type PreparedRescueOption = { candidate: ExperimentSlot; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsRescueOption = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditRescueOption = { verdict: GateVerdict; findings: string[]; repairTarget: string };
