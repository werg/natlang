export type PreparedExperimentSlot = { candidate: ResourceCapability; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsExperimentSlot = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditExperimentSlot = { verdict: GateVerdict; findings: string[]; repairTarget: string };
