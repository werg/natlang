export type PreparedResourceCapability = { candidate: ExperimentRequirement; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsResourceCapability = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditResourceCapability = { verdict: GateVerdict; findings: string[]; repairTarget: string };
