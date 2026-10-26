export type PreparedExperimentRequirement = { candidate: ResearchSchedule; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsExperimentRequirement = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditExperimentRequirement = { verdict: GateVerdict; findings: string[]; repairTarget: string };
