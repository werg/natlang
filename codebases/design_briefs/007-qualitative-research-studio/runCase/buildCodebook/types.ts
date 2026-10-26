export type PreparedCodeDefinition = { candidate: SamplingPlan; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsCodeDefinition = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditCodeDefinition = { verdict: GateVerdict; findings: string[]; repairTarget: string };
