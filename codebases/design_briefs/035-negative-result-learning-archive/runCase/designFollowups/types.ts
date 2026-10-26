export type PreparedDiscriminatingExperiment = { candidate: FailureExplanation; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsDiscriminatingExperiment = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditDiscriminatingExperiment = { verdict: GateVerdict; findings: string[]; repairTarget: string };
