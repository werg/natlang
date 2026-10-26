export type PreparedOnboardingTask = { candidate: EnvironmentConstraint; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsOnboardingTask = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditOnboardingTask = { verdict: GateVerdict; findings: string[]; repairTarget: string };
