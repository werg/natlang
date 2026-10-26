export type PreparedEnvironmentConstraint = { candidate: OnboardingOutcome; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsEnvironmentConstraint = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditEnvironmentConstraint = { verdict: GateVerdict; findings: string[]; repairTarget: string };
