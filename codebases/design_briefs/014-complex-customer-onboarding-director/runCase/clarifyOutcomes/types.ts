export type PreparedOnboardingOutcome = { candidate: Onboarding; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsOnboardingOutcome = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditOnboardingOutcome = { verdict: GateVerdict; findings: string[]; repairTarget: string };
