export type PreparedOnboardingRevision = { candidate: ProgressFinding; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsOnboardingRevision = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditOnboardingRevision = { verdict: GateVerdict; findings: string[]; repairTarget: string };
