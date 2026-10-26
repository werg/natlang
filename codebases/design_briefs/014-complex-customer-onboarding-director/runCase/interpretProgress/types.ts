export type PreparedProgressFinding = { candidate: OnboardingTask; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsProgressFinding = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditProgressFinding = { verdict: GateVerdict; findings: string[]; repairTarget: string };
