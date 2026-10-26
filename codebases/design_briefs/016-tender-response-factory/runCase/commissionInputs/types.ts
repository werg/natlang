export type PreparedContributionTask = { candidate: CapabilityMatch; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsContributionTask = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditContributionTask = { verdict: GateVerdict; findings: string[]; repairTarget: string };
