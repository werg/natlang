export type PreparedResponseSection = { candidate: ContributionTask; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsResponseSection = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditResponseSection = { verdict: GateVerdict; findings: string[]; repairTarget: string };
