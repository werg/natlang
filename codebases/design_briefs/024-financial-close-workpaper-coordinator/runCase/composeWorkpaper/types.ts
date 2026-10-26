export type PreparedWorkpaperConclusion = { candidate: CloseTask; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsWorkpaperConclusion = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditWorkpaperConclusion = { verdict: GateVerdict; findings: string[]; repairTarget: string };
