export type PreparedWorkpaperSignoff = { candidate: WorkpaperConclusion; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsWorkpaperSignoff = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditWorkpaperSignoff = { verdict: GateVerdict; findings: string[]; repairTarget: string };
