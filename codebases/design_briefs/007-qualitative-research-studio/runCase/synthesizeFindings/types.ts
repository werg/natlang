export type PreparedResearchFinding = { candidate: CodingDecision; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsResearchFinding = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditResearchFinding = { verdict: GateVerdict; findings: string[]; repairTarget: string };
