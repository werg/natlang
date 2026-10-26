export type PreparedIncidentGroup = { candidate: ProblemEpisode; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsIncidentGroup = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditIncidentGroup = { verdict: GateVerdict; findings: string[]; repairTarget: string };
