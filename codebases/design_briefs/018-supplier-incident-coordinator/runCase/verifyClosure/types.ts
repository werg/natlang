export type PreparedIncidentClosure = { candidate: CorrectiveAction; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsIncidentClosure = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditIncidentClosure = { verdict: GateVerdict; findings: string[]; repairTarget: string };
