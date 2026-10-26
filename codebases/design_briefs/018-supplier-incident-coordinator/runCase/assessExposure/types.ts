export type PreparedExposureFinding = { candidate: IncidentCluster; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsExposureFinding = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditExposureFinding = { verdict: GateVerdict; findings: string[]; repairTarget: string };
