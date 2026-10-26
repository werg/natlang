export type PreparedTermSense = { candidate: IntegrationProject; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsTermSense = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditTermSense = { verdict: GateVerdict; findings: string[]; repairTarget: string };
