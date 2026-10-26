export type PreparedIntegrationWave = { candidate: RoleInterface; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsIntegrationWave = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditIntegrationWave = { verdict: GateVerdict; findings: string[]; repairTarget: string };
