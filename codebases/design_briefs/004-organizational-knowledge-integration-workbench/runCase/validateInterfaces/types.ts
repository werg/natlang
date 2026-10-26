export type PreparedInterfaceScenario = { candidate: IntegrationWave; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsInterfaceScenario = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditInterfaceScenario = { verdict: GateVerdict; findings: string[]; repairTarget: string };
