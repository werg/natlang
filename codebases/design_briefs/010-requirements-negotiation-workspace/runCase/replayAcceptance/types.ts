export type PreparedAcceptanceScenario = { candidate: Requirement; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsAcceptanceScenario = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditAcceptanceScenario = { verdict: GateVerdict; findings: string[]; repairTarget: string };
