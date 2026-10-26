export type PreparedFunctionalRequirement = { candidate: SubstitutionRequest; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsFunctionalRequirement = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditFunctionalRequirement = { verdict: GateVerdict; findings: string[]; repairTarget: string };
