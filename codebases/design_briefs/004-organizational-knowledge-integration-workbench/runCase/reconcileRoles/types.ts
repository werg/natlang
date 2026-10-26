export type PreparedRoleInterface = { candidate: ConceptAlignment; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsRoleInterface = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditRoleInterface = { verdict: GateVerdict; findings: string[]; repairTarget: string };
