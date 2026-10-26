export type PreparedCompromiseOption = { candidate: RequirementConflict; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsCompromiseOption = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditCompromiseOption = { verdict: GateVerdict; findings: string[]; repairTarget: string };
