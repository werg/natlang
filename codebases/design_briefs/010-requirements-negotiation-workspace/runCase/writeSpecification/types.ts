export type PreparedRequirement = { candidate: CompromiseOption; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsRequirement = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditRequirement = { verdict: GateVerdict; findings: string[]; repairTarget: string };
