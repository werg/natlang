export type PreparedChangeOption = { candidate: ChangeConsequence; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsChangeOption = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditChangeOption = { verdict: GateVerdict; findings: string[]; repairTarget: string };
