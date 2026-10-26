export type PreparedChangeConsequence = { candidate: ChangeIntent; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsChangeConsequence = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditChangeConsequence = { verdict: GateVerdict; findings: string[]; repairTarget: string };
