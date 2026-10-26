export type PreparedAnalysisOrder = { candidate: MechanismFinding; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsAnalysisOrder = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditAnalysisOrder = { verdict: GateVerdict; findings: string[]; repairTarget: string };
