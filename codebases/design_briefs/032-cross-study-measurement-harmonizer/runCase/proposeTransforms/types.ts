export type PreparedHarmonizationRule = { candidate: MeasureRelation; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsHarmonizationRule = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditHarmonizationRule = { verdict: GateVerdict; findings: string[]; repairTarget: string };
