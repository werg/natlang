export type PreparedMeasureRelation = { candidate: MeasureDefinition; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsMeasureRelation = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditMeasureRelation = { verdict: GateVerdict; findings: string[]; repairTarget: string };
