export type PreparedMeasureDefinition = { candidate: Harmonization; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsMeasureDefinition = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditMeasureDefinition = { verdict: GateVerdict; findings: string[]; repairTarget: string };
