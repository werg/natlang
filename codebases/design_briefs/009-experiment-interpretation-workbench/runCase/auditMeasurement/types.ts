export type PreparedMeasurementFinding = { candidate: ExperimentDesign; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsMeasurementFinding = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditMeasurementFinding = { verdict: GateVerdict; findings: string[]; repairTarget: string };
