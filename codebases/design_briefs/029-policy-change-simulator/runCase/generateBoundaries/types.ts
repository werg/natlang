export type PreparedBoundaryScenario = { candidate: OutcomeDifference; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsBoundaryScenario = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditBoundaryScenario = { verdict: GateVerdict; findings: string[]; repairTarget: string };
