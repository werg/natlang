export type PreparedAssumptionBinding = { candidate: DecisionAssertion; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsAssumptionBinding = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditAssumptionBinding = { verdict: GateVerdict; findings: string[]; repairTarget: string };
