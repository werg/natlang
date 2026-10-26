export type PreparedCapabilityMatch = { candidate: TenderRequirement; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsCapabilityMatch = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditCapabilityMatch = { verdict: GateVerdict; findings: string[]; repairTarget: string };
