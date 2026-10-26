export type PreparedHarmonizationAudit = { candidate: HarmonizedSlice; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsHarmonizationAudit = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditHarmonizationAudit = { verdict: GateVerdict; findings: string[]; repairTarget: string };
