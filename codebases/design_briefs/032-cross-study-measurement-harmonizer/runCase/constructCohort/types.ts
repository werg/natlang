export type PreparedHarmonizedSlice = { candidate: HarmonizationRule; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsHarmonizedSlice = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditHarmonizedSlice = { verdict: GateVerdict; findings: string[]; repairTarget: string };
