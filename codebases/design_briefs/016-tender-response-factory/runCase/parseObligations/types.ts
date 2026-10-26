export type PreparedTenderRequirement = { candidate: TenderResponse; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsTenderRequirement = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditTenderRequirement = { verdict: GateVerdict; findings: string[]; repairTarget: string };
