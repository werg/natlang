export type PreparedContractObligation = { candidate: ContractCase; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsContractObligation = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditContractObligation = { verdict: GateVerdict; findings: string[]; repairTarget: string };
