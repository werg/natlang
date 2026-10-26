export type PreparedObligationRelation = { candidate: ContractObligation; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsObligationRelation = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditObligationRelation = { verdict: GateVerdict; findings: string[]; repairTarget: string };
