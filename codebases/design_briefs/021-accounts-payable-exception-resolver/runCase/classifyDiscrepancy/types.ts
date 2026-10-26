export type PreparedPayableDiscrepancy = { candidate: InvoiceMatch; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsPayableDiscrepancy = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditPayableDiscrepancy = { verdict: GateVerdict; findings: string[]; repairTarget: string };
