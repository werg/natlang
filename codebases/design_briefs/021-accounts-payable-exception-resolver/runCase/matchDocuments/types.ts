export type PreparedInvoiceMatch = { candidate: PayableCase; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsInvoiceMatch = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditInvoiceMatch = { verdict: GateVerdict; findings: string[]; repairTarget: string };
