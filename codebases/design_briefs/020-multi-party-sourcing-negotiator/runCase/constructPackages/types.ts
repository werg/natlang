export type PreparedSourcingPackage = { candidate: SupplierOffer; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsSourcingPackage = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditSourcingPackage = { verdict: GateVerdict; findings: string[]; repairTarget: string };
