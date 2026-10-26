export type PreparedSupplierOffer = { candidate: SourcingRequirement; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsSupplierOffer = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditSupplierOffer = { verdict: GateVerdict; findings: string[]; repairTarget: string };
