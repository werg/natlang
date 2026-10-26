export type PreparedPurchaseIntent = { candidate: PurchaseReview; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsPurchaseIntent = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditPurchaseIntent = { verdict: GateVerdict; findings: string[]; repairTarget: string };
