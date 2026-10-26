export type PreparedPurchaseDisposition = { candidate: PurchasingOption; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsPurchaseDisposition = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditPurchaseDisposition = { verdict: GateVerdict; findings: string[]; repairTarget: string };
