export type PreparedPurchasingOption = { candidate: ConformanceFinding; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsPurchasingOption = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditPurchasingOption = { verdict: GateVerdict; findings: string[]; repairTarget: string };
