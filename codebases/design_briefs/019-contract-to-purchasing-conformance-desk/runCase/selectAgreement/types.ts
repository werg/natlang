export type PreparedAgreementScope = { candidate: PurchaseIntent; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsAgreementScope = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditAgreementScope = { verdict: GateVerdict; findings: string[]; repairTarget: string };
