export type PreparedAcceptanceFinding = { candidate: DeliveryEvidence; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsAcceptanceFinding = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditAcceptanceFinding = { verdict: GateVerdict; findings: string[]; repairTarget: string };
