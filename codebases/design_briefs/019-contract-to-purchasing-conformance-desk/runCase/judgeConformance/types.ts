export type PreparedConformanceFinding = { candidate: AgreementScope; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsConformanceFinding = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditConformanceFinding = { verdict: GateVerdict; findings: string[]; repairTarget: string };
