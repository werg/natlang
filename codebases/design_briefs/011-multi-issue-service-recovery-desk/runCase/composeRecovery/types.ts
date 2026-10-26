export type PreparedRemedyBundle = { candidate: EntitlementFinding; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsRemedyBundle = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditRemedyBundle = { verdict: GateVerdict; findings: string[]; repairTarget: string };
