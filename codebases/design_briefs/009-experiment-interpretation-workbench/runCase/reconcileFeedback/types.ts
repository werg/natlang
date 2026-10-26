export type PreparedMechanismFinding = { candidate: ResultClaim; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsMechanismFinding = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditMechanismFinding = { verdict: GateVerdict; findings: string[]; repairTarget: string };
