export type PreparedCodingDecision = { candidate: CodeDefinition; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsCodingDecision = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditCodingDecision = { verdict: GateVerdict; findings: string[]; repairTarget: string };
