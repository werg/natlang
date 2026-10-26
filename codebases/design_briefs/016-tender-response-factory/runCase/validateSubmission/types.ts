export type PreparedSubmissionManifest = { candidate: ResponseSection; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsSubmissionManifest = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditSubmissionManifest = { verdict: GateVerdict; findings: string[]; repairTarget: string };
