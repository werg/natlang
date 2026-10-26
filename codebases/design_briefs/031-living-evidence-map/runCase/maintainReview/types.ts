export type PreparedEvidenceUpdate = { candidate: EvidenceClaim; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsEvidenceUpdate = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditEvidenceUpdate = { verdict: GateVerdict; findings: string[]; repairTarget: string };
