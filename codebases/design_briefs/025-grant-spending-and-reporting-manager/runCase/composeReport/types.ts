export type PreparedGrantNarrative = { candidate: ActivityEvidence; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsGrantNarrative = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditGrantNarrative = { verdict: GateVerdict; findings: string[]; repairTarget: string };
