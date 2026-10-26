export type PreparedSubstitutionPacket = { candidate: QualificationTask; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsSubstitutionPacket = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditSubstitutionPacket = { verdict: GateVerdict; findings: string[]; repairTarget: string };
