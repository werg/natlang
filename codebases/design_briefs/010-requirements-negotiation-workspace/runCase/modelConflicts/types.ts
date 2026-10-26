export type PreparedRequirementConflict = { candidate: StakeholderPosition; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsRequirementConflict = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditRequirementConflict = { verdict: GateVerdict; findings: string[]; repairTarget: string };
