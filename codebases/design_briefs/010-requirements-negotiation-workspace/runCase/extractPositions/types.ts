export type PreparedStakeholderPosition = { candidate: RequirementsProject; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsStakeholderPosition = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditStakeholderPosition = { verdict: GateVerdict; findings: string[]; repairTarget: string };
