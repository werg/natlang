export type PreparedDisclosureManifest = { candidate: DisclosureRisk; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsDisclosureManifest = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditDisclosureManifest = { verdict: GateVerdict; findings: string[]; repairTarget: string };
