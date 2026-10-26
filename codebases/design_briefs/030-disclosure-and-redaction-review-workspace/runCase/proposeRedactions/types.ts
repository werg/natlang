export type PreparedRedactionProposal = { candidate: DisclosureSelection; evidence: Evidence[]; alternatives: string[]; missingFields: string[] };
export type DecisionsRedactionProposal = { atoms: DecisionAtom[]; unresolved: string[]; evidenceRefs: string[] };
export type AuditRedactionProposal = { verdict: GateVerdict; findings: string[]; repairTarget: string };
