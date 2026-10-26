export type EvidenceItem = { id: string; source: string; version: string; text: string; status: 'current' | 'superseded' | 'unknown'; authority: string; };
export type EvidenceSlice = { subject: string; candidateIds: string[]; supporting: EvidenceItem[]; conflicting: EvidenceItem[]; missing: string[]; searchedScope: string; };

export type WorkRecord = { id: string; kind: string; subject: string; status: string; evidenceIds: string[]; fields: Record<string, string>; };

export type RecordMeta = { id: string; caseId: string; sourceRefs: string[]; state: 'draft' | 'qualified' | 'blocked'; };
export type Decimal = string;
export type Ratio = { numerator: Decimal; denominator: Decimal; };
export type Quantity = { value: Decimal; unit: string; };
export type Money = { minorUnits: string; currency: string; scale: number; };
export type HistoricalLinkage = RecordMeta & WorkRecord & {
  collectionIds: string[];
  entityKind: string;
  researchQuestion: string;
  archivistId: string;
  linkagePolicyId: string;
};
export type HistoricalObservation = RecordMeta & WorkRecord & {
  caseId: string;
  sourceRecordId: string;
  recordedName: string;
  dateExpression: string;
  placeExpression: string;
  attributes: string[];
};
export type HistoricalCandidate = RecordMeta & WorkRecord & {
  caseId: string;
  leftId: string;
  rightId: string;
  sharedEvidence: string[];
  incompatibleEvidence: string[];
};
export type HistoricalIdentityFinding = RecordMeta & WorkRecord & {
  caseId: string;
  candidateId: string;
  relation: 'same'|'different'|'related'|'uncertain';
  rationale: string;
};
export type HistoricalNetwork = RecordMeta & WorkRecord & {
  caseId: string;
  entityIds: string[];
  relationshipIds: string[];
  contradictionIds: string[];
  unresolvedAlternatives: string[];
};
export type HistoricalLinkageEdition = RecordMeta & WorkRecord & {
  caseId: string;
  networkId: string;
  acceptedLinkIds: string[];
  provisionalLinkIds: string[];
  reviewRefs: string[];
};
export type DraftHistoricalObservation = { values: {
    id?: string;
    caseId?: string;
    sourceRefs?: string[];
    state?: 'draft' | 'qualified' | 'blocked';
    kind?: string;
    subject?: string;
    evidenceIds?: string[];
    fields?: Record<string, string>;
    status?: string;
    sourceRecordId?: string;
    recordedName?: string;
    dateExpression?: string;
    placeExpression?: string;
    attributes?: string[];
  }; missingFields: string[]; conflicts: string[]; evidenceIds: string[]; };

export type LinkageArchivist = { id: string };
export type LinkageChange = { newRecords: EvidenceItem[]; transcriptionCorrections: EvidenceItem[]; geographyEvidence: EvidenceItem[]; relationshipEvidence: EvidenceItem[]; splitRequests: EvidenceItem[] };
export type LinkageDossier = { linkage: HistoricalLinkage; observations: HistoricalObservation[]; candidates: HistoricalCandidate[]; findings: HistoricalIdentityFinding[]; networks: HistoricalNetwork[]; editions: HistoricalLinkageEdition[]; evidence: EvidenceItem[]; checkpoint: LinkageCheckpoint };
export type DateInterval = { observationId: string; rawExpression: string; earliest: string; latest: string; precision: string; sourceIds: string[]; };
export type HistoricalPlace = { observationId: string; rawExpression: string; candidateJurisdictions: string[]; effectivePeriod: string; boundaryEvidenceIds: string[]; unresolved: string[]; };
export type LinkEdgeImpact = { candidateId: string; state: 'retained'|'reversed'|'review'; affectedFindings: string[]; affectedEditionStatements: string[]; sourceDeltaIds: string[]; };
export type LinkageCheckpoint = { revision: string; observations: HistoricalObservation[]; dates: DateInterval[]; places: HistoricalPlace[]; candidates: HistoricalCandidate[]; findings: HistoricalIdentityFinding[]; networks: HistoricalNetwork[]; impacts: LinkEdgeImpact[]; openQuestions: string[]; };
export type HistoricalLinkageProposal = { application: 'Historical Record Linkage Atelier'; status: 'ready-for-review'|'blocked'; observations: HistoricalObservation[]; dates: DateInterval[]; places: HistoricalPlace[]; candidates: HistoricalCandidate[]; findings: HistoricalIdentityFinding[]; network: HistoricalNetwork; impacts: LinkEdgeImpact[]; edition: HistoricalLinkageEdition; blockers: string[]; actionBoundary: string; resumeState: LinkageCheckpoint; };

export type HistoricalNormalization={observations:HistoricalObservation[];dates:DateInterval[];places:HistoricalPlace[];};

export type NetworkReconciliation={network:HistoricalNetwork;impacts:LinkEdgeImpact[];};
