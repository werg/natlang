export type EvidenceItem = { id: string; source: string; version: string; text: string; status: 'current' | 'superseded' | 'unknown'; authority: string; };
export type EvidenceSlice = { subject: string; candidateIds: string[]; supporting: EvidenceItem[]; conflicting: EvidenceItem[]; missing: string[]; searchedScope: string; };

export type WorkRecord = { id: string; kind: string; subject: string; status: string; evidenceIds: string[]; fields: Record<string, string>; };

export type RecordMeta = { id: string; caseId: string; sourceRefs: string[]; state: 'draft' | 'qualified' | 'blocked'; };
export type Decimal = string;
export type Ratio = { numerator: Decimal; denominator: Decimal; };
export type Quantity = { value: Decimal; unit: string; };
export type Money = { minorUnits: string; currency: string; scale: number; };
export type TerminologyProject = RecordMeta & WorkRecord & {
  corpusIds: string[];
  languageIds: string[];
  domainScope: string;
  terminologyOwnerId: string;
  releaseContext: string;
};
export type TermOccurrence = RecordMeta & WorkRecord & {
  caseId: string;
  term: string;
  language: string;
  context: string;
  candidateConcept: string;
  sourceRef: string;
};
export type MultilingualConcept = RecordMeta & WorkRecord & {
  caseId: string;
  occurrenceIds: string[];
  definition: string;
  broaderConceptIds: string[];
  disputedBoundary: string;
};
export type TermOption = RecordMeta & WorkRecord & {
  caseId: string;
  conceptId: string;
  language: string;
  proposedTerm: string;
  usageConditions: string;
};
export type TerminologyDecision = RecordMeta & WorkRecord & {
  caseId: string;
  optionIds: string[];
  acceptedTermIds: string[];
  dissent: string;
  ownerApprovalRef: string;
};
export type TerminologyImpact = RecordMeta & WorkRecord & {
  caseId: string;
  decisionId: string;
  affectedSegmentIds: string[];
  excludedOccurrences: string[];
  migrationReason: string;
};
export type DraftTermOccurrence = { values: {
    id?: string;
    caseId?: string;
    sourceRefs?: string[];
    state?: 'draft' | 'qualified' | 'blocked';
    kind?: string;
    subject?: string;
    evidenceIds?: string[];
    fields?: Record<string, string>;
    status?: string;
    term?: string;
    language?: string;
    context?: string;
    candidateConcept?: string;
    sourceRef?: string;
  }; missingFields: string[]; conflicts: string[]; evidenceIds: string[]; };

export type TerminologyOwner = { id: string };
export type TerminologyChange = { occurrences: EvidenceItem[]; expertOpinions: EvidenceItem[]; regionalGuidance: EvidenceItem[]; inflectionReviews: EvidenceItem[]; releasedContent: EvidenceItem[] };
export type TerminologyDossier = { project: TerminologyProject; occurrences: TermOccurrence[]; concepts: MultilingualConcept[]; options: TermOption[]; decisions: TerminologyDecision[]; impacts: TerminologyImpact[]; evidence: EvidenceItem[]; checkpoint: TerminologyCheckpoint };
export type SenseCluster = { occurrenceIds: string[]; conceptCandidate: string; definition: string; usageContexts: string[]; regionalScopes: string[]; disputedBoundary: string[]; evidenceIds: string[]; };
export type InflectionCandidate = { optionId: string; language: string; grammaticalContext: string; forms: string[]; reviewState: 'approved'|'pending'|'rejected'; evidenceIds: string[]; };
export type MigrationPlan = { decisionId: string; affectedContentIds: string[]; priorVersion: string; targetVersion: string; stagedChanges: string[]; mixedWorkflowBlockers: string[]; approvals: string[]; };
export type TerminologyCheckpoint = { revision: string; occurrences: TermOccurrence[]; clusters: SenseCluster[]; concepts: MultilingualConcept[]; options: TermOption[]; inflections: InflectionCandidate[]; decisions: TerminologyDecision[]; migrations: MigrationPlan[]; openDisagreements: string[]; };
export type TerminologyProposal = { application: 'Multilingual Terminology Negotiation Workbench'; status: 'ready-for-review'|'blocked'; occurrences: TermOccurrence[]; concepts: MultilingualConcept[]; options: TermOption[]; inflections: InflectionCandidate[]; decision: TerminologyDecision; migrations: MigrationPlan[]; openDisagreements: string[]; blockers: string[]; actionBoundary: string; resumeState: TerminologyCheckpoint; };

export type TermOptionSet = { options: TermOption[]; inflections: InflectionCandidate[]; blockers: string[]; };
