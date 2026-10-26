export type EvidenceItem = { id: string; source: string; version: string; text: string; status: 'current' | 'superseded' | 'unknown'; authority: string; };
export type EvidenceSlice = { subject: string; candidateIds: string[]; supporting: EvidenceItem[]; conflicting: EvidenceItem[]; missing: string[]; searchedScope: string; };

export type WorkRecord = { id: string; kind: string; subject: string; status: string; evidenceIds: string[]; fields: Record<string, string>; };

export type RecordMeta = { id: string; caseId: string; sourceRefs: string[]; state: 'draft' | 'qualified' | 'blocked'; };
export type Decimal = string;
export type Ratio = { numerator: Decimal; denominator: Decimal; };
export type Quantity = { value: Decimal; unit: string; };
export type Money = { minorUnits: string; currency: string; scale: number; };
export type OralHistoryProject = RecordMeta & WorkRecord & {
  interviewIds: string[];
  researchScope: string;
  archivistId: string;
  consentRegisterId: string;
  publicationAudience: string;
};
export type OralHistoryAccount = RecordMeta & WorkRecord & {
  caseId: string;
  speakerId: string;
  eventDescription: string;
  rememberedTime: string;
  certaintyLanguage: string;
};
export type OralHistoryReference = RecordMeta & WorkRecord & {
  caseId: string;
  accountId: string;
  mentionedName: string;
  candidateEntityIds: string[];
  unresolvedIdentity: string;
};
export type RecollectionComparison = RecordMeta & WorkRecord & {
  caseId: string;
  accountIds: string[];
  sharedEvent: string;
  agreement: string;
  disagreement: string;
};
export type OralHistoryQuestion = RecordMeta & WorkRecord & {
  caseId: string;
  accountIds: string[];
  unresolvedIssue: string;
  neutralQuestion: string;
  consentConstraint: string;
};
export type OralHistoryEdition = RecordMeta & WorkRecord & {
  caseId: string;
  accountIds: string[];
  narrative: string;
  restrictionIds: string[];
  provenanceManifestId: string;
};
export type DraftOralHistoryAccount = { values: {
    id?: string;
    caseId?: string;
    sourceRefs?: string[];
    state?: 'draft' | 'qualified' | 'blocked';
    kind?: string;
    subject?: string;
    evidenceIds?: string[];
    fields?: Record<string, string>;
    status?: string;
    speakerId?: string;
    eventDescription?: string;
    rememberedTime?: string;
    certaintyLanguage?: string;
  }; missingFields: string[]; conflicts: string[]; evidenceIds: string[]; };

export type Archivist = { id: string };
export type ArchiveChange = { transcriptUpdates: EvidenceItem[]; speakerResponses: EvidenceItem[]; consentChanges: EvidenceItem[]; correctionEvidence: EvidenceItem[]; disclosureReviews: EvidenceItem[] };
export type OralHistoryDossier = { project: OralHistoryProject; accounts: OralHistoryAccount[]; references: OralHistoryReference[]; comparisons: RecollectionComparison[]; questions: OralHistoryQuestion[]; editions: OralHistoryEdition[]; evidence: EvidenceItem[]; checkpoint: OralArchiveCheckpoint };
export type ConsentSpan = { speakerId: string; interviewId: string; sourceSpan: string; permittedUses: string[]; publicationAudience: string; state: 'current'|'withdrawn'|'uncertain'; evidenceIds: string[]; };

export type DerivativeLineage = { derivativeId: string; sourceSpanIds: string[]; consentIds: string[]; derivativeKind: string; status: 'permitted'|'retract'|'review'|'withheld'; };
export type OralArchiveCheckpoint = { revision: string; accounts: OralHistoryAccount[]; references: OralHistoryReference[]; comparisons: RecollectionComparison[]; consent: ConsentSpan[]; questions: OralHistoryQuestion[]; editions: OralHistoryEdition[]; lineage: DerivativeLineage[]; withdrawnSpans: string[]; obligations: string[]; };
export type OralArchiveProposal = { application: 'Oral History Reconciliation Archive'; status: 'ready-for-review'|'blocked'; accounts: OralHistoryAccount[]; references: OralHistoryReference[]; comparisons: RecollectionComparison[]; questions: OralHistoryQuestion[]; edition: OralHistoryEdition; consent: ConsentSpan[]; retractLineage: DerivativeLineage[]; disclosureFindings: string[]; blockers: string[]; actionBoundary: string; resumeState: OralArchiveCheckpoint; };
