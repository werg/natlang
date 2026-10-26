export type EvidenceItem = { id: string; source: string; version: string; text: string; status: 'current' | 'superseded' | 'unknown'; authority: string; };
export type EvidenceSlice = { subject: string; candidateIds: string[]; supporting: EvidenceItem[]; conflicting: EvidenceItem[]; missing: string[]; searchedScope: string; };

export type WorkRecord = { id: string; kind: string; subject: string; status: string; evidenceIds: string[]; fields: Record<string, string>; };

export type RecordMeta = { id: string; caseId: string; sourceRefs: string[]; state: 'draft' | 'qualified' | 'blocked'; };
export type Decimal = string;
export type Ratio = { numerator: Decimal; denominator: Decimal; };
export type Quantity = { value: Decimal; unit: string; };
export type Money = { minorUnits: string; currency: string; scale: number; };
export type Consultation = RecordMeta & WorkRecord & {
  topic: string;
  submissionIds: string[];
  scopeStatementId: string;
  analystId: string;
  responseOwnerId: string;
};
export type ConsultationArgument = RecordMeta & WorkRecord & {
  caseId: string;
  submissionId: string;
  position: string;
  reason: string;
  proposedAction: string;
  affectedContext: string;
};
export type SubmissionRelation = RecordMeta & WorkRecord & {
  caseId: string;
  submissionIds: string[];
  relation: 'template'|'duplicate'|'independent'|'unknown';
  uniqueContribution: string;
};
export type ConsultationTheme = RecordMeta & WorkRecord & {
  caseId: string;
  argumentIds: string[];
  commonConcern: string;
  minorityPositions: string[];
  unresolvedDisagreement: string;
};
export type ConsultationResponseOption = RecordMeta & WorkRecord & {
  caseId: string;
  themeId: string;
  proposedResponse: string;
  supportingRefs: string[];
  tradeoff: string;
};
export type ConsultationReport = RecordMeta & WorkRecord & {
  caseId: string;
  themeIds: string[];
  countArtifactId: string;
  omissionFindings: string[];
  approvalRefs: string[];
};
export type DraftConsultationArgument = { values: {
    id?: string;
    caseId?: string;
    sourceRefs?: string[];
    state?: 'draft' | 'qualified' | 'blocked';
    kind?: string;
    subject?: string;
    evidenceIds?: string[];
    fields?: Record<string, string>;
    status?: string;
    submissionId?: string;
    position?: string;
    reason?: string;
    proposedAction?: string;
    affectedContext?: string;
  }; missingFields: string[]; conflicts: string[]; evidenceIds: string[]; };

export type SynthesisEditor = { id: string };
export type ConsultationDocket = { consultation: Consultation; submissions: ConsultationArgument[]; relations: SubmissionRelation[]; themes: ConsultationTheme[]; responseOptions: ConsultationResponseOption[]; report: ConsultationReport | null; evidence: EvidenceItem[]; checkpoint: ConsultationCheckpoint };
export type SubmissionDelta = { newSubmissions: ConsultationArgument[]; lateSubmissionRefs: string[]; challengedThemeIds: string[]; privacyConsents: string[] };
export type ScopedArgument = { submissionId: string; position: string; reason: string; factualClaims: string[]; proposedAction: string; scope: string; evidenceIds: string[]; privacyFlags: string[] };
export type CampaignRelationGraph = { relations: SubmissionRelation[]; participantCount: string; templateClusterCount: string; uniqueReasonsByCluster: Record<string,string[]>; };
export type ThemeCoverage = { theme: ConsultationTheme; argumentIds: string[]; minorityPositions: string[]; disagreementKind: 'factual'|'value'|'mixed'|'unknown'; unsupportedConsensus: string[]; };
export type ResponseDraft = { options: ConsultationResponseOption[]; authorityQuestions: string[]; unsupportedChoices: string[]; };
export type ConsultationCheckpoint = { edition: string; argumentMap: ScopedArgument[]; relations: SubmissionRelation[]; themes: ConsultationTheme[]; responses: ConsultationResponseOption[]; report: ConsultationReport | null; includedLateSubmissionRefs: string[]; privacyRedactions: string[]; challengedCoverageDeltas: string[]; };
export type ConsultationUpdate = { responseOwnerId: string; decisionQuestions: string[]; challengedThemeIds: string[]; acceptedLateSubmissionRefs: string[]; };
export type SynthesisProposal = { application: 'Public Consultation Synthesis Studio'; status: 'ready-for-review'|'blocked'; scopedArguments: ScopedArgument[]; campaignGraph: CampaignRelationGraph; themeCoverage: ThemeCoverage[]; responses: ResponseDraft; report: ConsultationReport; countStatement: string; omissionFindings: string[]; privacyFindings: string[]; actionBoundary: string; resumeState: ConsultationCheckpoint; };
