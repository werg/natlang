export type EvidenceItem = { id: string; source: string; version: string; text: string; status: 'current' | 'superseded' | 'unknown'; authority: string; };
export type EvidenceSlice = { subject: string; candidateIds: string[]; supporting: EvidenceItem[]; conflicting: EvidenceItem[]; missing: string[]; searchedScope: string; };

export type WorkRecord = { id: string; kind: string; subject: string; status: string; evidenceIds: string[]; fields: Record<string, string>; };

export type RecordMeta = { id: string; caseId: string; sourceRefs: string[]; state: 'draft' | 'qualified' | 'blocked'; };
export type Decimal = string;
export type Ratio = { numerator: Decimal; denominator: Decimal; };
export type Quantity = { value: Decimal; unit: string; };
export type Money = { minorUnits: string; currency: string; scale: number; };
export type HealthRecordRequest = RecordMeta & WorkRecord & {
  requesterId: string;
  patientId: string;
  requestedScope: string;
  authorizationId: string;
  recordsOfficerId: string;
};
export type RecordAuthorization = RecordMeta & WorkRecord & {
  caseId: string;
  authorizedRecipientId: string;
  permittedCategories: string[];
  dateWindow: string;
  expiresAt: string;
};
export type RecordSelection = RecordMeta & WorkRecord & {
  caseId: string;
  patientId: string;
  episodeIds: string[];
  documentIds: string[];
  missingCoverage: string;
};
export type RecordDisclosureFinding = RecordMeta & WorkRecord & {
  caseId: string;
  documentId: string;
  responsiveSpans: string[];
  thirdPartySpans: string[];
  reviewReason: string;
};
export type RecordFulfillment = RecordMeta & WorkRecord & {
  caseId: string;
  selectionId: string;
  artifactIds: string[];
  withheldReasons: string[];
  reviewerApprovalRef: string;
};
export type RecordDelivery = RecordMeta & WorkRecord & {
  caseId: string;
  fulfillmentId: string;
  recipientBinding: string;
  deliveryReceipt: string;
  remainingObligations: string[];
};
export type DraftRecordAuthorization = { values: {
    id?: string;
    caseId?: string;
    sourceRefs?: string[];
    state?: 'draft' | 'qualified' | 'blocked';
    kind?: string;
    subject?: string;
    evidenceIds?: string[];
    fields?: Record<string, string>;
    status?: string;
    authorizedRecipientId?: string;
    permittedCategories?: string[];
    dateWindow?: string;
    expiresAt?: string;
  }; missingFields: string[]; conflicts: string[]; evidenceIds: string[]; };

export type RecordsOfficer = { id: string };
export type FulfillmentChange = { authorizationUpdates: EvidenceItem[]; indexedRecords: EvidenceItem[]; disclosureReviews: EvidenceItem[]; renderedArtifacts: EvidenceItem[]; metadataScans: EvidenceItem[]; approvalReceipts: EvidenceItem[]; deliveryReceipts: EvidenceItem[]; correctedRecords: EvidenceItem[] };
export type RecordsDossier = { request: HealthRecordRequest; authorization: RecordAuthorization; selection: RecordSelection; findings: RecordDisclosureFinding[]; fulfillment: RecordFulfillment | null; delivery: RecordDelivery | null; evidence: EvidenceItem[]; checkpoint: RecordsCheckpoint };
export type ReleaseAuthorization = { requesterId: string; patientId: string; recipientId: string; permittedCategories: string[]; dateWindow: string; expiresAt: string; current: boolean; blockers: string[]; evidenceIds: string[]; };
export type DocumentReview = { documentId: string; responsiveEpisodeIds: string[]; includedSpans: string[]; withheldSpans: string[]; thirdPartyReviewRefs: string[]; status: 'clear'|'targeted-review'|'withheld'|'out-of-scope'; };
export type RenderedPacket = { artifactId: string; contentHash: string; documentIds: string[]; metadataScanRef: string; hiddenMetadataFindings: string[]; reviewerApprovalRef: string; authorizationVersion: string; };
export type AmendmentObligation = { correctedRecordId: string; priorDeliveryId: string; reviewRequired: boolean; recipientNotificationRequired: boolean; notificationReceipt: string; blockers: string[]; };
export type RecordsCheckpoint = { revision: string; authorization: ReleaseAuthorization; episodeSearchRefs: string[]; documentReviews: DocumentReview[]; rendered: RenderedPacket | null; delivery: RecordDelivery | null; amendments: AmendmentObligation[]; };
export type FulfillmentProposal = { application: 'Health Record Request Fulfillment Workbench'; status: 'ready-for-review'|'blocked'; authorization: ReleaseAuthorization; selection: RecordSelection; documentReviews: DocumentReview[]; rendered: RenderedPacket | null; delivery: RecordDelivery | null; amendments: AmendmentObligation[]; blockers: string[]; actionBoundary: string; resumeState: RecordsCheckpoint; };
