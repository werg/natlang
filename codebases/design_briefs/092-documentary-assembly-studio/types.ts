export type EvidenceItem = { id: string; source: string; version: string; text: string; status: 'current' | 'superseded' | 'unknown'; authority: string; };
export type EvidenceSlice = { subject: string; candidateIds: string[]; supporting: EvidenceItem[]; conflicting: EvidenceItem[]; missing: string[]; searchedScope: string; };

export type WorkRecord = { id: string; kind: string; subject: string; status: string; evidenceIds: string[]; fields: Record<string, string>; };

export type RecordMeta = { id: string; caseId: string; sourceRefs: string[]; state: 'draft' | 'qualified' | 'blocked'; };
export type Decimal = string;
export type Ratio = { numerator: Decimal; denominator: Decimal; };
export type Quantity = { value: Decimal; unit: string; };
export type Money = { minorUnits: string; currency: string; scale: number; };
export type DocumentaryProject = RecordMeta & WorkRecord & {
  interviewIds: string[];
  archiveAssetIds: string[];
  editorialBrief: string;
  editorId: string;
  targetDurationSeconds: number;
};
export type DocumentarySegment = RecordMeta & WorkRecord & {
  caseId: string;
  assetId: string;
  timeRange: string;
  speakerClaim: string;
  context: string;
  rightsStatus: string;
};
export type DocumentaryStructure = RecordMeta & WorkRecord & {
  caseId: string;
  centralQuestion: string;
  sectionClaims: string[];
  supportingSegmentIds: string[];
  counterevidenceIds: string[];
};
export type DocumentaryEdit = RecordMeta & WorkRecord & {
  caseId: string;
  segmentIds: string[];
  order: number[];
  transitionText: string;
  impliedMeaning: string;
};
export type DocumentaryClearance = RecordMeta & WorkRecord & {
  caseId: string;
  editIds: string[];
  factualIssues: string[];
  rightsIssues: string[];
  accessibilityTasks: string[];
};
export type DocumentaryRelease = RecordMeta & WorkRecord & {
  caseId: string;
  editDecisionListId: string;
  approvedArtifactIds: string[];
  durationSeconds: number;
  approvalRefs: string[];
};
export type DraftDocumentarySegment = { values: {
    id?: string;
    caseId?: string;
    sourceRefs?: string[];
    state?: 'draft' | 'qualified' | 'blocked';
    kind?: string;
    subject?: string;
    evidenceIds?: string[];
    fields?: Record<string, string>;
    status?: string;
    assetId?: string;
    timeRange?: string;
    speakerClaim?: string;
    context?: string;
    rightsStatus?: string;
  }; missingFields: string[]; conflicts: string[]; evidenceIds: string[]; };

export type MediaAsset={id:string;revision:string;durationFrames:number;frameRate:number;speaker:string;claim:string;qualifier:string;contextBefore:string;contextAfter:string;rights:string[];sourceDate:string;correctedClaim:string;};
export type Cut={assetId:string;inFrame:number;outFrame:number;};
export type CutTransition={fromAssetId:string;toAssetId:string;narration:string;argument:string;};
export type AssemblyState={id:string;revision:number;assets:MediaAsset[];cuts:Cut[];transitions:CutTransition[];targetFrames:number;approvedClaimIds:string[];captionsRevision:string;descriptionsRevision:string;};
export type CutFinding={assetId:string;kind:'timecode'|'qualifier'|'rights'|'fact'|'implication'|'duration'|'correction';status:'clear'|'blocked'|'needs-review';evidenceIds:string[];reason:string;repair:string;};
export type DocumentaryCheckpoint={assemblyId:string;revision:number;digest:string;checkedAssets:string[];invalidatedOutputs:string[];clearanceRefs:string[];receiptStatus:'none'|'sent'|'unknown'|'confirmed';deliveryReceiptId?:string;deliveryObservedAt?:string;};
export type DocumentaryDeliverable={kind:'edit-list'|'script'|'captions'|'description'|'asset-manifest';revision:string;assemblyDigest:string;};
export type DocumentaryDeliveryEffect={kind:'deliver-documentary-package'|'read-documentary-delivery-status';assemblyId:string;revision:number;digest:string;deliverables:DocumentaryDeliverable[];clearanceRefs:string[];idempotencyKey:string;};
export type DocumentaryDeliveryReadback={assemblyId:string;revision:number;digest:string;receiptId:string;status:'confirmed'|'absent'|'conflict';observedAt:string;};
export type AssemblyRelease={assembly:AssemblyState;findings:CutFinding[];deliverables:DocumentaryDeliverable[];clearanceTasks:string[];factCheckTasks:string[];checkpoint:DocumentaryCheckpoint;status:'blocked'|'review'|'frozen';effects:DocumentaryDeliveryEffect[];};
export type DocumentaryDeliveryPreparation={effects:DocumentaryDeliveryEffect[];checkpoint:DocumentaryCheckpoint;};
