export type EvidenceItem = { id: string; source: string; version: string; text: string; status: 'current' | 'superseded' | 'unknown'; authority: string; };
export type EvidenceSlice = { subject: string; candidateIds: string[]; supporting: EvidenceItem[]; conflicting: EvidenceItem[]; missing: string[]; searchedScope: string; };

export type WorkRecord = { id: string; kind: string; subject: string; status: string; evidenceIds: string[]; fields: Record<string, string>; };

export type RecordMeta = { id: string; caseId: string; sourceRefs: string[]; state: 'draft' | 'qualified' | 'blocked'; };
export type Decimal = string;
export type Ratio = { numerator: Decimal; denominator: Decimal; };
export type Quantity = { value: Decimal; unit: string; };
export type Money = { minorUnits: string; currency: string; scale: number; };
export type BrandProject = RecordMeta & WorkRecord & {
  clientId: string;
  creativeBrief: string;
  productFactIds: string[];
  audienceGuidanceId: string;
  creativeDirectorId: string;
};
export type BrandConstraint = RecordMeta & WorkRecord & {
  caseId: string;
  desiredAssociation: string;
  factualClaim: string;
  prohibitedClaim: string;
  channelConstraint: string;
};
export type CreativeDirection = RecordMeta & WorkRecord & {
  caseId: string;
  conceptName: string;
  organizingIdea: string;
  voicePrinciples: string[];
  tradeoff: string;
};
export type BrandSystem = RecordMeta & WorkRecord & {
  caseId: string;
  directionId: string;
  terminologyRules: string[];
  voiceRules: string[];
  exampleAssets: string[];
};
export type BrandAsset = RecordMeta & WorkRecord & {
  caseId: string;
  systemId: string;
  channel: string;
  audienceTask: string;
  content: string;
};
export type BrandReview = RecordMeta & WorkRecord & {
  caseId: string;
  assetIds: string[];
  inconsistency: string;
  unsupportedClaim: string;
  revisionProposal: string;
};
export type DraftBrandConstraint = { values: {
    id?: string;
    caseId?: string;
    sourceRefs?: string[];
    state?: 'draft' | 'qualified' | 'blocked';
    kind?: string;
    subject?: string;
    evidenceIds?: string[];
    fields?: Record<string, string>;
    status?: string;
    desiredAssociation?: string;
    factualClaim?: string;
    prohibitedClaim?: string;
    channelConstraint?: string;
  }; missingFields: string[]; conflicts: string[]; evidenceIds: string[]; };

export type ProductBrief={id:string;product:string;capabilities:string[];prohibitions:string[];audiences:string[];goals:string[];channels:string[];};
export type ClaimEvidence={claimId:string;text:string;sourceId:string;revision:string;status:'supported'|'disputed'|'withdrawn';};
export type BrandDirection={id:string;name:string;positioning:string;emphasis:string[];exclusions:string[];messageLogic:string;claimIds:string[];rightsQuestions:string[];};
export type VoiceRule={context:string;principle:string;allowedExamples:string[];disallowedExamples:string[];disclosure:string[];};
export type BrandSystemState={id:string;revision:string;directionId:string;rules:VoiceRule[];approvedByClient:boolean;approvalScope:string[];};
export type BrandAssetDraft={id:string;channel:string;task:string;copy:string;claimIds:string[];systemRevision:string;requiredDisclosures:string[];};
export type BrandDefect={assetId:string;kind:'unsupported-claim'|'rights'|'voice'|'disclosure'|'inconsistent-term'|'approval-scope';state:'blocked'|'review';sourceIds:string[];repair:string;};
export type BrandCheckpoint={productRevision:string;systemRevision:string;assetRevisions:string[];invalidatedAssetIds:string[];approvedScopes:string[];};
export type BrandCoherenceAudit={defects:BrandDefect[];clearanceTasks:string[];checkpoint:BrandCheckpoint;status:'blocked'|'review'|'ready';};
export type BrandReviewPacket={directions:BrandDirection[];system:BrandSystemState;assets:BrandAssetDraft[];defects:BrandDefect[];clearanceTasks:string[];checkpoint:BrandCheckpoint;status:'blocked'|'review'|'ready';};
