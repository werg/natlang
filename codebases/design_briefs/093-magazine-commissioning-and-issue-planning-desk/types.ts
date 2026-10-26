export type EvidenceItem = { id: string; source: string; version: string; text: string; status: 'current' | 'superseded' | 'unknown'; authority: string; };
export type EvidenceSlice = { subject: string; candidateIds: string[]; supporting: EvidenceItem[]; conflicting: EvidenceItem[]; missing: string[]; searchedScope: string; };

export type WorkRecord = { id: string; kind: string; subject: string; status: string; evidenceIds: string[]; fields: Record<string, string>; };

export type RecordMeta = { id: string; caseId: string; sourceRefs: string[]; state: 'draft' | 'qualified' | 'blocked'; };
export type Decimal = string;
export type Ratio = { numerator: Decimal; denominator: Decimal; };
export type Quantity = { value: Decimal; unit: string; };
export type Money = { minorUnits: string; currency: string; scale: number; };
export type MagazineIssue = RecordMeta & WorkRecord & {
  issueTheme: string;
  audience: string;
  sectionIds: string[];
  editorId: string;
  publicationDate: string;
};
export type EditorialPitch = RecordMeta & WorkRecord & {
  caseId: string;
  contributorId: string;
  proposedAngle: string;
  evidencePlan: string;
  audienceValue: string;
};
export type IssueCoverage = RecordMeta & WorkRecord & {
  caseId: string;
  sectionId: string;
  coveredAngles: string[];
  missingPerspectives: string[];
  duplicatePitchIds: string[];
};
export type CommissionBrief = RecordMeta & WorkRecord & {
  caseId: string;
  pitchId: string;
  scope: string;
  deliverables: string[];
  evidenceRequirements: string[];
  dueAt: string;
};
export type EditorialTask = RecordMeta & WorkRecord & {
  caseId: string;
  pieceId: string;
  revisionNeed: string;
  factCheckIds: string[];
  productionDependencies: string[];
};
export type IssueReadiness = RecordMeta & WorkRecord & {
  caseId: string;
  pieceIds: string[];
  layoutArtifactIds: string[];
  unresolvedTasks: string[];
  approvalRefs: string[];
};
export type DraftEditorialPitch = { values: {
    id?: string;
    caseId?: string;
    sourceRefs?: string[];
    state?: 'draft' | 'qualified' | 'blocked';
    kind?: string;
    subject?: string;
    evidenceIds?: string[];
    fields?: Record<string, string>;
    status?: string;
    contributorId?: string;
    proposedAngle?: string;
    evidencePlan?: string;
    audienceValue?: string;
  }; missingFields: string[]; conflicts: string[]; evidenceIds: string[]; };

export type IssueBrief={id:string;theme:string;audience:string;mustCover:string[];exclusions:string[];publicationDate:string;formats:string[];};
export type StoryPitch={id:string;contributor:string;angle:string;reportingPlan:string[];sourceAccess:string[];format:string;estimatedWords:number;rightsTerms:string[];deadline:string;};
export type PitchAssessment={pitchId:string;audienceValue:string;distinctiveContribution:string;feasibility:'viable'|'conditional'|'unproven';overlaps:string[];coverageTags:string[];riskDependencies:string[];};
export type Commission={pitchId:string;angle:string;scope:string;wordCount:number;reportingDuties:string[];rights:string[];deadline:string;dependencies:string[];acceptance:{contributor:string;accepted:boolean;termsRevision:string;};state:'offered'|'accepted'|'declined'|'withdrawn';};
export type EditorialAsset={id:string;kind:'photo'|'illustration'|'headline'|'caption'|'promotion';text:string;credit:string;rights:string[];revision:string;pieceId:string;};
export type PieceStatus={pieceId:string;draftRevision:string;reportingComplete:boolean;factCheckRefs:string[];editorialTasks:string[];assets:EditorialAsset[];state:'pitched'|'commissioned'|'reporting'|'editing'|'ready'|'withdrawn';};
export type IssueState={id:string;revision:number;brief:IssueBrief;pitches:StoryPitch[];assessments:PitchAssessment[];commissions:Commission[];pieces:PieceStatus[];contingencies:StoryPitch[];layoutPieceIds:string[];creditsRevision:string;};
export type IssueFinding={kind:'gap'|'overlap'|'terms'|'access'|'reporting'|'fact'|'rights'|'headline'|'withdrawal'|'credit';pieceIds:string[];status:'clear'|'blocked'|'needs-review';reason:string;dependencies:string[];};
export type EditorialCheckpoint={issueId:string;revision:number;acceptedTerms:string[];checkedPieceRevisions:string[];invalidatedAssets:string[];freezeDigest:string;};
export type IssueReadinessPacket={issue:IssueState;findings:IssueFinding[];coverageGaps:string[];withdrawalPlan:string[];invalidatedAssets:string[];checkpoint:EditorialCheckpoint;status:'blocked'|'review'|'ready-to-freeze';};

export type MagazineSourceBinding = { brief: IssueBrief; issue: IssueState; editorId: string; sectionIds: string[]; sourceEvidenceIds: string[]; blockers: string[] };
