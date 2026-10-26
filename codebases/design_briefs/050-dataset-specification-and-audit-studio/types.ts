export type ID = string;
export type Decimal = string;
export type Ratio = { numerator: string; denominator: string };
export type Quantity = { value: Decimal; unit: string };
export type Money = { minorUnits: string; currency: string; scale: number };
export type DateTime = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };
export type RecordMeta = { id: ID; version: string; createdAt: DateTime; updatedAt: DateTime };
export type DatasetProject = RecordMeta & {
  intendedUse: string;
  targetPopulation: string;
  sourceCollectionIds: ID[];
  ownerId: ID;
  releaseAudience: string;
};
export type DatasetUse = RecordMeta & {
  caseId: ID;
  task: string;
  targetPopulation: string;
  prohibitedUses: string[];
  requiredSlices: string[];
};
export type AnnotationRule = RecordMeta & {
  caseId: ID;
  label: string;
  inclusionRule: string;
  exclusionRule: string;
  ambiguityPolicy: string;
};
export type DatasetRecordAudit = RecordMeta & {
  caseId: ID;
  recordId: ID;
  eligibility: string;
  qualityIssues: string[];
  rightsStatus: string;
};
export type CoverageFinding = RecordMeta & {
  caseId: ID;
  slice: string;
  eligibleCount: number;
  targetCount: number;
  gapDescription: string;
};
export type DatasetRelease = RecordMeta & {
  caseId: ID;
  acquisitionTaskIds: ID[];
  annotationVersion: string;
  artifactIds: ID[];
  unresolvedLimits: string[];
};

export type DatasetAuditSession = { projectId: string; instruction: string; proposedUse: DatasetUse; sourceEvents: Evidence[]; suppliedRules: string[]; authority: "authorized" | "unclear" | "unauthorized"; changedSourceIds: string[]; reviewer: string; candidates?: DatasetCandidate[] };
export type DatasetCandidate = { id: string; sourceId: string; worldId: string | null; content: string; consentIds: string[]; licenseEvidenceIds: string[]; proposedSlices: string[]; proposedSplit: "train" | "dev" | "test" | "unassigned" };
export type ObservableLabel = { label: string; observableEvidence: string[]; excludedInferences: string[]; ambiguityDisposition: string; adjudicationRule: string; blockers: string[] };
export type SourceFamily = { id: string; recordIds: string[]; sourceIds: string[]; worldIds: string[]; independenceBasis: string; disputedLinks: string[]; split: "train" | "dev" | "test" | "unassigned" };
export type RecordEligibility = { recordId: string; familyId: string; permittedUse: string; permittedAudience: string; eligible: boolean; quarantineReasons: string[]; permissionEvidenceIds: string[]; slices: string[] };
export type SliceAccounting = { slice: string; recordIds: string[]; independentFamilyIds: string[]; recordCount: number; independentCount: number; missingMechanisms: string[]; representativenessLimits: string[] };
export type FrozenDatasetEdition = { editionId: string; useId: string; ruleVersion: string; admittedRecordIds: string[]; familyAssignments: SourceFamily[]; permissionEvidenceIds: string[]; artifactIds: string[]; ownerApprovalId: string | null; downstreamRecipientIds: string[]; released: boolean };
export type ConsentRemoval = { withdrawalSourceId: string; subjectSourceIds: string[]; descendantArtifactIds: string[]; recipientIds: string[]; unresolvedLineage: string[]; removalReceiptIds: string[]; notificationReceiptIds: string[] };
export type DatasetEditionState = { revision: number; use: DatasetUse | null; labels: ObservableLabel[]; families: SourceFamily[]; eligibility: RecordEligibility[]; accounting: SliceAccounting[]; editions: FrozenDatasetEdition[]; removals: ConsentRemoval[] };
export type DatasetEditionReview = { edition: DatasetEditionState; outcome: "revise-task" | "acquire-evidence" | "review-release" | "withdrawal-open"; proposedRelease: FrozenDatasetEdition | null; acquisitionBriefs: string[]; ownerQuestions: string[]; limitations: string[] };
