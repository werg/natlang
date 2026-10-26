export type EvidenceItem = { id: string; source: string; version: string; text: string; status: 'current' | 'superseded' | 'unknown'; authority: string; };
export type EvidenceSlice = { subject: string; candidateIds: string[]; supporting: EvidenceItem[]; conflicting: EvidenceItem[]; missing: string[]; searchedScope: string; };
export type DecisionAtom = { field: string; value: string; status: 'established' | 'disputed' | 'unknown' | 'inapplicable'; evidenceIds: string[]; rationale: string; alternative: string; };
export type StageDraft = { stage: string; subject: string; fields: Record<string, string>; decisions: DecisionAtom[]; evidenceIds: string[]; unresolved: string[]; conflicts: string[]; };
export type StageAudit = { verdict: 'pass' | 'fail' | 'needs-review'; gate: string; findings: string[]; repairTargets: string[]; };
export type WorkRecord = { id: string; kind: string; subject: string; workflowStatus: string; evidenceIds: string[]; fields: Record<string, string>; };
export type StageOutputs = { extractPermissions: StageResultCollectionPermission; interpretUse: StageResultProposedCollectionUse; assessCompatibility: StageResultRightsFinding; requestClearance: StageResultClearanceTask; maintainUseRegister: StageResultCollectionUseRecord };
export type RunProposal = { application: string; goal: string; status: 'ready-for-review' | 'blocked'; stageOutputs: StageOutputs; view: ViewModel; questions: string[]; coverage: string[]; checks: string[]; actionBoundary: string; };
export type QualifyInput = { evidence: EvidenceSlice; draft: StageDraft; };
export type RecordMeta = { id: string; caseId: string; sourceRefs: string[]; state: 'draft' | 'qualified' | 'blocked'; };
export type Decimal = string;
export type Ratio = { numerator: Decimal; denominator: Decimal; };
export type Quantity = { value: Decimal; unit: string; };
export type Money = { minorUnits: string; currency: string; scale: number; };
export type CollectionUse = RecordMeta & WorkRecord & {
  collectionItemIds: string[];
  proposedUse: string;
  audience: string;
  requesterId: string;
  rightsOfficerId: string;
};
export type CollectionPermission = RecordMeta & WorkRecord & {
  caseId: string;
  itemId: string;
  permittedUse: string;
  prohibitedUse: string;
  attributionRule: string;
  expiry: string;
};
export type ProposedCollectionUse = RecordMeta & WorkRecord & {
  caseId: string;
  itemIds: string[];
  channel: string;
  transformation: string;
  audience: string;
  duration: string;
};
export type RightsFinding = RecordMeta & WorkRecord & {
  caseId: string;
  useId: string;
  permissionIds: string[];
  status: 'allowed'|'restricted'|'unknown';
  unmetCondition: string;
};
export type ClearanceTask = RecordMeta & WorkRecord & {
  caseId: string;
  findingId: string;
  rightsholderId: string;
  requestedPermission: string;
  requiredEvidence: string;
};
export type CollectionUseRecord = RecordMeta & WorkRecord & {
  caseId: string;
  useId: string;
  approvalRefs: string[];
  attributionText: string;
  expiryTasks: string[];
};
export type DraftCollectionPermission = { values: {
    id?: string;
    caseId?: string;
    sourceRefs?: string[];
    state?: 'draft' | 'qualified' | 'blocked';
    kind?: string;
    subject?: string;
    evidenceIds?: string[];
    fields?: Record<string, string>;
    workflowStatus?: string;
    itemId?: string;
    permittedUse?: string;
    prohibitedUse?: string;
    attributionRule?: string;
    expiry?: string;
  }; missingFields: string[]; conflicts: string[]; evidenceIds: string[]; };
export type StageResultCollectionPermission = { accepted: CollectionPermission[]; unresolved: DraftCollectionPermission[]; rejected: string[]; consumedEvidence: string[]; nextQuestions: string[]; gate: StageAudit[]; };
export type DraftProposedCollectionUse = { values: {
    id?: string;
    caseId?: string;
    sourceRefs?: string[];
    state?: 'draft' | 'qualified' | 'blocked';
    kind?: string;
    subject?: string;
    evidenceIds?: string[];
    fields?: Record<string, string>;
    workflowStatus?: string;
    itemIds?: string[];
    channel?: string;
    transformation?: string;
    audience?: string;
    duration?: string;
  }; missingFields: string[]; conflicts: string[]; evidenceIds: string[]; };
export type StageResultProposedCollectionUse = { accepted: ProposedCollectionUse[]; unresolved: DraftProposedCollectionUse[]; rejected: string[]; consumedEvidence: string[]; nextQuestions: string[]; gate: StageAudit[]; };
export type DraftRightsFinding = { values: {
    id?: string;
    caseId?: string;
    sourceRefs?: string[];
    state?: 'draft' | 'qualified' | 'blocked';
    kind?: string;
    subject?: string;
    evidenceIds?: string[];
    fields?: Record<string, string>;
    useId?: string;
    permissionIds?: string[];
    status?: 'allowed'|'restricted'|'unknown';
    unmetCondition?: string;
  }; missingFields: string[]; conflicts: string[]; evidenceIds: string[]; };
export type StageResultRightsFinding = { accepted: RightsFinding[]; unresolved: DraftRightsFinding[]; rejected: string[]; consumedEvidence: string[]; nextQuestions: string[]; gate: StageAudit[]; };
export type DraftClearanceTask = { values: {
    id?: string;
    caseId?: string;
    sourceRefs?: string[];
    state?: 'draft' | 'qualified' | 'blocked';
    kind?: string;
    subject?: string;
    evidenceIds?: string[];
    fields?: Record<string, string>;
    workflowStatus?: string;
    findingId?: string;
    rightsholderId?: string;
    requestedPermission?: string;
    requiredEvidence?: string;
  }; missingFields: string[]; conflicts: string[]; evidenceIds: string[]; };
export type StageResultClearanceTask = { accepted: ClearanceTask[]; unresolved: DraftClearanceTask[]; rejected: string[]; consumedEvidence: string[]; nextQuestions: string[]; gate: StageAudit[]; };
export type DraftCollectionUseRecord = { values: {
    id?: string;
    caseId?: string;
    sourceRefs?: string[];
    state?: 'draft' | 'qualified' | 'blocked';
    kind?: string;
    subject?: string;
    evidenceIds?: string[];
    fields?: Record<string, string>;
    workflowStatus?: string;
    useId?: string;
    approvalRefs?: string[];
    attributionText?: string;
    expiryTasks?: string[];
  }; missingFields: string[]; conflicts: string[]; evidenceIds: string[]; };
export type StageResultCollectionUseRecord = { accepted: CollectionUseRecord[]; unresolved: DraftCollectionUseRecord[]; rejected: string[]; consumedEvidence: string[]; nextQuestions: string[]; gate: StageAudit[]; };
export type CollectionFinding = { conflicts: string[]; uncovered: string[]; duplicateIds: string[]; disposition: string; };
export type GoalContract = { outcome: string; scope: string[]; exclusions: string[]; acceptanceEvidence: string[]; ambiguities: string[]; };
export type PolicySelection = { clauses: string[]; precedence: string; unresolvedAuthority: string[]; };
export type WorkGraph = { nodes: string[]; prerequisites: string[]; blockedQuestions: string[]; };
export type EvidencePlan = { required: string[]; searched: string[]; omittedPartitions: string[]; };
export type ReviewPacket = { stageOutputs: StageOutputs; blockers: string[]; approvalRequired: string[]; };
export type ViewModel = { title: string; summary: string; sections: string[]; unresolved: string[]; };
export type RevisionProposal = { target: string; change: string; expectedEffect: string; regressionRisks: string[]; };
export type InvalidationProposal = { changedRefs: string[]; affectedArtifacts: string[]; unaffectedArtifacts: string[]; unresolvedImpact: string[]; };

export type RightsSteward = { id: string };
export type RightsChange = { permissions: EvidenceItem[]; useChanges: EvidenceItem[]; expiryUpdates: EvidenceItem[]; attributionCorrections: EvidenceItem[]; assetInventory: EvidenceItem[] };
export type RightsDossier = { collectionUse: CollectionUse; permissions: CollectionPermission[]; proposedUses: ProposedCollectionUse[]; findings: RightsFinding[]; clearanceTasks: ClearanceTask[]; useRecords: CollectionUseRecord[]; evidence: EvidenceItem[]; checkpoint: RightsCheckpoint };
export type UseDimension = { itemId: string; rightType: string; permittedUses: string[]; prohibitedUses: string[]; channels: string[]; audiences: string[]; attributionRule: string; validThrough: string; sourceIds: string[]; }
export type UseCompatibility = { proposedUseId: string; itemId: string; compatible: boolean; missingRights: string[]; expiredRights: string[]; incompatibleDimensions: string[]; evidenceIds: string[]; }
export type RightsDerivative = { assetId: string; sourceItemIds: string[]; channel: string; attributionVersion: string; permissionIds: string[]; state: 'cleared'|'blocked'|'remove'|'review'; }
export type RightsCheckpoint = { revision: string; permissions: CollectionPermission[]; uses: ProposedCollectionUse[]; compatibility: UseCompatibility[]; clearanceTasks: ClearanceTask[]; derivatives: RightsDerivative[]; expiryActions: string[]; attributionUpdates: string[]; };
export type RightsProposal = { application: 'Collection Rights and Consent Steward'; status: 'ready-for-review'|'blocked'; permissions: CollectionPermission[]; compatibility: UseCompatibility[]; clearances: ClearanceTask[]; register: CollectionUseRecord; derivatives: RightsDerivative[]; expiryActions: string[]; blockers: string[]; actionBoundary: string; resumeState: RightsCheckpoint; };
