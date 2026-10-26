export type EvidenceItem = { id: string; source: string; version: string; text: string; status: 'current' | 'superseded' | 'unknown'; authority: string; };
export type EvidenceSlice = { subject: string; candidateIds: string[]; supporting: EvidenceItem[]; conflicting: EvidenceItem[]; missing: string[]; searchedScope: string; };

export type WorkRecord = { id: string; kind: string; subject: string; workflowStatus: string; evidenceIds: string[]; fields: Record<string, string>; };

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

export type RightsSteward = { id: string };
export type RightsChange = { permissions: EvidenceItem[]; useChanges: EvidenceItem[]; expiryUpdates: EvidenceItem[]; attributionCorrections: EvidenceItem[]; assetInventory: EvidenceItem[] };
export type RightsDossier = { collectionUse: CollectionUse; permissions: CollectionPermission[]; proposedUses: ProposedCollectionUse[]; findings: RightsFinding[]; clearanceTasks: ClearanceTask[]; useRecords: CollectionUseRecord[]; evidence: EvidenceItem[]; checkpoint: RightsCheckpoint };

export type UseCompatibility = { proposedUseId: string; itemId: string; compatible: boolean; missingRights: string[]; expiredRights: string[]; incompatibleDimensions: string[]; evidenceIds: string[]; }
export type RightsDerivative = { assetId: string; sourceItemIds: string[]; channel: string; attributionVersion: string; permissionIds: string[]; state: 'cleared'|'blocked'|'remove'|'review'; }
export type RightsCheckpoint = { revision: string; permissions: CollectionPermission[]; uses: ProposedCollectionUse[]; compatibility: UseCompatibility[]; clearanceTasks: ClearanceTask[]; derivatives: RightsDerivative[]; expiryActions: string[]; attributionUpdates: string[]; };
export type RightsProposal = { application: 'Collection Rights and Consent Steward'; status: 'ready-for-review'|'blocked'; permissions: CollectionPermission[]; compatibility: UseCompatibility[]; clearances: ClearanceTask[]; register: CollectionUseRecord; derivatives: RightsDerivative[]; expiryActions: string[]; blockers: string[]; actionBoundary: string; resumeState: RightsCheckpoint; };
