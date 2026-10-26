export type EvidenceItem = { id: string; source: string; version: string; text: string; status: 'current' | 'superseded' | 'unknown'; authority: string; };
export type EvidenceSlice = { subject: string; candidateIds: string[]; supporting: EvidenceItem[]; conflicting: EvidenceItem[]; missing: string[]; searchedScope: string; };

export type WorkRecord = { id: string; kind: string; subject: string; workflowStatus: string; evidenceIds: string[]; fields: Record<string, string>; };

export type RecordMeta = { id: string; caseId: string; sourceRefs: string[]; state: 'draft' | 'qualified' | 'blocked'; };
export type Decimal = string;
export type Ratio = { numerator: Decimal; denominator: Decimal; };
export type Quantity = { value: Decimal; unit: string; };
export type Money = { minorUnits: string; currency: string; scale: number; };
export type ReferralCase = RecordMeta & WorkRecord & {
  patientId: string;
  referringOrganizationId: string;
  receivingServiceId: string;
  coordinatorId: string;
  clinicianPriorityRef: string;
};
export type ReferralRequirement = RecordMeta & WorkRecord & {
  caseId: string;
  serviceId: string;
  requiredDocument: string;
  validityRule: string;
  acceptingRole: string;
};
export type ReferralRecordMatch = RecordMeta & WorkRecord & {
  caseId: string;
  patientId: string;
  documentId: string;
  identityEvidence: string;
  mismatch: string;
};
export type ReferralGap = RecordMeta & WorkRecord & {
  caseId: string;
  requirementId: string;
  documentIds: string[];
  status: 'met'|'missing'|'expired'|'uncertain';
  missingDetail: string;
};
export type ReferralRequest = RecordMeta & WorkRecord & {
  caseId: string;
  gapIds: string[];
  recipientId: string;
  requestedMaterial: string;
  purpose: string;
};
export type ReferralPacket = RecordMeta & WorkRecord & {
  caseId: string;
  documentIds: string[];
  approvalRef: string;
  deliveryReceipt: string;
  receivingStatus: string;
};
export type DraftReferralRequirement = { values: {
    id?: string;
    caseId?: string;
    sourceRefs?: string[];
    state?: 'draft' | 'qualified' | 'blocked';
    kind?: string;
    subject?: string;
    evidenceIds?: string[];
    fields?: Record<string, string>;
    workflowStatus?: string;
    serviceId?: string;
    requiredDocument?: string;
    validityRule?: string;
    acceptingRole?: string;
  }; missingFields: string[]; conflicts: string[]; evidenceIds: string[]; };

export type ReferralCoordinator = { id: string };
export type PacketChange = { checklistUpdates: EvidenceItem[]; identityEvidence: EvidenceItem[]; documents: EvidenceItem[]; clinicianUpdates: EvidenceItem[]; recipientReceipts: EvidenceItem[] };
export type ReferralDossier = { referral: ReferralCase; requirements: ReferralRequirement[]; matches: ReferralRecordMatch[]; gaps: ReferralGap[]; requests: ReferralRequest[]; packets: ReferralPacket[]; evidence: EvidenceItem[]; checkpoint: ReferralCheckpoint };
export type IdentityBinding = { patientId: string; documentIds: string[]; verifiedRefs: string[]; mismatches: string[]; status: 'verified'|'uncertain'|'mismatch' };
export type ChecklistDelta = { version: string; addedRequirementIds: string[]; changedRequirementIds: string[]; unchangedRequirementIds: string[]; missingAuthority: string[] };
export type DocumentFinding = { documentId: string; requirementId: string; state: 'present-valid'|'unsigned'|'expired'|'wrong-episode'|'identity-uncertain'|'missing'; evidenceIds: string[]; }
export type ReferralTransferEffect = { kind: 'send-referral-packet'|'read-recipient-status'; effectId: string; packetId: string; packetRevision: string; patientId: string; episodeScope: string; recipientServiceId: string; documentIds: string[]; authorizationRef: string; };
export type TransferLedger = { packetId: string; state: 'prepared'|'authorized'|'dispatched'|'acknowledged'|'unknown'; receiptRefs: string[]; recipientReadback: string[]; resendAllowed: boolean; blockers: string[]; effects?: ReferralTransferEffect[]; };
export type ReferralCheckpoint = { revision: string; referral: ReferralCase; identity: IdentityBinding; checklist: ChecklistDelta; documents: DocumentFinding[]; priorityRef: string; episodeScope: string; requests: ReferralRequest[]; transfer: TransferLedger; disclosureRefs: string[]; };
export type ReferralProposal = { application: 'Referral Packet Completeness Manager'; status: 'ready-for-review'|'blocked'; identity: IdentityBinding; checklist: ChecklistDelta; documents: DocumentFinding[]; gaps: ReferralGap[]; requests: ReferralRequest[]; clinicianPriorityRef: string; transfer: TransferLedger; effects: ReferralTransferEffect[]; blockers: string[]; actionBoundary: string; resumeState: ReferralCheckpoint; };
