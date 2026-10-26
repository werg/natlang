export type EvidenceItem = { id: string; source: string; version: string; text: string; status: 'current' | 'superseded' | 'unknown'; authority: string; };
export type EvidenceSlice = { subject: string; candidateIds: string[]; supporting: EvidenceItem[]; conflicting: EvidenceItem[]; missing: string[]; searchedScope: string; };

export type WorkRecord = { id: string; kind: string; subject: string; status: string; evidenceIds: string[]; fields: Record<string, string>; };

export type RecordMeta = { id: string; caseId: string; sourceRefs: string[]; state: 'draft' | 'qualified' | 'blocked'; };
export type Decimal = string;
export type Ratio = { numerator: Decimal; denominator: Decimal; };
export type Quantity = { value: Decimal; unit: string; };
export type Money = { minorUnits: string; currency: string; scale: number; };
export type CommunityDispute = RecordMeta & WorkRecord & {
  participantIds: string[];
  sharedResourceIds: string[];
  issue: string;
  facilitatorId: string;
  agreementHistoryIds: string[];
};
export type ExistingAgreement = RecordMeta & WorkRecord & {
  caseId: string;
  participantIds: string[];
  acceptedTerms: string[];
  effectiveWindow: string;
  disputedTerm: string;
};
export type MediationPosition = RecordMeta & WorkRecord & {
  caseId: string;
  participantId: string;
  desiredOutcome: string;
  underlyingConcern: string;
  statedLimit: string;
};
export type CommunityOption = RecordMeta & WorkRecord & {
  caseId: string;
  positionIds: string[];
  proposedArrangement: string;
  concessions: string[];
  unresolvedConflict: string;
};
export type AgreementTrial = RecordMeta & WorkRecord & {
  caseId: string;
  optionId: string;
  duration: string;
  successCriteria: string[];
  exitConditions: string[];
};
export type AgreementReview = RecordMeta & WorkRecord & {
  caseId: string;
  trialId: string;
  participantResponses: string[];
  observedOutcomes: string[];
  proposedRevision: string;
};
export type DraftExistingAgreement = { values: {
    id?: string;
    caseId?: string;
    sourceRefs?: string[];
    state?: 'draft' | 'qualified' | 'blocked';
    kind?: string;
    subject?: string;
    evidenceIds?: string[];
    fields?: Record<string, string>;
    status?: string;
    participantIds?: string[];
    acceptedTerms?: string[];
    effectiveWindow?: string;
    disputedTerm?: string;
  }; missingFields: string[]; conflicts: string[]; evidenceIds: string[]; };

export type Mediator = { id: string };
export type MediationChange = { participantResponses: EvidenceItem[]; disputedAccounts: EvidenceItem[]; trialObservations: EvidenceItem[]; withdrawals: EvidenceItem[]; safetyOrRightsFlags: EvidenceItem[] };
export type MediationDossier = { dispute: CommunityDispute; priorAgreements: ExistingAgreement[]; positions: MediationPosition[]; options: CommunityOption[]; trials: AgreementTrial[]; reviews: AgreementReview[]; evidence: EvidenceItem[]; checkpoint: MediationCheckpoint };
export type ParticipantTermState = { participantId: string; termId: string; state: 'accepted'|'declined'|'pending'|'withdrawn'; evidenceIds: string[] };
export type TrialTerms = { id: string; participants: string[]; terms: string[]; start: string; end: string; reviewTrigger: string; exitProcess: string; acceptance: ParticipantTermState[]; };
export type DivergentTrialFinding = { participantId: string; outcome: string; sourceIds: string[]; unresolved: string[]; };
export type MediationCheckpoint = { revision: string; priorAgreements: ExistingAgreement[]; competingAccounts: EvidenceItem[]; participantTerms: ParticipantTermState[]; options: CommunityOption[]; trialTerms: TrialTerms[]; findings: DivergentTrialFinding[]; exitsCompleted: string[]; unresolved: string[]; };
export type MediationProposal = { application: 'Community Agreement Mediator'; status: 'ready-for-review'|'blocked'; positions: MediationPosition[]; options: CommunityOption[]; trial: TrialTerms | null; participantTerms: ParticipantTermState[]; divergentFindings: DivergentTrialFinding[]; referrals: string[]; exitActions: string[]; blockers: string[]; actionBoundary: string; resumeState: MediationCheckpoint; };
