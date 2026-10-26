export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };

export type PreparedEvidence = { subject: string; evidence: Evidence[]; missing: string[]; conflicts: string[]; alternatives: string[] };
export type DecisionAtom = { field: string; value: string; evidenceIds: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable"; alternative: string };
export type Qualification = "pass" | "fail" | "needs-review";

export type FieldConflict = { field: string; alternatives: string[]; evidenceIds: string[] };
export type EvidenceBinding = { field: string; evidenceIds: string[]; sourceVersions: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable" };

export type Question = { id: string; question: string; respondent: string; acceptableEvidence: string[] };

export type RecordMeta = { id: ID; version: string; createdAt: DateTime; updatedAt: DateTime };
export type ReleaseReview = RecordMeta & {
  releaseId: ID;
  baselineRevision: string;
  candidateRevision: string;
  integrationIds: ID[];
  releaseOwnerId: ID;
};
export type BehavioralChange = RecordMeta & {
  caseId: ID;
  componentId: ID;
  oldBehavior: string;
  newBehavior: string;
  affectedContract: string;
};
export type ConsumerDependency = RecordMeta & {
  caseId: ID;
  consumerId: ID;
  reliedOnBehavior: string;
  evidenceRefs: Ref[];
  exposure: string;
};
export type CompatibilityTest = RecordMeta & {
  caseId: ID;
  consumerId: ID;
  scenario: string;
  baselineResult: string;
  candidateResult: string;
};
export type MigrationWave = RecordMeta & {
  caseId: ID;
  consumerIds: ID[];
  prerequisiteIds: ID[];
  fallbackPlan: string;
  acceptanceCriteria: string[];
};
export type ReleaseReadiness = RecordMeta & {
  caseId: ID;
  waveIds: ID[];
  documentationIds: ID[];
  ownerApprovals: Ref[];
  unresolvedRisk: string;
};
export type DraftBehavioralChange = { values: { caseId?: ID; componentId?: ID; oldBehavior?: string; newBehavior?: string; affectedContract?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };

export type CompatibilityReleaseRequest = { releaseId: ID; baselineRevision: string; candidateRevision: string; changeEvidence: Evidence[]; consumerEvidence: Evidence[]; policyClauses: string[]; authorityStatus: 'authorized' | 'unclear' | 'unauthorized'; changedRequirementIds: ID[]; existingBehavior: BehavioralChange; pinnedVersions: string[]; unownedIntegrationIds: ID[]; rollbackDataPlan: DataRollbackPlan };
export type ReleaseBehaviorDelta = { changeId: ID; behavior: string; baseline: string; candidate: string; defaultChanged: boolean; orderingChanged: boolean; retryChanged: boolean; errorChanged: boolean; evidenceRefs: ID[]; verificationNeeded: string[] };
export type BehaviorConsumerMap = { consumerId: ID; behaviorIds: ID[]; owner: string; pinnedVersion: string; confidence: 'supported' | 'uncertain' | 'unowned'; evidenceRefs: ID[] };
export type PairedCompatibilityReplay = { consumerId: ID; input: string; baselineOutcome: string; candidateOutcome: string; acceptanceCriterion: string; disposition: 'compatible' | 'breaking' | 'unresolved'; environmentRefs: ID[] };
export type CoordinatedChangeWave = { waveId: ID; consumerIds: ID[]; prerequisites: string[]; owner: string; entryEvidence: string[]; rollbackTrigger: string };
export type MigrationWavePlan = { waves: CoordinatedChangeWave[]; blockedConsumers: ID[]; supportedOldVersionWindow: string[] };
export type DataRollbackPlan = { candidateWrites: string[]; baselineReadableFields: string[]; reconciliationSteps: string[]; irreversibleLosses: string[]; rehearsalRefs: ID[] };
export type WaveRollbackQualification = { qualified: boolean; missingEvidence: string[]; dataLossRisk: string[] };
export type ReleaseCommunications = { candidateOnlyDocs: string[]; supportNotices: string[]; audienceVersionBindings: string[]; mismatchedRevisionRefs: string[] };
export type ReleaseReadinessDecision = { status: 'blocked' | 'review' | 'ready'; blockers: string[]; unresolvedContracts: string[]; requiredApprovals: string[] };
export type ReleaseTransitionProposal = { releaseId: ID; changes: ReleaseBehaviorDelta[]; consumers: BehaviorConsumerMap[]; replay: PairedCompatibilityReplay[]; waves: MigrationWavePlan; rollback: DataRollbackPlan; communications: ReleaseCommunications; readiness: ReleaseReadinessDecision; commitAllowed: boolean };
export type ReleaseWaveReceipt = { receiptId: ID; releaseId: ID; waveId: ID; baselineRevision: string; candidateRevision: string; outcome: "completed" | "partial" | "rolled-back" | "unknown"; acceptedConsumerIds: ID[]; replayEvidence: { consumerId: ID; input: string; baselineOutcome: string; candidateOutcome: string; evidenceRefs: ID[] }[]; candidateWriteFields: string[]; baselineReadableFields: string[]; baselineReadbackEvidence: ID[]; issueIds: ID[]; evidenceRefs: ID[] };
export type ReleaseWaveState = { waveId: ID; status: "pending" | "in-progress" | "completed" | "held" | "rolled-back" | "unknown"; acceptedConsumerIds: ID[]; unresolvedConsumerIds: ID[]; issues: ID[] };
export type ReleaseRolloutBook = { revision: number; activeRevision: string | null; candidateRevision: string | null; state: "not-started" | "in-progress" | "active" | "rolled-back" | "unknown"; waves: ReleaseWaveState[]; receipts: ReleaseWaveReceipt[]; consumedReceiptIds: ID[] };
export type ReleaseWaveAssessment = { advanceAllowed: boolean; allConsumersAccepted: boolean; rollbackSafe: boolean; reasons: string[]; unresolvedConsumerIds: ID[] };
export type ReleaseRolloutReview = { book: ReleaseRolloutBook; outcome: "recorded" | "replayed" | "held"; releaseState: "not-started" | "in-progress" | "active" | "rolled-back" | "unknown"; followUp: string[] };
