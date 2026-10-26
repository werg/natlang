

export type ID = string; export type Hash = string; export type LocalDate = string; export type DateTime = string; export type Decimal = string; export type Ratio = { numerator: Decimal; denominator: Decimal }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number };
export type Ref = { tenantId: ID; objectId: ID; version: string; hash: Hash };
export type SpanRef = { source: Ref; locatorKind: 'text' | 'table' | 'timecode' | 'region'; locator: string; excerptHash: Hash };

export type FieldConflict = { field: string; alternatives: string[]; evidenceRefs: string[] };

export type ActionProposal = { action: string; target: string; requiresApproval: boolean };

export type Evidence = { id: string; source: string; version: string; observedAt: string; authority: string; scope: string; text: string };
export type EvidencePage = { evidence: Evidence[]; searchedPartition: string; complete: boolean };

export type GoalContract = { outcome: string; exclusions: string[]; entities: string[]; acceptanceEvidence: string[]; unresolvedScope: string[] };
export type PolicyCandidate = { id: string; authority: string; scope: string; effectiveFrom: LocalDate; effectiveTo: LocalDate; supersedes: string[]; clauses: string[] };
export type PolicySelection = { selected: string[]; precedenceQuestions: string[]; applicableScope: string };
export type Candidate = { id: string; subject: string; existingRecord: string; evidenceRefs: string[] };

export type GateVerdict = 'pass' | 'fail' | 'needs-review';
export type PolicySimulation = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; baselinePolicyId: string; candidatePolicyId: string; caseCollectionId: string; policyOwnerId: string; purpose: string };
export type PolicyDelta = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; clauseIds: string[]; changedCondition: string; changedConsequence: string; unchangedSafeguards: string[] };
export type PolicyCaseFacts = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; establishedFacts: string[]; unknownFacts: string[]; protectedAttributes: string[] };
export type OutcomeDifference = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; baselineOutcome: string; candidateOutcome: string; causalDeltaIds: string[] };
export type BoundaryScenario = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; deltaId: string; initialFacts: string[]; perturbedFact: string; expectedDistinction: string };
export type PolicyOption = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; deltaIds: string[]; revisedText: string; affectedCaseIds: string[]; unresolvedTradeoff: string };
export type DraftPolicyDelta = { value: { tenantId?: string; id?: string; version?: number; schemaVersion?: string; state?: string; createdAt?: string; updatedAt?: string; createdBy?: string; sourceRefs?: Ref[]; decisionRefs?: Ref[]; sensitivity?: string[]; purposeIds?: string[]; retentionPolicyId?: string; caseId?: string; clauseIds?: string[]; changedCondition?: string; changedConsequence?: string; unchangedSafeguards?: string[] }; missingFields: string[]; conflicts: FieldConflict[]; evidenceRefs: string[] };

export type PolicyScenarioRequest = { caseId: string; baselinePolicyId: string; candidatePolicyId: string; caseCollectionId: string; policyOwnerId: string; purpose: string; expectedVersion: string };
export type PolicyReplayResult = { policyId: string; caseId: string; inputDigest: string; outcome: string; duties: string[]; permissions: string[]; timing: string[]; escalations: string[]; engineVersion: string; executionId: string; evidenceRefs: string[] };
export type PolicyReplayPair = { caseId: string; baseline: PolicyReplayResult; candidate: PolicyReplayResult };
export type PolicySimulationDossier = { checkpoint?: PolicyScenarioCheckpoint; simulations: PolicySimulation[]; evidencePages: EvidencePage[]; policyCandidates: PolicyCandidate[]; searchedPartitions: string[]; cases: PolicyCaseFacts[]; replayPairs: PolicyReplayPair[]; ownerEdits: EvidencePage[]; dedupChanges: EvidencePage[]; adoptionReceipts: EvidencePage[] };
export type PolicyGuardrail = { disposition: string; blockedCaseIds: string[]; preservedCaseIds: string[]; evidenceRefs: string[]; proposedActions: ActionProposal[] };
export type PolicySimulationResult = { caseId: string; delta: PolicyDeltaSet; cohort: CohortReview; replayBinding: PolicyReplayBinding; outcomes: PolicyOutcomeSet | null; regressions: RegressionSuite | null; boundaries: BoundaryMatrix | null; adoption: PolicyGuardrail; proposal: PolicyOptionProposal | null; openQuestions: string[]; checkpoint: PolicyScenarioCheckpoint; commitAllowed: boolean };
export type PolicyReplayBinding = { verifiedPairs: PolicyReplayPair[]; blockedCaseIds: string[]; issues: string[]; evidenceRefs: string[] };
export type PolicyDeltaSet = { baselineHash: string; candidateHash: string; changedConditionIds: string[]; changedConsequences: string[]; protectedSafeguards: string[]; effectiveWindow: string; unresolvedPrecedence: string[] };
export type CohortReview = { snapshotHash: string; cases: PolicyCaseFacts[]; indeterminateIds: string[]; dedupChangedIds: string[]; sourceRefs: string[] };
export type PolicyOutcomePair = { caseId: string; baseline: PolicyReplayResult; candidate: PolicyReplayResult; changedDuties: string[]; changedPermissions: string[]; changedTiming: string[]; changedEscalations: string[]; causalConditions: string[]; unresolved: string[] };
export type PolicyOutcomeSet = { baselineHash: string; candidateHash: string; cohortHash: string; pairs: PolicyOutcomePair[]; unpairedCaseIds: string[]; evidenceRefs: string[] };
export type RegressionSuite = { preservedExceptionCaseIds: string[]; lostSafeguardCaseIds: string[]; missingRegressionIds: string[]; nextTests: string[] };
export type BoundaryMatrix = { testedBoundaryPairs: string[]; untestedCombinations: string[]; indeterminateCaseIds: string[]; sourceOutcomeRefs: string[] };
export type PolicyOptionProposal = { candidateHash: string; affectedCohortIds: string[]; expectedOutcomeChanges: string[]; retainedSafeguards: string[]; unresolvedTradeoffs: string[]; approvalOwnerId: string; adoptionRevalidationRequired: string[] };

export type PolicyScenarioCheckpoint={caseId:string;revision:number;policyVersionHashes:string[];cohortIds:string[];pairedRunIds:string[];outcomeEditions:PolicyOutcomeSet[];regressionRunIds:string[];boundaryEditions:BoundaryMatrix[];adoptionReceiptRefs:string[];stalePairIds:string[]};
