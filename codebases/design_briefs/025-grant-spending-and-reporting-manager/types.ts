
export type ID = string; export type Hash = string; export type LocalDate = string; export type DateTime = string; export type Decimal = string; export type Ratio = { numerator: Decimal; denominator: Decimal }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number };
export type Ref = { tenantId: ID; objectId: ID; version: string; hash: Hash };
export type SpanRef = { source: Ref; locatorKind: 'text' | 'table' | 'timecode' | 'region'; locator: string; excerptHash: Hash };

export type FieldConflict = { field: string; alternatives: string[]; evidenceRefs: string[] };

export type ActionProposal = { action: string; target: string; requiresApproval: boolean };

export type Evidence = { id: string; source: string; version: string; observedAt: string; authority: string; scope: string; text: string };
export type EvidencePage = { evidence: Evidence[]; searchedPartition: string; complete: boolean };

export type PolicyCandidate = { id: string; authority: string; scope: string; effectiveFrom: LocalDate; effectiveTo: LocalDate; supersedes: string[]; clauses: string[] };
export type PolicySelection = { selected: string[]; precedenceQuestions: string[]; applicableScope: string };

export type GateVerdict = 'pass' | 'fail' | 'needs-review';
export type GrantReport = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; grantId: string; reportingWindow: string; projectId: string; principalOwnerId: string; funderId: string };
export type GrantCondition = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; category: string; allowedUse: string; excludedUse: string; reportingTrigger: string };
export type GrantAllocation = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; expenseId: string; conditionId: string; allocatedAmount: { minorUnits: string; currency: string; scale: number }; allocationBasis: string };
export type ActivityEvidence = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; milestoneId: string; activityRefs: Ref[]; deliveredOutcome: string; remainingRequirement: string };
export type GrantNarrative = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; conditionIds: string[]; claimRefs: Ref[]; narrative: string; limitation: string };
export type GrantSubmission = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; artifactIds: string[]; approvalRefs: Ref[]; submittedAt: string; receiptRef: Ref };

export type GrantCycleRequest = { caseId: string; grantId: string; reportingWindow: string; projectId: string; principalOwnerId: string; funderId: string; expectedVersion: string };
export type GrantEvidenceLedger = { checkpoint?: GrantCheckpoint; reports: GrantReport[]; evidencePages: EvidencePage[]; policyCandidates: PolicyCandidate[]; searchedPartitions: string[]; periodAllocations: EvidencePage[]; amendments: EvidencePage[]; sharedActivityLinks: EvidencePage[]; participantDisclosureRequests: EvidencePage[]; priorSubmissions: EvidencePage[] };
export type GrantGuardrail = { disposition: string; blockers: string[]; evidenceRefs: string[]; proposedActions: ActionProposal[] };
export type GrantConditionSet = { conditions: GrantCondition[]; unresolvedConditions: string[]; unresolvedEffectiveDates: string[]; reportableMilestones: string[] };
export type GrantPeriodSlice = { sliceId: string; expenseId: string; periodId: string; amount: Money; originalAmount: Money; basisRef: string };
export type PeriodAllocationResult = { slices: GrantPeriodSlice[]; unresolvedExpenseIds: string[]; approvedBasisIds: string[] };
export type PeriodConservationResult = { verifiedSlices: GrantPeriodSlice[]; withheldExpenseIds: string[]; blockers: string[] };
export type SharedActivityResult = { acceptedActivityLinks: string[]; withheldLinks: string[]; blockers: string[]; evidenceRefs: Ref[] };
export type SharedActivityDecision = { activityId: string; acceptedGrantShares: string[]; withheldGrantShares: string[]; duplicateBenefitIds: string[]; evidenceRefs: Ref[]; reason: string };
export type ParticipantDisclosureReview = { allowedParticipantIds: string[]; withheldParticipantIds: string[]; allowedAggregateRefs: Ref[]; blockers: string[]; evidenceRefs: Ref[] };
export type GrantAllocationResult = { allocations: GrantAllocation[]; verifiedSlices: GrantPeriodSlice[]; withheldExpenseIds: string[]; conservationBlockers: string[]; unresolvedBasis: string[] };
export type GrantActivityRegister = { accepted: ActivityEvidence[]; partialMilestoneIds: string[]; unsupportedClaims: string[]; evidenceRefs: Ref[] };
export type GrantNarrativeDraft = { narrative: GrantNarrative | null; blockedClaims: string[]; limitation: string[] };
export type GrantCorrectionPlan = { amendment: GrantSubmission | null; blockers: string[]; oldReceiptRefs: Ref[] };
export type GrantSubmissionPlan = { submission: GrantSubmission | null; artifactHash: string; dispatchReady: boolean; receiptRefs: Ref[]; blockers: string[] };
export type GrantProgramResult = { caseId: string; conditions: GrantConditionSet; allocations: GrantAllocationResult | null; activities: GrantActivityRegister | null; narrative: GrantNarrativeDraft | null; submission: GrantSubmissionPlan | null; guardrails: GrantGuardrail[]; checkpoint: GrantCheckpoint; commitAllowed: boolean };

export type DraftGrantCondition = { value: { tenantId?: string; id?: string; version?: number; schemaVersion?: string; state?: string; createdAt?: string; updatedAt?: string; createdBy?: string; sourceRefs?: Ref[]; decisionRefs?: Ref[]; sensitivity?: string[]; purposeIds?: string[]; retentionPolicyId?: string; caseId?: string; category?: string; allowedUse?: string; excludedUse?: string; reportingTrigger?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidenceRefs: string[] };

export type GrantCheckpoint={caseId:string;revision:number;awardEditionIds:string[];periodAllocationVersions:string[];activityLineageRefs:string[];disclosureReviewRefs:string[];submissionEditions:GrantSubmissionPlan[];amendmentReceiptRefs:string[];funderReceiptRefs:string[];unresolvedCorrections:string[]};
