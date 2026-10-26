
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

export type GateVerdict = 'pass' | 'fail' | 'needs-review';
export type Onboarding = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; customerId: string; objectives: string[]; sponsorId: string; targetDate: string; scopeBaselineId: string };
export type OnboardingOutcome = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; objective: string; acceptanceEvidence: string; customerOwnerId: string; excludedScope: string };
export type EnvironmentConstraint = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; systemId: string; capability: string; accessRequirement: string; incompatibility: string };
export type OnboardingTask = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; outcomeIds: string[]; prerequisiteIds: string[]; assignedTo: string; completionEvidence: string };
export type ProgressFinding = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; taskId: string; evidenceRefs: Ref[]; residualWork: string };
export type OnboardingRevision = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; changedTaskIds: string[]; scopeDelta: string; customerApprovalRef: Ref; acceptanceResult: string };
export type DraftOnboardingOutcome = { value: { tenantId?: string; id?: string; version?: number; schemaVersion?: string; state?: string; createdAt?: string; updatedAt?: string; createdBy?: string; sourceRefs?: Ref[]; decisionRefs?: Ref[]; sensitivity?: string[]; purposeIds?: string[]; retentionPolicyId?: string; caseId?: string; objective?: string; acceptanceEvidence?: string; customerOwnerId?: string; excludedScope?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidenceRefs: string[] };

export type CustomerLaunchRequest = { caseId: string; requestedBy: string; scope: string; expectedVersion: string; excludedScope: string[] };
export type CustomerImplementationFile = { onboardingRecords: Onboarding[]; evidencePages: EvidencePage[]; governingTerms: PolicyCandidate[]; searchedPartitions: string[]; accountTransferReceipts: EvidencePage[]; productionSampleComparisons: EvidencePage[]; integrationFallbackApprovals: EvidencePage[]; operatorCapabilityEvidence: EvidencePage[]; contractualDateAmendments: EvidencePage[] };
export type OnboardingGuardrail = { proposedActions: ActionProposal[]; blockers: string[]; evidenceRefs: string[]; applicability: string };

export type EnvironmentAssessment={caseId:string;compatibleSystems:string[];unknownInterfaces:string[];securityBlockers:string[];sampleProductionRisks:string[];evidenceRefs:string[];};
export type OnboardingTaskPlan={caseId:string;revision:string;tasks:OnboardingTask[];dependencies:string[];sharedPrerequisiteIds:string[];pilotTaskIds:string[];blockedOutcomes:string[];};
export type TaskProgressLedger={caseId:string;taskFindings:ProgressFinding[];attemptedButUnverified:string[];completedWithEvidence:string[];residualWork:string[];};
export type OnboardingChangeProposal={caseId:string;revision:string;preservedTaskIds:string[];changedTaskIds:string[];scopeDelta:string[];approvalRequired:boolean;acceptanceReplay:string[];};
export type OnboardingProgramResult={caseId:string;outcomes:OnboardingOutcome[];environment:EnvironmentAssessment;plan:OnboardingTaskPlan|null;progress:TaskProgressLedger|null;replan:OnboardingChangeProposal|null;guardrails:OnboardingGuardrail[];unresolved:string[];actions:ActionProposal[];checkpoint:OnboardingCheckpoint;commitAllowed:boolean};
export type OnboardingCheckpoint={caseId:string;revision:number;outcomeEditions:OnboardingOutcome[];activePlan:OnboardingTaskPlan|null;completedTaskReceiptRefs:string[];pendingTaskKeys:string[];ownerTransferReceipts:string[];integrationFallbackRefs:string[];unresolvedCustomerAcceptances:string[]};
