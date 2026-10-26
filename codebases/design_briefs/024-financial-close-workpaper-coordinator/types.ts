
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
export type CloseWorkpaper = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; entityId: string; period: string; accountIds: string[]; preparerId: string; reviewerId: string };
export type ReconciliationScope = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; accountIds: string[]; sourceSnapshotIds: string[]; assertion: string; materialityRule: string };
export type ReconciliationBreak = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; sourceLineIds: string[]; difference: { minorUnits: string; currency: string; scale: number }; explanationKind: string; unresolvedFact: string };
export type CloseTask = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; breakIds: string[]; requestedEvidence: string; ownerId: string; dueAt: string };
export type WorkpaperConclusion = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; scopeId: string; explainedBreakIds: string[]; residualBreakIds: string[]; conclusion: string };
export type WorkpaperSignoff = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; conclusionId: string; reviewerId: string; approvedHash: string; invalidatedByIds: string[] };

export type CloseCycleRequest = { caseId: string; entityId: string; period: string; accountIds: string[]; preparerId: string; reviewerId: string; expectedVersion: string };
export type CloseSourceBook = { checkpoint?: CloseCheckpoint; workpapers: CloseWorkpaper[]; evidencePages: EvidencePage[]; policyCandidates: PolicyCandidate[]; searchedPartitions: string[]; priorSignoffs: WorkpaperSignoff[]; lateJournalReceipts: EvidencePage[]; reportRegenerations: EvidencePage[]; preparerTransfers: EvidencePage[]; formulaValidationReceipts: EvidencePage[]; deadlineEvents: EvidencePage[] };
export type CloseGuardrail = { disposition: string; invalidatedIds: string[]; preservedIds: string[]; blockers: string[]; evidenceRefs: string[]; proposedActions: ActionProposal[] };
export type CloseAssertionScope = { caseId: string; entityId: string; period: string; assertion: string; accountIds: string[]; sourceRefs: Ref[]; missingAccounts: string[]; blockers: string[] };
export type CloseBreakFinding = { sourceLineIds: string[]; explanationKind: string; amountMinorUnits: string; currency: string; scale: number; establishedExplanation: string; unresolvedFact: string; evidenceRefs: Ref[]; material: boolean };
export type CloseBreakLedger = { findings: CloseBreakFinding[]; blockers: string[]; complete: boolean };
export type CloseTaskQueue = { tasks: CloseTask[]; unassignedBreakIds: string[]; openEvidenceNeeds: string[]; complete: boolean };
export type CloseConclusionDraft = { conclusion: WorkpaperConclusion | null; verifiedFacts: string[]; managementExplanations: string[]; residualBreakIds: string[]; blockers: string[]; sourceRefs: Ref[] };
export type CloseSignoffReview = { eligible: boolean; reviewerId: string; exactBinding: string; invalidatedSignoffIds: string[]; blockers: string[] };
export type CloseProgramResult = { caseId: string; scope: CloseAssertionScope; breaks: CloseBreakLedger | null; tasks: CloseTaskQueue | null; conclusion: CloseConclusionDraft | null; signoff: CloseSignoffReview | null; guardrails: CloseGuardrail[]; checkpoint: CloseCheckpoint; commitAllowed: boolean };

export type DraftReconciliationScope = { value: { tenantId?: string; id?: string; version?: number; schemaVersion?: string; state?: string; createdAt?: string; updatedAt?: string; createdBy?: string; sourceRefs?: Ref[]; decisionRefs?: Ref[]; sensitivity?: string[]; purposeIds?: string[]; retentionPolicyId?: string; caseId?: string; accountIds?: string[]; sourceSnapshotIds?: string[]; assertion?: string; materialityRule?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidenceRefs: string[] };

export type CloseCheckpoint={caseId:string;revision:number;sourceEditionRefs:string[];workpaperEditions:CloseWorkpaper[];openBreakIds:string[];taskReceiptRefs:string[];invalidatedConclusionIds:string[];formulaFailureRefs:string[];signoffReceiptRefs:string[];missedDeadlineIds:string[]};
