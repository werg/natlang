
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
export type AmendmentCase = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; baseAgreementId: string; amendmentIds: string[]; effectiveDate: string; reviewOwnerId: string; baselineHash: string };
export type AmendmentTarget = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; amendmentId: string; targetClauseIds: string[]; operation: 'replace'|'add'|'delete'|'narrow'|'extend'; ambiguity: string };
export type AmendmentScope = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; targetId: string; validFrom: string; validTo: string; coveredEntities: string[]; excludedCases: string[] };
export type AmendedObligation = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; priorId: string; newText: string; preservedConditions: string[]; removedConditions: string[] };
export type AmendmentImpact = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; obligationId: string; affectedTaskIds: string[]; affectedArtifactIds: string[]; unaffectedReason: string };
export type AmendmentRelease = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; impactIds: string[]; reviewerIds: string[]; approvedHash: string; migrationPlan: string };

export type AmendmentReviewRequest = { caseId: string; baseAgreementId: string; amendmentIds: string[]; effectiveDate: string; reviewOwnerId: string; baselineHash: string; expectedVersion: string };
export type ExecutedAgreementBundle = { checkpoint?: AmendmentCheckpoint; agreements: AmendmentCase[]; evidencePages: EvidencePage[]; policyCandidates: PolicyCandidate[]; searchedPartitions: string[]; executedEditions: EvidencePage[]; billingHistory: EvidencePage[]; overlapEvidence: EvidencePage[]; cutoverReceipts: EvidencePage[]; exceptionRegressions: EvidencePage[] };
export type AmendmentGuardrail = { disposition: string; affectedIds: string[]; preservedIds: string[]; blockers: string[]; evidenceRefs: string[]; proposedActions: ActionProposal[] };
export type AmendmentTargetMap = { targets: AmendmentTarget[]; ambiguousSectionIds: string[]; unresolvedOperations: string[] };
export type AmendmentScopeMap = { scopes: AmendmentScope[]; uncoveredTargetIds: string[]; precedenceQuestions: string[]; retroactiveQuestions: string[] };
export type ObligationComposition = { obligations: AmendedObligation[]; preservedConditionsById: Record<string,string[]>; removedConditionIds: string[]; blockers: string[]; diffHash: string };
export type AmendmentImpactMap = { impacts: AmendmentImpact[]; unaffectedIds: string[]; cutoverRevalidationIds: string[]; historicalBillingIds: string[]; blockers: string[] };
export type AmendmentReleasePlan = { release: AmendmentRelease | null; approvedDiffHash: string; migrationDecisionsRequired: string[]; blockers: string[] };
export type AmendmentProgramResult = { caseId: string; baseline: GoalContract; targets: AmendmentTargetMap; scope: AmendmentScopeMap | null; composition: ObligationComposition | null; impacts: AmendmentImpactMap | null; release: AmendmentReleasePlan | null; guardrails: AmendmentGuardrail[]; checkpoint: AmendmentCheckpoint; commitAllowed: boolean };

export type DraftAmendmentTarget = { value: { tenantId?: string; id?: string; version?: number; schemaVersion?: string; state?: string; createdAt?: string; updatedAt?: string; createdBy?: string; sourceRefs?: Ref[]; decisionRefs?: Ref[]; sensitivity?: string[]; purposeIds?: string[]; retentionPolicyId?: string; caseId?: string; amendmentId?: string; targetClauseIds?: string[]; operation?: 'replace'|'add'|'delete'|'narrow'|'extend'; ambiguity?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidenceRefs: string[] };

export type AmendmentCheckpoint={caseId:string;revision:number;executedEditionRefs:string[];amendmentEditions:AmendmentCase[];effectiveScopeRefs:string[];changedDependencyIds:string[];preservedExceptionIds:string[];billingCorrectionRefs:string[];cutoverReceiptRefs:string[];pendingReleaseBlockers:string[]};
