
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
export type ContractCase = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; agreementIds: string[]; partyIds: string[]; operationalScope: string; ownerId: string; governingPolicyId: string };
export type ContractObligation = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; actorId: string; modality: 'duty'|'permission'|'prohibition'; action: string; trigger: string; deadlineRule: string };
export type ObligationRelation = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; fromId: string; toId: string; relation: 'depends'|'excepts'|'supersedes'|'conflicts'; scope: string };
export type TriggerFinding = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; obligationId: string; eventIds: string[]; status: 'active'|'inactive'|'unknown'; missingFact: string };
export type ObligationTask = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; obligationId: string; ownerId: string; prerequisiteIds: string[]; dueAt: string; completionEvidence: string };
export type ObligationStatus = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; obligationId: string; status: 'open'|'satisfied'|'waived'|'disputed'; evidenceRefs: Ref[] };

export type ObligationReviewRequest = { caseId: string; agreementIds: string[]; partyIds: string[]; operationalScope: string; ownerId: string; expectedVersion: string };
export type ContractEvidenceDossier = { checkpoint?: ContractObligationCheckpoint; contracts: ContractCase[]; evidencePages: EvidencePage[]; policyCandidates: PolicyCandidate[]; searchedPartitions: string[]; businessCalendars: EvidencePage[]; partyIdentityChanges: EvidencePage[]; noticeReceipts: EvidencePage[]; disputedClauseRefs: EvidencePage[]; confidentialTaskRefs: EvidencePage[] };
export type ObligationGuardrail = { disposition: string; alternatives: string[]; blockers: string[]; evidenceRefs: string[]; proposedActions: ActionProposal[] };
export type DutyFacet = { obligation: ContractObligation; exceptionText: string[]; scopeText: string[]; sourceEdition: string; unresolvedFacets: string[] };
export type DutyFacetLedger = { duties: DutyFacet[]; unresolved: string[]; sourceRefs: Ref[] };
export type ClauseRelationGraph = { relations: ObligationRelation[]; isolatedObligationIds: string[]; disputedPaths: string[]; precedenceQuestions: string[] };
export type TriggerLedger = { findings: TriggerFinding[]; missingFacts: string[]; effectiveNoticeBlockers: string[] };
export type ObligationTaskMap = { tasks: ObligationTask[]; withheldObligationIds: string[]; unscheduledTaskIds: string[]; blockers: string[] };
export type ObligationDischargeLedger = { statuses: ObligationStatus[]; partialObligationIds: string[]; disputedObligationIds: string[]; evidenceGaps: string[] };
export type ObligationProgramResult = { caseId: string; duties: DutyFacetLedger; relations: ClauseRelationGraph | null; triggers: TriggerLedger | null; tasks: ObligationTaskMap | null; statuses: ObligationDischargeLedger | null; guardrails: ObligationGuardrail[]; checkpoint: ContractObligationCheckpoint; commitAllowed: boolean };

export type DraftContractObligation = { value: { tenantId?: string; id?: string; version?: number; schemaVersion?: string; state?: string; createdAt?: string; updatedAt?: string; createdBy?: string; sourceRefs?: Ref[]; decisionRefs?: Ref[]; sensitivity?: string[]; purposeIds?: string[]; retentionPolicyId?: string; caseId?: string; actorId?: string; modality?: 'duty'|'permission'|'prohibition'; action?: string; trigger?: string; deadlineRule?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidenceRefs: string[] };

export type ContractObligationCheckpoint={caseId:string;revision:number;contractEditionRefs:string[];dutyEditions:DutyFacetLedger[];clauseRelationEditions:ClauseRelationGraph[];activeTriggerIds:string[];performanceTaskIds:string[];dischargeEditions:ObligationDischargeLedger[];noticeEffectRefs:string[];openDisputes:string[]};
