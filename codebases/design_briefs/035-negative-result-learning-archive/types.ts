
export type ID = string; export type Hash = string; export type LocalDate = string; export type DateTime = string; export type Decimal = string; export type Ratio = { numerator: Decimal; denominator: Decimal }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number };
export type Ref = { tenantId: ID; objectId: ID; version: string; hash: Hash };
export type SpanRef = { source: Ref; locatorKind: 'text' | 'table' | 'timecode' | 'region'; locator: string; excerptHash: Hash };

export type FieldConflict = { field: string; alternatives: string[]; evidenceRefs: string[] };

export type Question = { question: string; respondent: string; evidenceNeeded: string };

export type Evidence = { id: string; source: string; version: string; observedAt: string; authority: string; scope: string; text: string };
export type EvidencePage = { evidence: Evidence[]; searchedPartition: string; complete: boolean };

export type PolicyCandidate = { id: string; authority: string; scope: string; effectiveFrom: LocalDate; effectiveTo: LocalDate; supersedes: string[]; clauses: string[] };
export type PolicySelection = { selected: string[]; precedenceQuestions: string[]; applicableScope: string };

export type GateVerdict = 'pass' | 'fail' | 'needs-review';
export type NegativeResultCase = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; projectId: string; originalQuestion: string; investigatorId: string; recordIds: string[]; archiveScope: string };
export type ResearchAttempt = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; hypothesis: string; procedure: string; actualConditions: string[]; completionStatus: string };
export type NegativeFinding = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; attemptId: string; kind: 'negative'|'invalid'|'incomplete'|'inconclusive'; supportedConclusion: string };
export type FailureExplanation = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; findingId: string; mechanism: string; supportingRefs: Ref[]; rivalExplanationIds: string[] };
export type DiscriminatingExperiment = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; explanationIds: string[]; proposedChange: string; expectedDistinction: string; stopRule: string };
export type ResearchLesson = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; attemptIds: string[]; reusableLesson: string; applicability: string; supersededBy: string };
export type DraftResearchAttempt = { value: { tenantId?: string; id?: string; version?: number; schemaVersion?: string; state?: string; createdAt?: string; updatedAt?: string; createdBy?: string; sourceRefs?: Ref[]; decisionRefs?: Ref[]; sensitivity?: string[]; purposeIds?: string[]; retentionPolicyId?: string; caseId?: string; hypothesis?: string; procedure?: string; actualConditions?: string[]; completionStatus?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidenceRefs: string[] };

export type AttemptTimeline = { attemptId: string; intendedSteps: string[]; executedSteps: string[]; omittedSteps: string[]; deviations: string[]; measuredOutcomes: string[]; sourceRefs: Ref[] };
export type InferentialStatus = { attemptId: string; class: 'valid-negative' | 'measurement-invalid' | 'incomplete' | 'uncertain'; hypothesisTested: string; supportedConclusion: string; limits: string[]; sourceRefs: Ref[] };
export type RivalMechanism = { id: string; mechanism: string; status: 'demonstrated' | 'plausible' | 'contradicted'; supportingRefs: Ref[]; contraryRefs: Ref[]; conditions: string[] };
export type FollowupPrediction = { explanationId: string; singleChange: string; predictedIfTrue: string; predictedIfFalse: string; safetyConstraints: string[]; resourceConstraints: string[]; stopRule: string };
export type LessonRevision = { lessonId: string; applicabilityConditions: string[]; contradictedBy: string[]; supersedes: string[]; preservesEarlierVersion: boolean; restrictedAttribution: boolean; methodAccessRoles: string[] };
export type NegativeResultDossier = { caseId: string; attempts: AttemptTimeline[]; inferences: InferentialStatus[]; rivalMechanisms: RivalMechanism[]; followupPredictions: FollowupPrediction[]; lessonRevisions: LessonRevision[]; sharedProcessIssues: string[]; openQuestions: Question[]; commitAllowed: boolean };
export type ArchiveRequest = { caseId: ID; researchQuestion: string; scope: string; archivist: ID; versionAtReview: string };
export type AttemptArchiveCorpus = { archiveVersion: string; attempts: NegativeResultCase[]; evidencePages: EvidencePage[]; authorityRules: PolicyCandidate[]; searchedPartitions: string[] };
export type FollowupExecutionReceipt = { receiptId: ID; predictionRef: Ref; explanationId: ID; changedVariable: string; observedConditions: string[]; actualOutcome: string; measurementValid: "valid" | "invalid" | "unknown"; deviations: string[]; sourceRefs: Ref[]; completedAt: DateTime };
export type FollowupOutcome = { receiptId: ID; predictionRef: Ref; explanationId: ID; status: "valid-result" | "invalid-run" | "incomplete" | "uncertain"; favors: "if-true" | "if-false" | "neither" | "unresolved"; basis: string[]; inferenceLimits: string[]; sourceRefs: Ref[] };
export type FollowupLearningLedger = { revision: number; outcomes: FollowupOutcome[]; receipts: FollowupExecutionReceipt[]; consumedReceiptIds: ID[] };
export type FollowupReconciliation = { ledger: FollowupLearningLedger; outcome: FollowupOutcome; receiptDisposition: "recorded" | "replayed" | "held"; lessonUpdateAllowed: boolean; nextQuestions: Question[] };
