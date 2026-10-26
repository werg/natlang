

export type ID = string; export type Hash = string; export type LocalDate = string; export type DateTime = string; export type Decimal = string; export type Ratio = { numerator: Decimal; denominator: Decimal }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number };
export type Ref = { tenantId: ID; objectId: ID; version: string; hash: Hash };
export type SpanRef = { source: Ref; locatorKind: 'text' | 'table' | 'timecode' | 'region'; locator: string; excerptHash: Hash };

export type FieldConflict = { field: string; alternatives: string[]; evidenceRefs: string[] };

export type Evidence = { id: string; source: string; version: string; observedAt: string; authority: string; scope: string; text: string };
export type EvidencePage = { evidence: Evidence[]; searchedPartition: string; complete: boolean };

export type PolicyCandidate = { id: string; authority: string; scope: string; effectiveFrom: LocalDate; effectiveTo: LocalDate; supersedes: string[]; clauses: string[] };
export type PolicySelection = { selected: string[]; precedenceQuestions: string[]; applicableScope: string };

export type GateVerdict = 'pass' | 'fail' | 'needs-review';
export type ReplicationProject = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; studyId: string; replicationGoal: string; sourceIds: string[]; investigatorId: string; computeBudget: { minorUnits: string; currency: string; scale: number } };
export type MethodStep = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; inputIds: string[]; operation: string; outputDefinition: string; parameterSources: Ref[] };
export type ReplicationDependency = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; stepId: string; prerequisiteIds: string[]; resourceIds: string[]; compatibilityIssue: string };
export type MethodQuestion = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; stepId: string; ambiguity: string; alternatives: string[]; discriminatingEvidence: string };
export type ReplicationTask = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; stepId: string; commandSpecId: string; inputHashIds: string[]; expectedArtifact: string };
export type ReplicationFinding = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; taskIds: string[]; referenceResult: string; observedResult: string; explanationStatus: string };
export type DraftMethodStep = { value: { tenantId?: string; id?: string; version?: number; schemaVersion?: string; state?: string; createdAt?: string; updatedAt?: string; createdBy?: string; sourceRefs?: Ref[]; decisionRefs?: Ref[]; sensitivity?: string[]; purposeIds?: string[]; retentionPolicyId?: string; caseId?: string; inputIds?: string[]; operation?: string; outputDefinition?: string; parameterSources?: Ref[] }; missingFields: string[]; conflicts: FieldConflict[]; evidenceRefs: string[] };

export type ReplicationPlanRequest = { caseId: string; studyId: string; replicationGoal: string; sourceIds: string[]; investigatorId: string; computeBudget: Money; expectedVersion: string };
export type ReplicationEvidenceBundle = { checkpoint?: ReplicationCheckpoint; projects: ReplicationProject[]; evidencePages: EvidencePage[]; policyCandidates: PolicyCandidate[]; searchedPartitions: string[]; dependencyReceipts: EvidencePage[]; seedPlanEvidence: EvidencePage[]; datasetVersionReceipts: EvidencePage[]; jobReceipts: ReplicationJobReceipt[]; methodConflictEvidence: EvidencePage[] };
export type ReplicationJobReceipt = { taskId: string; commandHash: string; sourceHash: string; datasetHash: string; environmentHash: string; seed: string; outputHash: string; completed: boolean; expectedArtifactRefs: string[]; actualArtifactRefs: string[]; errorRef: string };
export type MethodProcedure = { stepIds: string[]; exactParameters: string[]; sourceAndCodeConflicts: string[]; unresolvedChoices: string[]; evidenceRefs: string[] };
export type ReplicationGraph = { dependencies: ReplicationDependency[]; cycles: string[]; unavailableInputs: string[]; environmentRequirements: string[]; evidenceRefs: string[] };
export type SeedSensitivityPlan = { preregisteredSeeds: string[]; seedSourceRef: string; omissions: string[]; outcomeSelectionForbidden: string[] };
export type DatasetPin = { expectedHash: string; observedHash: string; datasetVersion: string; exactMatch: boolean; allowedAdaptation: string; evidenceRefs: string[] };
export type ArtifactCompleteness = { taskId: string; complete: boolean; expected: string[]; verified: string[]; missing: string[]; reusable: boolean; evidenceRefs: string[] };
export type ReplicationComparison = { taskId: string; referenceResult: string; replicatedResult: string; seed: string; sourceHash: string; datasetHash: string; environmentHash: string; differenceStatus: string; unresolved: string[]; evidenceRefs: string[] };
export type ReplicationWorkbenchResult = { caseId: string; frame: ReplicationFrame; procedure: MethodProcedure; dependencyGraph: ReplicationGraph; seedPlan: SeedSensitivityPlan; datasetPin: DatasetPin; executionPlan: ReplicationTask[]; artifactChecks: ArtifactCompleteness[]; comparisons: ReplicationComparison[]; adaptations: string[]; blockers: string[]; checkpoint: ReplicationCheckpoint; commitAllowed: boolean };
export type ReplicationFrame = { studyId: string; replicationGoal: string; sourceIds: string[]; investigatorId: string; computeBudget: Money; expectedVersion: string; blockers: string[] };

export type ReplicationCheckpoint={caseId:string;revision:number;methodEditionRefs:string[];dependencyVersions:string[];seedPlans:SeedSensitivityPlan[];datasetPins:DatasetPin[];executionPlanEditions:ReplicationTask[][];jobReceiptIds:string[];completeRunIds:string[];incompleteRunIds:string[];adaptationRefs:string[]};
