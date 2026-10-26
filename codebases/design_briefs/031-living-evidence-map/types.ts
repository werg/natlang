

export type ID = string; export type Hash = string; export type LocalDate = string; export type DateTime = string; export type Decimal = string; export type Ratio = { numerator: Decimal; denominator: Decimal }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number };
export type Ref = { tenantId: ID; objectId: ID; version: string; hash: Hash };
export type SpanRef = { source: Ref; locatorKind: 'text' | 'table' | 'timecode' | 'region'; locator: string; excerptHash: Hash };

export type FieldConflict = { field: string; alternatives: string[]; evidenceRefs: string[] };

export type Evidence = { id: string; source: string; version: string; observedAt: string; authority: string; scope: string; text: string };
export type EvidencePage = { evidence: Evidence[]; searchedPartition: string; complete: boolean };

export type PolicyCandidate = { id: string; authority: string; scope: string; effectiveFrom: LocalDate; effectiveTo: LocalDate; supersedes: string[]; clauses: string[] };
export type PolicySelection = { selected: string[]; precedenceQuestions: string[]; applicableScope: string };

export type GateVerdict = 'pass' | 'fail' | 'needs-review';
export type EvidenceReview = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; question: string; inclusionPolicyId: string; corpusIds: string[]; reviewOwnerId: string; cutoffDate: string };
export type EligibilityCriterion = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; population: string; interventionOrExposure: string; outcome: string; exclusion: string };
export type StudyIdentity = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; reportIds: string[]; registryIds: string[]; cohortDescription: string; linkageBasis: string };
export type StudyObservation = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; studyId: string; eligible: boolean; measuredOutcome: string; resultRef: Ref; limitation: string };
export type EvidenceClaim = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; proposition: string; supportingStudyIds: string[]; opposingStudyIds: string[]; applicability: string };
export type EvidenceUpdate = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; changedStudyIds: string[]; staleClaimIds: string[]; revisionReason: string; remainingGap: string };
export type DraftEligibilityCriterion = { value: { tenantId?: string; id?: string; version?: number; schemaVersion?: string; state?: string; createdAt?: string; updatedAt?: string; createdBy?: string; sourceRefs?: Ref[]; decisionRefs?: Ref[]; sensitivity?: string[]; purposeIds?: string[]; retentionPolicyId?: string; caseId?: string; population?: string; interventionOrExposure?: string; outcome?: string; exclusion?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidenceRefs: string[] };

export type EvidenceMapRequest = { caseId: string; question: string; inclusionPolicyId: string; corpusIds: string[]; reviewOwnerId: string; cutoffDate: string; expectedVersion: string };

export type EligibilitySnapshot = { policyId: string; cutoffDate: string; includeRules: string[]; excludeRules: string[]; unresolvedRules: string[]; hash: string };
export type StudyLineage = { canonicalStudyIds: string[]; versionIds: string[]; preprintJournalLinks: string[]; duplicateCandidates: string[]; unresolvedIdentity: string[] };
export type ExtractedObservation = { studyId: string; outcome: string; originalValue: string; originalUnit: string; design: string; population: string; followUp: string; limitation: string; evidenceRefs: string[] };
export type ObservationLedger = ExtractedObservation[];
export type MeasurementAlignment = { alignedMeasures: string[]; unconvertedMeasures: string[]; conversionRefs: string[]; incomparableReasons: string[] };
export type EvidenceClaimGraph = { claims: EvidenceClaim[]; supportingStudyIds: string[]; opposingStudyIds: string[]; applicabilityLimits: string[]; confidenceRationale: string[]; evidenceRefs: string[] };
export type ReviewUpdateLedger = { invalidatedClaimIds: string[]; preservedClaimIds: string[]; rescreenedStudyIds: string[]; staleStudyIds: string[]; revisionReason: string; ownerQuestions: string[] };
export type EvidenceMapResult = { review: ReviewFrame; eligibility: EligibilitySnapshot; lineage: StudyLineage; observations: ObservationLedger; rescreenedObservations: ObservationLedger; measures: MeasurementAlignment; claims: EvidenceClaimGraph; retractionUpdate: ReviewUpdateLedger; rescreenUpdate: ReviewUpdateLedger; unresolved: string[]; checkpoint: EvidenceMapCheckpoint; commitAllowed: boolean };
export type ReviewFrame = { reviewId: string; question: string; corpusIds: string[]; reviewOwnerId: string; cutoffDate: string; expectedVersion: string; openAuthorityIssues: string[] };
export type ResearchEvidencePack = { checkpoint?: EvidenceMapCheckpoint; reviews: EvidenceReview[]; evidencePages: EvidencePage[]; studyRecords: Evidence[]; eligibilityPolicies: PolicyCandidate[]; searchedPartitions: string[]; versionLinks: EvidencePage[]; measureConversions: EvidencePage[]; accessLimitations: EvidencePage[]; retractions: EvidencePage[]; eligibilityChanges: EvidencePage[] };

export type EvidenceMapCheckpoint={caseId:string;revision:number;reviewQuestion:string;cutoff:string;eligibilityVersions:string[];studyLineageEditions:StudyLineage[];observationEditions:ObservationLedger[];claimGraphEditions:EvidenceClaimGraph[];retractionReceiptRefs:string[];rescreenedStudyIds:string[];invalidatedClaimIds:string[]};
