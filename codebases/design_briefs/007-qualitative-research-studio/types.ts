
export type ID = string; export type Hash = string; export type LocalDate = string; export type DateTime = string; export type Decimal = string; export type Ratio = { numerator: Decimal; denominator: Decimal }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number };
export type Ref = { tenantId: ID; objectId: ID; version: string; hash: Hash };
export type SpanRef = { source: Ref; locatorKind: 'text' | 'table' | 'timecode' | 'region'; locator: string; excerptHash: Hash };

export type FieldConflict = { field: string; alternatives: string[]; evidenceRefs: string[] };

export type Question = { question: string; respondent: string; evidenceNeeded: string };

export type Evidence = { id: string; source: string; version: string; observedAt: string; authority: string; scope: string; text: string };
export type EvidencePage = { evidence: Evidence[]; searchedPartition: string; complete: boolean };

export type GoalContract = { outcome: string; exclusions: string[]; entities: string[]; acceptanceEvidence: string[]; unresolvedScope: string[] };
export type PolicyCandidate = { id: string; authority: string; scope: string; effectiveFrom: LocalDate; effectiveTo: LocalDate; supersedes: string[]; clauses: string[] };
export type PolicySelection = { selected: string[]; precedenceQuestions: string[]; applicableScope: string };
export type Candidate = { id: string; subject: string; existingRecord: string; evidenceRefs: string[] };

export type GateVerdict = 'pass' | 'fail' | 'needs-review';
export type ResearchProject = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; researchQuestion: string; decisionContext: string; corpusIds: string[]; consentPolicyId: string; principalInvestigatorId: string };
export type InquiryFrame = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; subquestions: string[]; rivalExplanations: string[]; excludedClaims: string[]; stoppingRule: string };
export type SamplingPlan = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; strata: string[]; selectedRecordIds: string[]; excludedRecordIds: string[]; selectionReason: string };
export type CodeDefinition = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; label: string; inclusionRule: string; exclusionRule: string; anchorRefs: Ref[]; parentCodeId: string };
export type CodingDecision = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; passageRef: Ref; codeIds: string[]; disagreement: string; adjudication: string };
export type ResearchFinding = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; question: string; supportingRefs: Ref[]; contradictoryRefs: Ref[]; claim: string; limitation: string };
export type DraftInquiryFrame = { value: { tenantId?: string; id?: string; version?: number; schemaVersion?: string; state?: string; createdAt?: string; updatedAt?: string; createdBy?: string; sourceRefs?: Ref[]; decisionRefs?: Ref[]; sensitivity?: string[]; purposeIds?: string[]; retentionPolicyId?: string; caseId?: string; subquestions?: string[]; rivalExplanations?: string[]; excludedClaims?: string[]; stoppingRule?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidenceRefs: string[] };

export type ResearchInquiry = { caseId: string; researchQuestion: string; decisionToInform: string; requestedBy: string; expectedCorpusVersion: string; targetPopulation?: string; requestedStoppingRule?: string };
export type InterviewCorpus = { version: string; projects: ResearchProject[]; interviewsAndFollowups: EvidencePage[]; sampleFrame: EvidencePage[]; translationReviews: EvidencePage[]; privacyAndConsentEvents: EvidencePage[]; investigatorRules: PolicyCandidate[]; searchedPartitions: string[] };

export type InterviewPassage = { passageId: string; participantKey: string; language: string; originalText: string; reviewedTranslation?: string; context: string; consentScope: string; reidentificationCues: string[]; evidenceIds: string[] };
export type CodeBoundary = { codeId: string; label: string; descriptiveOrExplanatory: 'descriptive'|'inferred'; includeExamples: string[]; excludeExamples: string[]; boundaryCounterexamples: string[]; version: string };
export type CodeExampleJudgment = { passageId: string; role: 'include'|'exclude'|'counterexample'|'ambiguous'; observedMeaning: string; boundaryReason: string; evidenceIds: string[] };
export type CodeBoundaryReview = { boundary: CodeBoundary; judgments: CodeExampleJudgment[]; disposition: 'bounded'|'overlapping'|'unresolved'; rationale: string; questions: string[] };
export type CodebookConstruction = { boundaries: CodeBoundary[]; reviews: CodeBoundaryReview[] };
export type CoderDisagreementReview = { passageId: string; codeId: string; codeVersion: string; coderLabels: string[]; disagreementKind: 'boundary'|'translation'|'interpretive'|'insufficient-record'; retainedMinorityReading: string; evidenceIds: string[]; resolutionQuestion: string | null };
export type CoderComparison = { passageId: string; codeVersion: string; coderA: string; coderB: string; agreements: string[]; disagreements: string[]; minorityInterpretations: string[]; disagreementReviews?: CoderDisagreementReview[]; stale: boolean; evidenceIds: string[] };

export type QuotationRiskReview = { highRiskPassageIds: string[]; maskedPassageIds: string[]; restrictedPassageIds: string[]; reviewerQuestions: Question[] };
export type InquiryDecomposition = { decision: string; subquestions: string[]; observableEvidence: string[]; inferentialLimits: string[] };
export type RivalExplanation = { explanation: string; distinguishingObservation: string; contradictingCases: string[]; unresolved: string[] };
export type SamplingStratum = { condition: string; rationale: string; candidateIds: string[]; expectedContrast: string };
export type SamplingManifest = { selectedIds: string[]; excludedIds: string[]; nonresponseIds: string[]; selectionMethod: string; inclusionProbabilities: string[]; frozenAt: string };
export type CodeMechanismSearch = { codeId: string; mechanism: string; supportingRefs: string[]; contradictoryRefs: string[]; narrowedInterpretation: string | null; missingPartitions: string[]; searchQuestion: string };
export type CounterevidenceSearch = { supportingRefs: string[]; contradictoryRefs: string[]; searchedPartitions: string[]; missingPartitions: string[]; absenceClaimAllowed: boolean; codeMechanismSearches?: CodeMechanismSearch[] };
export type QualitativeStudyProposal = { caseId: string; corpusVersion: string; inquiryMaps: InquiryDecomposition[]; rivalExplanations: RivalExplanation[]; strata: SamplingStratum[]; sampleManifest: SamplingManifest; codeBoundaries: CodeBoundary[]; codebookReviews?: CodeBoundaryReview[]; passages: InterviewPassage[]; coderComparisons: CoderComparison[]; counterevidence: CounterevidenceSearch[]; quotationRisk: QuotationRiskReview; unmetStoppingConditions: string[]; questions: Question[]; commitAllowed: boolean };
export type QualitativeStudyCheckpoint = { caseId: string; revision: number; activeCorpusVersion: string; proposals: QualitativeStudyProposal[]; frozenSampleVersion: string; codebookVersions: string[]; codingComparisons: CoderComparison[]; invalidatedCodingIds: string[]; pendingPrivacyReviews: string[]; exportReceipts: string[] };
export type QualitativeStudyRun = { proposal: QualitativeStudyProposal; checkpoint: QualitativeStudyCheckpoint; staleCodingIds: string[]; candidateExportPassageIds: string[]; blockedPassageIds: string[] };
export type StudyRevisionImpact = { staleCodingIds: string[]; reopenedPassageIds: string[]; changedCodeIds: string[]; changedSampleIds: string[]; reasons: string[] };
