export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };
export type CaseInput = { caseId: string; goal: string; evidence: Evidence[]; policy: string[]; authorityStatus: "authorized" | "unclear" | "unauthorized"; changedEvidenceIds: string[]; audience: string; domainRecords: { firstStage: DocumentedPromise } };
export type GoalContract = { outcome: string; exclusions: string[]; affectedEntities: string[]; acceptanceEvidence: string[]; clarificationNeeded: string[] };
export type PolicySelection = { selected: string[]; unresolvedPrecedence: string[]; authorityStatus: "authorized" | "unclear" | "unauthorized" };
export type WorkGraph = { nodes: string[]; prerequisites: string[]; blockedQuestions: string[] };
export type EvidencePlan = { requiredEvidence: string[]; selectedIds: string[]; searchedUniverse: string[]; missingCoverage: string[] };
export type PreparedEvidence = { subject: string; evidence: Evidence[]; missing: string[]; conflicts: string[]; alternatives: string[] };
export type DecisionAtom = { field: string; value: string; evidenceIds: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable"; alternative: string };
export type Qualification = "pass" | "fail" | "needs-review";
export type AuditResult = { verdict: Qualification; gate: string; reason: string; repairTargets: string[] };
export type FieldConflict = { field: string; alternatives: string[]; evidenceIds: string[] };
export type EvidenceBinding = { field: string; evidenceIds: string[]; sourceVersions: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable" };
export type Rejection = { id: string; reason: string; evidenceIds: string[] };
export type CoverageManifest = { eligible: string[]; processed: string[]; deferred: string[]; rejected: string[]; outOfScope: string[] };
export type Question = { id: string; question: string; respondent: string; acceptableEvidence: string[] };
export type ActionProposal = { kind: string; targetIds: string[]; reason: string; approvalRequired: boolean };
export type CheckResult = { check: string; verdict: Qualification; evidenceIds: string[] };
export type CollectionFinding = { acceptedRefs: string[]; unresolvedRefs: string[]; conflicts: string[]; coverageGaps: string[] };
export type QuestionBatch = { questions: string[]; authorizedRespondent: string; acceptableEvidence: string[] };
export type InvalidationProposal = { invalidated: string[]; preserved: string[]; unresolvedImpact: string[] };
export type ReviewPacket = { supportedDecisions: string[]; unresolvedAlternatives: string[]; actionScope: string[]; approvalBindings: string[] };
export type ViewModel = { title: string; status: "ready" | "blocked" | "review"; sections: string[]; uncertainties: string[] };
export type RevisionProposal = { target: string; change: string; contracts: string[]; expectedBenefit: string; regressionRisks: string[]; tests: string[] };

export type RecordMeta = { id: ID; version: string; createdAt: DateTime; updatedAt: DateTime };
export type DocumentationReview = RecordMeta & {
  repositoryId: ID;
  documentIds: ID[];
  targetRelease: string;
  ownerId: ID;
  behaviorSnapshotId: ID;
};
export type DocumentedPromise = RecordMeta & {
  caseId: ID;
  documentId: ID;
  behaviorClaim: string;
  applicableVersion: string;
  exampleRefs: Ref[];
};
export type BehaviorObservation = RecordMeta & {
  caseId: ID;
  promiseId: ID;
  probeSpecId: ID;
  observedOutcome: string;
  environmentHash: string;
};
export type DocumentationMismatch = RecordMeta & {
  caseId: ID;
  promiseId: ID;
  kind: 'stale-doc'|'product-defect'|'intentional-change'|'uncertain';
  rationale: string;
};
export type DocumentationPatch = RecordMeta & {
  caseId: ID;
  artifactIds: ID[];
  baseHashes: string[];
  proposedText: string;
  protectedClaims: string[];
};
export type DocumentationRelease = RecordMeta & {
  caseId: ID;
  patchIds: ID[];
  exampleTestIds: ID[];
  residualMismatchIds: ID[];
  approvalRefs: Ref[];
};
export type DraftDocumentedPromise = { values: { caseId?: ID; documentId?: ID; behaviorClaim?: string; applicableVersion?: string; exampleRefs?: Ref[] }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultDocumentedPromise = { accepted: DocumentedPromise[]; unresolved: DraftDocumentedPromise[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftBehaviorObservation = { values: { caseId?: ID; promiseId?: ID; probeSpecId?: ID; observedOutcome?: string; environmentHash?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultBehaviorObservation = { accepted: BehaviorObservation[]; unresolved: DraftBehaviorObservation[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftDocumentationMismatch = { values: { caseId?: ID; promiseId?: ID; kind?: 'stale-doc'|'product-defect'|'intentional-change'|'uncertain'; rationale?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultDocumentationMismatch = { accepted: DocumentationMismatch[]; unresolved: DraftDocumentationMismatch[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftDocumentationPatch = { values: { caseId?: ID; artifactIds?: ID[]; baseHashes?: string[]; proposedText?: string; protectedClaims?: string[] }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultDocumentationPatch = { accepted: DocumentationPatch[]; unresolved: DraftDocumentationPatch[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftDocumentationRelease = { values: { caseId?: ID; patchIds?: ID[]; exampleTestIds?: ID[]; residualMismatchIds?: ID[]; approvalRefs?: Ref[] }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultDocumentationRelease = { accepted: DocumentationRelease[]; unresolved: DraftDocumentationRelease[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type RunProposal = { caseId: string; goal: GoalContract; policy: PolicySelection; work: WorkGraph; evidencePlan: EvidencePlan; extractPromises: StageResultDocumentedPromise; probeBehavior: StageResultBehaviorObservation; classifyMismatch: StageResultDocumentationMismatch; constructRepairs: StageResultDocumentationPatch; verifyConsistency: StageResultDocumentationRelease; collection: CollectionFinding; questions: QuestionBatch; invalidation: InvalidationProposal; review: ReviewPacket; view: ViewModel; revisions: RevisionProposal[]; disposition: "proposed" | "blocked" | "needs-review"; rationale: string[] };

export type CaseStageSet = { extractPromises: StageResultDocumentedPromise; probeBehavior: StageResultBehaviorObservation; classifyMismatch: StageResultDocumentationMismatch; constructRepairs: StageResultDocumentationPatch; verifyConsistency: StageResultDocumentationRelease; };
export type DocumentationEditionRequest = { reviewId: ID; targetRelease: string; sourceEvidence: Evidence[]; requirements: Evidence[]; releaseDecisions: Evidence[]; approvedIntent: string[]; policyClauses: string[]; authorityStatus: 'authorized' | 'unclear' | 'unauthorized'; exampleSources: Evidence[]; priorPromise: DocumentedPromise; currentPageBases: string[]; translatedEditions: string[] };
export type VersionedDocumentationPromise = { promiseId: ID; claim: string; sourceSpan: string; pageEdition: string; audience: string; normative: boolean; applicableRelease: string; exampleRefs: ID[] };
export type BehaviorProbePlan = { promiseId: ID; input: string; expected: string; environmentIdentity: string; targetRelease: string; observationRefs: ID[]; status: 'observed' | 'prepared' | 'environment-mismatch' };
export type DocumentationDiscrepancy = { promiseId: ID; kind: 'stale-doc' | 'product-defect' | 'intentional-change' | 'uncertain'; authorityRefs: ID[]; rationale: string; ownerQuestion: string };
export type DocumentationRepairBundle = { pagePatches: string[]; examplePatches: string[]; testPatches: string[]; releaseNotePatches: string[]; patchBase: string; blockedSecretSpans: string[]; reviewedReplacementRefs: ID[] };
export type DocumentationRebaseReview = { staleSpans: string[]; revalidatedClaims: string[]; preservedConcurrentEdits: string[]; invalidatedPatchParts: string[] };
export type DocumentationReleaseDecision = { status: 'blocked' | 'review' | 'ready'; version: string; unresolvedMismatches: ID[]; brokenExamples: ID[]; staleBaseSpans: string[]; translationEditionLag: string[]; approvals: string[] };
export type SecretExportFinding = { sourceSpan: string; category: string; exportBlocked: boolean };
export type DocumentationEditionProposal = { reviewId: ID; promises: VersionedDocumentationPromise[]; probes: BehaviorProbePlan[]; mismatches: DocumentationDiscrepancy[]; patch: DocumentationRepairBundle; rebase: DocumentationRebaseReview; release: DocumentationReleaseDecision; commitAllowed: boolean };
