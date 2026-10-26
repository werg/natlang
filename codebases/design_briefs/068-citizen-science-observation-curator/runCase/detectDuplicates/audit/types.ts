export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };
export type CaseInput = { caseId: string; goal: string; evidence: Evidence[]; policy: string[]; authorityStatus: "authorized" | "unclear" | "unauthorized"; changedEvidenceIds: string[]; audience: string; domainRecords: { firstStage: FieldObservation } };
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
export type ObservationReview = RecordMeta & {
  collectionId: ID;
  projectId: ID;
  observationType: string;
  curatorId: ID;
  disclosurePolicyId: ID;
};
export type FieldObservation = RecordMeta & {
  caseId: ID;
  observerId: ID;
  observedAt: DateTime;
  locationRef: Ref;
  description: string;
  method: string;
};
export type ObservationInterpretation = RecordMeta & {
  caseId: ID;
  observationId: ID;
  candidateConceptIds: ID[];
  interpretation: string;
  ambiguity: string;
};
export type ObservationLink = RecordMeta & {
  caseId: ID;
  observationIds: ID[];
  relation: 'same-event'|'independent'|'related'|'unknown';
  basis: string;
};
export type ObservationQuestion = RecordMeta & {
  caseId: ID;
  observationId: ID;
  decisiveUnknown: string;
  requestedEvidence: string;
  recipientId: ID;
};
export type CuratedObservation = RecordMeta & {
  caseId: ID;
  observationId: ID;
  acceptedInterpretation: string;
  qualityFlags: string[];
  publicLocationPrecision: string;
};
export type DraftFieldObservation = { values: { caseId?: ID; observerId?: ID; observedAt?: DateTime; locationRef?: Ref; description?: string; method?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultFieldObservation = { accepted: FieldObservation[]; unresolved: DraftFieldObservation[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftObservationInterpretation = { values: { caseId?: ID; observationId?: ID; candidateConceptIds?: ID[]; interpretation?: string; ambiguity?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultObservationInterpretation = { accepted: ObservationInterpretation[]; unresolved: DraftObservationInterpretation[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftObservationLink = { values: { caseId?: ID; observationIds?: ID[]; relation?: 'same-event'|'independent'|'related'|'unknown'; basis?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultObservationLink = { accepted: ObservationLink[]; unresolved: DraftObservationLink[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftObservationQuestion = { values: { caseId?: ID; observationId?: ID; decisiveUnknown?: string; requestedEvidence?: string; recipientId?: ID }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultObservationQuestion = { accepted: ObservationQuestion[]; unresolved: DraftObservationQuestion[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftCuratedObservation = { values: { caseId?: ID; observationId?: ID; acceptedInterpretation?: string; qualityFlags?: string[]; publicLocationPrecision?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultCuratedObservation = { accepted: CuratedObservation[]; unresolved: DraftCuratedObservation[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type RunProposal = { caseId: string; goal: GoalContract; policy: PolicySelection; work: WorkGraph; evidencePlan: EvidencePlan; normalizeObservations: StageResultFieldObservation; resolveTerminology: StageResultObservationInterpretation; detectDuplicates: StageResultObservationLink; requestVerification: StageResultObservationQuestion; publishCurated: StageResultCuratedObservation; collection: CollectionFinding; questions: QuestionBatch; invalidation: InvalidationProposal; review: ReviewPacket; view: ViewModel; revisions: RevisionProposal[]; disposition: "proposed" | "blocked" | "needs-review"; rationale: string[] };

export type CaseStageSet = { normalizeObservations: StageResultFieldObservation; resolveTerminology: StageResultObservationInterpretation; detectDuplicates: StageResultObservationLink; requestVerification: StageResultObservationQuestion; publishCurated: StageResultCuratedObservation; };
