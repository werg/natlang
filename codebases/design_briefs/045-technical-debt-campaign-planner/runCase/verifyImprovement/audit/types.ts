export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };
export type CaseInput = { caseId: string; goal: string; evidence: Evidence[]; policy: string[]; authorityStatus: "authorized" | "unclear" | "unauthorized"; changedEvidenceIds: string[]; audience: string; domainRecords: { firstStage: DebtSymptom } };
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
export type DebtCampaign = RecordMeta & {
  repositoryIds: ID[];
  serviceIds: ID[];
  planningHorizon: string;
  sponsorId: ID;
  capacityBudget: number;
};
export type DebtSymptom = RecordMeta & {
  caseId: ID;
  affectedWorkflow: string;
  observation: string;
  frequencyBasis: string;
  evidenceRefs: Ref[];
};
export type DebtMechanism = RecordMeta & {
  caseId: ID;
  symptomIds: ID[];
  proposedCause: string;
  rivalCause: string;
  validationTask: string;
};
export type DebtWorkPackage = RecordMeta & {
  caseId: ID;
  mechanismId: ID;
  boundedChange: string;
  excludedScope: string;
  completionEvidence: string;
};
export type DebtWave = RecordMeta & {
  caseId: ID;
  packageIds: ID[];
  prerequisiteIds: ID[];
  capacityUse: number;
  rollbackPlan: string;
};
export type DebtOutcome = RecordMeta & {
  caseId: ID;
  packageId: ID;
  beforeMeasure: string;
  afterMeasure: string;
  residualSymptoms: string[];
};
export type DraftDebtSymptom = { values: { caseId?: ID; affectedWorkflow?: string; observation?: string; frequencyBasis?: string; evidenceRefs?: Ref[] }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultDebtSymptom = { accepted: DebtSymptom[]; unresolved: DraftDebtSymptom[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftDebtMechanism = { values: { caseId?: ID; symptomIds?: ID[]; proposedCause?: string; rivalCause?: string; validationTask?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultDebtMechanism = { accepted: DebtMechanism[]; unresolved: DraftDebtMechanism[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftDebtWorkPackage = { values: { caseId?: ID; mechanismId?: ID; boundedChange?: string; excludedScope?: string; completionEvidence?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultDebtWorkPackage = { accepted: DebtWorkPackage[]; unresolved: DraftDebtWorkPackage[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftDebtWave = { values: { caseId?: ID; packageIds?: ID[]; prerequisiteIds?: ID[]; capacityUse?: number; rollbackPlan?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultDebtWave = { accepted: DebtWave[]; unresolved: DraftDebtWave[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftDebtOutcome = { values: { caseId?: ID; packageId?: ID; beforeMeasure?: string; afterMeasure?: string; residualSymptoms?: string[] }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultDebtOutcome = { accepted: DebtOutcome[]; unresolved: DraftDebtOutcome[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type RunProposal = { caseId: string; goal: GoalContract; policy: PolicySelection; work: WorkGraph; evidencePlan: EvidencePlan; collectSymptoms: StageResultDebtSymptom; formMechanisms: StageResultDebtMechanism; designWorkPackages: StageResultDebtWorkPackage; sequenceCampaign: StageResultDebtWave; verifyImprovement: StageResultDebtOutcome; collection: CollectionFinding; questions: QuestionBatch; invalidation: InvalidationProposal; review: ReviewPacket; view: ViewModel; revisions: RevisionProposal[]; disposition: "proposed" | "blocked" | "needs-review"; rationale: string[] };

export type CaseStageSet = { collectSymptoms: StageResultDebtSymptom; formMechanisms: StageResultDebtMechanism; designWorkPackages: StageResultDebtWorkPackage; sequenceCampaign: StageResultDebtWave; verifyImprovement: StageResultDebtOutcome; };
