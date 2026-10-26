export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };
export type CaseInput = { caseId: string; goal: string; evidence: Evidence[]; policy: string[]; authorityStatus: "authorized" | "unclear" | "unauthorized"; changedEvidenceIds: string[]; audience: string; domainRecords: { firstStage: ProcedureRequirement } };
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
export type InstructionAdaptation = RecordMeta & {
  masterProcedureId: ID;
  siteId: ID;
  equipmentVariantIds: ID[];
  ownerId: ID;
  intendedAudience: string;
};
export type ProcedureRequirement = RecordMeta & {
  caseId: ID;
  action: string;
  prerequisite: string;
  safetyConstraint: string;
  expectedEvidence: string;
};
export type LocalProcedureConstraint = RecordMeta & {
  caseId: ID;
  siteId: ID;
  equipmentDifference: string;
  localResource: string;
  incompatibility: string;
};
export type InstructionVariant = RecordMeta & {
  caseId: ID;
  requirementIds: ID[];
  adaptedSteps: string[];
  retainedControls: string[];
  deviationIds: ID[];
};
export type ProcedureScenario = RecordMeta & {
  caseId: ID;
  variantId: ID;
  initialCondition: string;
  expectedAction: string;
  counterexample: string;
};
export type InstructionRelease = RecordMeta & {
  caseId: ID;
  variantId: ID;
  approvalRefs: Ref[];
  trainingTaskIds: ID[];
  supersededVersion: string;
};
export type DraftProcedureRequirement = { values: { caseId?: ID; action?: string; prerequisite?: string; safetyConstraint?: string; expectedEvidence?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultProcedureRequirement = { accepted: ProcedureRequirement[]; unresolved: DraftProcedureRequirement[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftLocalProcedureConstraint = { values: { caseId?: ID; siteId?: ID; equipmentDifference?: string; localResource?: string; incompatibility?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultLocalProcedureConstraint = { accepted: LocalProcedureConstraint[]; unresolved: DraftLocalProcedureConstraint[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftInstructionVariant = { values: { caseId?: ID; requirementIds?: ID[]; adaptedSteps?: string[]; retainedControls?: string[]; deviationIds?: ID[] }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultInstructionVariant = { accepted: InstructionVariant[]; unresolved: DraftInstructionVariant[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftProcedureScenario = { values: { caseId?: ID; variantId?: ID; initialCondition?: string; expectedAction?: string; counterexample?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultProcedureScenario = { accepted: ProcedureScenario[]; unresolved: DraftProcedureScenario[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftInstructionRelease = { values: { caseId?: ID; variantId?: ID; approvalRefs?: Ref[]; trainingTaskIds?: ID[]; supersededVersion?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultInstructionRelease = { accepted: InstructionRelease[]; unresolved: DraftInstructionRelease[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type RunProposal = { caseId: string; goal: GoalContract; policy: PolicySelection; work: WorkGraph; evidencePlan: EvidencePlan; recoverProcedure: StageResultProcedureRequirement; mapLocalContext: StageResultLocalProcedureConstraint; constructVariant: StageResultInstructionVariant; replayProcedure: StageResultProcedureScenario; releaseInstruction: StageResultInstructionRelease; collection: CollectionFinding; questions: QuestionBatch; invalidation: InvalidationProposal; review: ReviewPacket; view: ViewModel; revisions: RevisionProposal[]; disposition: "proposed" | "blocked" | "needs-review"; rationale: string[] };

export type CaseStageSet = { recoverProcedure: StageResultProcedureRequirement; mapLocalContext: StageResultLocalProcedureConstraint; constructVariant: StageResultInstructionVariant; replayProcedure: StageResultProcedureScenario; releaseInstruction: StageResultInstructionRelease; };
