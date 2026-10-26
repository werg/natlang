export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };
export type CaseInput = { caseId: string; goal: string; evidence: Evidence[]; policy: string[]; authorityStatus: "authorized" | "unclear" | "unauthorized"; changedEvidenceIds: string[]; audience: string; domainRecords: { firstStage: SiteInstruction } };
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
export type ChangeOrderCase = RecordMeta & {
  projectId: ID;
  instructionIds: ID[];
  baselineScopeId: ID;
  coordinatorId: ID;
  affectedArea: string;
};
export type SiteInstruction = RecordMeta & {
  caseId: ID;
  issuerId: ID;
  requestedChange: string;
  issuedAt: DateTime;
  authorityScope: string;
};
export type ScopeDifference = RecordMeta & {
  caseId: ID;
  instructionId: ID;
  baselineRequirement: string;
  proposedRequirement: string;
  changeKind: string;
};
export type ConstructionImpact = RecordMeta & {
  caseId: ID;
  differenceId: ID;
  affectedTaskIds: ID[];
  interfaceIds: ID[];
  scheduleConsequence: string;
};
export type ChangeOption = RecordMeta & {
  caseId: ID;
  impactIds: ID[];
  proposedWork: string;
  costBasis: string;
  scheduleAlternative: string;
};
export type ChangePackage = RecordMeta & {
  caseId: ID;
  optionId: ID;
  evidenceRefs: Ref[];
  approvalRefs: Ref[];
  residualDispute: string;
};
export type DraftSiteInstruction = { values: { caseId?: ID; issuerId?: ID; requestedChange?: string; issuedAt?: DateTime; authorityScope?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultSiteInstruction = { accepted: SiteInstruction[]; unresolved: DraftSiteInstruction[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftScopeDifference = { values: { caseId?: ID; instructionId?: ID; baselineRequirement?: string; proposedRequirement?: string; changeKind?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultScopeDifference = { accepted: ScopeDifference[]; unresolved: DraftScopeDifference[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftConstructionImpact = { values: { caseId?: ID; differenceId?: ID; affectedTaskIds?: ID[]; interfaceIds?: ID[]; scheduleConsequence?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultConstructionImpact = { accepted: ConstructionImpact[]; unresolved: DraftConstructionImpact[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftChangeOption = { values: { caseId?: ID; impactIds?: ID[]; proposedWork?: string; costBasis?: string; scheduleAlternative?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultChangeOption = { accepted: ChangeOption[]; unresolved: DraftChangeOption[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftChangePackage = { values: { caseId?: ID; optionId?: ID; evidenceRefs?: Ref[]; approvalRefs?: Ref[]; residualDispute?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultChangePackage = { accepted: ChangePackage[]; unresolved: DraftChangePackage[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type RunProposal = { caseId: string; goal: GoalContract; policy: PolicySelection; work: WorkGraph; evidencePlan: EvidencePlan; reconstructInstruction: StageResultSiteInstruction; compareScope: StageResultScopeDifference; traceDependencies: StageResultConstructionImpact; formOptions: StageResultChangeOption; assembleChange: StageResultChangePackage; collection: CollectionFinding; questions: QuestionBatch; invalidation: InvalidationProposal; review: ReviewPacket; view: ViewModel; revisions: RevisionProposal[]; disposition: "proposed" | "blocked" | "needs-review"; rationale: string[] };

export type CaseStageSet = { reconstructInstruction: StageResultSiteInstruction; compareScope: StageResultScopeDifference; traceDependencies: StageResultConstructionImpact; formOptions: StageResultChangeOption; assembleChange: StageResultChangePackage; };
