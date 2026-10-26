export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };
export type CaseInput = { caseId: string; goal: string; evidence: Evidence[]; policy: string[]; authorityStatus: "authorized" | "unclear" | "unauthorized"; changedEvidenceIds: string[]; audience: string; domainRecords: { firstStage: DesignDelta } };
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
export type EngineeringChange = RecordMeta & {
  changeId: ID;
  affectedDesignIds: ID[];
  effectiveRule: string;
  engineeringOwnerId: ID;
  baselineRevision: string;
};
export type DesignDelta = RecordMeta & {
  caseId: ID;
  designId: ID;
  changedFunction: string;
  changedInterface: string;
  preservedRequirements: string[];
};
export type ChangeApplicability = RecordMeta & {
  caseId: ID;
  deltaId: ID;
  serialRanges: string[];
  lotIds: ID[];
  grandfatheredCases: string[];
};
export type EngineeringImpact = RecordMeta & {
  caseId: ID;
  deltaId: ID;
  affectedItemIds: ID[];
  affectedDocumentIds: ID[];
  indirectReason: string;
};
export type ChangeWorkPackage = RecordMeta & {
  caseId: ID;
  impactIds: ID[];
  taskIds: ID[];
  stockDisposition: string;
  prerequisiteIds: ID[];
};
export type ChangeVerification = RecordMeta & {
  caseId: ID;
  packageId: ID;
  configurationIds: ID[];
  verificationRefs: Ref[];
  residualExceptions: string[];
};
export type DraftDesignDelta = { values: { caseId?: ID; designId?: ID; changedFunction?: string; changedInterface?: string; preservedRequirements?: string[] }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultDesignDelta = { accepted: DesignDelta[]; unresolved: DraftDesignDelta[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftChangeApplicability = { values: { caseId?: ID; deltaId?: ID; serialRanges?: string[]; lotIds?: ID[]; grandfatheredCases?: string[] }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultChangeApplicability = { accepted: ChangeApplicability[]; unresolved: DraftChangeApplicability[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftEngineeringImpact = { values: { caseId?: ID; deltaId?: ID; affectedItemIds?: ID[]; affectedDocumentIds?: ID[]; indirectReason?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultEngineeringImpact = { accepted: EngineeringImpact[]; unresolved: DraftEngineeringImpact[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftChangeWorkPackage = { values: { caseId?: ID; impactIds?: ID[]; taskIds?: ID[]; stockDisposition?: string; prerequisiteIds?: ID[] }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultChangeWorkPackage = { accepted: ChangeWorkPackage[]; unresolved: DraftChangeWorkPackage[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftChangeVerification = { values: { caseId?: ID; packageId?: ID; configurationIds?: ID[]; verificationRefs?: Ref[]; residualExceptions?: string[] }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultChangeVerification = { accepted: ChangeVerification[]; unresolved: DraftChangeVerification[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type RunProposal = { caseId: string; goal: GoalContract; policy: PolicySelection; work: WorkGraph; evidencePlan: EvidencePlan; interpretChange: StageResultDesignDelta; resolveApplicability: StageResultChangeApplicability; traceDependencies: StageResultEngineeringImpact; planImplementation: StageResultChangeWorkPackage; verifyConfiguration: StageResultChangeVerification; collection: CollectionFinding; questions: QuestionBatch; invalidation: InvalidationProposal; review: ReviewPacket; view: ViewModel; revisions: RevisionProposal[]; disposition: "proposed" | "blocked" | "needs-review"; rationale: string[] };

export type CaseStageSet = { interpretChange: StageResultDesignDelta; resolveApplicability: StageResultChangeApplicability; traceDependencies: StageResultEngineeringImpact; planImplementation: StageResultChangeWorkPackage; verifyConfiguration: StageResultChangeVerification; };
