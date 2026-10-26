export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };
export type CaseInput = { caseId: string; goal: string; evidence: Evidence[]; policy: string[]; authorityStatus: "authorized" | "unclear" | "unauthorized"; changedEvidenceIds: string[]; audience: string; domainRecords: { firstStage: FabricationRequirement } };
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
export type FabricationQuote = RecordMeta & {
  customerId: ID;
  requestedObject: string;
  intendedUse: string;
  dueDate: LocalDate;
  estimatorId: ID;
};
export type FabricationRequirement = RecordMeta & {
  caseId: ID;
  function: string;
  dimensions: string;
  appearancePreference: string;
  essentialConstraint: string;
};
export type FabricationConcept = RecordMeta & {
  caseId: ID;
  requirementIds: ID[];
  materialChoice: string;
  constructionApproach: string;
  tradeoff: string;
};
export type FabricationRoute = RecordMeta & {
  caseId: ID;
  conceptId: ID;
  operationIds: ID[];
  resourceIds: ID[];
  inspectionCriteria: string[];
};
export type FabricationEstimate = RecordMeta & {
  caseId: ID;
  routeId: ID;
  materialCost: Money;
  laborCost: Money;
  contingencyBasis: string;
};
export type FabricationAgreement = RecordMeta & {
  caseId: ID;
  conceptId: ID;
  estimateId: ID;
  acceptedAssumptions: string[];
  approvalRef: Ref;
};
export type DraftFabricationRequirement = { values: { caseId?: ID; function?: string; dimensions?: string; appearancePreference?: string; essentialConstraint?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultFabricationRequirement = { accepted: FabricationRequirement[]; unresolved: DraftFabricationRequirement[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftFabricationConcept = { values: { caseId?: ID; requirementIds?: ID[]; materialChoice?: string; constructionApproach?: string; tradeoff?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultFabricationConcept = { accepted: FabricationConcept[]; unresolved: DraftFabricationConcept[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftFabricationRoute = { values: { caseId?: ID; conceptId?: ID; operationIds?: ID[]; resourceIds?: ID[]; inspectionCriteria?: string[] }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultFabricationRoute = { accepted: FabricationRoute[]; unresolved: DraftFabricationRoute[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftFabricationEstimate = { values: { caseId?: ID; routeId?: ID; materialCost?: Money; laborCost?: Money; contingencyBasis?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultFabricationEstimate = { accepted: FabricationEstimate[]; unresolved: DraftFabricationEstimate[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftFabricationAgreement = { values: { caseId?: ID; conceptId?: ID; estimateId?: ID; acceptedAssumptions?: string[]; approvalRef?: Ref }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultFabricationAgreement = { accepted: FabricationAgreement[]; unresolved: DraftFabricationAgreement[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type RunProposal = { caseId: string; goal: GoalContract; policy: PolicySelection; work: WorkGraph; evidencePlan: EvidencePlan; recoverRequirements: StageResultFabricationRequirement; generateConcepts: StageResultFabricationConcept; planProduction: StageResultFabricationRoute; calculateQuote: StageResultFabricationEstimate; confirmScope: StageResultFabricationAgreement; collection: CollectionFinding; questions: QuestionBatch; invalidation: InvalidationProposal; review: ReviewPacket; view: ViewModel; revisions: RevisionProposal[]; disposition: "proposed" | "blocked" | "needs-review"; rationale: string[] };

export type CaseStageSet = { recoverRequirements: StageResultFabricationRequirement; generateConcepts: StageResultFabricationConcept; planProduction: StageResultFabricationRoute; calculateQuote: StageResultFabricationEstimate; confirmScope: StageResultFabricationAgreement; };
