export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };
export type CaseInput = { caseId: string; goal: string; evidence: Evidence[]; policy: string[]; authorityStatus: "authorized" | "unclear" | "unauthorized"; changedEvidenceIds: string[]; audience: string; domainRecords: { firstStage: SurplusOffer } };
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
export type SurplusExchange = RecordMeta & {
  offerIds: ID[];
  needIds: ID[];
  coordinatorId: ID;
  exchangeRegion: string;
  allocationPolicyId: ID;
};
export type SurplusOffer = RecordMeta & {
  caseId: ID;
  ownerId: ID;
  materialFunction: string;
  quantity: Quantity;
  condition: string;
  transferRestrictions: string[];
};
export type ReuseNeed = RecordMeta & {
  caseId: ID;
  recipientId: ID;
  desiredFunction: string;
  essentialProperties: string[];
  adaptationTolerance: string;
};
export type ReuseMatch = RecordMeta & {
  caseId: ID;
  offerIds: ID[];
  needId: ID;
  reuseConcept: string;
  requiredAdaptations: string[];
};
export type ExchangePackage = RecordMeta & {
  caseId: ID;
  matchIds: ID[];
  transferLegIds: ID[];
  reservedQuantities: Quantity[];
  logisticsCost: Money;
};
export type ReuseReceipt = RecordMeta & {
  caseId: ID;
  packageId: ID;
  custodyRefs: Ref[];
  recipientAcceptance: Ref;
  residualIssue: string;
};
export type DraftSurplusOffer = { values: { caseId?: ID; ownerId?: ID; materialFunction?: string; quantity?: Quantity; condition?: string; transferRestrictions?: string[] }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultSurplusOffer = { accepted: SurplusOffer[]; unresolved: DraftSurplusOffer[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftReuseNeed = { values: { caseId?: ID; recipientId?: ID; desiredFunction?: string; essentialProperties?: string[]; adaptationTolerance?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultReuseNeed = { accepted: ReuseNeed[]; unresolved: DraftReuseNeed[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftReuseMatch = { values: { caseId?: ID; offerIds?: ID[]; needId?: ID; reuseConcept?: string; requiredAdaptations?: string[] }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultReuseMatch = { accepted: ReuseMatch[]; unresolved: DraftReuseMatch[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftExchangePackage = { values: { caseId?: ID; matchIds?: ID[]; transferLegIds?: ID[]; reservedQuantities?: Quantity[]; logisticsCost?: Money }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultExchangePackage = { accepted: ExchangePackage[]; unresolved: DraftExchangePackage[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftReuseReceipt = { values: { caseId?: ID; packageId?: ID; custodyRefs?: Ref[]; recipientAcceptance?: Ref; residualIssue?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultReuseReceipt = { accepted: ReuseReceipt[]; unresolved: DraftReuseReceipt[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type RunProposal = { caseId: string; goal: GoalContract; policy: PolicySelection; work: WorkGraph; evidencePlan: EvidencePlan; interpretOffers: StageResultSurplusOffer; interpretNeeds: StageResultReuseNeed; generateReuse: StageResultReuseMatch; assembleExchange: StageResultExchangePackage; verifyTransfer: StageResultReuseReceipt; collection: CollectionFinding; questions: QuestionBatch; invalidation: InvalidationProposal; review: ReviewPacket; view: ViewModel; revisions: RevisionProposal[]; disposition: "proposed" | "blocked" | "needs-review"; rationale: string[] };

export type CaseStageSet = { interpretOffers: StageResultSurplusOffer; interpretNeeds: StageResultReuseNeed; generateReuse: StageResultReuseMatch; assembleExchange: StageResultExchangePackage; verifyTransfer: StageResultReuseReceipt; };
