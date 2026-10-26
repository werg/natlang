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

export type CustomQuoteProposalCase = { caseId: ID; customerRequirements: FabricationRequirement[]; measurements: ShopMeasurement[]; concepts: ShopConcept[]; equipment: ShopCapability[]; materials: MaterialOffer[]; finishReferences: FinishReference[]; certificateRequirements: CertificateRequirement[]; acceptedSamples: Ref[]; designRevisions: FabricationRevision[]; evidence: Evidence[]; selectedConceptId: ID | null };
export type QuoteRequirementRecord = { requirementId: ID; function: string; dimension: string; tolerance: string; finish: string; requiredCertificate: string; approvalRef: Ref | null };
export type ShopMeasurement = { featureId: ID; value: string; unit: string; method: string; tolerance: string; verified: boolean; evidenceRef: Ref };
export type ShopConcept = { conceptId: ID; featureIds: ID[]; materialId: ID; process: string; toolingChange: boolean; finishRef: Ref; certification: string[] };
export type ShopCapability = { process: string; equipmentId: ID; capacity: string; certificationScope: string[]; active: boolean };
export type MaterialOffer = { materialId: ID; unitCost: string; currency: string; validUntil: string; supplierRef: Ref; processCompatible: boolean };
export type FinishReference = { referenceId: Ref; physicalSample: boolean; customerAccepted: boolean; appearanceDescription: string };
export type CertificateRequirement = { scope: string; requiredStandard: string; evidenceRef: Ref | null };
export type FabricationRevision = { revisionId: ID; changedFeatureIds: ID[]; toolingAffected: boolean; customerApproved: boolean };
export type QuoteFeasibility = { missingMeasurements: string[]; conceptBlockers: string[]; capabilityGaps: string[]; finishAcceptanceGaps: string[]; certificationGaps: string[]; provisional: boolean };
export type CustomQuoteProposal = { caseId: ID; selectedConceptId: ID | null; feasibility: QuoteFeasibility; productionRoute: string[]; materialCostBasis: string[]; toolingCostChanges: string[]; quoteExpiry: string; quoteStatus: string; scopeConfirmation: string[]; commitAllowed: boolean };
export type MeasurementQualification = { acceptedMeasurements: ShopMeasurement[]; missingMeasurements: string[]; unverifiedMeasurementIds: ID[]; toleranceConflicts: string[] };
export type ConceptSelection = { conceptId: ID | null; blockers: string[]; finishAcceptanceGaps: string[]; concept: ShopConcept | null };
export type ShopRouteAssessment = { steps: string[]; capabilityGaps: string[]; certificationGaps: string[]; specialistTasks: string[]; scopedCertificates: string[] };
export type MaterialPriceReview = { costBasis: string[]; expiredMaterialIds: ID[]; toolingChanges: string[]; toolingRouteImpact: string[]; staleCostLines: ID[]; earliestExpiry: string };

export type MeasurementTrace = { matches: string[]; unverifiedIds: ID[]; missingFeatures: string[] };
export type FinishAcceptanceReview = { acceptedReferenceIds: Ref[]; unacceptedConceptIds: ID[]; unresolvedConceptIds: ID[] };
export type CapabilityMatch = { supportedProcesses: string[]; unsupportedProcesses: string[]; routeSteps: string[] };
export type CertificationMatch = { acceptedScopes: string[]; missingScopes: string[]; specialistTasks: string[] };
export type CurrentMaterialOffers = { acceptedOffers: MaterialOffer[]; expiredMaterialIds: ID[]; incompatibleOfferIds: ID[] };
export type ToolingRevisionImpact = { toolingChanges: string[]; routeChanges: string[]; staleCostLineIds: ID[] };
