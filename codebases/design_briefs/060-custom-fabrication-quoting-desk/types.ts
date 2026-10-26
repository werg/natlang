export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };

export type PreparedEvidence = { subject: string; evidence: Evidence[]; missing: string[]; conflicts: string[]; alternatives: string[] };
export type DecisionAtom = { field: string; value: string; evidenceIds: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable"; alternative: string };
export type Qualification = "pass" | "fail" | "needs-review";

export type FieldConflict = { field: string; alternatives: string[]; evidenceIds: string[] };
export type EvidenceBinding = { field: string; evidenceIds: string[]; sourceVersions: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable" };

export type Question = { id: string; question: string; respondent: string; acceptableEvidence: string[] };

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

export type CustomQuoteProposalCase = { caseId: ID; customerRequirements: FabricationRequirement[]; measurements: ShopMeasurement[]; concepts: ShopConcept[]; equipment: ShopCapability[]; materials: MaterialOffer[]; finishReferences: FinishReference[]; certificateRequirements: CertificateRequirement[]; acceptedSamples: Ref[]; designRevisions: FabricationRevision[]; evidence: Evidence[]; selectedConceptId: ID | null };

export type ShopMeasurement = { featureId: ID; value: string; unit: string; method: string; tolerance: string; verified: boolean; evidenceRef: Ref };
export type ShopConcept = { conceptId: ID; featureIds: ID[]; materialId: ID; process: string; toolingChange: boolean; finishRef: Ref; certification: string[] };
export type ShopCapability = { process: string; equipmentId: ID; capacity: string; certificationScope: string[]; active: boolean };
export type MaterialOffer = { materialId: ID; unitCost: string; currency: string; validUntil: string; supplierRef: Ref; processCompatible: boolean };
export type FinishReference = { referenceId: Ref; physicalSample: boolean; customerAccepted: boolean; appearanceDescription: string };
export type CertificateRequirement = { scope: string; requiredStandard: string; evidenceRef: Ref | null };
export type FabricationRevision = { revisionId: ID; changedFeatureIds: ID[]; toolingAffected: boolean; customerApproved: boolean };
export type QuoteFeasibility = { missingMeasurements: string[]; conceptBlockers: string[]; capabilityGaps: string[]; finishAcceptanceGaps: string[]; certificationGaps: string[]; expiredMaterialIds: ID[]; provisional: boolean };
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

// A client response is reconciled against a revisioned quote book. The
// response itself is evidence; it cannot replace the authorized commercial
// terms or establish a quote that is stale, incomplete, or provisional.
export type QuoteDecisionBook = { revision: number; quotes: QuoteDecisionRecord[]; responseEvents: CustomerQuoteResponse[]; consumedResponseIds: ID[] };
export type QuoteDecisionRecord = { quoteId: ID; quoteRevision: string; issuedTermDigest: string; status: "issued" | "accepted" | "declined" | "change-requested" | "superseded" | "uncertain"; acceptedTermDigest: string | null; customerResponseId: ID | null; responseEvidence: Ref[]; pendingChanges: string[] };
export type CustomerQuoteResponse = { responseId: ID; quoteId: ID; quoteRevision: string; respondingPartyId: ID; response: "accept" | "decline" | "request-change"; presentedTermDigest: string; evidenceRef: Ref; authorityEvidence: Ref; receivedAt: DateTime };
export type QuoteCurrentFacts = { quoteId: ID; currentRevision: string; currentTermDigest: string; validity: "current" | "expired" | "unknown"; feasibility: "ready" | "blocked" | "unknown"; feasibilityEvidence: Ref[]; allRequiredApprovalsPresent: boolean; approvalRefs: Ref[]; authorizedRespondingPartyIds: ID[]; responseAuthority: "verified" | "invalid" | "unknown" };
export type QuoteDecisionReview = { book: QuoteDecisionBook; outcome: "accepted" | "declined" | "change-requested" | "replayed" | "held"; reasons: string[]; nextTasks: string[] };
export type QuoteApprovalEvidence = { approvalKind: string; evidenceRef: Ref };
export type FabricationQuoteIssue = { quoteId: ID; quoteRevision: string; termDigest: string; renderedAt: DateTime; proposal: CustomQuoteProposal; current: QuoteCurrentFacts; requiredApprovalKinds: string[]; approvalEvidence: QuoteApprovalEvidence[] };
export type QuoteIssueReview = { book: QuoteDecisionBook; outcome: "issued" | "replayed" | "held"; reasons: string[]; nextTasks: string[] };
export type QuoteIssueReadiness = { issuable: boolean; reasons: string[]; approvalRefs: Ref[]; evidenceRefs: Ref[] };
export type QuoteResponseReadiness = { decisionMayBeRecorded: boolean; acceptanceMayBeRecorded: boolean; reasons: string[] };
