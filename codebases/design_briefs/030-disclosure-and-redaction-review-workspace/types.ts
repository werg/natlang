
export type ID = string; export type Hash = string; export type LocalDate = string; export type DateTime = string; export type Decimal = string; export type Ratio = { numerator: Decimal; denominator: Decimal }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number };
export type Ref = { tenantId: ID; objectId: ID; version: string; hash: Hash };
export type SpanRef = { source: Ref; locatorKind: 'text' | 'table' | 'timecode' | 'region'; locator: string; excerptHash: Hash };

export type FieldConflict = { field: string; alternatives: string[]; evidenceRefs: string[] };

export type Evidence = { id: string; source: string; version: string; observedAt: string; authority: string; scope: string; text: string };
export type EvidencePage = { evidence: Evidence[]; searchedPartition: string; complete: boolean };

export type PolicyCandidate = { id: string; authority: string; scope: string; effectiveFrom: LocalDate; effectiveTo: LocalDate; supersedes: string[]; clauses: string[] };
export type PolicySelection = { selected: string[]; precedenceQuestions: string[]; applicableScope: string };

export type GateVerdict = 'pass' | 'fail' | 'needs-review';
export type DisclosureCase = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; recipientId: string; purpose: string; documentIds: string[]; authorityRef: Ref; reviewOwnerId: string };
export type DisclosureScope = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; requestedTopics: string[]; dateWindow: string; permittedCategories: string[]; excludedCategories: string[] };
export type DisclosureSelection = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; documentId: string; relevantSpans: SpanRef[]; excludedReason: string; partialDisclosure: boolean };
export type RedactionProposal = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; documentId: string; spanRefs: SpanRef[]; reasonCode: string; retainedMeaning: string };
export type DisclosureRisk = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; artifactIds: string[]; combinedInference: string; restrictedFact: string; mitigation: string };
export type DisclosureManifest = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; artifactIds: string[]; redactionIds: string[]; reviewerIds: string[]; recipientBinding: string };
export type DraftDisclosureScope = { value: { tenantId?: string; id?: string; version?: number; schemaVersion?: string; state?: string; createdAt?: string; updatedAt?: string; createdBy?: string; sourceRefs?: Ref[]; decisionRefs?: Ref[]; sensitivity?: string[]; purposeIds?: string[]; retentionPolicyId?: string; caseId?: string; requestedTopics?: string[]; dateWindow?: string; permittedCategories?: string[]; excludedCategories?: string[] }; missingFields: string[]; conflicts: FieldConflict[]; evidenceRefs: string[] };

export type DisclosureReviewRequest = { caseId: string; recipientId: string; purpose: string; documentIds: string[]; reviewOwnerId: string; expectedVersion: string };
export type DisclosureEvidencePack = { checkpoint?: DisclosureCheckpoint; cases: DisclosureCase[]; evidencePages: EvidencePage[]; policyCandidates: PolicyCandidate[]; searchedPartitions: string[]; documentInspections: DocumentInspection[]; redactionReceipts: RedactionReceipt[]; renderReceipts: RenderReceipt[]; recipientApprovals: RecipientApproval[]; challengeMappings: EvidencePage[] };
export type DocumentInspection = { documentId: string; sourceHash: string; textLayers: string[]; embeddedMetadata: string[]; ocrInspectionRef: string; evidenceRefs: string[] };
export type RedactionReceipt = { documentId: string; inputHash: string; outputHash: string; redactionMapRef: string; engineVersion: string; textLayerSanitized: boolean; metadataSanitized: boolean };
export type RenderReceipt = { documentId: string; outputHash: string; renderedHash: string; extractedTextHash: string; visualReviewRef: string; residualFindings: string[] };
export type RecipientApproval = { recipientId: string; purpose: string; artifactHash: string; approvalRef: string; version: string };
export type DisclosureScopeDecision = { permittedTopics: string[]; excludedTopics: string[]; dateWindow: string; blockers: string[]; authorityRefs: string[] };
export type SpanSelection = { documentId: string; sourceHash: string; permittedSpans: string[]; excludedSpans: string[]; reasonBySpan: string[]; sourceRefs: string[] };
export type RedactionPlan = { spanSelections: SpanSelection[]; sensitiveFields: string[]; necessaryQualifiers: string[]; inferenceRisks: string[]; proposedRedactions: string[] };
export type CombinedInferencePathway = { artifactIds: string[]; restrictedFact: string; combinationLogic: string; mitigation: string; evidenceRefs: string[] };
export type ReleaseArtifact = { documentId: string; originalHash: string; sanitizedHash: string; renderedHash: string; extractedTextHash: string; redactionMapRef: string; recipientId: string; purpose: string };
export type DisclosureReleaseProposal = { caseId: string; recipientId: string; purpose: string; scope: DisclosureScopeDecision; plan: RedactionPlan; artifacts: ReleaseArtifact[]; withheldDocuments: string[]; unresolvedChallenges: string[]; readyForAuthorizedRelease: boolean; checkpoint: DisclosureCheckpoint };

export type DisclosureCheckpoint={caseId:string;revision:number;recipientId:string;purpose:string;scopeEditions:DisclosureScopeDecision[];artifactHashes:string[];redactionReceiptIds:string[];renderReceiptIds:string[];approvalReceiptIds:string[];dispatchReceiptIds:string[];withheldDocumentIds:string[];reopenedChallengeIds:string[]};

export type RecipientApprovalReview = { authorizedArtifactHashes: string[]; approvalRefs: string[]; blockers: string[] };
export type DisclosureChallengeMapping = { challengeRef: string; documentId: string; artifactHash: string; sourceRefs: string[]; redactionReason: string; approvalRefs: string[] };
export type DisclosureChallengeMap = { mappings: DisclosureChallengeMapping[]; withheldDocumentIds: string[]; unresolvedChallenges: string[] };
