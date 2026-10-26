
export type ID = string; export type Hash = string; export type LocalDate = string; export type DateTime = string; export type Decimal = string; export type Ratio = { numerator: Decimal; denominator: Decimal }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number };
export type Ref = { tenantId: ID; objectId: ID; version: string; hash: Hash };
export type SpanRef = { source: Ref; locatorKind: 'text' | 'table' | 'timecode' | 'region'; locator: string; excerptHash: Hash };

export type FieldConflict = { field: string; alternatives: string[]; evidenceRefs: string[] };

export type ActionProposal = { action: string; target: string; requiresApproval: boolean };

export type Evidence = { id: string; source: string; version: string; observedAt: string; authority: string; scope: string; text: string };
export type EvidencePage = { evidence: Evidence[]; searchedPartition: string; complete: boolean };

export type GoalContract = { outcome: string; exclusions: string[]; entities: string[]; acceptanceEvidence: string[]; unresolvedScope: string[] };
export type PolicyCandidate = { id: string; authority: string; scope: string; effectiveFrom: LocalDate; effectiveTo: LocalDate; supersedes: string[]; clauses: string[] };
export type PolicySelection = { selected: string[]; precedenceQuestions: string[]; applicableScope: string };

export type GateVerdict = 'pass' | 'fail' | 'needs-review';
export type TenderResponse = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; tenderId: string; customerId: string; dueAt: string; responseOwnerId: string; submissionChannel: string };
export type TenderRequirement = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; requirementText: string; responseLocation: string; mandatory: boolean; scoringHint: string };
export type CapabilityMatch = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; requirementId: string; capabilityId: string; match: 'full'|'partial'|'none'|'unverified'; limitation: string };
export type ContributionTask = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; requirementIds: string[]; ownerId: string; requestedEvidence: string; dueAt: string };
export type ResponseSection = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; requirementIds: string[]; text: string; claimRefs: Ref[]; exceptionIds: string[] };
export type SubmissionManifest = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; artifactIds: string[]; requirementIds: string[]; unresolvedIds: string[]; receiptRef: Ref };
export type DraftTenderRequirement = { value: { tenantId?: string; id?: string; version?: number; schemaVersion?: string; state?: string; createdAt?: string; updatedAt?: string; createdBy?: string; sourceRefs?: Ref[]; decisionRefs?: Ref[]; sensitivity?: string[]; purposeIds?: string[]; retentionPolicyId?: string; caseId?: string; requirementText?: string; responseLocation?: string; mandatory?: boolean; scoringHint?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidenceRefs: string[] };

export type BidPreparationRequest = { responseCaseId: string; tenderId: string; requestedBy: string; deadline: string; expectedVersion: string };
export type TenderSourceRoom = { responses: TenderResponse[]; evidencePages: EvidencePage[]; policyCandidates: PolicyCandidate[]; searchedPartitions: string[]; addenda: EvidencePage[]; bundleConstraints: EvidencePage[]; certificates: EvidencePage[]; pricingSchedules: EvidencePage[]; portalReceipts: EvidencePage[]; bundleValidations:PortalBundleValidation[]; };
export type TenderGuardrail = { disposition: string; blockers: string[]; evidenceRefs: string[]; proposedActions: ActionProposal[] };

export type PortalBundleValidation={tenderId:string;responseHash:string;fileSetHash:string;addendumRevision:string;filenames:string[];fileSizes:string[];requiredForms:string[];missingForms:string[];status:'pass'|'fail'|'unknown';validatorId:string;};
export type TenderBundleProposal={caseId:string;responseHash:string;fileSetHash:string;addendumRevision:string;approvalRequired:boolean;portalState:'not-submitted'|'unknown'|'receipt-confirmed';portalReceiptRefs:string[];blockingIssues:string[];};
export type TenderProgramResult={caseId:string;goal:GoalContract;policy:PolicySelection;obligations:TenderRequirement[];capabilities:CapabilityMatch[];contributions:ContributionTask[];response:ResponseSection[];bundle:TenderBundleProposal|null;guardrails:TenderGuardrail[];unresolved:string[];checkpoint:TenderCheckpoint;commitAllowed:boolean};
export type TenderCheckpoint={caseId:string;revision:number;tenderId:string;responseEditions:ResponseSection[][];addendumOrder:string[];bundleHashes:string[];approvalRefs:string[];portalReceiptRefs:string[];unknownSubmissionHashes:string[];staleResponseHashes:string[]};
