
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
export type SubscriptionCase = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; customerId: string; subscriptionIds: string[]; requestedOutcome: string; requestedEffectiveDate: string; consentRef: Ref };
export type ChangeIntent = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; affectedIds: string[]; requestedAction: string; preservedServices: string[]; ambiguity: string };
export type ChangeConsequence = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; subscriptionId: string; accessEnd: string; billingDelta: { minorUnits: string; currency: string; scale: number }; dataConsequence: string };
export type ChangeOption = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; actionIds: string[]; effectiveAt: string; explanation: string; requiresConsent: boolean };
export type SubscriptionOperation = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; selectedOptionId: string; operationId: string; approvalRef: Ref; externalStatus: string };
export type ExitReceipt = { tenantId: ID; id: ID; version: number; schemaVersion: string; state: string; createdAt: DateTime; updatedAt: DateTime; createdBy: ID; sourceRefs: Ref[]; decisionRefs: Ref[]; sensitivity: string[]; purposeIds: ID[]; retentionPolicyId: ID; caseId: string; operationIds: string[]; remainingCharges: string; remainingAccess: string; confirmationRef: Ref };
export type DraftChangeIntent = { value: { tenantId?: string; id?: string; version?: number; schemaVersion?: string; state?: string; createdAt?: string; updatedAt?: string; createdBy?: string; sourceRefs?: Ref[]; decisionRefs?: Ref[]; sensitivity?: string[]; purposeIds?: string[]; retentionPolicyId?: string; caseId?: string; affectedIds?: string[]; requestedAction?: string; preservedServices?: string[]; ambiguity?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidenceRefs: string[] };

export type SubscriptionExitRequest = { caseId: string; customerId: string; subscriptionIds: string[]; requestedOutcome: string; requestedEffectiveDate: string; requestedBy: string; expectedVersion: string };
export type ExitServiceSnapshot = { subscriptions: SubscriptionCase[]; evidencePages: EvidencePage[]; policyCandidates: PolicyCandidate[]; searchedPartitions: string[]; entitlementReceipts: EvidencePage[]; billingReceipts: EvidencePage[]; mutationReceipts: EvidencePage[] };
export type ExitGuardrail = { decisions: string[]; blockers: string[]; evidenceRefs: string[]; proposedActions: ActionProposal[] };

export type ExitProgramResult={caseId:string;scope:GoalContract;policy:PolicySelection;intent:ChangeIntent|null;consequences:ChangeConsequence[];choices:ChangeOption[];operation:SubscriptionOperation|null;receipt:ExitReceipt|null;guardrails:ExitGuardrail[];unresolved:string[];checkpoint:ExitCheckpoint;commitAllowed:boolean};
export type ExitCheckpoint={caseId:string;revision:number;requestedSubscriptionIds:string[];consentReceiptRefs:string[];operationReceiptRefs:string[];billingReceiptRefs:string[];completedChanges:string[];unknownOperationIds:string[];pendingRefundRefs:string[];historicalChoices:ChangeOption[]};
