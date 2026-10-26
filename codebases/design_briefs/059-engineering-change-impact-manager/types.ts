export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };

export type PreparedEvidence = { subject: string; evidence: Evidence[]; missing: string[]; conflicts: string[]; alternatives: string[] };
export type DecisionAtom = { field: string; value: string; evidenceIds: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable"; alternative: string };
export type Qualification = "pass" | "fail" | "needs-review";

export type FieldConflict = { field: string; alternatives: string[]; evidenceIds: string[] };
export type EvidenceBinding = { field: string; evidenceIds: string[]; sourceVersions: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable" };

export type Question = { id: string; question: string; respondent: string; acceptableEvidence: string[] };

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

export type EngineeringChangeDossier = { caseId: ID; change: DesignDelta; baselineConfiguration: ConfigurationState[]; inventory: PartInventory[]; fieldedUnits: FieldedConfiguration[]; supplierShipments: SupplierShipment[]; retrofitProgress: RetrofitProgress[]; dependencyEdges: ConfigurationDependency[]; evidence: Evidence[] };
export type ConfigurationState = { configurationId: ID; designRevision: string; serialRange: string; approvedRevision: string; grandfathered: boolean; evidenceRefs: Ref[] };
export type PartInventory = { partId: ID; revision: string; quantity: number; intendedConfigurations: ID[]; disposition: string };
export type FieldedConfiguration = { unitId: ID; configurationId: ID; installedPartRevision: string; serviceStatus: string; maintenanceRefs: Ref[] };
export type SupplierShipment = { shipmentId: ID; partRevision: string; quantity: number; received: boolean; ownerDecision: string };
export type RetrofitProgress = { unitId: ID; fromConfiguration: ID; targetConfiguration: ID; completedSteps: string[]; verifiedSteps: string[] };
export type ConfigurationDependency = { sourceId: ID; targetId: ID; dependencyKind: string; evidenceRef: Ref };
export type ApplicabilityPartition = { changedConfigurations: ID[]; preservedGrandfatheredUnits: ID[]; unresolvedConfigurations: ID[]; documentationOnly: boolean; affectedDependencies: ConfigurationDependency[]; closureEvidence: ID[]; closureBlockers: string[] };
export type InventoryDisposition = { useByConfiguration: string[]; quarantineShipments: ID[]; fieldedOldRevisionUnits: ID[]; unaffectedStock: string[]; unresolvedItems: string[]; unresolvedShipments: ID[] };
export type RetrofitFence = { completeUnitIds: ID[]; partialUnitIds: ID[]; mixedStateConstraints: string[]; nextStepsByUnit: string[] };
export type EngineeringChangePlan = { changeId: ID; applicability: ApplicabilityPartition; inventory: InventoryDisposition; retrofit: RetrofitFence; affectedDownstream: ID[]; verificationRequirements: string[]; planDisposition: string; commitAllowed: boolean };

export type ConfigSeedSet = { directConfigurationIds: ID[]; unresolvedConfigurationIds: ID[]; documentationOnly: boolean };
export type ConfigImpactClosure = { changedConfigurationIds: ID[]; preservedConfigurationIds: ID[]; traversalEvidence: ID[]; missingDependencyEvidence: string[] };
export type ShipmentDispositionReview = { unresolvedShipmentIds: ID[]; oldRevisionUnitIds: ID[]; dispositionTasks: string[] };
export type RetrofitExecutionReceipt = { receiptId: ID; changeId: ID; unitId: ID; fromConfiguration: ID; targetConfiguration: ID; stepId: ID; outcome: "completed" | "failed" | "unknown" | "reversed"; installedRevision: string; verificationRefs: Ref[]; reversalRefs: Ref[]; observedAt: DateTime };
export type RetrofitUnitState = { unitId: ID; currentConfigurationId: ID | null; state: "baseline" | "mixed" | "target" | "unknown"; verifiedStepIds: ID[]; blockers: string[] };
export type RetrofitExecutionBook = { revision: number; units: RetrofitUnitState[]; receipts: RetrofitExecutionReceipt[]; consumedReceiptIds: ID[] };
export type RetrofitExecutionAssessment = { unitState: "baseline" | "mixed" | "target" | "unknown"; advanceAllowed: boolean; reasons: string[] };
export type RetrofitExecutionReconciliation = { book: RetrofitExecutionBook; outcome: "recorded" | "replayed" | "held"; incompleteStepIds: ID[]; nextSteps: string[] };
