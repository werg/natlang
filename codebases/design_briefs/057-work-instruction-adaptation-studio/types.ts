export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };

export type PreparedEvidence = { subject: string; evidence: Evidence[]; missing: string[]; conflicts: string[]; alternatives: string[] };
export type DecisionAtom = { field: string; value: string; evidenceIds: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable"; alternative: string };
export type Qualification = "pass" | "fail" | "needs-review";

export type FieldConflict = { field: string; alternatives: string[]; evidenceIds: string[] };
export type EvidenceBinding = { field: string; evidenceIds: string[]; sourceVersions: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable" };

export type Question = { id: string; question: string; respondent: string; acceptableEvidence: string[] };

export type RecordMeta = { id: ID; version: string; createdAt: DateTime; updatedAt: DateTime };
export type InstructionAdaptation = RecordMeta & {
  masterProcedureId: ID;
  siteId: ID;
  equipmentVariantIds: ID[];
  ownerId: ID;
  intendedAudience: string;
};
export type ProcedureRequirement = RecordMeta & {
  caseId: ID;
  action: string;
  prerequisite: string;
  safetyConstraint: string;
  expectedEvidence: string;
};
export type LocalProcedureConstraint = RecordMeta & {
  caseId: ID;
  siteId: ID;
  equipmentDifference: string;
  localResource: string;
  incompatibility: string;
};
export type InstructionVariant = RecordMeta & {
  caseId: ID;
  requirementIds: ID[];
  adaptedSteps: string[];
  retainedControls: string[];
  deviationIds: ID[];
};
export type ProcedureScenario = RecordMeta & {
  caseId: ID;
  variantId: ID;
  initialCondition: string;
  expectedAction: string;
  counterexample: string;
};
export type InstructionRelease = RecordMeta & {
  caseId: ID;
  variantId: ID;
  approvalRefs: Ref[];
  trainingTaskIds: ID[];
  supersededVersion: string;
};
export type DraftProcedureRequirement = { values: { caseId?: ID; action?: string; prerequisite?: string; safetyConstraint?: string; expectedEvidence?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };

export type InstructionAdaptationCase = { caseId: ID; baseline: ProcedureRequirement; sourceVersion: string; localTerminology: TerminologyEntry[]; translations: WarningTranslation[]; equipmentReferences: EquipmentAlias[]; interruptionRules: RestartCondition[]; localExceptions: LocalException[]; floorCopies: FloorCopy[]; evidence: Evidence[]; changeNote: string };
export type TerminologyEntry = { term: string; meaning: string; locale: string; evidenceRef: Ref };
export type WarningTranslation = { appliesToSteps: ID[]; sourceText: string; renderedText: string; locale: string; preservedHazards: string[]; omittedHazards: string[]; evidenceRef: Ref };
export type EquipmentAlias = { localName: string; assetIds: ID[]; controlIds: ID[]; verifiedRef: Ref | null };
export type RestartCondition = { stepId: ID; interruptionState: string; requiredIsolationChecks: string[]; repeatChecks: string[]; evidenceRef: Ref };
export type LocalException = { stepId: ID; waivedConstraint: string; preservedInspections: string[]; authorityRef: Ref };
export type FloorCopy = { copyId: ID; version: string; location: string; owner: string; withdrawn: boolean };
export type AmbiguitySet = { ambiguousTerms: string[]; equipmentConflicts: string[]; unresolvedRefs: string[] };
export type WarningIntegrity = { safe: boolean; missingHazards: string[]; untranslatedConditions: string[]; evidenceRefs: Ref[]; stepBindings: WarningStepBinding[] };
export type WorkfloorInstructionDraft = { sourceVersion: string; localSteps: string[]; equipmentIds: ID[]; preservedInspectionGates: string[]; restartText: string[]; warningText: string[]; stepWarnings: WarningStepBinding[]; inspectionGuards: InspectionGuardMap[]; restartProtocols: RestartProtocol[]; blockedReason: string[] };
export type ReplayFinding = { passedSteps: string[]; failedSteps: string[]; interruptionFailures: string[]; requiredRevisions: string[] };
export type InstructionReleasePacket = { variant: WorkfloorInstructionDraft; replay: ReplayFinding; withdrawalTasks: string[]; applicableCopyIds: ID[]; releasable: boolean; approvalRequired: boolean };

export type WarningStepBinding = { stepId: ID; translationRefs: Ref[]; missingHazards: string[]; missingConditions: string[]; safeToRelease: boolean };
export type InspectionGuardMap = { stepId: ID; mandatoryInspections: string[]; waivedConstraint: string | null; authorityRef: Ref | null };
export type RestartProtocol = { stepId: ID; resumeState: string; isolationChecks: string[]; repeatVerifications: string[]; incomplete: boolean };
export type FloorCopyReceipt = { receiptId: ID; caseId: ID; copyId: ID; sourceVersion: string; outcome: "withdrawn" | "unavailable" | "replaced" | "unknown"; observedLocation: string; witnessRef: Ref | null; replacementCopy: FloorCopy | null; replacementPostedRef: Ref | null; receivedAt: DateTime };
export type FloorCopyLedger = { revision: number; copies: FloorCopy[]; receipts: FloorCopyReceipt[]; consumedReceiptIds: ID[] };
export type FloorCopyReconciliation = { ledger: FloorCopyLedger; disposition: "recorded" | "replayed" | "held"; activeOldCopyIds: ID[]; followUp: string[] };
export type CopyDispositionAssessment = { valid: boolean; oldCopyWithdrawn: boolean; replacementMayBeAdded: boolean; reasons: string[] };
