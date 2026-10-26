export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };

export type PreparedEvidence = { subject: string; evidence: Evidence[]; missing: string[]; conflicts: string[]; alternatives: string[] };
export type DecisionAtom = { field: string; value: string; evidenceIds: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable"; alternative: string };
export type Qualification = "pass" | "fail" | "needs-review";

export type FieldConflict = { field: string; alternatives: string[]; evidenceIds: string[] };
export type EvidenceBinding = { field: string; evidenceIds: string[]; sourceVersions: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable" };

export type Question = { id: string; question: string; respondent: string; acceptableEvidence: string[] };

export type RecordMeta = { id: ID; version: string; createdAt: DateTime; updatedAt: DateTime };
export type MigrationProject = RecordMeta & {
  sourceSystemIds: ID[];
  targetSchemaId: ID;
  migrationOwnerId: ID;
  cutoverWindow: string;
  rollbackPolicyId: ID;
};
export type SourceFieldMeaning = RecordMeta & {
  caseId: ID;
  systemId: ID;
  field: string;
  meaning: string;
  missingStates: string[];
  conditionalUse: string;
};
export type MigrationMapping = RecordMeta & {
  caseId: ID;
  sourceFields: string[];
  targetField: string;
  condition: string;
  transformation: string;
};
export type MigrationDecision = RecordMeta & {
  caseId: ID;
  sourceRecordId: ID;
  mappingIds: ID[];
  targetDraft: string;
  unresolvedFields: string[];
};
export type MigrationRehearsal = RecordMeta & {
  caseId: ID;
  manifestId: ID;
  targetRecordIds: ID[];
  invariantFailures: string[];
  rollbackArtifactId: ID;
};
export type MigrationCutover = RecordMeta & {
  caseId: ID;
  rehearsalId: ID;
  approvedHash: string;
  finalDeltaIds: ID[];
  reconciliationStatus: string;
};
export type DraftSourceFieldMeaning = { values: { caseId?: ID; systemId?: ID; field?: string; meaning?: string; missingStates?: string[]; conditionalUse?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };

export type DomainMigrationRequest = { migrationId: ID; sourceEvidence: Evidence[]; priorSourceMeaning: SourceFieldMeaning; targetSchema: string; targetSchemaFields: string[]; sourceRecords: SourceRecord[]; sourceDeltas: Evidence[]; authorityStatus: 'authorized' | 'unclear' | 'unauthorized'; approvedMappingVersion: string | null; targetWriteEvidence: Evidence[]; rollbackPolicy: string[] };
export type SourceMeaningDecision = { sourceSystemId: ID; field: string; conditionalMeaning: string; missingStates: string[]; supportingRefs: ID[]; conflictingRefs: ID[]; status: 'established' | 'review' | 'unknown' };
export type DomainFieldMapping = { mappingId: ID; sourceSystemId: ID; sourceFields: string[]; validityCondition: string; targetField: string; transformation: string; missingStateMap: string[]; informationLoss: string[]; approved: boolean };
export type MissingStateMapping = { sourceState: string; targetState: string; collision: boolean; ownerApprovalRef: ID | null };
export type SourceRecord = { sourceSystemId: ID; rawId: ID; validFrom: string; validTo: string; values: string[]; relationshipRefs: ID[] };
export type HistoricalIdentityFinding = { identityKey: string; recordIds: ID[]; reusedId: boolean; overlappingHistories: boolean; ownerReviewNeeded: boolean };
export type SourceRecordDisposition = { sourceId: string; status: 'accepted' | 'quarantined' | 'rejected'; targetValue: string; mappingVersion: string; reason: string; relationshipRefs: ID[] };
export type MigrationConservationReport = { sourceCounts: string[]; targetCounts: string[]; relationshipChecks: string[]; invariantFailures: string[]; roundTripLosses: string[]; unaccountedDeltas: ID[]; passed: boolean };
export type TargetWriteReconciliationPlan = { targetWritesPreserved: ID[]; sourceConflicts: string[]; resolutionPlans: string[]; rollbackAllowed: boolean };
export type CutoverReconciliation = { frozenMappingVersion: string; replayedDeltaRefs: ID[]; unaccountedSourceIds: ID[]; changedMappings: string[]; targetWritePlan: TargetWriteReconciliationPlan; switchAllowed: boolean };
export type DomainMigrationProposal = { migrationId: ID; meanings: SourceMeaningDecision[]; mappings: DomainFieldMapping[]; dispositions: SourceRecordDisposition[]; rehearsal: MigrationConservationReport; cutover: CutoverReconciliation; commitAllowed: boolean };
export type MigrationCutoverReceipt = { receiptId: ID; migrationId: ID; frozenMappingVersion: string; priorTargetVersion: string; observedTargetVersion: string; disposition: "switched" | "partial" | "unknown" | "rolled-back"; appliedSourceIds: ID[]; replayedDeltaRefs: ID[]; unaccountedSourceIds: ID[]; preservedTargetWriteIds: ID[]; rollbackEvidenceRefs: ID[]; evidenceRefs: Ref[] };
export type MigrationCutoverBook = { revision: number; currentTargetVersion: string | null; receipts: MigrationCutoverReceipt[]; consumedReceiptIds: ID[] };
export type MigrationCutoverReview = { book: MigrationCutoverBook; outcome: "switched" | "partial" | "unknown" | "rolled-back" | "held" | "replayed"; unresolvedSourceIds: ID[]; followUp: string[] };
export type MigrationCutoverAssessment = { safeToAdvance: boolean; reasons: string[]; unresolvedSourceIds: ID[] };
