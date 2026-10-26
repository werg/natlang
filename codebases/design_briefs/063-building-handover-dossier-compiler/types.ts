export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };

export type PreparedEvidence = { subject: string; evidence: Evidence[]; missing: string[]; conflicts: string[]; alternatives: string[] };
export type DecisionAtom = { field: string; value: string; evidenceIds: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable"; alternative: string };
export type Qualification = "pass" | "fail" | "needs-review";

export type FieldConflict = { field: string; alternatives: string[]; evidenceIds: string[] };
export type EvidenceBinding = { field: string; evidenceIds: string[]; sourceVersions: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable" };

export type Question = { id: string; question: string; respondent: string; acceptableEvidence: string[] };

export type RecordMeta = { id: ID; version: string; createdAt: DateTime; updatedAt: DateTime };
export type BuildingHandover = RecordMeta & {
  projectId: ID;
  buildingId: ID;
  assetIds: ID[];
  handoverOwnerId: ID;
  targetDate: LocalDate;
};
export type HandoverRequirement = RecordMeta & {
  caseId: ID;
  assetClass: string;
  requiredArtifact: string;
  acceptanceRole: string;
  completionCondition: string;
};
export type AssetIdentity = RecordMeta & {
  caseId: ID;
  assetId: ID;
  aliases: string[];
  location: string;
  equipmentRevision: string;
};
export type HandoverEvidence = RecordMeta & {
  caseId: ID;
  requirementId: ID;
  assetId: ID;
  artifactIds: ID[];
  sufficiency: string;
};
export type HandoverClosureTask = RecordMeta & {
  caseId: ID;
  requirementIds: ID[];
  ownerId: ID;
  missingEvidence: string;
  dueAt: DateTime;
};
export type HandoverDossier = RecordMeta & {
  caseId: ID;
  assetIds: ID[];
  evidenceIds: ID[];
  outstandingIds: ID[];
  acceptanceRefs: Ref[];
};
export type DraftHandoverRequirement = { values: { caseId?: ID; assetClass?: string; requiredArtifact?: string; acceptanceRole?: string; completionCondition?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };

export type HandoverCompilationInput = { caseId: ID; requirements: HandoverRequirement[]; installedAssets: InstalledAssetVariant[]; manuals: VariantManual[]; certificates: AssetCertificate[]; portalDeliverables: PortalDeliverable[]; exportProfiles: AccessibleExportProfile[]; temporaryArrangements: TemporaryArrangement[]; evidence: Evidence[] };
export type InstalledAssetVariant = { assetId: ID; officialTag: string; model: string; variant: string; location: string; evidenceRef: Ref };
export type VariantManual = { manualId: ID; model: string; variantScope: string[]; sections: ManualSection[]; format: string; sourceVersion: string };
export type ManualSection = { sectionId: ID; topic: string; applicableVariantIds: string[] };
export type AssetCertificate = { certificateId: ID; namedAssetId: ID; model: string; revision: string; status: string; evidenceRef: Ref };
export type PortalDeliverable = { deliverableId: ID; portalUrl: string; expiry: string; permittedArchive: boolean; contentHash: string | null; format: string; owner: string };
export type AccessibleExportProfile = { requiredBy: string; acceptedFormats: string[]; accessibilityRequirement: string; approvedConverter: string | null };
export type TemporaryArrangement = { arrangementId: ID; affectedAssetIds: ID[]; description: string; expiresOn: string; followUpTask: string; owner: string; evidenceRef: Ref };
export type AssetManualMatch = { assetId: ID; variant: string; manualId: ID | null; relevantSections: ID[]; blockers: string[] };
export type CertificateMatch = { certificateId: ID; assetId: ID; accepted: boolean; reason: string };
export type DossierArchiveAction = { deliverableId: ID; action: string; hash: string | null; blocker: string };
export type DossierPackage = { caseId: ID; manuals: AssetManualMatch[]; certificates: CertificateMatch[]; archiveActions: DossierArchiveAction[]; accessibleExports: string[]; transferredObligations: string[]; missingRequirements: string[]; readyForAcceptance: boolean; commitAllowed: boolean };
export type HandoverAcceptanceContext = { caseId: ID; currentDossierVersion: string; requiredDeliverableIds: ID[]; missingDeliverableIds: ID[]; permittedReceivingRoles: string[]; readyForAcceptance: boolean };
export type HandoverAcceptanceReceipt = { receiptId: ID; caseId: ID; dossierVersion: string; receivingPartyId: ID; receivingRole: string; acceptedDeliverableIds: ID[]; exceptions: string[]; signedEvidence: Ref[]; receivedAt: DateTime };
export type HandoverAcceptanceBook = { revision: number; receipts: HandoverAcceptanceReceipt[]; validReceiptIds: ID[]; consumedReceiptIds: ID[] };
export type HandoverAcceptanceReview = { book: HandoverAcceptanceBook; outcome: "accepted" | "partial" | "held" | "replayed"; missingDeliverableIds: ID[]; nextTasks: string[] };
export type HandoverAcceptanceAssessment = { disposition: "complete" | "partial" | "held"; missingDeliverableIds: ID[]; unexpectedDeliverableIds: ID[]; exceptionItems: string[]; reasons: string[] };

export type CertificateObligationFinding = { requirementId: ID; applicability: "yes" | "no" | "unknown"; acceptable: boolean; reasons: string[]; evidenceRefs: Ref[] };
export type VariantManualFinding = { manualId: ID; applicable: boolean; relevantSections: ID[]; blockers: string[] };
export type DeliverableAcceptanceTrace = { deliverableId: ID; covered: boolean; receiptIds: ID[]; blockers: string[] };
