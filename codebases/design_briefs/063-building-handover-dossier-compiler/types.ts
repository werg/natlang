export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };
export type CaseInput = { caseId: string; goal: string; evidence: Evidence[]; policy: string[]; authorityStatus: "authorized" | "unclear" | "unauthorized"; changedEvidenceIds: string[]; audience: string; domainRecords: { firstStage: HandoverRequirement } };
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
export type StageResultHandoverRequirement = { accepted: HandoverRequirement[]; unresolved: DraftHandoverRequirement[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftAssetIdentity = { values: { caseId?: ID; assetId?: ID; aliases?: string[]; location?: string; equipmentRevision?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultAssetIdentity = { accepted: AssetIdentity[]; unresolved: DraftAssetIdentity[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftHandoverEvidence = { values: { caseId?: ID; requirementId?: ID; assetId?: ID; artifactIds?: ID[]; sufficiency?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultHandoverEvidence = { accepted: HandoverEvidence[]; unresolved: DraftHandoverEvidence[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftHandoverClosureTask = { values: { caseId?: ID; requirementIds?: ID[]; ownerId?: ID; missingEvidence?: string; dueAt?: DateTime }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultHandoverClosureTask = { accepted: HandoverClosureTask[]; unresolved: DraftHandoverClosureTask[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type DraftHandoverDossier = { values: { caseId?: ID; assetIds?: ID[]; evidenceIds?: ID[]; outstandingIds?: ID[]; acceptanceRefs?: Ref[] }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };
export type StageResultHandoverDossier = { accepted: HandoverDossier[]; unresolved: DraftHandoverDossier[]; rejected: Rejection[]; consumedRefs: Ref[]; coverage: CoverageManifest; nextQuestions: Question[]; proposedActions: ActionProposal[]; checks: CheckResult[] };
export type RunProposal = { caseId: string; goal: GoalContract; policy: PolicySelection; work: WorkGraph; evidencePlan: EvidencePlan; recoverRequirements: StageResultHandoverRequirement; resolveAssets: StageResultAssetIdentity; matchArtifacts: StageResultHandoverEvidence; planClosure: StageResultHandoverClosureTask; assembleDossier: StageResultHandoverDossier; collection: CollectionFinding; questions: QuestionBatch; invalidation: InvalidationProposal; review: ReviewPacket; view: ViewModel; revisions: RevisionProposal[]; disposition: "proposed" | "blocked" | "needs-review"; rationale: string[] };

export type CaseStageSet = { recoverRequirements: StageResultHandoverRequirement; resolveAssets: StageResultAssetIdentity; matchArtifacts: StageResultHandoverEvidence; planClosure: StageResultHandoverClosureTask; assembleDossier: StageResultHandoverDossier; };

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
