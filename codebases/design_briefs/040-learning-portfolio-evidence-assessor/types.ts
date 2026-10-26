export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };

export type PreparedEvidence = { subject: string; evidence: Evidence[]; missing: string[]; conflicts: string[]; alternatives: string[] };
export type DecisionAtom = { field: string; value: string; evidenceIds: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable"; alternative: string };
export type Qualification = "pass" | "fail" | "needs-review";

export type FieldConflict = { field: string; alternatives: string[]; evidenceIds: string[] };
export type EvidenceBinding = { field: string; evidenceIds: string[]; sourceVersions: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable" };

export type Question = { id: string; question: string; respondent: string; acceptableEvidence: string[] };

export type RecordMeta = { id: ID; version: string; createdAt: DateTime; updatedAt: DateTime };
export type PortfolioReview = RecordMeta & {
  learnerId: ID;
  competencyFrameworkId: ID;
  artifactIds: ID[];
  assessorId: ID;
  reviewPurpose: string;
};
export type PortfolioEvidence = RecordMeta & {
  caseId: ID;
  artifactId: ID;
  contribution: string;
  context: string;
  authorshipStatus: string;
};
export type CompetencyMapping = RecordMeta & {
  caseId: ID;
  evidenceId: ID;
  competencyId: ID;
  demonstratedFacet: string;
  missingFacet: string;
};
export type CompetencyFinding = RecordMeta & {
  caseId: ID;
  competencyId: ID;
  evidenceIds: ID[];
  sufficiency: 'met'|'partial'|'insufficient';
  limitation: string;
};
export type EvidenceTask = RecordMeta & {
  caseId: ID;
  competencyId: ID;
  task: string;
  requiredDemonstration: string;
  acceptanceCriteria: string[];
};
export type PortfolioDisposition = RecordMeta & {
  caseId: ID;
  findingIds: ID[];
  assessorNotes: string;
  nextTaskIds: ID[];
  approvalRef: Ref;
};
export type DraftPortfolioEvidence = { values: { caseId?: ID; artifactId?: ID; contribution?: string; context?: string; authorshipStatus?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };

export type PortfolioAssessmentRequest = { caseId: ID; requestedOutcome: string; sourceEvidence: Evidence[]; priorPortfolioEvidence: PortfolioEvidence; frameworkStandard: string; frameworkVersion: string; policyClauses: string[]; authorityStatus: 'authorized' | 'unclear' | 'unauthorized'; recencyRule: string; accommodationRule: string; stakes: 'high' | 'ordinary' | 'unknown'; priorFinding: PortfolioAssessmentFinding | null; changedEvidenceIds: ID[]; assessmentState?: PortfolioAssessmentState };
export type IndexedPortfolioEvidence = { evidenceId: ID; evidenceType: string; artifactId: ID; author: string; contribution: string; sourceLocation: string; authorshipStatus: 'individual' | 'shared-known' | 'unresolved' | 'self-report'; contradictoryRefs: ID[] };
export type PortfolioCompetencyMap = { competencyId: ID; evidenceId: ID; demonstratedBehavior: string; sourceLocation: string; demonstratedFacets: string[]; missingFacets: string[]; accommodationRef: string };
export type PortfolioCompetencyFinding = { competencyId: ID; result: 'met' | 'partial' | 'insufficient' | 'review'; evidenceIds: ID[]; independentDemonstrations: number; recencyDecision: string; limitations: string[]; conflicts: ID[] };
export type PortfolioEvidenceTask = { competencyId: ID; task: string; constraints: string[]; successEvidence: string[]; feasible: boolean };
export type PortfolioAssessmentFinding = { findingId: ID; version: number; competencyResults: PortfolioCompetencyFinding[]; sourceEvidenceIds: ID[]; assessorId: ID; status: string };
export type AttributionAppealReview = { priorFindingVersion: number; affectedCompetencyIds: ID[]; evidenceVersionRefs: ID[]; newDecisionAuthorized: boolean; preservedPriorFinding: boolean };
export type UnsupportedOutcome = { outcome: string; reason: string; missingEvidence: string[] };
export type AssessorDisposition = { recommendation: string; unsupported: UnsupportedOutcome[]; assessorApprovalNeeded: boolean; appealPath: string; conflictsPreserved: ID[]; pendingFindingVersion: number };
export type PortfolioAssessmentRevision = { caseId: ID; revision: number; frameworkVersion: string; sourceEvidenceVersions: { evidenceId: ID; version: string }[]; changedEvidenceIds: ID[]; mappings: PortfolioCompetencyMap[]; findings: PortfolioCompetencyFinding[]; tasks: PortfolioEvidenceTask[]; attributionAppeal: AttributionAppealReview; disposition: AssessorDisposition; priorFindingId: ID | null };
export type PortfolioAssessmentState = { revision: number; authoritativeFindings: PortfolioAssessmentFinding[]; reviewHistory: PortfolioAssessmentRevision[]; appealHistory: AttributionAppealReview[]; frameworkVersions: string[] };
export type PortfolioAssessmentProposal = { caseId: ID; indexed: IndexedPortfolioEvidence[]; mappings: PortfolioCompetencyMap[]; findings: PortfolioCompetencyFinding[]; tasks: PortfolioEvidenceTask[]; attributionAppeal: AttributionAppealReview; disposition: AssessorDisposition; approvalRequired: boolean; assessmentState: PortfolioAssessmentState; commitAllowed: boolean };
