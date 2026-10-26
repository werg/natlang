export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };

export type PreparedEvidence = { subject: string; evidence: Evidence[]; missing: string[]; conflicts: string[]; alternatives: string[] };
export type DecisionAtom = { field: string; value: string; evidenceIds: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable"; alternative: string };
export type Qualification = "pass" | "fail" | "needs-review";

export type FieldConflict = { field: string; alternatives: string[]; evidenceIds: string[] };
export type EvidenceBinding = { field: string; evidenceIds: string[]; sourceVersions: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable" };

export type Question = { id: string; question: string; respondent: string; acceptableEvidence: string[] };

export type RecordMeta = { id: ID; version: string; createdAt: DateTime; updatedAt: DateTime };
export type ObservationReview = RecordMeta & {
  collectionId: ID;
  projectId: ID;
  observationType: string;
  curatorId: ID;
  disclosurePolicyId: ID;
};
export type FieldObservation = RecordMeta & {
  caseId: ID;
  observerId: ID;
  observedAt: DateTime;
  locationRef: Ref;
  description: string;
  method: string;
};
export type ObservationInterpretation = RecordMeta & {
  caseId: ID;
  observationId: ID;
  candidateConceptIds: ID[];
  interpretation: string;
  ambiguity: string;
};
export type ObservationLink = RecordMeta & {
  caseId: ID;
  observationIds: ID[];
  relation: 'same-event'|'independent'|'related'|'unknown';
  basis: string;
};
export type ObservationQuestion = RecordMeta & {
  caseId: ID;
  observationId: ID;
  decisiveUnknown: string;
  requestedEvidence: string;
  recipientId: ID;
};
export type CuratedObservation = RecordMeta & {
  caseId: ID;
  observationId: ID;
  acceptedInterpretation: string;
  qualityFlags: string[];
  publicLocationPrecision: string;
};
export type DraftFieldObservation = { values: { caseId?: ID; observerId?: ID; observedAt?: DateTime; locationRef?: Ref; description?: string; method?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };

export type CuratorIdentity = { id: string };
export type ObservationSubmission = { observation: FieldObservation; originalText: string; originalTimestamp: string; declaredTimezone: string; locationPrecision: string; mediaHashes: string[]; evidence: Evidence[] };
export type ObservationCorrection = { observationId: string; expertId: string; priorInterpretationId: string; newConceptIds: string[]; rationale: string; evidence: Evidence[] };
export type CurationCollection = { review: ObservationReview; vocabulary: string[]; disclosureRule: string; restrictedPrecision: string; publicPrecision: string };
export type CurationCheckpoint = { revision: string; submissions: ObservationSubmission[]; interpretations: ObservationInterpretation[]; links: ObservationLink[]; questions: ObservationQuestion[]; curated: CuratedObservation[]; correctionHistory: ObservationCorrection[]; invalidatedDerivativeIds: string[]; };
export type CurationRequest = { submissions: ObservationSubmission[]; corrections: ObservationCorrection[]; answerEvidence: Evidence[]; requestedObservationIds: string[] };
export type ObservationComparison = { leftId: string; rightId: string; relation: 'same-event'|'independent'|'related'|'unknown'; basis: string; evidenceIds: string[]; blockers: string[] };
export type CurationProposal = { application: 'Citizen-science observation curator'; status: 'ready-for-review'|'blocked'; normalized: FieldObservation[]; interpretations: ObservationInterpretation[]; links: ObservationLink[]; questions: ObservationQuestion[]; publicRecords: CuratedObservation[]; rawProvenance: string[]; privacyFindings: string[]; invalidatedDerivativeIds: string[]; blockers: string[]; actionBoundary: string; resumeState: CurationCheckpoint; };
export type NormalizedBatch = { observations: FieldObservation[]; rawProvenance: string[]; unresolvedMetadata: string[]; };
export type CandidateSet = { interpretations: ObservationInterpretation[]; unresolved: string[]; questions: string[]; };
export type DuplicateGraph = { links: ObservationLink[]; unresolved: ObservationComparison[]; mediaAttachments: Record<string,string[]>; };
export type QuestionSet = { questions: ObservationQuestion[]; safetyBlockers: string[]; };
export type PublicProjection = { records: CuratedObservation[]; privacyFindings: string[]; invalidatedDerivativeIds: string[]; blockers: string[]; };
