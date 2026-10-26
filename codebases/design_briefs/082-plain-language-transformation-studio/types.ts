export type EvidenceItem = { id: string; source: string; version: string; text: string; status: 'current' | 'superseded' | 'unknown'; authority: string; };
export type WorkRecord = { id: string; kind: string; subject: string; status: string; evidenceIds: string[]; fields: Record<string, string>; };
export type RecordMeta = { id: string; caseId: string; sourceRefs: string[]; state: 'draft' | 'qualified' | 'blocked'; };
export type Decimal = string;
export type Ratio = { numerator: Decimal; denominator: Decimal; };
export type Quantity = { value: Decimal; unit: string; };
export type Money = { minorUnits: string; currency: string; scale: number; };
export type PlainLanguageProject = RecordMeta & WorkRecord & {
  documentIds: string[];
  audienceDescription: string;
  purpose: string;
  reviewerId: string;
  fidelityPolicyId: string;
};
export type MeaningUnit = RecordMeta & WorkRecord & {
  caseId: string;
  proposition: string;
  modality: string;
  condition: string;
  exception: string;
  sourceSpan: string;
};
export type ReaderTask = RecordMeta & WorkRecord & {
  caseId: string;
  audience: string;
  task: string;
  requiredMeaningIds: string[];
  likelyAmbiguity: string;
};
export type PlainLanguageDraft = RecordMeta & WorkRecord & {
  caseId: string;
  meaningIds: string[];
  rewrittenText: string;
  exampleText: string;
  retainedTerminology: string[];
};
export type ComprehensionProbe = RecordMeta & WorkRecord & {
  caseId: string;
  taskId: string;
  question: string;
  expectedUnderstanding: string;
  misleadingAnswer: string;
};
export type PlainLanguageRelease = RecordMeta & WorkRecord & {
  caseId: string;
  draftId: string;
  fidelityReviewRef: string;
  comprehensionFindings: string[];
  sourceVersion: string;
};

export type ReaderEditionRequest = { instruction: string; projects: PlainLanguageProject[]; authoritativeSources: EvidenceItem[]; fidelityRules: EvidenceItem[]; requiredProjectIds: string[]; reviewer: string; previouslyApprovedIds: string[]; sourceChanges: EvidenceItem[] };
export type OperativeUnit = { id: string; sourceId: string; sourceVersion: string; start: number; end: number; quotation: string; kind: "claim" | "duty" | "permission" | "condition" | "exception" | "uncertainty"; actor: string; action: string; conditionIds: string[]; exceptionToIds: string[]; protectedTerms: string[]; unresolvedScope: string[] };
export type ReaderDecisionTask = { id: string; readerGoal: string; decisionPoint: string; requiredUnitIds: string[]; backgroundNeeded: string[]; forbiddenOmissions: string[] };
export type ReaderParagraph = { id: string; taskId: string; heading: string; text: string; unitIds: string[]; illustrativeExamples: string[]; retainedTechnicalTerms: string[]; linkDescriptions: string[] };
export type ConsequenceProbe = { id: string; situation: string; testedUnitIds: string[]; boundary: "ordinary" | "exception" | "negative" | "deadline" | "ambiguity"; readerQuestion: string };
export type ConsequenceAnswer = { probeId: string; actor: string; allowedAction: string; requiredAction: string; prohibitedAction: string; conditions: string[]; uncertainty: string[]; citedIds: string[] };
export type FidelityDifference = { probeId: string; equivalent: boolean; changedConsequence: string; affectedParagraphIds: string[]; repairUnitIds: string[] };
export type ReaderEditionState = { revision: number; units: OperativeUnit[]; tasks: ReaderDecisionTask[]; paragraphs: ReaderParagraph[]; probes: ConsequenceProbe[]; differences: FidelityDifference[]; approvedSourceVersions: string[]; fidelityReviewIds: string[]; accessibilityReviewIds: string[]; releaseReceiptIds: string[] };
export type ReaderEditionReview = { edition: ReaderEditionState; plainText: string; sourceMap: string[]; blockedParagraphIds: string[]; invalidatedReviewIds: string[]; reviewerQuestions: string[]; releaseReady: boolean };
