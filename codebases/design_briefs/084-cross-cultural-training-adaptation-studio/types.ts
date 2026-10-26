export type EvidenceItem = { id: string; source: string; version: string; text: string; status: 'current' | 'superseded' | 'unknown'; authority: string; };
export type EvidenceSlice = { subject: string; candidateIds: string[]; supporting: EvidenceItem[]; conflicting: EvidenceItem[]; missing: string[]; searchedScope: string; };

export type WorkRecord = { id: string; kind: string; subject: string; status: string; evidenceIds: string[]; fields: Record<string, string>; };

export type RecordMeta = { id: string; caseId: string; sourceRefs: string[]; state: 'draft' | 'qualified' | 'blocked'; };
export type Decimal = string;
export type Ratio = { numerator: Decimal; denominator: Decimal; };
export type Quantity = { value: Decimal; unit: string; };
export type Money = { minorUnits: string; currency: string; scale: number; };
export type TrainingAdaptation = RecordMeta & WorkRecord & {
  trainingProgramId: string;
  audienceGuidanceId: string;
  competencyIds: string[];
  reviewerId: string;
  targetContext: string;
};
export type TrainingMechanism = RecordMeta & WorkRecord & {
  caseId: string;
  competencyId: string;
  scenarioFunction: string;
  requiredDecision: string;
  incidentalElements: string[];
};
export type ContextFitFinding = RecordMeta & WorkRecord & {
  caseId: string;
  mechanismId: string;
  sourceElement: string;
  suppliedAudienceIssue: string;
  adaptationNeed: string;
};
export type AdaptedScenario = RecordMeta & WorkRecord & {
  caseId: string;
  mechanismId: string;
  newSetting: string;
  scenarioText: string;
  preservedDecisionStructure: string;
};
export type AdaptationTest = RecordMeta & WorkRecord & {
  caseId: string;
  scenarioId: string;
  expectedCompetency: string;
  responseClasses: string[];
  driftFinding: string;
};
export type AdaptedTrainingRelease = RecordMeta & WorkRecord & {
  caseId: string;
  scenarioIds: string[];
  audienceReviewRefs: string[];
  sourceVersion: string;
  limitations: string[];
};
export type DraftTrainingMechanism = { values: {
    id?: string;
    caseId?: string;
    sourceRefs?: string[];
    state?: 'draft' | 'qualified' | 'blocked';
    kind?: string;
    subject?: string;
    evidenceIds?: string[];
    fields?: Record<string, string>;
    status?: string;
    competencyId?: string;
    scenarioFunction?: string;
    requiredDecision?: string;
    incidentalElements?: string[];
  }; missingFields: string[]; conflicts: string[]; evidenceIds: string[]; };

export type AdaptationReviewer = { id: string };
export type AdaptationChange = { audienceEvidence: EvidenceItem[]; reviewerResponses: EvidenceItem[]; learnerTestResults: EvidenceItem[]; curriculumUpdates: EvidenceItem[] };
export type LearningDossier = { program: TrainingAdaptation; mechanisms: TrainingMechanism[]; contextFindings: ContextFitFinding[]; scenarios: AdaptedScenario[]; tests: AdaptationTest[]; releases: AdaptedTrainingRelease[]; evidence: EvidenceItem[]; checkpoint: LearningCheckpoint };
export type CompetencyKernel = { competencyId: string; requiredDecision: string; decisionComplexity: string[]; incidentalElements: string[]; sourceVersion: string; evidenceIds: string[]; };
export type AudienceFit = { mechanismId: string; fitEvidence: string[]; unfamiliarElements: string[]; reviewerDisagreements: string[]; stereotypeRisks: string[]; scopeLimit: string; blockers: string[]; };
export type ComplexityComparison = { mechanismId: string; decisionStepsBefore: string[]; decisionStepsAfter: string[]; preservedTradeoffs: string[]; omittedDemands: string[]; blockers: string[]; };
export type LearningCheckpoint = { revision: string; program: TrainingAdaptation; kernels: CompetencyKernel[]; fit: AudienceFit[]; scenarios: AdaptedScenario[]; tests: AdaptationTest[]; releases: AdaptedTrainingRelease[]; invalidatedScenarioIds: string[]; reviewerApprovals: string[]; };
export type AdaptationProposal = { application: 'Cross Cultural Training Adaptation Studio'; status: 'ready-for-review'|'blocked'; kernels: CompetencyKernel[]; fit: AudienceFit[]; scenarios: AdaptedScenario[]; complexity: ComplexityComparison[]; tests: AdaptationTest[]; release: AdaptedTrainingRelease; blockers: string[]; actionBoundary: string; resumeState: LearningCheckpoint; };
