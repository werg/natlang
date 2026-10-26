export type EvidenceItem = { id: string; source: string; version: string; text: string; status: 'current' | 'superseded' | 'unknown'; authority: string; };
export type WorkRecord = { id: string; kind: string; subject: string; status: string; evidenceIds: string[]; fields: Record<string, string>; };
export type RecordMeta = { id: string; caseId: string; sourceRefs: string[]; state: 'draft' | 'qualified' | 'blocked'; };
export type Decimal = string;
export type Ratio = { numerator: Decimal; denominator: Decimal; };
export type Quantity = { value: Decimal; unit: string; };
export type Money = { minorUnits: string; currency: string; scale: number; };
export type ExperienceStudy = RecordMeta & WorkRecord & {
  serviceIds: string[];
  researchQuestion: string;
  feedbackCollectionIds: string[];
  improvementOwnerId: string;
  consentPolicyId: string;
};
export type ExperienceEpisode = RecordMeta & WorkRecord & {
  caseId: string;
  serviceId: string;
  administrativeJourney: string;
  reportedObstacle: string;
  consequence: string;
};
export type JourneyLink = RecordMeta & WorkRecord & {
  caseId: string;
  episodeId: string;
  eventIds: string[];
  relationship: string;
  uncertainty: string;
};
export type ServiceMechanism = RecordMeta & WorkRecord & {
  caseId: string;
  episodeIds: string[];
  proposedExplanation: string;
  rivalExplanation: string;
  neededEvidence: string;
};
export type ServiceExperiment = RecordMeta & WorkRecord & {
  caseId: string;
  mechanismId: string;
  proposedChange: string;
  operationalMetric: string;
  decisionRule: string;
};
export type ExperienceOutcome = RecordMeta & WorkRecord & {
  caseId: string;
  experimentId: string;
  exactResultArtifactId: string;
  qualitativeRefs: string[];
  limitation: string;
};

export type ExperienceInbox = { instruction: string; studies: ExperienceStudy[]; feedback: EvidenceItem[]; governance: EvidenceItem[]; expectedStudyIds: string[]; reviewer: string; importedRecordIds: string[]; changes: EvidenceItem[]; metrics?: AdministrativeMetric[] };
export type AdministrativeEpisode = { id: string; patientScope: string | null; serviceId: string; obstacle: string; reportedConsequence: string; channel: string; sourceIds: string[]; mixedClinicalConcern: boolean; consentScope: string; linkageQuestions: string[] };
export type JourneyAssociation = { episodeId: string; eventIds: string[]; sameEpisodeAs: string[]; independentExperience: boolean; patientScope: string | null; unresolved: string[] };
export type AdministrativeMechanism = { id: string; episodeIds: string[]; explanation: string; rivals: string[]; contraryCaseIds: string[]; discriminatingEvidence: string[]; confidenceScope: string };
export type AdministrativeTrial = { id: string; mechanismId: string; approvedChange: string; affectedTeams: string[]; primaryMeasure: string; burdenMeasure: string; decisionRule: string; implementationReceiptIds: string[]; stopConditions: string[] };
export type AdministrativeMetric = { artifactId: string; trialId: string; period: string; populationDefinition: string; channelIds: string[]; availableChannelIds: string[]; numerator: string; denominator: string; subgroup: string; workloadDefinition: string; measuredBy: string };
export type AdministrativeOutcome = { trialId: string; disposition: "adopt" | "revise" | "abandon" | "inconclusive"; comparableMetricIds: string[]; excludedMetricIds: string[]; minorityFindings: string[]; downstreamBurden: string[]; limitations: string[] };
export type ExperienceConsentRemoval = { sourceIds: string[]; episodeIds: string[]; derivativeIds: string[]; pendingRemovalIds: string[]; actualReceiptIds: string[] };
export type ExperienceStudyState = { revision: number; episodes: AdministrativeEpisode[]; associations: JourneyAssociation[]; mechanisms: AdministrativeMechanism[]; trials: AdministrativeTrial[]; outcomes: AdministrativeOutcome[]; removals: ExperienceConsentRemoval[]; clinicalRoutingReceiptIds: string[] };
export type ExperienceInvestigation = { studyState: ExperienceStudyState; administrativeFindings: string[]; pendingClinicalRoutes: string[]; nextObservations: string[]; outcomeDecisions: AdministrativeOutcome[]; consentActions: ExperienceConsentRemoval[]; studyLimits: string[] };
