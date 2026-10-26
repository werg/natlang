export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };

export type PreparedEvidence = { subject: string; evidence: Evidence[]; missing: string[]; conflicts: string[]; alternatives: string[] };
export type DecisionAtom = { field: string; value: string; evidenceIds: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable"; alternative: string };
export type Qualification = "pass" | "fail" | "needs-review";

export type FieldConflict = { field: string; alternatives: string[]; evidenceIds: string[] };
export type EvidenceBinding = { field: string; evidenceIds: string[]; sourceVersions: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable" };

export type Question = { id: string; question: string; respondent: string; acceptableEvidence: string[] };

export type RecordMeta = { id: ID; version: string; createdAt: DateTime; updatedAt: DateTime };
export type RestorationProject = RecordMeta & {
  siteId: ID;
  landholderIds: ID[];
  restorationGoals: string[];
  expertOwnerId: ID;
  planningHorizon: string;
};
export type HabitatObservation = RecordMeta & {
  caseId: ID;
  locationId: ID;
  observedFeature: string;
  surveyMethod: string;
  seasonalContext: string;
};
export type RestorationObjective = RecordMeta & {
  caseId: ID;
  desiredCondition: string;
  indicator: string;
  landholderConstraint: string;
  uncertainty: string;
};
export type RestorationOption = RecordMeta & {
  caseId: ID;
  objectiveIds: ID[];
  approvedAction: string;
  prerequisites: string[];
  expectedMechanism: string;
};
export type MonitoringTask = RecordMeta & {
  caseId: ID;
  objectiveId: ID;
  observationMethod: string;
  samplingWindow: string;
  decisionRule: string;
};
export type RestorationRevision = RecordMeta & {
  caseId: ID;
  observationIds: ID[];
  affectedOptionIds: ID[];
  proposedChange: string;
  expertApprovalRef: Ref;
};
export type DraftHabitatObservation = { values: { caseId?: ID; locationId?: ID; observedFeature?: string; surveyMethod?: string; seasonalContext?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };

export type RestorationPlanningInput = { caseId: ID; baseline: HabitatObservation[]; objectives: RestorationObjectiveRecord[]; surveys: HabitatSurvey[]; speciesLocations: SensitiveSpeciesRecord[]; interventions: RestorationIntervention[]; landPermissions: FieldAccessPermission[]; impactReports: InterventionImpact[]; volunteerSamples: VolunteerSample[]; expertRoster: ExpertReviewScope[]; evidence: Evidence[] };
export type RestorationObjectiveRecord = { objectiveId: ID; target: string; areaIds: ID[]; indicatorIds: ID[]; seasonWindow: string; authorityRef: Ref };
export type HabitatSurvey = { surveyId: ID; areaId: ID; method: string; season: string; observedAt: string; effort: string; observations: string[]; evidenceRef: Ref };
export type SensitiveSpeciesRecord = { recordId: ID; exactCoordinate: string; sensitivity: string; accessRule: string; evidenceRef: Ref };
export type RestorationIntervention = { interventionId: ID; areaIds: ID[]; action: string; objectiveIds: ID[]; permissionIds: ID[]; stopConditions: string[]; qualifiedLead: string };
export type FieldAccessPermission = { permissionId: ID; areaId: ID; permittedActions: string[]; expiresAt: string; ownerRef: Ref };
export type InterventionImpact = { reportId: ID; interventionId: ID; observation: string; severity: string; observedAt: string; evidenceRef: Ref };
export type VolunteerSample = { sampleId: ID; areaId: ID; method: string; season: string; effortDuration: string; observerCount: number; observations: string[]; evidenceRef: Ref };
export type ExpertReviewScope = { expertId: ID; taxaOrSystem: string[]; qualificationRef: Ref; available: boolean };
export type SpeciesDisclosurePlan = { volunteerView: string[]; publicView: string[]; exactLocationView: string[]; restrictedRecordIds: ID[] };
export type SurveyComparison = { comparablePairs: string[]; incomparablePairs: string[]; confoundingChanges: string[] };
export type MonitoringDesign = { objectiveIndicators: string[]; seasonalRounds: string[]; effortFields: string[]; repeatMethods: string[]; comparabilityWarnings: string[] };
export type HabitatActionPlan = { caseId: ID; disclosure: SpeciesDisclosurePlan; comparison: SurveyComparison; monitoring: MonitoringDesign; permittedActions: ID[]; stoppedActions: ID[]; suspendedInterventions: ID[]; expertEscalations: string[]; unresolvedPermissions: ID[]; commitAllowed: boolean };
export type RestorationFieldReceipt = { receiptId: ID; caseId: ID; interventionId: ID; permissionId: ID; areaIds: ID[]; outcome: "completed" | "partial" | "paused" | "unknown"; observedAt: DateTime; actionsPerformed: string[]; impacts: InterventionImpact[]; evidenceRefs: Ref[] };
export type RestorationActionState = { interventionId: ID; status: "permitted" | "paused" | "suspended" | "unknown"; lastReceiptId: ID; reasons: string[] };
export type RestorationActionBook = { revision: number; actionStates: RestorationActionState[]; receipts: RestorationFieldReceipt[]; consumedReceiptIds: ID[] };
export type RestorationFieldReconciliation = { book: RestorationActionBook; disposition: "recorded" | "replayed" | "held"; interventionStatus: "permitted" | "paused" | "suspended" | "unknown"; expertEscalations: string[]; followUp: string[] };
export type ImpactLinkage = { reportId: ID; interventionId: ID | null; status: "linked" | "ambiguous" | "unlinked"; effect: "adverse" | "unclear" | "benign"; affectedSystem: string; evidenceRefs: Ref[] };
export type QualifiedExpertRoute = { reportId: ID; disposition: "assigned" | "unavailable" | "no-qualified-expert" | "out-of-scope"; expertId: ID | null; reason: string };
export type FieldActionAuthorization = { permittedActions: ID[]; stoppedActions: ID[]; unresolvedPermissionIds: ID[] };
export type EcologicalIncidentReview = { suspendedInterventions: ID[]; reviewRequests: string[]; preservedObservationRefs: Ref[] };
