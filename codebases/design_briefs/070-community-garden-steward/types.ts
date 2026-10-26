export type ID = string; export type Ref = string; export type Decimal = string; export type Ratio = { numerator: string; denominator: string }; export type Quantity = { value: Decimal; unit: string }; export type Money = { minorUnits: string; currency: string; scale: number }; export type LocalDate = string; export type DateTime = string; export type SpanRef = string;
export type Evidence = { id: string; source: string; version: string; authority: string; observedAt: string; text: string; supports: string[]; contradicts: string[]; status: "current" | "superseded" | "uncertain" };

export type PreparedEvidence = { subject: string; evidence: Evidence[]; missing: string[]; conflicts: string[]; alternatives: string[] };
export type DecisionAtom = { field: string; value: string; evidenceIds: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable"; alternative: string };
export type Qualification = "pass" | "fail" | "needs-review";

export type FieldConflict = { field: string; alternatives: string[]; evidenceIds: string[] };
export type EvidenceBinding = { field: string; evidenceIds: string[]; sourceVersions: string[]; authority: string; status: "established" | "disputed" | "unknown" | "inapplicable" };

export type Question = { id: string; question: string; respondent: string; acceptableEvidence: string[] };

export type RecordMeta = { id: ID; version: string; createdAt: DateTime; updatedAt: DateTime };
export type GardenSeason = RecordMeta & {
  gardenId: ID;
  memberIds: ID[];
  plotIds: ID[];
  season: string;
  coordinatorId: ID;
};
export type GardenPreference = RecordMeta & {
  caseId: ID;
  memberId: ID;
  desiredUse: string;
  availability: string;
  accessibilityNeed: string;
};
export type PlotConstraint = RecordMeta & {
  caseId: ID;
  plotId: ID;
  observedConditions: string[];
  sharedResourceIds: ID[];
  useRestrictions: string[];
};
export type GardenPlan = RecordMeta & {
  caseId: ID;
  plotIds: ID[];
  plantingConcept: string;
  sharedActivities: string[];
  requiredResources: string[];
};
export type GardenAgreement = RecordMeta & {
  caseId: ID;
  memberIds: ID[];
  plotAssignments: string[];
  sharedDuties: string[];
  acceptanceRefs: Ref[];
};
export type GardenTask = RecordMeta & {
  caseId: ID;
  agreementId: ID;
  activity: string;
  ownerId: ID;
  dueWindow: string;
};
export type DraftGardenPreference = { values: { caseId?: ID; memberId?: ID; desiredUse?: string; availability?: string; accessibilityNeed?: string }; missingFields: string[]; conflicts: FieldConflict[]; evidence: EvidenceBinding[]; questions: Question[] };

export type GardenSteward = { id: string };
export type GardenChange = { unavailableMemberIds: string[]; brokenToolIds: string[]; newAssessments: Evidence[]; memberResponses: Evidence[]; departures: string[] };
export type GardenDossier = { season: GardenSeason; preferences: GardenPreference[]; constraints: PlotConstraint[]; plans: GardenPlan[]; agreements: GardenAgreement[]; tasks: GardenTask[]; evidence: Evidence[]; checkpoint: GardenCheckpoint };
export type GardenCheckpoint = { revision: string; preferences: GardenPreference[]; constraints: PlotConstraint[]; plans: GardenPlan[]; agreements: GardenAgreement[]; tasks: GardenTask[]; toolReservations: string[]; completedTaskRefs: string[]; transferredRights: string[]; };
export type PlotConditionMap = { plotId: string; knownConditions: string[]; uncertainConditions: string[]; interventionsRequiringExpert: string[]; evidenceIds: string[] };
export type PlotAlternative = { plan: GardenPlan; tradeoffs: string[]; hardConstraintFindings: string[]; unresolved: string[] };
export type AllocationNegotiation = { proposals: GardenAgreement[]; unacceptedDuties: string[]; trials: string[]; fairnessReasons: string[]; blockers: string[] };
export type SeasonBoard = { tasks: GardenTask[]; reservations: string[]; brokenToolHolds: string[]; careTransfers: string[]; completionRefs: string[]; conflicts: string[] };
export type GardenProposal = { application: 'Community garden steward'; status: 'ready-for-review'|'blocked'; preferences: GardenPreference[]; plotMap: PlotConditionMap[]; alternatives: PlotAlternative[]; negotiation: AllocationNegotiation; seasonBoard: SeasonBoard; blockers: string[]; actionBoundary: string; resumeState: GardenCheckpoint; };
