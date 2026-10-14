/** Closed fictional mission. The application proposes a policy and never controls a robot. */
export type MissionRequest = { caseId: string; start: string; goal: string; initialEnergy: number; objective: string };
export type Witness = { sourceId: string; path: string; quote: string };
export type SourceDocument = { id: string; path: string; roleHint: string; text: string };
export type SourceManifest = { caseId: string; documents: { id: string; path: string; roleHint: string }[]; source_scope: string; output_contract: Record<string, string> };
export type SourceClause = { id: string; text: string; witnesses: Witness[] };
export type Point = { x: number; y: number };
export type FrameAlternative = {
  alternativeId: string; sourceFrame: string; origin: Point;
  uVector: Point; vVector: Point; witnesses: Witness[];
};
export type LandmarkMention = { localId: string; frame: string; position: Point | null; description: string; witnesses: Witness[] };
export type FrameSurvey = { alternatives: FrameAlternative[]; landmarks: LandmarkMention[]; unitResults: FrameUnitResult[]; unresolved: string[] };
export type PassageAssertion = {
  from: string; to: string; frame: string; direction: 'one-way' | 'two-way';
  kind: 'corridor' | 'water-gate' | 'tunnel' | 'chimney'; cost: number | null;
  condition: string; physicalExistence: 'present' | 'absent' | 'unknown'; witnesses: Witness[];
};
export type GearRule = { ruleId: string; subject: string; condition: string; consequence: string; witnesses: Witness[] };
export type SensorRule = { sensorId: string; availableAt: string; cost: number; answerDescriptions: { answer: string; meaning: string }[]; scope: string; witnesses: Witness[] };
export type EquipmentSurvey = { gear: GearRule[]; sensors: SensorRule[]; unresolved: string[] };
export type AnchorAssessment = { compatible: boolean | null; reason: string; witnesses: Witness[] };
export type ConditionalAnchorBinding = {
  alternativeId: string; localId: string; localFrame: string; globalId: string; globalFrame: string;
  transformedPoint: Point; assessment: AnchorAssessment;
};
export type SensorWorldAnswer = { worldId: string; answer: string | null; reason: string; witnesses: Witness[] };
export type SurveyEvidence = {
  manifest: SourceManifest; documents: SourceDocument[]; clauses: SourceClause[];
  frames: FrameSurvey; passages: PassageAssertion[]; gear: GearRule[]; sensors: SensorRule[];
  coverage: { sourceId: string; clauseIds: string[] }[]; unresolved: string[];
};
export type LandmarkDecision = {
  leftId: string; rightId: string; relation: 'same' | 'different' | 'unresolved';
  sharedFrameAnchor: boolean; reason: string; witnesses: Witness[];
};
export type LandmarkPartition = { decisions: LandmarkDecision[]; fixedBindings: { localId: string; globalId: string }[]; unresolvedPairs: string[][] };
export type FrameHypothesis = { hypothesisId: string; transform: FrameAlternative; bindings: { localId: string; globalId: string; position: Point }[]; witnesses: Witness[] };
export type WorldPassage = {
  passageId: string; from: string; to: string; kind: 'corridor' | 'water-gate' | 'tunnel' | 'chimney';
  cost: number; condition: string; witnesses: Witness[];
};
export type WorldModel = {
  worldId: string; frame: FrameHypothesis; passages: WorldPassage[];
  sampleLocation: string; correctRigSide: string; facts: SourceClause[]; witnesses: Witness[];
};
export type RejectedWorld = { hypothesisId: string; reason: string; witnesses: Witness[] };
export type WorldFrontier = { pending: FrameHypothesis[]; worlds: WorldModel[]; rejected: RejectedWorld[]; unresolved: string[]; survey: SurveyEvidence };
export type WorldConstruction = { worlds: WorldModel[]; rejected: RejectedWorld[]; unresolved: string[] };
export type WorldSet = { worlds: WorldModel[]; rejected: RejectedWorld[]; landmarks: LandmarkPartition; unresolved: string[] };
export type RobotState = { location: string; energy: number; sampleHeld: boolean; rope: 'unused' | 'rigged'; rigSide: string | null };
export type BeliefState = { stateId: string; robot: RobotState; worldIds: string[] };
export type MissionAction = {
  actionId: string; kind: 'move' | 'rig' | 'observe' | 'pickup';
  from: string; to: string | null; rigSide: string | null; sensorId: string | null;
  cost: number; witnesses: Witness[];
};
export type CrossingDecision = { worldId: string; allowed: boolean; nextRobot: RobotState | null; reason: string; witnesses: Witness[] };
export type ObservationCell = { answer: string; worldIds: string[]; witnesses: Witness[] };
export type ObservationPartition = { sensorId: string; cells: ObservationCell[]; complete: boolean; discriminating: boolean; reason: string; witnesses: Witness[] };
export type ActionEdge = { action: MissionAction; outcomes: { answer: string | null; nextStateId: string; worldIds: string[] }[]; witnesses: Witness[] };
export type StateExpansion = { stateId: string; actions: ActionEdge[]; rejectedActions: { action: MissionAction; reason: string; worldIds: string[]; witnesses: Witness[] }[]; newStates: BeliefState[] };
export type BeliefGraph = { states: BeliefState[]; expansions: StateExpansion[]; initialStateId: string; goalStateIds: string[]; deadStateIds: string[] };
export type BeliefFrontier = { universeSize: number; pendingStateIds: string[]; expandedStateIds: string[]; graph: BeliefGraph; worlds: WorldSet; survey: SurveyEvidence; request: MissionRequest };
export type StateValue = { stateId: string; winning: boolean; worstRemainingEnergy: number | null; worstRemainingObservations: number | null; chosenActionId: string | null; reason: string; witnesses: Witness[] };
export type ValueFrontier = { remainingStateIds: string[]; values: StateValue[]; graph: BeliefGraph; request: MissionRequest };
export type PolicyNode = { nodeId: string; stateId: string; terminal: boolean; action: MissionAction | null; outcomes: { answer: string | null; nextNodeId: string; worldIds: string[] }[] };
export type BeliefPolicy = { rootNodeId: string | null; nodes: PolicyNode[]; worstCaseEnergy: number | null; worstCaseObservations: number | null; blockedReasons: string[] };
export type MissionFinding = { kind: 'defect' | 'unresolved' | 'defended'; proposition: string; worldIds: string[]; witnesses: Witness[] };
export type WorldReplay = { worldId: string; nodeIds: string[]; actionIds: string[]; terminal: RobotState; totalEnergy: number; observations: number; success: boolean; findings: MissionFinding[] };
export type AuditedMission = { caseId: string; worlds: WorldSet; policy: BeliefPolicy; replays: WorldReplay[]; findings: MissionFinding[]; sourceCoverage: { sourceId: string; clauseIds: string[] }[] };
export type PlanningResult = { graph: BeliefGraph; values: StateValue[]; policy: BeliefPolicy };

/** Survey v5: NL-owned reading units and meanings; original provenance is bound by the parent. */
export type SurveyQuote = { clauseId: string; exactQuote: string };
export type SurveyReadingUnit = { unitId: string; kind: 'frame' | 'landmark' | 'passage' | 'gear' | 'sensor'; sourceFrame: string | null; originalLocalName: string | null; primaryClauseIds: string[]; supportingClauseIds: string[]; focus: string };
export type SurveyUnitInventory = { units: SurveyReadingUnit[]; unresolved: string[] };
export type LandmarkMeaning = { localId: string; frame: string; position: Point | null; description: string };
export type LandmarkItemMeaning = { meaning: LandmarkMeaning | null; evidence: SurveyQuote[]; unresolved: string[] };
export type ConventionMeaning = { sourceFrame: string; origin: Point; uVector: Point; vVector: Point; evidence: SurveyQuote[] };
export type ConventionInterpretation = { alternatives: ConventionMeaning[]; unresolved: string[] };
export type PassageMeaning = { from: string; to: string; frame: string; direction: 'one-way' | 'two-way'; kind: 'corridor' | 'water-gate' | 'tunnel' | 'chimney'; cost: number | null; condition: string; physicalExistence: 'present' | 'absent' | 'unknown' };
export type PassageItemMeaning = { assertion: PassageMeaning | null; evidence: SurveyQuote[]; unresolved: string[] };
export type GearMeaning = { ruleId: string; subject: string; condition: string; consequence: string };
export type GearItemMeaning = { meaning: GearMeaning | null; evidence: SurveyQuote[]; unresolved: string[] };
export type SensorMeaning = { sensorId: string; availableAt: string; cost: number; answerDescriptions: { answer: string; meaning: string }[]; scope: string };
export type SensorItemMeaning = { meaning: SensorMeaning | null; evidence: SurveyQuote[]; unresolved: string[] };

/** Frame v8: actual per-unit returns and checked projections survive later failures. */
export type FrameUnitResult = { unit: SurveyReadingUnit; status: 'accepted' | 'unresolved' | 'failed'; supportClosure: FrameSupportClosure | null; convention: ConventionInterpretation | null; landmark: LandmarkItemMeaning | null; alternatives: FrameAlternative[]; landmarks: LandmarkMention[]; failure: string | null; unresolved: string[] };
export type FrameReadFrontier = { clauses: SourceClause[]; pendingUnits: SurveyReadingUnit[]; completed: FrameUnitResult[]; unresolved: string[] };

export type FrameSupportClosure = { frameRole: 'reference-grid' | 'local-export' | 'unresolved'; ownedConventionClauseIds: string[]; localOriginClauseIds: string[]; identityBridgeClauseIds: string[]; fixedPointClauseIds: string[]; orientationClauseIds: string[]; supportingClauseIds: string[]; evidence: SurveyQuote[]; unresolved: string[] };
