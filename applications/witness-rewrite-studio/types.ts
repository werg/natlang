/** All story identities and rules come from the visible input snapshot. */
export type SourceRecord = { id: string; path: string; text: string; sha256: string };
export type SourceSnapshot = { revision: string; records: SourceRecord[] };
export type Witness = { sourceId: string; path: string; quote: string };
export type LocalDecision = { id: string; site: string; inputIds: string[]; ownedContribution: string; witnesses: Witness[] };
export type RewriteRequest = {
  id: string;
  requestedChange: string;
  protectedMeanings: string[];
  allowedEditPaths: string[];
};
export type PreservationRule = {
  id: string;
  kind: 'canon' | 'knowledge' | 'physical' | 'motivation' | 'ambiguity' | 'scope';
  meaning: string;
  witnesses: Witness[];
};
export type FactDelta = { removedFactIds: string[]; addedMeanings: string[]; scopeFrameIds: string[]; witnesses: Witness[] };
export type RequestMeaning = { requestId: string; delta: FactDelta; rules: PreservationRule[]; unknowns: string[] };
export type RecordInventory = {
  selectedScenes: SourceRecord[];
  recollections: SourceRecord[];
  canon: SourceRecord[];
  proposals: SourceRecord[];
  instructions: SourceRecord[];
  identityRecords: SourceRecord[];
  unclassified: SourceRecord[];
};
export type IntakeResult = { snapshot: SourceSnapshot; inventory: RecordInventory; meaning: RequestMeaning };
export type FrameDefinition = {
  id: string;
  kind: 'selected' | 'memory' | 'quotation' | 'speculation';
  parentId: string | null;
  owner: string | null;
  sceneIds: string[];
  witnesses: Witness[];
};
export type EventView = {
  id: string;
  sceneId: string;
  frameId: string;
  participants: string[];
  assertedMeanings: string[];
  observationMeanings: string[];
  reportedMeanings: string[];
  witnesses: Witness[];
};
export type FrameIndex = { frames: FrameDefinition[]; events: EventView[]; unresolved: string[]; localDecisions: LocalDecision[] };
export type AliasLink = { names: string[]; entityId: string; kind: 'person' | 'object' | 'event'; witnesses: Witness[] };
export type EntityIndex = { aliases: AliasLink[]; distinctPairs: { left: string; right: string; witnesses: Witness[] }[]; unresolved: string[] };
export type Precedence = { earlier: string; later: string; frameId: string; witnesses: Witness[] };
export type TemporalGraph = { edges: Precedence[]; unresolvedPairs: { left: string; right: string }[]; cycles: string[][] };
export type AuthorizedFact = { id: string; meaning: string; frameId: string; authorityKind: string; witnesses: Witness[] };
export type UncertaintyFact = { id: string; question: string; possibleMeanings: string[]; protected: boolean; witnesses: Witness[] };
export type AuthorityView = { facts: AuthorizedFact[]; beliefsOnly: AuthorizedFact[]; uncertainty: UncertaintyFact[] };
export type WorldIndex = { frames: FrameIndex; entities: EntityIndex; temporal: TemporalGraph; authority: AuthorityView };
export type InformationEvent = {
  id: string;
  eventId: string;
  factId: string;
  sender: string | null;
  receivers: string[];
  mode: 'observe' | 'read' | 'hear' | 'report' | 'mislead';
  credibleToReceivers: boolean;
  witnesses: Witness[];
};
export type BeliefState = { actor: string; beforeEventId: string; knownFactIds: string[]; believedMeanings: string[]; unresolved: string[] };
export type DialogueDecision = {
  localDecision: LocalDecision;
  sceneId: string;
  actor: string;
  assertedFactIds: string[];
  status: 'licensed' | 'unsupported' | 'unresolved';
  informationPath: string[];
  reason: string;
  witnesses: Witness[];
};
export type KnowledgeIndex = { acquisitions: InformationEvent[]; states: BeliefState[]; dialogue: DialogueDecision[]; localDecisions: LocalDecision[] };
export type ObjectTransition = {
  eventId: string;
  objectId: string;
  frameId: string;
  kind: 'transfer' | 'appearance' | 'copy' | 'state-change';
  sourceHolders: string[];
  targetHolders: string[];
  state: string | null;
  witnesses: Witness[];
};
export type PropIndex = { transitions: ObjectTransition[]; actualHolders: { objectId: string; afterEventId: string; holders: string[] }[]; unresolved: string[] };
export type MotifLink = { earlierSceneId: string; laterSceneId: string; meaning: string; requiredFactIds: string[]; witnesses: Witness[] };
export type MotifIndex = { links: MotifLink[]; unlicensedEchoes: string[]; unresolved: string[] };
export type Analysis = { intake: IntakeResult; world: WorldIndex; knowledge: KnowledgeIndex; props: PropIndex; motifs: MotifIndex };
export type MeaningDependency = {
  id: string;
  premiseId: string;
  consumerId: string;
  relation: 'knowledge' | 'physical' | 'motivation' | 'callback' | 'authority';
  necessary: boolean | null;
  affected: boolean | null;
  witnesses: Witness[];
};
export type Obligation = {
  id: string;
  sceneIds: string[];
  kind: 'requested-change' | 'knowledge' | 'physical' | 'motivation' | 'ambiguity' | 'callback';
  requiredMeaning: string;
  protectedRuleIds: string[];
  dependencies: MeaningDependency[];
  witnesses: Witness[];
};
export type ImpactFrontier = {
  localDecisions: LocalDecision[];
  nodes: Obligation[];
  pendingNodeIds: string[];
  expandedNodeIds: string[];
  affectedNodeIds: string[];
  links: MeaningDependency[];
  unresolved: string[];
};
export type AffectedComponent = { id: string; obligationIds: string[]; sceneIds: string[]; protectedRuleIds: string[] };
export type ImpactResult = { localDecisions: LocalDecision[]; obligations: Obligation[]; components: AffectedComponent[]; links: MeaningDependency[]; unaffectedSceneIds: string[]; unresolved: string[] };
export type ProposedEdit = {
  id: string;
  sourceId: string;
  sourcePath: string;
  oldQuote: string;
  replacement: string;
  sceneId: string;
  removedFactIds: string[];
  addedMeanings: string[];
  witnesses: Witness[];
};
export type Candidate = { id: string; componentId: string; edits: ProposedEdit[]; claimedCoverage: string[]; intendedPreservations: string[]; witnesses: Witness[] };
export type ReplayClaimJudgment = { coveredIds: string[]; unsupportedIds: string[]; explanation: string; witnesses: Witness[] };
export type Replay = { candidate: Candidate; snapshot: SourceSnapshot; analysis: Analysis; claimJudgment: ReplayClaimJudgment; consumedLocalDecisions: string[] };
export type SemanticFinding = {
  id: string;
  obligationId: string;
  status: 'satisfied' | 'violated' | 'unresolved';
  reason: string;
  requiredSourceIds: string[];
  witnesses: Witness[];
};
export type Audit = { candidateId: string; findings: SemanticFinding[]; missingObligationIds: string[]; sourceChecks: ExactCheck[] };
export type CandidateReview = { replay: Replay; audit: Audit };
export type CandidateChoice = { localDecisions: LocalDecision[]; selectedCandidateIds: string[]; rejected: { candidateId: string; violatedObligationIds: string[]; reason: string }[]; equivalentAlternatives: string[]; unresolved: string[] };
export type RepairDecision = { componentId: string; choice: CandidateChoice; audits: Audit[]; consumedLocalDecisionIds: string[] };
export type RepairState = {
  remainingComponentIds: string[];
  settledComponentIds: string[];
  decisions: RepairDecision[];
  selectedEdits: ProposedEdit[];
  unresolved: string[];
};
export type ExactCheck = { id: string; valid: boolean; detail: string };
export type PatchPacket = {
  baseRevision: string;
  requestId: string;
  selectedProposalIds: string[];
  edits: ProposedEdit[];
  unchangedSceneIds: string[];
  preservedUncertaintyIds: string[];
  obligationCoverage: { obligationId: string; status: 'satisfied' | 'unresolved'; sourceIds: string[] }[];
  rejectedProposals: { candidateId: string; violatedObligationIds: string[]; reason: string }[];
  unresolved: string[];
  sourceChecks: ExactCheck[];
  consumedLocalDecisionIds: string[];
};
