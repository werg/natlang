# Cave Mission — implemented decomposition

Status: source-only application; no live execution or training admission. Program `astra-app-cave-mission-v1`.

## Typed natural-language ownership

The application reconciles two uncertain coordinate frames, derives physical worlds, explores a belief-state graph and constructs a policy whose every observation outcome returns a sealed sample. All decisions live in numbered NL functions. The visible TypeScript file contains declarations only. The external checker replays a supplied policy and checks exact arithmetic/topology; it never selects an action or interprets evidence for the application.

Current size: 29 NL files, 745 physical lines, 526 body lines and 210 numbered steps;57 named typed contracts. Four branches own survey interpretation, world construction, all-outcome planning and independent replay. The initial architecture was reduced from 29 proposed modules by folding small phrase interpretations into their owning functions; there are no placeholder stages.

## Callable graph and stage deltas

Companion folders define visibility. The base rows below are actual files; the survey-v5 table below adds four connected item interpreters. Current `project.json` records all 28 parent-child edges and exact signatures.

| Callable | Input → result | Owned semantic delta |
|---|---|---|
| `planMission/auditMission/replayBranch/challengeAssumption.nl` | `MissionAction, RobotState, WorldModel, SurveyEvidence → CrossingDecision` | Independently rejudge one action against its world and carried-state premises. |
| `planMission/auditMission/replayBranch.nl` | `WorldModel, MissionRequest, SurveyEvidence, BeliefPolicy → WorldReplay` | Execute a proposed policy as pure state transitions in one declared world. |
| `planMission/auditMission.nl` | `MissionRequest, SurveyEvidence, WorldSet, PlanningResult → AuditedMission` | Replay the chosen policy in every supported physical world and challenge its assumptions. |
| `planMission/designPolicy/closePolicy.nl` | `BeliefGraph, StateValue[], MissionRequest → BeliefPolicy` | Project winning decisions into a complete rooted observation policy without changing them. |
| `planMission/designPolicy/evaluateBranches/settleValue.nl` | `ValueFrontier → ValueFrontier` | Own one all-outcome dynamic-programming value and its chosen action. |
| `planMission/designPolicy/evaluateBranches.nl` | `BeliefGraph, MissionRequest → StateValue[]` | Propagate winning values in increasing-energy order over the positive-cost belief DAG. |
| `planMission/designPolicy/expandBeliefState/availableActions/judgeCrossing.nl` | `MissionAction, RobotState, WorldModel, GearRule[] → CrossingDecision` | Invoke a carried-state scope judgment for one directed passage in one world. |
| `planMission/designPolicy/expandBeliefState/availableActions.nl` | `BeliefState, WorldSet, SurveyEvidence → StateExpansion` | Generate physically legal action candidates and their deterministic state changes across the belief. |
| `planMission/designPolicy/expandBeliefState/predictObservation/interpretSensorScope.nl` | `SensorRule, WorldModel, RobotState → SensorWorldAnswer` | Judge one sensor answer against the feature it can actually observe in one physical world. |
| `planMission/designPolicy/expandBeliefState/predictObservation.nl` | `MissionAction, BeliefState, WorldSet, SurveyEvidence → ObservationPartition` | Build the actual observation partition by consuming scoped sensor meanings for every world. |
| `planMission/designPolicy/expandBeliefState.nl` | `BeliefFrontier → BeliefFrontier` | Expand one belief state with physical actions and source-supported observation partitions. |
| `planMission/designPolicy.nl` | `MissionRequest, SurveyEvidence, WorldSet → PlanningResult` | Explore the finite belief-state graph and solve its all-outcome cost recurrence. |
| `planMission/reconcileMaps/alignFrames/assessAnchorPair.nl` | `LandmarkMention, LandmarkMention, Point, LandmarkPartition → AnchorAssessment` | Check a proposed coordinate binding against identity evidence rather than appearance alone. |
| `planMission/reconcileMaps/alignFrames.nl` | `SurveyEvidence, LandmarkPartition → FrameHypothesis[]` | Apply finite basis alternatives and test them against independent anchors. |
| `planMission/reconcileMaps/bindLandmarks.nl` | `SurveyEvidence → LandmarkPartition` | Use scoped description judgments to separate genuine anchors from repeated visual markers. |
| `planMission/reconcileMaps/buildWorlds/extendHypothesis/resolveContradiction.nl` | `PassageAssertion[], FrameHypothesis, SurveyEvidence → AnchorAssessment` | Decide whether overlapping survey claims genuinely conflict under one proposed transform. |
| `planMission/reconcileMaps/buildWorlds/extendHypothesis.nl` | `WorldFrontier → WorldFrontier` | Translate one hypothesis into a directed, conditioned world model and reconcile overlapping reports. |
| `planMission/reconcileMaps/buildWorlds.nl` | `SurveyEvidence, FrameHypothesis[] → WorldConstruction` | Construct and retain each consistent physical map through a shrinking hypothesis frontier. |
| `planMission/reconcileMaps.nl` | `SurveyEvidence → WorldSet` | Join landmarks and frame alternatives into all supported physical worlds. |
| `planMission/survey/readEquipment.nl` | `SourceClause[] → EquipmentSurvey` | Recover resource transitions and the distinct scopes of available observations. |
| `planMission/survey/readFrames/interpretConvention.nl` | `SourceClause[], SourceClause[] → FrameAlternative[]` | Translate one frame convention into explicit origin and basis alternatives. |
| `planMission/survey/readFrames.nl` | `SourceClause[] → FrameSurvey` | Interpret coordinate frames and inventory landmarks before choosing any physical alignment. |
| `planMission/survey/readPassages.nl` | `SourceClause[] → PassageAssertion[]` | Distinguish directed movement, geometric adjacency and known absence in passage descriptions. |
| `planMission/survey.nl` | `MissionRequest → SurveyEvidence` | Reduce nested survey collections into distinct frame, passage and equipment assertions. |
| `planMission.nl` | `MissionRequest → AuditedMission` | Reconcile uncertain cave maps and produce an all-world mission policy. |

## Different state machines

1. The nested folder reducer preserves all 31 clause IDs and witnesses, then joins frame, passage, equipment and sensor interpretations. Document counts alone confer no depth.
2. Landmark judgments distinguish the unique origin disk from duplicated blue circles. A directly awaited scoped NL literal owns same/different/unresolved decisions consumed by frame alignment.
3. World construction uses a shrinking frame-hypothesis frontier. The two u conventions survive as different directed physical maps. Unknown evidence is retained separately from a contradicted hypothesis.
4. Belief search keys location, energy, sample state, rope state and remaining world IDs. Each `iterateOn` step expands one exact state; its finite universe comes from source locations/gear, energies0..9 and nonempty world subsets. A source energy allowance is a problem fact, not a runtime cap.
5. Scoped crossing decisions gate physical transitions. Scoped sensor judgments produce the actual answer partition. Every outcome must remain in the AND branch. A one-cell observation whose only effect spends energy is safely omitted.
6. Positive source costs give a lower-energy dependency DAG. A separate reducer settles minimax values in increasing-energy order; each action wins only when all outcomes win. Choice compares worst energy, then worst observation count.
7. Policy closure projects selected edges without inventing routes. Per-world audit invokes a separate scoped challenge and consumes its nextRobot, then checks terminal sample/resource state and all-world coverage.

The final AuditedMission exposes worlds, policy, replays, findings and source coverage. Survey, full graph and values remain typed internal outputs in the trajectory; duplicating the entire search graph in the user certificate adds no decision.

## Exact seams and justification

- Existing Folder/loader: immutable bytes, paths and callable visibility.
- Type declarations: exact shape validation, no executable policy.
- Ordinary arithmetic/set operations inside generated stage code: apply already interpreted transforms and state identities.
- External `verify-result.mjs`: independently replay supplied actions in both fixture worlds, check source IDs/quotes, exact transforms/directions, sensor partitions, rope and carrier transitions, chronology and objective. It cannot certify quotation entailment or prose explanations; those require independent semantic review.
- External `check-verifier.mjs`: five equivalent positives and seventeen deliberate corruptions qualify verifier mechanism only.

No crisp planning engine, semantic regex interpreter or service chooses the policy. Reference-report.json is a hand-authored full typed witness outside model-visible roots.

## Connected first world

Ten nested documents contain 31 connected clauses. Lost handedness leaves east/west sample routes unresolved. A unique disk anchors the origin while repeated blue circles cannot identify a landing. ECHO observes the east chamber's structure; GLOW observes a repeated surface mark and cannot separate worlds. Rigging consumes the only spool and reveals nothing. Water gates are one-way. Picking up the sealed sample prevents tunnel return, making the correctly rigged wide chimney necessary. Therefore the observation must precede the irreversible rope/route commitment. Both eight-action branches cost nine and return with zero energy.

## Qualification

Source loader, typed reference and twenty-two external checker controls pass. Original fixture/oracle and canonical policy independently reviewed by Sol (439ba667). The initial live diagnostic confirms several child-result handoffs but exposes conditional-compatibility wording and quote-provenance defects. The separate conditional-anchor-v2 source revision needs peer review and a new pinned diagnostic; no complete live success is credited. The manifest keeps source-only status, shared source-world split identity and separate model-facing debt until live qualification.

## Conditional-anchor-v2 source delta

The NL anchor assessment now decides consistency conditional on one candidate transform: matching source-backed coordinates admit a possible binding, contradictions reject, and missing conditional premises remain unknown. Selection of the actual reflection stays unresolved across worlds. The landmark literal returns semantic judgments and clause-keyed exact quote strings; its parent owns original source ID/path binding and substring checks before consuming the result. Alignment validates returned quotation provenance before inserting conditional bindings. These exact checks do not decide place identity or select a map. No crisp decision engine was added.

## Conditional-binding-table-v3 source delta

The second live diagnostic established conditional true assessments but lost their named pair identities before constructing FrameHypothesis. Alignment now owns a typed ConditionalAnchorBinding table, immediately inserting each actual child return with the candidate pair and transform. Numbered NL steps interpret the rows, handle conflicting or missing identity evidence, and project accepted named bindings before considering source-local chamber names. The local-name rule has a semantic naming obligation, not a blanket missing-result fallback. Passage reconciliation conditions its judgment on the supplied world instead of asking which reflection is actual.

The new table is internal; external report, fixture, oracle and canonical policy remain unchanged. Coordinate arithmetic and source quotation checks are exact mechanics. Candidate identity, naming scope and conditional consistency remain NL policy. No JavaScript semantic engine or new wrapper stage was added. Source-only pending independent review and a fresh pinned live diagnostic.

## Physical-existence-v4 source delta

The v3 trace consumed conditional binding rows correctly, then conflated a physically present outward gate with its forbidden reverse traversal. PassageAssertion now names physicalExistence explicitly. The NL reader separately interprets connection presence, ordered permission and carried-state conditions, preserving reverse-prohibition evidence on the present one-way row. World construction creates travel edges only from present connections and keeps complete negative claims for reconciliation; the conditional comparison respects ordered endpoints. This is an internal typed semantic distinction; final report, source fixture, oracle and canonical policy remain stable. Exact mechanics only project already interpreted allowed directions and validate source membership. The stopped v3 partial remains a source diagnostic without model-negative or admission credit.

## Survey-item-natural-language v5 decomposition

The stopped physical-v4 diagnostic remains frozen: its extraction calls constructed case-specific
passage, landmark and equipment tables inside eval. One equipment-wide literal still delegated to a
whole-survey table. V5 makes extraction a connected typed NL pipeline before another live diagnostic.

| Unit | Natural-language ownership | Actual returned contribution | Exact seam |
| --- | --- | --- | --- |
| Reader inventory literals | Identify every original reading unit, including several units in one clause; join only source-established supporting scope | SurveyReadingUnit identities, original clause IDs and focus question | Original ID membership; no assertion fields precomputed |
| readFrames/interpretLandmark | Judge one framed landmark's position, features and identity scope | One nullable LandmarkMeaning plus clause-keyed exact quotes | Parent binds original source ID/path, validates quote bytes |
| readFrames/interpretConvention | Judge one local frame's complete allowed origin/basis alternatives | ConventionInterpretation with actual typed alternatives and unresolved premises | Exact vector arithmetic after NL interpretation; parent quote binding |
| readPassages/interpretPassage | Judge one physical connection separately from direction, load and existence | One nullable PassageMeaning; preserves absence, reverse prohibition and unknown | Endpoint membership in actual FrameSurvey; parent quote binding |
| readEquipment/interpretGear | Judge one original resource condition or transition | One nullable GearMeaning; consumption, sealed carrier and irreversibility remain distinct | Parent source-domain checks |
| readEquipment/interpretSensor | Judge one sensor's answer categories, scope, costs and state effects | One nullable SensorMeaning, not an inferred answer in a chosen world | Parent source-domain checks |
| survey reader joins | Consume each actual child result, preserve disagreements and union original support | Existing FrameSurvey/EquipmentSurvey/passage collection | Exact concatenation and provenance, no semantic lookup tables |

Frame extraction precedes passage extraction. The actual FrameSurvey supplies endpoint/frame keys;
passage interpretation cannot re-create a second landmark table. A frame's local origin remains a
separate mention from its named global anchor. Complete finite handedness alternatives are supported
uncertainty, not a blocking unresolved premise. Conditional alignment still chooses no actual reflection
and an optional false candidate rejects only that pair.

Each original inventory unit is visited once. Unit focus identifies its source subject/question without
preselecting coordinates, directions, costs, effects or sensor capability. A child scoped literal receives
that unit and its original clauses, is directly awaited, and its actual semantic row supplies the parent's
fields. Supporting quotes are nonempty exact substrings of supplied original clause witnesses; the
parent owns source identity. Missing decisive evidence is explicit unresolved data. Existing policy,
world, fixture and external canonical certificate remain unchanged. No crisp extraction classifier is
introduced. The new source needs native type/source replay, per-item control review, independent peer
review and roughly48 live samples; none of those preparation receipts is training admission.

## Connected value-consumer depth revision

The audit found no visible executable TypeScript policy. Survey-v5 already obtains actual per-unit NL meanings; its frozen package stays separately reviewed. settleValue expands seven steps to ten, making the action-choice versus uncontrollable-observation distinction explicit, joining every actual world/outcome dependency, separating missing graph evidence from a losing world, valuing both worst-case coordinates and retaining tied optimal witnesses. It consumes existing typed expansions and child StateValues without reselecting routes or rebuilding source assertions. No new leaf or crisp decision helper is added. Exact maxima/arithmetic project already established policy alternatives. Fixture, report types and canonical policy remain unchanged; the source delta needs its own peer/live qualification.

## Original local names and one-unit frame progress v8

The v7 live reader accepted a generated landmark name because SurveyReadingUnit lacked the original
local name; it then repeated an already returned convention judgment. V8 gives inventory an exact
originalLocalName for each landmark, preserves null for other unit kinds, and validates the complete
original name token before interpretation. The child judges position/description/uncertainty while
copying that supplied name; physical identity still belongs to reconciliation.

A named advanceFrameUnit consumes one original unit in FrameReadFrontier, retaining actual nullable
child results, source-bound projected rows and accepted/unresolved/failed outcomes. readFrames folds
those actual completed projections into FrameSurvey.unitResults/alternatives/landmarks. Support
selection remains scoped NL in the single-unit step; exact provenance/progress checks are mechanism.
Existing convention/landmark helpers are reused through explicit uses, without duplicate implementations.
Fixture/oracle/final mission reference and downstream physical/directional policy remain unchanged.
