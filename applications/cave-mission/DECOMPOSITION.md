# Cave Mission — implemented decomposition

Status: source-only application; no live execution or training admission. Program `astra-app-cave-mission-v1`.

## Typed natural-language ownership

The application reconciles two uncertain coordinate frames, derives physical worlds, explores a belief-state graph and constructs a policy whose every observation outcome returns a sealed sample. All decisions live in numbered NL functions. The visible TypeScript file contains declarations only. The external checker replays a supplied policy and checks exact arithmetic/topology; it never selects an action or interprets evidence for the application.

Actual size: 25 NL files, 539 total lines, 355 body lines and 172 numbered steps; 43 named typed contracts. Four branches own survey interpretation, world construction, all-outcome planning and independent replay. The initial architecture was reduced from 29 proposed modules by folding small phrase interpretations into their owning functions; there are no placeholder stages.

## Callable graph and stage deltas

Companion folders define visibility. Every row below is an actual file; `project.json` records all 24 parent-child edges and exact signatures.

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
