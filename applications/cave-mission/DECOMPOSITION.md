# Cave Mission — application decomposition

Status: architecture for independent review; no runnable or training claim. Program ID: `astra-app-cave-mission-v1`.

## Behavior

Compile conflicting local survey notes into a set of still-possible cave maps, then construct a mission policy that succeeds under every supported map. The policy may make an observation before choosing a route. It must preserve the distinction between an unobserved corridor, a blocked corridor and a corridor that does not exist. Inputs include nested survey directories, frame conventions, landmark descriptions, crossing affordances, instrument observations, gear state and a source-defined energy allowance.

The result is a contingent policy and an evidence-backed model set, not a single guessed map. The application controls no physical robot. All geometry and mission rules describe a closed fictional world.

Natural-language stages own map interpretation, source reconciliation, affordance judgment, observation choice and planning. Exact arithmetic and transition checks remain verifiers or narrow services. Existing Folder, `nl`, `iterateOn`, callable-folder and service abstractions supply the mechanism.

## Module graph

```text
planMission.nl
planMission/
  survey.nl
  survey/
    readFrame.nl
    readFrame/interpretOrientation.nl
    readPassages.nl
    readPassages/interpretPassageClaim.nl
    readAffordances.nl
    readAffordances/interpretGearScope.nl
  reconcileMaps.nl
  reconcileMaps/
    bindLandmarks.nl
    bindLandmarks/compareLandmarkMeaning.nl
    alignFrames.nl
    alignFrames/assessAnchorPair.nl
    buildWorlds.nl
    buildWorlds/extendHypothesis.nl
    buildWorlds/extendHypothesis/resolveContradiction.nl
  designPolicy.nl
  designPolicy/
    expandBeliefState.nl
    expandBeliefState/availableActions.nl
    expandBeliefState/availableActions/judgeCrossing.nl
    expandBeliefState/predictObservation.nl
    expandBeliefState/predictObservation/interpretSensorScope.nl
    evaluateBranches.nl
    evaluateBranches/chooseObservation.nl
    evaluateBranches/chooseObservation/comparePartitions.nl
    closePolicy.nl
  auditMission.nl
  auditMission/replayBranch.nl
  auditMission/replayBranch/challengeAssumption.nl
```

Estimated size: 29 meaningful `.nl` functions and 650–850 NL lines, plus substantial typed contracts. Reconciliation explores world hypotheses; planning explores belief states and observation partitions. They are different state machines, not two repetitions of a generic per-record loop. Deep nested judgments occur where frame alignment or crossing scope actually needs them.

## Typed contracts and owned deltas

| Unit | Input → owned result | Semantic transformation |
|---|---|---|
| `survey` | `Folder, MissionRequest → SurveyEvidence` | Interpret each survey folder in its own frame and provenance context. |
| `readFrame` | `SourceDocument, FrameContext → FrameAssertion[]` | Separate heading conventions, origin claims, handedness and local restart notes. |
| `interpretOrientation` | `OrientationPhrase, Anchors → OrientationConstraint` | Resolve observer-facing versus map-facing directions and which frame they modify. |
| `readPassages` | `SourceDocument, LocalFrame → PassageAssertion[]` | Distinguish traversal, line of sight, attempted entry, conditional opening and inferred adjacency. |
| `readAffordances` | `GearDescriptions, PassageDescriptions → AffordanceClaim[]` | Interpret who or what can cross under which carried-state conditions. |
| `bindLandmarks` | `SurveyEvidence → LandmarkPartition` | Decide whether differently named landmarks are the same physical place; retain plausible alternatives. |
| `alignFrames` | `LandmarkPartition, FrameAssertions → FrameHypotheses` | Join anchor pairs and orientation scope into candidate transforms. |
| `assessAnchorPair` | `AnchorPair, LocalDescriptions → AnchorAssessment` | Reject same-looking but context-incompatible anchors before exact coordinate calculations. |
| `buildWorlds` | `FrameHypotheses, PassageAssertions → WorldSet` | Extend partial maps, classify contradictions and preserve all supported alternatives. |
| `resolveContradiction` | `ConflictingClaims, SourceContext → ContradictionDelta` | Distinguish stale state, different referent, perspective and actual inconsistent evidence. |
| `designPolicy` | `WorldSet, MissionState → BeliefPolicy` | Search a finite graph of location, gear, energy and remaining knowledge. |
| `availableActions` | `BeliefState, AffordanceClaims → ActionCandidates` | Determine actions known safe and actions requiring an observation before commitment. |
| `judgeCrossing` | `Passage, GearState, ModelSubset → CrossingDecision` | Interpret clearance, direction and one-use gear semantics for the current carried state. |
| `predictObservation` | `Query, BeliefState → ObservationPartition` | Derive what each sensor answer would mean in each possible world. |
| `interpretSensorScope` | `SensorClause, Target, Context → ObservationMeaning` | Separate a nondetection from proof of absence and respect orientation or occlusion restrictions. |
| `evaluateBranches` | `ExpandedBeliefGraph → PolicyValues` | Propagate successful/failed branches and compare worst-case mission cost without choosing a single favored world. |
| `chooseObservation` | `CompetingPartitions, MissionGoal → QueryDecision` | Prefer a query that separates action-relevant worlds; explain a superficially informative but useless query. |
| `closePolicy` | `PolicyValues, BeliefGraph → BeliefPolicy` | Construct complete observation branches and witness every terminal goal/return state. |
| `auditMission` | `Policy, WorldSet, Evidence → AuditedMission` | Challenge frame assumptions and replay every surviving world through the policy. |

Core types include `SourceDocument`, `Witness`, `FrameConstraint`, `LandmarkHypothesis`, `WorldModel`, `BeliefState`, `ObservationPartition`, `ActionDelta`, `PolicyNode`, `BranchFailure` and `AuditedMission`. Every inferred edge or affordance carries exact source quotations. Unknown direction or clearance is an explicit unresolved variant, never a default numeric coordinate or empty description.

## Structured pseudocode commitments

1. Reduce the actual nested survey folders using their local frame records. Invoke different interpretation branches for frame, passage and gear evidence. Preserve the record-to-fact ownership map.
2. Iterate landmark candidate pairs. A scoped typed NL literal compares the two descriptions with the current anchor context; consume its same/different/unresolved result to modify the candidate partition.
3. Build map hypotheses through `iterateOn` over a finite assignment frontier. An extension resolves one previously unassigned anchor/frame alternative. Impossible assignments are retained as explicit counterexamples with the conflicting source clauses.
4. Explore belief states with a worklist. A state key consists of location, remaining world IDs, source-defined energy and gear state. Each iteration removes one not-yet-expanded state; newly discovered states come from the finite source-defined universe. Termination is exhaustion or a proved complete goal policy, not an arbitrary number of model turns.
5. Inside `judgeCrossing`, formulate an NL literal specific to the carried gear and passage language. Its typed decision must gate whether a traversal edge is inserted.
6. Inside `predictObservation`, invoke a separate literal for each instrument/target scope. Its model partition must become the actual observation branches; a prompt that describes a literal without calling it earns no credit.
7. Propagate branch values across the finite belief graph. A policy wins only if every supported observation outcome wins. Compare exact energy/path costs after semantic legality is established.
8. `auditMission` replays the proposed policy against all remaining worlds and asks a local counterexample literal to challenge each dependence on orientation, visibility or gear consumption. A failed branch returns the first unsupported assumption and its evidence path.

## Exact seams

- **Folder snapshots and stable IDs:** exact input mechanism; no interpretation.
- **Grid transforms and distance arithmetic:** apply an already selected finite transform; never choose a frame from ambiguous language.
- **State identity and set operations:** exact serialization, deduplication and subset checks; never decide which worlds are supported.
- **Transition verifier:** check a proposed action against the formal world and carried state, preserving resource conservation and one-use effects. It returns counterexamples, not the next action.
- **Witness verifier:** check exact quote/path membership; independent semantic review checks entailment.
- **Policy verifier:** enumerate the declared possible worlds and execute all observation branches to verify coverage, termination and the stated objective. Natural-language planning creates the candidate policy.

The source-defined energy allowance is part of the problem world. It is not a runtime budget or a cap on reasoning.

## First connected world

The first source world will have 28–36 relevant facts across three survey teams, two plausible map alignments, a duplicated landmark description, a temporary one-way water gate and gear whose bulk changes which passages are traversable. Two sensors have different observation scopes: one can discriminate the route-relevant map ambiguity; the other returns the same observation in both surviving worlds. A successful plan must gather the right evidence before consuming the only return-enabling resource.

The exact oracle accepts equivalent policy encodings that preserve all world branches, legal transitions, mission goals and optimality requirements. Node names and prose explanations are not gold spellings. An incomplete observation branch, silently discarded world or unsafe retry is a semantic failure even if one sampled route succeeds.

## Readiness

First prove typed/callable-folder integrity, source visibility, independent source-oracle derivation and exact verifier positive/negative cases. Then execute the actual NL application live and inspect stage deltas, literal consumption, structural iteration and branch coverage. Adoption requires the repository's model-facing measurement gate; training admission remains separate. The entire project and its scenario variants share their declared source-world split identities.
