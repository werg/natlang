import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createNatlangRuntime, loadNatlang } from "../../../../ts-host/dist/index.js";
import { scriptedModel } from "../../../../ts-host/test/support/natlang.mjs";

const casesUrl = new URL("../../../../training/semantic_applications/design_briefs/039-apprenticeship-practice-simulator/cases.json", import.meta.url);
const rootUrl = new URL("../simulateApprenticeshipTurn.nl", import.meta.url);
const fixtureTurn = async id => {
  const cases = JSON.parse(await readFile(casesUrl, "utf8"));
  return structuredClone(cases.find(item => item.id.endsWith(id)).inputs.turn);
};
const rootCode = `const replay = await inspectTurnIdentifier(turn);
if (replay.status === "replay") { const r = replay.receipt; return { sessionId: turn.sessionId, status: "simulated", utterance: r.utterance, typedAction: r.typedAction, transition: r.transition, challenge: r.challenge, debrief: r.debrief, blockers: [], checkpoint: turn.checkpoint, replayed: true, commitAllowed: false }; }
if (replay.status === "conflict") return { sessionId: turn.sessionId, status: "replay-conflict", utterance: null, typedAction: null, transition: null, challenge: null, debrief: null, blockers: replay.blockers, checkpoint: turn.checkpoint, replayed: false, commitAllowed: false };
if (replay.status === "unresolved") return { sessionId: turn.sessionId, status: "replay-unresolved", utterance: null, typedAction: null, transition: null, challenge: null, debrief: null, blockers: replay.blockers, checkpoint: turn.checkpoint, replayed: false, commitAllowed: false };
const utterance = await interpretTraineeUtterance(turn);
if (utterance.kind.kind === "question") return { sessionId: turn.sessionId, status: "question-answered", utterance, typedAction: null, transition: null, challenge: null, debrief: null, blockers: [], checkpoint: turn.checkpoint, replayed: false, commitAllowed: false };
if (utterance.kind.kind === "ambiguous") return { sessionId: turn.sessionId, status: "clarification-needed", utterance, typedAction: null, transition: null, challenge: null, debrief: null, blockers: utterance.clarificationNeeded, checkpoint: turn.checkpoint, replayed: false, commitAllowed: false };
const typedAction = await matchPermittedProcedureStep(utterance, turn);
if (typedAction.status === "clarification") return { sessionId: turn.sessionId, status: "clarification-needed", utterance, typedAction, transition: null, challenge: null, debrief: null, blockers: typedAction.clarificationNeeded, checkpoint: turn.checkpoint, replayed: false, commitAllowed: false };
throw new Error("expected a clarification action");`;

const matchingReplayInspector = `const records = turn.checkpoint.turnReceipts ?? [];
const found = records.filter(record => record.binding.turnId === turn.turnId);
if (found.length > 1) return { status: "conflict", receipt: null, blockers: ["multiple receipts share this turn ID"] };
if (found.length === 1) {
  const binding = found[0].binding;
  const same = binding.sessionId === turn.sessionId && binding.turnId === turn.turnId &&
    JSON.stringify(binding.sourceScenario) === JSON.stringify(turn.sourceScenario) && binding.utterance === turn.utterance &&
    JSON.stringify(binding.sourceEvidence) === JSON.stringify(turn.sourceEvidence) && binding.authorityStatus === turn.authorityStatus &&
    JSON.stringify(binding.actor) === JSON.stringify(turn.actor) && JSON.stringify(binding.world) === JSON.stringify(turn.world) &&
    JSON.stringify(binding.procedure) === JSON.stringify(turn.procedure) && binding.scenarioFamily === turn.scenarioFamily &&
    binding.learningObjective === turn.learningObjective && binding.seed === turn.seed &&
    JSON.stringify(binding.actionHistory) === JSON.stringify(turn.actionHistory);
  return same ? { status: "replay", receipt: found[0], blockers: [] } : { status: "conflict", receipt: null, blockers: ["turn ID is bound to different input"] };
}
if (turn.checkpoint.appliedTransitionIds.includes(turn.turnId) || turn.checkpoint.seenChallengeIds.includes(turn.turnId) || turn.checkpoint.lastTransitionId === turn.turnId)
  return { status: "unresolved", receipt: null, blockers: ["legacy checkpoint has no complete turn receipt"] };
return { status: "new", receipt: null, blockers: [] };`;

test("procedure clarification returns before simulator advancement and preserves the checkpoint", async () => {
  const turn = await fixtureTurn("operational-5");
  const script = scriptedModel(opening => {
    if (opening.includes("If replay.status is replay")) return rootCode;
    if (opening.includes("Search turn.checkpoint.turnReceipts")) return `return { status: "new", receipt: null, blockers: [] };`;
    if (opening.includes("Preserve its exact kind, intent, and target"))
      return `const kind = await classifyQuestionOrAction(turn.utterance); return { kind, clarificationNeeded: [] };`;
    if (opening.includes("Classify the communicative act"))
      return `return { kind: "action-proposal", intent: "isolate press", target: "press" };`;
    if (opening.includes("Compare the proposed operation with the scenario"))
      return `return { turnId: turn.turnId, kind: "isolate", target: "press", preconditions: [], policyRef: "procedure:lockout", status: "clarification", clarificationNeeded: ["Which isolation control will be applied before reaching into the press?"] };`;
    return null;
  });
  const simulate = loadNatlang(rootUrl.pathname);
  const result = await createNatlangRuntime({ model: script.driver }).run(() => simulate(turn));

  assert.equal(result.status, "clarification-needed");
  assert.equal(result.typedAction.status, "clarification");
  assert.deepEqual(result.blockers, ["Which isolation control will be applied before reaching into the press?"]);
  assert.equal(result.transition, null);
  assert.equal(result.challenge, null);
  assert.equal(result.debrief, null);
  assert.deepEqual(result.checkpoint, turn.checkpoint);
  assert.equal(result.replayed, false);
  assert.equal(script.openings.length, 5, "clarification stops before world advancement, challenge selection and debrief");
});

test("complete turn receipt replays without reclassification and preserves newer checkpoint state", async () => {
  const turn = await fixtureTurn("operational-5");
  const binding = {
    sessionId: turn.sessionId, turnId: turn.turnId, sourceScenario: turn.sourceScenario, utterance: turn.utterance,
    sourceEvidence: turn.sourceEvidence, authorityStatus: turn.authorityStatus, actor: turn.actor, world: turn.world,
    procedure: turn.procedure, scenarioFamily: turn.scenarioFamily, learningObjective: turn.learningObjective,
    seed: turn.seed, actionHistory: turn.actionHistory,
  };
  const utterance = { kind: { kind: "action-proposal", intent: "isolate press", target: "press" }, clarificationNeeded: [] };
  const typedAction = { turnId: turn.turnId, kind: "isolate", target: "press", preconditions: ["stop"], policyRef: "procedure:lockout", status: "simulated", clarificationNeeded: [] };
  const transition = { transitionId: `${turn.turnId}:transition`, actionId: turn.turnId, priorStateJson: turn.world.stateJson,
    nextStateJson: "later simulated world", revealedFactIds: [], consequence: "in-simulator isolation recorded", invariantFailures: [] };
  const challenge = { id: `${turn.turnId}:challenge`, seed: turn.seed, trigger: "new pressure reading", visibleFactSnapshot: ["press jammed"], recoveryPaths: ["stop"], deferredReason: "" };
  const debrief = { plannedActionKind: "isolate", plannedTarget: "press", simulatedActionId: turn.turnId,
    simulatedTransitionId: transition.transitionId, simulatedConsequence: transition.consequence, actionEvidence: [], currentActionObserved: false,
    missedObligations: ["no learner action record was supplied"], effectiveChecks: [], outcomeLuck: "unassessed", nextPracticeTarget: "record the required precondition check" };
  const receipt = { binding, utterance, typedAction, transition, challenge, debrief };
  const currentCheckpoint = { ...turn.checkpoint, version: 8, appliedTransitionIds: [turn.turnId, "later-turn"],
    lastTransitionId: "later-transition", worldStateJson: "state after later-turn", turnReceipts: [receipt] };
  const replayTurn = { ...turn, checkpoint: currentCheckpoint };
  const script = scriptedModel(opening => {
    if (opening.includes("If replay.status is replay")) return rootCode;
    if (opening.includes("Search turn.checkpoint.turnReceipts")) return matchingReplayInspector;
    return null;
  });
  const simulate = loadNatlang(rootUrl.pathname);
  const result = await createNatlangRuntime({ model: script.driver }).run(() => simulate(replayTurn));

  assert.equal(result.status, "simulated");
  assert.equal(result.replayed, true);
  assert.deepEqual(result.utterance, receipt.utterance);
  assert.deepEqual(result.typedAction, receipt.typedAction);
  assert.deepEqual(result.transition, receipt.transition);
  assert.deepEqual(result.challenge, receipt.challenge);
  assert.deepEqual(result.debrief, receipt.debrief);
  assert.deepEqual(result.checkpoint, currentCheckpoint, "replay does not rewind newer state or checkpoint history");
  assert.equal(script.openings.length, 2, "replay uses only the root and exact receipt lookup");

  const changedTurn = { ...replayTurn, utterance: `${turn.utterance} changed input` };
  const changedResult = await createNatlangRuntime({ model: script.driver }).run(() => simulate(changedTurn));
  assert.equal(changedResult.status, "replay-conflict");
  assert.equal(changedResult.transition, null);
  assert.deepEqual(changedResult.checkpoint, currentCheckpoint);
});

test("legacy applied turn ID without its complete receipt stays unresolved", async () => {
  const turn = await fixtureTurn("operational-5");
  const legacyCheckpoint = { ...turn.checkpoint, appliedTransitionIds: [turn.turnId], lastTransitionId: turn.turnId };
  const script = scriptedModel(opening => {
    if (opening.includes("If replay.status is replay")) return rootCode;
    if (opening.includes("Search turn.checkpoint.turnReceipts")) return matchingReplayInspector;
    return null;
  });
  const simulate = loadNatlang(rootUrl.pathname);
  const result = await createNatlangRuntime({ model: script.driver }).run(() => simulate({ ...turn, checkpoint: legacyCheckpoint }));
  assert.equal(result.status, "replay-unresolved");
  assert.equal(result.transition, null);
  assert.deepEqual(result.checkpoint, legacyCheckpoint);
  assert.equal(script.openings.length, 2);

  const seenOnlyCheckpoint = { ...turn.checkpoint, seenChallengeIds: [turn.turnId] };
  const seenOnly = await createNatlangRuntime({ model: script.driver }).run(() => simulate({ ...turn, checkpoint: seenOnlyCheckpoint }));
  assert.equal(seenOnly.status, "replay-unresolved");
  assert.deepEqual(seenOnly.checkpoint, seenOnlyCheckpoint);
});
