import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createNatlangRuntime, loadNatlang } from "../../../../ts-host/dist/index.js";
import { scriptedModel } from "../../../../ts-host/test/support/natlang.mjs";

test("question turn consumes the classifier and visible-fact answer without advancing state", async () => {
  const cases = JSON.parse(await readFile(new URL("../../../../training/semantic_applications/design_briefs/039-apprenticeship-practice-simulator/cases.json", import.meta.url), "utf8"));
  const turn = cases.find(item => item.id.endsWith("operational-1")).inputs.turn;
  const script = scriptedModel(opening => {
    if (opening.includes("If utterance.kind.kind is question"))
      return `const utterance = await interpretTraineeUtterance(turn);
if (utterance.kind.kind === "question") return { sessionId: turn.sessionId, status: "question-answered", utterance, typedAction: null, transition: null, challenge: null, debrief: null, blockers: [], checkpoint: turn.checkpoint, replayed: false, commitAllowed: false };
if (utterance.kind.kind === "ambiguous") return { sessionId: turn.sessionId, status: "clarification-needed", utterance, typedAction: null, transition: null, challenge: null, debrief: null, blockers: utterance.clarificationNeeded, checkpoint: turn.checkpoint, replayed: false, commitAllowed: false };
throw new Error("question fixture was not classified as a question");`;
    if (opening.includes("Preserve its exact kind, intent, and target"))
      return `const kind = await classifyQuestionOrAction(turn.utterance);
if (kind.kind === "question") { const answer = await resolveFromVisibleActorFacts(turn.utterance, turn.actor, turn.world); return { kind, answer, clarificationNeeded: [] }; }
if (kind.kind === "ambiguous") return { kind, clarificationNeeded: ["Clarify whether this is a question or a proposed operation, and identify its intended target."] };
return { kind, clarificationNeeded: [] };`;
    if (opening.includes("Classify the communicative act"))
      return `return { kind: "question", intent: "learn the gauge position", target: "G39" };`;
    if (opening.includes("Search the named actor's visible facts"))
      return `return { answer: "Gauge G39 is upstream of valve V39.", factRefs: ["equipment diagram G39"], withheld: [] };`;
    return null;
  });
  const runTurn = loadNatlang(new URL("../simulateApprenticeshipTurn.nl", import.meta.url).pathname);
  const result = await createNatlangRuntime({ model: script.driver }).run(() => runTurn(turn));

  assert.equal(result.status, "question-answered");
  assert.equal(result.utterance.kind.kind, "question");
  assert.equal(result.utterance.kind.target, "G39");
  assert.deepEqual(result.utterance.answer, {
    answer: "Gauge G39 is upstream of valve V39.", factRefs: ["equipment diagram G39"], withheld: [],
  });
  assert.equal(result.transition, null);
  assert.equal(result.challenge, null);
  assert.deepEqual(result.checkpoint, turn.checkpoint);
  assert.equal(script.openings.length, 4, "only root, interpretation, classifier and visible-fact answer execute");
});
