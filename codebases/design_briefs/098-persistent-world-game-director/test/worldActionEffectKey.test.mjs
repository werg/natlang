import test from "node:test";
import assert from "node:assert/strict";
import { samePendingWorldActionBinding, worldActionEffectKey } from "../directPersistentWorld/prepareWorldActionEffects/prepareWorldActionEffects.ts";
import { createNatlangRuntime, loadNatlang } from "../../../../ts-host/dist/index.js";
import { scriptedModel } from "../../../../ts-host/test/support/natlang.mjs";

test("effect kind separates apply from status read for the same logical action", () => {
  const apply = worldActionEffectKey("apply-player-actions", "world-7", 12, "rules-a", "client-44");
  const read = worldActionEffectKey("read-action-status", "world-7", 12, "rules-a", "client-44");
  assert.notEqual(apply, read);
});

test("same logical action identity keeps one stable key across retries", () => {
  const first = worldActionEffectKey("apply-player-actions", "world-7", 12, "rules-a", "client-44");
  const retry = worldActionEffectKey("apply-player-actions", "world-7", 12, "rules-a", "client-44");
  assert.equal(first, retry);
});

test("pending binding comparator rejects changed payload or prior revision", () => {
  const pending = {
    clientActionId: "client-44", worldId: "world-7", priorRevision: 12,
    ruleVersion: "rules-a", actions: [{ id: "move", actorId: "player-1", kind: "move", targetId: "door", location: "hall", resourceDelta: "0", requires: [], clientActionId: "client-44" }],
  };
  assert.equal(samePendingWorldActionBinding(pending, "world-7", 12, "rules-a", "client-44", pending.actions), true);
  assert.equal(samePendingWorldActionBinding(pending, "world-7", 12, "rules-a", "client-44", [{ ...pending.actions[0], targetId: "vault" }]), false);
  assert.equal(samePendingWorldActionBinding(pending, "world-7", 13, "rules-a", "client-44", pending.actions), false);
});

test("Natlang callable resolves both named helpers through its TypeScript module namespace", async () => {
  const script = scriptedModel(opening => {
    assert.match(opening, /prepareWorldActionEffects\.worldActionEffectKey/);
    assert.match(opening, /prepareWorldActionEffects\.samePendingWorldActionBinding/);
    return `const action = { id: "move", actorId: "player-1", kind: "move", targetId: "door", location: "hall", resourceDelta: "0", requires: [], clientActionId: "client-44" };
const pending = { clientActionId: "client-44", worldId: "world-7", priorRevision: 12, ruleVersion: "rules-a", actions: [action] };
const key = prepareWorldActionEffects.worldActionEffectKey("apply-player-actions", "world-7", 12, "rules-a", "client-44");
const matches = prepareWorldActionEffects.samePendingWorldActionBinding(pending, "world-7", 12, "rules-a", "client-44", [action]);
return { effects: [], checkpoint, blockers: [key, String(matches)] };`;
  });
  const prepare = loadNatlang(new URL("../directPersistentWorld/prepareWorldActionEffects.nl", import.meta.url).pathname);
  const runtime = createNatlangRuntime({ model: script.driver });
  const world = { id: "world-7", revision: 12, ruleVersion: "rules-a", locations: [], resources: [], characters: [], events: [], boundaries: [] };
  const action = { id: "move", actorId: "player-1", kind: "move", targetId: "door", location: "hall", resourceDelta: "0", requires: [], clientActionId: "client-44" };
  const pending = { clientActionId: "client-44", worldId: "world-7", priorRevision: 12, ruleVersion: "rules-a", actions: [action] };
  const checkpoint = { worldId: "world-7", revision: 12, ruleVersion: "rules-a", pendingActionIds: ["client-44"], unknownReceiptIds: [],
    pendingActions: [pending], boundaryRevisions: [], replayCursor: "r12" };
  const reconciliation = { receipts: [], pendingActionIds: ["client-44"], unknownReceiptIds: [], committedEvents: [], updatedWorld: world,
    status: "awaiting-simulator", blockers: [] };
  const result = await runtime.run(() => prepare(world, [action], checkpoint, reconciliation));
  assert.deepEqual(result.blockers, ["apply-player-actions:world-7:12:rules-a:client-44", "true"]);
  assert.deepEqual(result.effects, []);
});
