import test from "node:test";
import assert from "node:assert/strict";
import reconcileScaledBatches from "../developMenu/reconcileKitchenBatches/reconcileScaledBatches.ts";
import { createNatlangRuntime, loadNatlang } from "../../../../ts-host/dist/index.js";
import { scriptedModel } from "../../../../ts-host/test/support/natlang.mjs";

const dish = (id, quantityMinor = "100", yieldServings = "10") => ({
  id,
  recipe: [{ ingredientId: "flour", quantityMinor, unit: "g", yieldServings, wasteBasis: "none" }],
});
const batch = (dishId, quantityMinor = "200") => ({ dishId, ingredientId: "flour", quantityMinor, unit: "g" });

test("reconciles exact integer scaling against the supplied batch ledger", () => {
  const result = reconcileScaledBatches(20, [dish("bread")], [batch("bread")]);
  assert.deepEqual(result.expected, [batch("bread")]);
  assert.deepEqual(result.discrepancies, []);
  assert.deepEqual(result.blockers, []);
});

test("returns typed blockers for invalid covers, malformed/negative quantities, yields and fractional scaling", () => {
  assert.match(reconcileScaledBatches(0, [dish("bread")], []).blockers[0], /covers/);
  assert.ok(reconcileScaledBatches(20, [dish("bread", "-1")], []).blockers.length);
  assert.ok(reconcileScaledBatches(20, [dish("bread", "1x")], []).blockers.length);
  assert.ok(reconcileScaledBatches(20, [dish("bread", "100", "0")], []).blockers.length);
  assert.ok(reconcileScaledBatches(3, [dish("bread", "1", "2")], []).blockers.some(x => x.includes("nonintegral")));
});

test("retains duplicate recipe and supplied rows as discrepancies", () => {
  const result = reconcileScaledBatches(20, [dish("bread"), dish("bread")], [batch("bread"), batch("bread")]);
  assert.ok(result.discrepancies.some(x => x.includes("duplicate recipe line")));
  assert.ok(result.discrepancies.some(x => x.includes("duplicate supplied batch")));
});

test("Natlang callable resolves the default kitchen scaling module export", async () => {
  const script = scriptedModel(opening => {
    assert.match(opening, /reconcileScaledBatches/);
    return "return reconcileScaledBatches(ledger.brief.covers, dishes, ledger.exactScaledBatches);";
  });
  const reconcile = loadNatlang(new URL("../developMenu/reconcileKitchenBatches.nl", import.meta.url).pathname);
  const runtime = createNatlangRuntime({ model: script.driver });
  const dish = { id: "bread", revision: "r1", name: "Bread", intent: "side", recipe: [
    { ingredientId: "flour", quantityMinor: "100", unit: "g", yieldServings: "10", wasteBasis: "none" },
  ], equipment: [], prepMinutes: 1, serviceMinutes: 1, dietaryLabels: [] };
  const ledger = { brief: { id: "brief", serviceTime: "12:00", covers: 20, mandatoryRestrictions: [], preferences: [],
    budgetMinor: "1000", currency: "USD", equipmentCapacity: {} }, ingredientFacts: [], dishes: [dish], exactScaledBatches: [batch("bread")],
    sharedComponents: [], schedule: [], approvedControls: [], revision: 1 };
  const result = await runtime.run(() => reconcile(ledger, [dish]));
  assert.deepEqual(result.expected, [batch("bread")]);
  assert.deepEqual(result.discrepancies, []);
  assert.deepEqual(result.blockers, []);
});
