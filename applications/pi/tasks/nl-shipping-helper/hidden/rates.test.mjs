import { test } from 'node:test';
import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';
const { shippingCost } = await import(pathToFileURL(`${process.cwd()}/helpers/rates.js`).href);
test('billable weight', () => {
  assert.equal(shippingCost(2.2, 'domestic', 0), 500 + 3 * 150);
  assert.equal(shippingCost(2, 'domestic', 0), 500 + 2 * 150);
  assert.equal(shippingCost(0, 'domestic', 0), 650);
  assert.equal(shippingCost(0.3, 'domestic', 0), 650);
  assert.equal(shippingCost(4.5, 'domestic', 4999), 500 + 5 * 150);
});
test('international', () => {
  assert.equal(shippingCost(3, 'international', 0), 1900);
  assert.equal(shippingCost(0.5, 'international', 999999), 1300);
});
test('free domestic', () => {
  assert.equal(shippingCost(10, 'domestic', 5000), 0);
  assert.equal(shippingCost(10, 'domestic', 7000), 0);
});
test('unknown zone', () => {
  assert.throws(() => shippingCost(1, 'mars', 0), { message: 'unknown zone: mars' });
});
test('integers', () => {
  for (const w of [0.1, 1.7, 3.3, 7, 11.01]) for (const z of ['domestic', 'international']) assert.ok(Number.isInteger(shippingCost(w, z, 0)));
});
