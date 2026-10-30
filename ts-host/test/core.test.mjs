import assert from 'node:assert/strict';
import test from 'node:test';
import * as core from '../dist/core/index.js';
import { constructorRealm } from '../dist/core/index.js';
import { scenarios } from './support/core-scenarios.mjs';

// The core's scripted-agent scenarios under Node, over a realm made from the Function constructor. The same scenarios
// run inside workerd in workerd-core.test.mjs.
const realm = constructorRealm((body, name, ...parameters) => new Function(...parameters, body));
for (const [name, scenario] of Object.entries(scenarios(core, realm))) test(name, scenario);

test('the core exposes no Node platform, and a runtime without a platform is refused', () => {
  assert.equal(core.nodePlatform, undefined);
  assert.throws(() => core.createNatlangRuntime({}), /platform/);
});
