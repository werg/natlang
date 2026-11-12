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

test('the core exposes the type expansion editors draw forms from: aliases inlined, named, with field docs; a recursive name stays a name', () => {
  const env = new core.TypeEnv({
    Ticket: core.parseType('{ title: string; tags?: string[] }'),
    Tree: core.parseType('{ children: Tree[] }'),
  });
  const ticket = core.expandType(core.parseType('Ticket'), env, { Ticket: { title: 'One line summary' } });
  assert.equal(ticket.alias, 'Ticket');
  assert.deepEqual(ticket.fields.map(field => [field.name, field.optional, field.doc]), [['title', false, 'One line summary'], ['tags', true, undefined]]);
  const tree = core.expandType(core.parseType('Tree'), env);
  assert.deepEqual(tree.fields[0].type.element, { kind: 'name', name: 'Tree' });
  assert.deepEqual(core.expandType(core.parseType('Missing'), env), { kind: 'name', name: 'Missing' });
});
