import assert from 'node:assert/strict';
import { test } from 'node:test';
import { ONCE_EFFECTS, recordingServices } from '../dist/native/effects.js';

// A host may hand over frozen or hardened service objects (a proxy over a frozen property must return that property's
// own value, so the recorder wraps a shell, not the service).
test('recording wraps frozen services', async () => {
  const seen = [];
  const service = Object.freeze({ greet: async name => `hello ${name}`, [ONCE_EFFECTS]: ['greet'], label: 'frozen' });
  const services = recordingServices({ people: service }, event => seen.push(`${event.phase}:${event.method}`));
  assert.equal(Object.isFrozen(service), true);
  assert.equal(await services.people.greet('Ada'), 'hello Ada');
  assert.equal(services.people.label, 'frozen');
  assert.deepEqual(Object.keys(services.people), ['greet', 'label']);
  assert.equal('greet' in services.people, true);
  assert.deepEqual(seen, ['requested:greet', 'completed:greet']);
  // The once-only rule still keys on the service object itself.
  assert.equal(await services.people.greet('Ada'), 'hello Ada');
  assert.equal(seen.filter(item => item.startsWith('completed')).length, 2);
});
