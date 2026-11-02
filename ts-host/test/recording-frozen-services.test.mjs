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

test('once-only cached calls retain their exact sync and async result observations', async () => {
  const events=[];let syncCalls=0,asyncCalls=0;
  const source={ [ONCE_EFFECTS]:['sync','later'], sync:()=>{syncCalls++;return {value:'sync'};},
    later:async()=>{asyncCalls++;return {value:'async'};} };
  const wrapped=recordingServices({source},event=>events.push(event));
  assert.deepEqual(wrapped.source.sync(),{value:'sync'});
  assert.deepEqual(wrapped.source.sync(),{value:'sync'});
  assert.deepEqual(await wrapped.source.later(),{value:'async'});
  assert.deepEqual(await wrapped.source.later(),{value:'async'});
  assert.equal(syncCalls,1);assert.equal(asyncCalls,1);
  assert.deepEqual(events.filter(x=>x.phase==='completed').map(x=>x.exact),[
    {result:{value:'sync'},async:false},{result:{value:'sync'},async:false},
    {result:{value:'async'},async:true},{result:{value:'async'},async:true}]);
});
