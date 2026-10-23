import assert from 'node:assert/strict';
import { test } from 'node:test';
import { NativeToolAgent } from '../dist/native/agent.js';
import { nativeProgramGuidance } from '../dist/adaptation/index.js';
import { NatlangRuntime, NatlangTask } from '../dist/runtime/runtime.js';
import { interpreter, lambda } from './support/natlang.mjs';

const identity = checkpoint => ({ architecture: 'fixture-native', checkpoint_sha256: checkpoint.repeat(64),
  tokenizer_fingerprint: 'b'.repeat(64), embedding_weight_sha256: 'c'.repeat(64), consumption_modes: [] });
const binding = (student, text, id) => ({ schema: 'natlang.memetic-program-guidance/1', program_id: 'fixture-program',
  student: { architecture: student.architecture, checkpoint_sha256: student.checkpoint_sha256,
    tokenizer_fingerprint: student.tokenizer_fingerprint, embedding_weight_sha256: student.embedding_weight_sha256 },
  text, native_state_id: id, native_state_artifact: { path: '/tmp/guidance.nzs', sha256: 'd'.repeat(64), bytes: 1 },
  program_guidance: `${id}` });

test('native program guidance is stored for the exact student and appears in the actual system call', async () => {
  const writes = [], requests = [];
  const id = 'nzs1_' + 'a'.repeat(52);
  const current = identity('a');
  const store = { async pin() {}, async getNativeState(stateId) {
    assert.equal(stateId, id);
    return { id, metadata: { schema: 'neuralese.native-state/1', text: 'Use the approved procedure.', ...current } };
  } };
  const driver = async request => {
    requests.push(request);
    return { calls: [['return_result', { status: 'success', value: 'done' }]] };
  };
  driver.neuralese = true;
  const agent = new NativeToolAgent(driver, { maxTurns: 2, programGuidance: nativeProgramGuidance('Use the approved procedure.', 'fixture-program'),
    neuralese: { store, nativeStateConsumer: current, port: { dialect: 'native-state/fixture-native', async write(text) {
      writes.push(text);
      return { id, dialect: 'native-state/fixture-native', length: 4, width: 8, dtype: 'f32',
        native_state: { schema: 'neuralese.native-state/1', text, ...current } };
    } } } });
  const runtime = interpreter({ agent: currentSession => agent.run(currentSession) });
  const result = await runtime.run(lambda({ type: '() => string', instructions: 'Return a string.' }));
  assert.equal(result.value, 'done');
  assert.deepEqual(writes, ['Use the approved procedure.']);
  const system = requests[0].messages[0].content;
  assert.ok(Array.isArray(system));
  assert.ok(system.some(part => part.type === 'native_state' && part.id === id));
  assert.ok(!system.some(part => part.type === 'text' && part.text.includes('Use the approved procedure.')));
});

test('an exact stored memetic binding reaches the prompt by its native state ID', async () => {
  const student = identity('a'), id = 'nzs1_' + 'c'.repeat(52), text = 'Use the approved procedure.';
  let request;
  const store = { async pin() {}, async getNativeState(stateId) {
    assert.equal(stateId, id);
    return { id, metadata: { schema: 'neuralese.native-state/1', text, ...student } };
  } };
  const driver = async value => { request = value; return { calls: [['return_result', { status: 'success', value: 'done' }]] }; };
  driver.neuralese = true;
  const agent = new NativeToolAgent(driver, { programGuidance: nativeProgramGuidance(binding(student, text, id)),
    neuralese: { store, nativeStateConsumer: student } });
  const runtime = interpreter({ agent: currentSession => agent.run(currentSession) });
  const result = await runtime.run(lambda({ type: '() => string', instructions: 'Return a string.' }));
  assert.equal(result.value, 'done');
  assert.ok(request.messages[0].content.some(part => part.type === 'native_state' && part.id === id));
  assert.ok(!request.messages[0].content.some(part => part.type === 'text' && part.text.includes(text)));
});

test('native program guidance fails closed when the writer uses a stale checkpoint', async () => {
  const expected = identity('a'), stale = identity('d'), originalId = 'nzs1_' + 'e'.repeat(52);
  const driver = async () => { throw new Error('model must not be called'); };
  driver.neuralese = true;
  const agent = new NativeToolAgent(driver, { programGuidance: nativeProgramGuidance(binding(stale,
    'Use the approved procedure.', originalId)),
    neuralese: { store: {}, nativeStateConsumer: expected, port: { dialect: 'native-state/fixture-native', async write(text) {
      return { id: 'nzs1_' + 'a'.repeat(52), dialect: 'native-state/fixture-native', length: 4, width: 8, dtype: 'f32',
        native_state: { schema: 'neuralese.native-state/1', text, ...stale } };
    } } } });
  const runtime = interpreter({ agent: currentSession => agent.run(currentSession) });
  await assert.rejects(() => runtime.run(lambda({ type: '() => string', instructions: 'Return a string.' })), /exact current student checkpoint/);
});

test('a binding from an older checkpoint is re-encoded for the current student', async () => {
  const old = identity('a'), current = identity('d'), oldId = 'nzs1_' + 'e'.repeat(52), currentId = 'nzs1_' + 'f'.repeat(52);
  let writes = 0, request;
  const store = { async pin() {}, async getNativeState(stateId) {
    return stateId === currentId ? { id: currentId, metadata: { schema: 'neuralese.native-state/1',
      text: 'Use the approved procedure.', ...current } } : undefined;
  } };
  const driver = async value => { request = value; return { calls: [['return_result', { status: 'success', value: 'done' }]] }; };
  driver.neuralese = true;
  const agent = new NativeToolAgent(driver, { programGuidance: nativeProgramGuidance(binding(old,
    'Use the approved procedure.', oldId)), neuralese: { store, nativeStateConsumer: old,
    nativeStateConsumerForCall: async () => current,
    port: { dialect: 'native-state/fixture-native', async write(text) {
      writes++;
      return { id: currentId, dialect: 'native-state/fixture-native', length: 4, width: 8, dtype: 'f32',
        native_state: { schema: 'neuralese.native-state/1', text, ...current } };
    } } } });
  const runtime = interpreter({ agent: currentSession => agent.run(currentSession) });
  const result = await runtime.run(lambda({ type: '() => string', instructions: 'Return a string.' }));
  assert.equal(result.value, 'done');
  assert.equal(writes, 1);
  assert.ok(request.messages[0].content.some(part => part.type === 'native_state' && part.id === currentId));
  assert.ok(!request.messages[0].content.some(part => part.type === 'native_state' && part.id === oldId));
});

test('concurrent calls share one persisted re-encoded guidance state per program and student', async () => {
  const student = identity('f'), id = 'nzs1_' + 'g'.repeat(52), text = 'Use the approved procedure.';
  let writes = 0;
  const states = new Map();
  const store = { async pin() {}, async getNativeState(stateId) { return states.get(stateId); } };
  const makeRuntime = () => {
    const driver = async () => ({ calls: [['return_result', { status: 'success', value: 'done' }]] });
    driver.neuralese = true;
    const agent = new NativeToolAgent(driver, { programGuidance: nativeProgramGuidance(text, 'fixture-program'),
      neuralese: { store, nativeStateConsumer: student, port: { dialect: 'native-state/fixture-native', async write(value) {
        writes++;
        await new Promise(resolve => setTimeout(resolve, 15));
        states.set(id, { id, metadata: { schema: 'neuralese.native-state/1', text: value, ...student } });
        return { id, dialect: 'native-state/fixture-native', length: 4, width: 8, dtype: 'f32',
          native_state: { schema: 'neuralese.native-state/1', text: value, ...student } };
      } } } });
    return interpreter({ agent: currentSession => agent.run(currentSession) });
  };
  const invoke = () => makeRuntime().run(lambda({ type: '() => string', instructions: 'Return a string.' }));
  const [left, right] = await Promise.all([invoke(), invoke()]);
  assert.equal(left.value, 'done');
  assert.equal(right.value, 'done');
  assert.equal(writes, 1);
});

test('native writer output must exist as an actual stored artifact before prompt use', async () => {
  const student = identity('e'), id = 'nzs1_' + 'i'.repeat(52);
  const driver = async () => { throw new Error('model must not be called'); };
  driver.neuralese = true;
  const agent = new NativeToolAgent(driver, { programGuidance: nativeProgramGuidance('Use the approved procedure.', 'fixture-program'),
    neuralese: { store: { async pin() {}, async getNativeState() { return undefined; } }, nativeStateConsumer: student,
      port: { dialect: 'native-state/fixture-native', async write(text) {
        return { id, dialect: 'native-state/fixture-native', length: 4, width: 8, dtype: 'f32',
          native_state: { schema: 'neuralese.native-state/1', text, ...student } };
      } } } });
  const runtime = interpreter({ agent: currentSession => agent.run(currentSession) });
  await assert.rejects(() => runtime.run(lambda({ type: '() => string', instructions: 'Return a string.' })), /did not persist the produced native state/);
});

test('native task binding requires its exact program descriptor', () => {
  const student = identity('a'), id = 'nzs1_' + 'h'.repeat(52);
  const guidance = binding(student, 'Use the approved procedure.', id);
  assert.throws(() => new NatlangTask(new NatlangRuntime({ nativeProgramGuidance: guidance })), /exact program descriptor/);
  assert.throws(() => new NatlangTask(new NatlangRuntime({ nativeProgramGuidance: guidance,
    program: { id: 'different-program' } })), /belongs to a different program/);
});

test('plain authored text guidance is inserted into the composed system prompt once', async () => {
  const requests = [];
  const driver = async request => {
    requests.push(request);
    return { calls: [['return_result', { status: 'success', value: 'done' }]] };
  };
  const agent = new NativeToolAgent(driver, { programGuidance: 'Use the approved procedure.' });
  const runtime = interpreter({ agent: currentSession => agent.run(currentSession) });
  await runtime.run(lambda({ type: '() => string', instructions: 'Return a string.' }));
  const prompt = requests[0].messages[0].content;
  assert.equal(prompt.split('<natlang_program_guidance>').length - 1, 1);
  assert.equal(prompt.split('Use the approved procedure.').length - 1, 1);
});
