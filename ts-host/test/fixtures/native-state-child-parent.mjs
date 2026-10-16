// Real native HTTP forwards and state splices; deterministic tool readout is not a quality claim.
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { AsyncLocalStorage } from 'node:async_hooks';
import { FileNeuraleseStore } from '../../dist/neuralese/node-block-store.js';
import { neuraleseServerModelTurn } from '../../dist/model/neuralese-server.js';
import { NativeToolAgent } from '../../dist/native/agent.js';
import { session as open } from '../support/natlang.mjs';
import { validateSchema } from '../../dist/native/graph.js';
import { ModelTurnRecorder, setRecorderSource } from '../../dist/native/model-recording.js';
import { nativeStateConsumerFromInfo } from '../../dist/native/neuralese.js';
const [endpoint, archive, recording] = process.argv.slice(2);
const recorder = recording ? new ModelTurnRecorder() : undefined;
const recordingScope=new AsyncLocalStorage();
if (recorder) setRecorderSource(() => recordingScope.getStore() ?? recorder);
const info = await fetch(endpoint+'/v1/neuralese/info').then(r => r.json());
const store = await FileNeuraleseStore.open(archive, { residentBytes: 0 });
const neuralese = { store, dialect: info.dialects[0], nativeStateConsumer: nativeStateConsumerFromInfo(info) };
const server = neuraleseServerModelTurn({ endpoint, model: 'native-fixture', store });
const driver = Object.assign(async request => {
  const raw = await server({ ...request, max_tokens: 8 });
  return { ...raw, text: undefined, calls: [['return_result', { status: 'success', value: raw.text || 'consumed' }]] };
}, { neuralese: true, decide: server.decide });
const child = open({ type: '() => Neuralese<string>', instructions: 'Produce a useful memo.' }, { neuralese }).session;
await new NativeToolAgent(driver, { maxTurns: 2, neuralese }).run(child);
assert.ok(child.lam.return?.$neuralese, JSON.stringify(child.lam));
const id = child.lam.return.$neuralese.id;
assert.ok(id.startsWith('nzs1_'));
const write = child.runtime.trace.events.find(event => event.kind === 'block_write' && event.block === id);
assert.ok(write?.native_identity, 'native producer is recorded through the shared block-write graph');
const reopened = await FileNeuraleseStore.open(archive, { residentBytes: 0 });
assert.equal((await reopened.getNativeState(id)).id, id);
assert.deepEqual(await server.blocks.collect(new Set()), [id]);
assert.equal(await server.blocks.hasNativeState(id), false, 'parent must restore exact native state from its persistent archive');
const parent = open({ type: '(memo: Neuralese<string>) => string', instructions: 'Consume the child memo.',
  args: { memo: child.lam.return } }, { neuralese }).session;
await new NativeToolAgent(driver, { maxTurns: 2, neuralese }).run(parent);
assert.equal(parent.lam.return, 'consumed');
const read = parent.runtime.trace.events.find(event => event.kind === 'block_read' && event.block === id);
assert.ok(read?.inputs.some(input => input.node === write.node), 'parent reads the actual child producer, not an invented external edge');
if (recorder) {
  const decisions = [0,1].map(index => open({ type: "(memo: Neuralese<string>) => 'yes' | 'no'",
    instructions:'Choose the result from the memo.', args:{memo:child.lam.return} }, {neuralese}).session);
  const decisionRecorders=decisions.map(()=>new ModelTurnRecorder());
  await Promise.all(decisions.map((session,index)=>recordingScope.run(decisionRecorders[index],()=>
    new NativeToolAgent(driver,{maxTurns:2,neuralese,decisionReadout:'finite-returns'}).run(session))));
  for(const own of decisionRecorders){
    assert.equal(own.nativeTurns().length,1,'coalesced decisions retain the actual caller recorder');
    recorder.nested.push(...own.turns);recorder.absorb(own);
  }
  const turns = recorder.nativeTurns();
  assert.equal(turns.length, 4);
  const graphSchema=JSON.parse(await readFile(new URL('../../../spec/neuralese-graph.schema.json',import.meta.url),'utf8'));
  const decided=turns.filter(turn=>turn.native.kind==='decision');
  assert.equal(decided.length,2);
  for(const turn of decided){
    assert.equal(turn.native.decision.schema,'natlang.native-decision-capture/1');
    assert.equal(turn.native.decision.continuations.length,2);
    assert.ok(turn.native.decision.continuations.every(ids=>ids.length>0));
    assert.equal(turn.native.graph.read_bindings[id],write.node);
    const node=recorder.graphNodes.get(turn.native.graph.graph_turn);
    assert.equal(node.readout,'decision');assert.equal(node.temperature,null);assert.equal(node.seed,null);
    assert.deepEqual(validateSchema(graphSchema,node),[]);
    assert.ok(validateSchema(graphSchema,{...node,seed:1}).length,'deterministic readout cannot claim a sampling seed');
  }
  const [produced, consumed] = turns.map(turn => turn.native);
  assert.equal(produced.graph.written_nodes[id], write.node);
  assert.equal(consumed.graph.read_bindings[id], write.node);
  for (const turn of turns.filter(turn=>turn.native.kind==='generation')) {
    assert.ok(turn.native.sampling.prompt.length);
    assert.ok(turn.native.sampling.samples.length);
    assert.equal(turn.native.sampling.schema, 'natlang.native-structural-capture/1');
    assert.equal(turn.native.sampling.sampling_law.temperature, 0);
    assert.equal(turn.native.sampling.sampling_law.probabilities, 'greedy-argmax');
    assert.ok(turn.native.sampling.samples.every(sample => sample.logp === null));
    assert.equal(recorder.graphNodes.get(turn.native.graph.graph_turn).kind, 'model_turn');
    for (const output of turn.native.sampling.writes ?? []) {
      for (const artifactId of [output.state_id, output.raw_state_id].filter(Boolean))
        assert.ok(await store.getNativeState(artifactId), 'captured producer dependencies are durable');
    }
  }
  assert.throws(() => recorder.linkNativeTurn(produced.id, consumed.graph.graph_turn), /two model-turn occurrences/);
  await writeFile(archive + '/native-recorded-turns.json', JSON.stringify(turns));
  const conflicting = new ModelTurnRecorder();
  conflicting.record(turns[0]);
  conflicting.linkNativeTurn(produced.id, consumed.graph.graph_turn);
  assert.throws(() => recorder.absorb(conflicting), /two model-turn occurrences/);
  const duplicateWrite = new ModelTurnRecorder();
  duplicateWrite.record(turns[0]);
  duplicateWrite.absorb(recorder);
  duplicateWrite.recordGraphNode({ ...recorder.graphNodes.get(write.node), node: write.node + ':duplicate' });
  assert.throws(() => duplicateWrite.nativeTurns(), /repeated native writes/);
  const duplicateRead = new ModelTurnRecorder();
  duplicateRead.record(turns[1]);
  duplicateRead.absorb(recorder);
  duplicateRead.recordGraphNode({ ...recorder.graphNodes.get(read.node), node: read.node + ':duplicate' });
  assert.throws(() => duplicateRead.nativeTurns(), /repeated native reads/);
  const original = recorder.graphNodes.get(write.node);
  recorder.recordGraphNode({ ...original, length: Number(original.length) + 1 });
  assert.throws(() => recorder.nativeTurns(), /conflicting source node identities/);
  setRecorderSource(() => undefined);
}
console.log(JSON.stringify({ state_id: id, parent_result: parent.lam.return, archived: true,
  ...(recorder ? { recorded: true } : {}) }));
