import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import * as host from '../dist/index.js';
import { activeNativeAdapters } from '../dist/native/model-recording.js';
import { currentFrame } from '../dist/runtime/context.js';
import { scriptedModel } from './support/natlang.mjs';

const base = { architecture: 'lfm2', checkpoint_sha256: 'c'.repeat(64), tokenizer_fingerprint: 't', embedding_weight_sha256: 'e'.repeat(64) };
const named = body => `---\nargs: { value: string }\nreturns: string\n---\n${body}\n`;
function compile(files) {
  const app = host.compileVirtualProject({ files }, host, { target: 'node' });
  assert.equal(app.ok, true, host.formatDiagnostics(app.diagnostics));
  return app;
}
const files = {
  'outer.nl': named('Pass value to inner.'),
  'outer/inner.nl': named('Return value in uppercase.'),
  'main.ts': "import outer from './outer.nl'; export default outer;",
};
/** A release giving the program root, the `outer` folder and both lambdas their own artifact. */
function release(program, { scales = {}, requires = {}, reductions = [], memberships = true } = {}) {
  const { tree, bindings } = host.compileScopeTree(program);
  const artifacts = tree.map(node => ({ node: node.id, artifact: 'nnt1:' + node.id, rank: node.level === 'lambda' ? 2 : 8,
    parameterization: 'native-subspace', requires: requires[node.id] ?? [] }));
  const outer = bindings.find(binding => binding.component.endsWith('outer.nl::instructions'));
  return host.sealManifest({ schema: 'natlang.adapters/v1', program: { id: program.id, buildHash: program.buildHash }, base, tree, bindings, artifacts,
    families: [{ family: 'classify', version: 'v1', artifact: 'nnt1:family-classify', amplitudeHex: '0x1.0000000000000p+0' }],
    memberships: memberships ? [{ binding: outer.id, family: 'classify', version: 'v1', gate: 0.5 }] : [],
    softStates: [], scales, validatedReductions: reductions, fallback: 'base', receipts: {} });
}
/** Runs the program, recording the adapters each model request carried, per instruction. */
async function run(app, weights, options = {}) {
  const seen = [];
  const model = scriptedModel(opening => opening.includes('Pass value') ? 'return await inner(value)' : 'return value.toUpperCase()');
  const driver = async (request, ...rest) => {
    const text = String(request.messages[1].content);
    seen.push({ who: text.includes('Pass value') ? 'outer' : 'inner', adapters: activeNativeAdapters().map(item => item.id) });
    return model.driver(request, ...rest);
  };
  const traces = [];
  const runtime = host.createNatlangRuntime({ program: app.manifest.adaptation, model: driver, weights, trace: trace => traces.push(trace), ...options });
  const value = await runtime.run(() => app.require('main.ts').default('hi'));
  return { value, seen, traces };
}

test('the scope tree follows source folders and every definition keeps its own binding', () => {
  const app = compile(files);
  const { tree, bindings } = host.compileScopeTree(app.manifest.adaptation);
  const ancestry = name => bindings.find(binding => binding.component.endsWith(name + '::instructions')).ancestry;
  assert.deepEqual(ancestry('outer.nl'), ['tree:', 'lambda:' + bindings.find(b => b.component.endsWith('outer.nl::instructions')).component]);
  assert.deepEqual(ancestry('outer/inner.nl').slice(0, 2), ['tree:', 'folder:outer']);
  assert.equal(tree.filter(node => node.parent === null).length, 1);
  const trees = host.compileScopeTree(app.manifest.adaptation, { treeRoots: ['outer'] });
  assert.equal(trees.tree.find(node => node.path === 'outer').level, 'tree');
  assert.throws(() => host.compileScopeTree(app.manifest.adaptation, { treeRoots: ['missing'] }), /contains no bound definition/);
});

test('a call runs its own lexical stack; the caller\'s adapters are replaced, not inherited', async () => {
  const app = compile(files);
  const manifest = release(app.manifest.adaptation);
  const binding = new host.WeightBinding(manifest, { executor: base, program: app.manifest.adaptation });
  const { value, seen, traces } = await run(app, [binding]);
  assert.equal(value, 'HI');
  const outer = seen.find(item => item.who === 'outer').adapters, inner = seen.find(item => item.who === 'inner').adapters;
  const lambda = id => manifest.tree.find(node => node.level === 'lambda' && node.path === id).id;
  assert.deepEqual(outer, ['nnt1:family-classify', 'nnt1:tree:', 'nnt1:' + lambda('outer.nl')]);
  assert.deepEqual(inner, ['nnt1:tree:', 'nnt1:folder:outer', 'nnt1:' + lambda('outer/inner.nl')]);
  const recorded = traces.map(trace => trace.adaptation.weights);
  assert.ok(recorded.every(item => item.release === manifest.digest && item.status === 'ready'));
  assert.equal(new Set(recorded.map(item => item.identity)).size, 2);
  // Outside any natlang call nothing lexical is active.
  assert.deepEqual(activeNativeAdapters(), []);
});

test('scale zero removes a node; unavailable artifacts drop their dependents and fall back coherently', async () => {
  const app = compile(files), program = app.manifest.adaptation;
  const plain = release(program);
  const innerLambda = plain.tree.find(node => node.path === 'outer/inner.nl').id;
  const manifest = release(program, { scales: { 'folder:outer': 0 }, requires: { [innerLambda]: ['nnt1:tree:'] } });
  const resolve = (options, m = manifest) => new host.WeightBinding(m, { executor: base, ...options })
    .resolve('ignored', plain.bindings.find(b => b.ancestry.at(-1) === innerLambda).component, null);
  assert.deepEqual(resolve({}).effective.map(item => item.node), ['tree:', innerLambda]);
  // The root is missing: the inner lambda required it, so both go and the unvalidated empty reduction runs the base.
  const missing = resolve({ available: new Set(['nnt1:' + innerLambda]) });
  assert.equal(missing.status, 'degraded');
  assert.deepEqual(missing.effective, []);
  assert.deepEqual(missing.skipped.map(item => item.reason), ['not loaded by the serving backend', 'requires a dropped parent version']);
  // A validated reduction runs instead of the base.
  const reduced = release(program, { reductions: [['nnt1:folder:outer', 'nnt1:' + innerLambda]] });
  const partial = resolve({ available: new Set(['nnt1:' + innerLambda, 'nnt1:folder:outer']) }, reduced);
  assert.equal(partial.status, 'degraded');
  assert.deepEqual(partial.effective.map(item => item.node), ['folder:outer', innerLambda]);
  assert.throws(() => resolve({ available: new Set(), strict: true }), /weights-unavailable/);
});

test('another base or a changed definition never runs as tuned execution', async () => {
  const app = compile(files), program = app.manifest.adaptation;
  const manifest = release(program);
  const other = new host.WeightBinding(manifest, { executor: { ...base, checkpoint_sha256: 'd'.repeat(64) } });
  assert.equal(other.status, 'degraded');
  assert.ok(other.manifest.bindings.every(entry => other.resolve(entry.definitionId ?? '', entry.component, null).effective.length === 0));
  assert.throws(() => new host.WeightBinding(manifest, { executor: { ...base, checkpoint_sha256: 'd'.repeat(64) }, strict: true }), /weights-incompatible/);
  const edited = compile({ ...files, 'outer/inner.nl': named('Return value in lowercase.') });
  const stale = new host.WeightBinding(manifest, { executor: base, program: edited.manifest.adaptation });
  assert.equal(stale.status, 'stale');
  const { seen } = await run(edited, [stale]);
  assert.deepEqual(seen.find(item => item.who === 'inner').adapters, []);
  assert.ok(seen.find(item => item.who === 'outer').adapters.length > 0, 'unchanged definitions keep their stack');
});

test('tasks pin the release active at creation; manifests are tamper-evident', async () => {
  const app = compile(files), program = app.manifest.adaptation;
  const first = new host.WeightBinding(release(program), { executor: base });
  const second = new host.WeightBinding(release(program, { memberships: false }), { executor: base });
  let current = [first];
  const runtime = host.createNatlangRuntime({ program, weights: () => current,
    model: scriptedModel(opening => opening.includes('Pass value') ? 'return await inner(value)' : 'return value.toUpperCase()').driver });
  const pending = runtime.run(async () => { await new Promise(resolve => setTimeout(resolve, 5)); return activeTaskRelease(); });
  current = [second];
  assert.equal(await pending, first.release, 'a task created before activation keeps its release');
  assert.equal(await runtime.run(() => activeTaskRelease()), second.release);
  const tampered = JSON.parse(JSON.stringify(first.manifest)); tampered.artifacts[0].rank = 99;
  assert.throws(() => host.validateManifest(tampered), /digest does not match/);
});
function activeTaskRelease() { return currentFrame().task.weights[0].release; }

test('release store: immutable releases, compare-before-activate, rollback', () => {
  const root = mkdtempSync(join(tmpdir(), 'natlang-weights-'));
  try {
    const app = compile(files), program = app.manifest.adaptation;
    const a = release(program), b = release(program, { memberships: false });
    host.storeRelease(root, a); host.storeRelease(root, b); host.storeRelease(root, a);
    const receipts = { backend: { kind: 'native', loaded: a.artifacts.map(item => item.artifact), identity: base },
      readiness: { definition: 'outer', ok: true, effectiveIdentity: 'x', ms: 1 } };
    assert.equal(host.tuneStatus(root).status, 'untuned');
    host.activateRelease(root, a.digest, receipts, null);
    assert.throws(() => host.activateRelease(root, b.digest, receipts, null), /active release changed/);
    host.activateRelease(root, b.digest, receipts, a.digest);
    assert.equal(host.currentManifest(root).digest, b.digest);
    const weights = host.activeWeights(root, { executor: base });
    assert.equal(weights()[0].release, b.digest);
    host.rollbackRelease(root);
    assert.equal(weights()[0].release, a.digest);
    assert.equal(host.tuneStatus(root, { executor: base, program }).status, 'ready');
    assert.throws(() => host.activateRelease(root, a.digest, { ...receipts, readiness: { ...receipts.readiness, ok: false } }, a.digest), /readiness probe/);
  } finally { rmSync(root, { recursive: true, force: true }); }
});

test('soft instruction states replace exactly their trained text, need native-state import, and fall back when text changes', async () => {
  const { hexDigest } = await import('../dist/native/hash.js');
  const app = compile(files), program = app.manifest.adaptation;
  const plain = release(program);
  const inner = plain.bindings.find(binding => binding.component.endsWith('outer/inner.nl::instructions'));
  const state = 'nzs1_' + 'a'.repeat(52);
  const withSoft = text => host.sealManifest({ ...Object.fromEntries(Object.entries(plain).filter(([key]) => key !== 'digest')),
    softStates: [{ binding: inner.id, kind: 'instructions', target: null, state, textSha256: hexDigest(text), tokens: 7 }] });
  const manifest = withSoft('Return value in uppercase.\n'); // the exact body text, as collection records it
  // Without native-state import the soft binding is not served as tuned.
  const noImport = new host.WeightBinding(manifest, { executor: base });
  assert.equal(noImport.resolve('', inner.component, null).status, 'degraded');
  assert.throws(() => new host.WeightBinding(manifest, { executor: base, strict: true }).resolve('', inner.component, null), /weights-unavailable/);
  const prompts = [];
  const runWith = async (binding, project = app) => {
    const model = scriptedModel(opening => opening.includes('Pass value') ? 'return await inner(value)' : 'return value.toUpperCase()');
    const traces = [];
    const runtime = host.createNatlangRuntime({ program: project.manifest.adaptation, weights: [binding], trace: trace => traces.push(trace),
      // A transport that carries content parts (native state); it records what each prompt carried.
      model: Object.assign(async (request, ...rest) => {
        const content = request.messages[1].content;
        prompts.push(typeof content === 'string' ? content : JSON.stringify(content));
        return model.driver({ ...request, messages: request.messages.map(message => Array.isArray(message.content)
          ? { ...message, content: message.content.map(part => part.type === 'text' ? part.text : `[${part.id}]`).join('') } : message) }, ...rest);
      }, { neuralese: true }) });
    await runtime.run(() => project.require('main.ts').default('hi'));
    return traces;
  };
  // The stand-in model answers whatever it sees; the point is what the prompt carried.
  const soft = new host.WeightBinding(manifest, { executor: base, nativeState: true });
  prompts.length = 0;
  // The inner call asks for its state in place of the text (resolving the artifact needs a native store; with none
  // the call stops rather than running on text it was not trained with).
  await assert.rejects(runWith(soft), new RegExp(`${state} is unavailable`));
  assert.ok(!prompts.some(prompt => prompt.includes('Return value in uppercase.')), 'the trained text was never sent');
  // The trained text no longer matches (the release says another text): the call runs untuned, with its text.
  const drifted = new host.WeightBinding(withSoft('Return value in lowercase.\n'), { executor: base, nativeState: true });
  prompts.length = 0;
  const traces = await runWith(drifted);
  assert.ok(prompts.some(prompt => prompt.includes('Return value in uppercase.')));
  const innerTrace = traces.find(trace => trace.adaptation?.weights?.binding === inner.id);
  assert.equal(innerTrace.adaptation.weights.status, 'degraded');
  assert.deepEqual(innerTrace.adaptation.weights.effective, []);
});

test('vLLM serving loads whole-stack exports and routes each call to its own stack by name', async () => {
  const { createServer } = await import('node:http');
  const { mkdtempSync, mkdirSync, writeFileSync } = await import('node:fs');
  const { tmpdir } = await import('node:os');
  const { join } = await import('node:path');
  const { loadVllmStacks } = await import('../dist/tuned-serving.js');
  const { createResolvedModelSession } = await import('../dist/model/index.js');
  const app = compile(files), program = app.manifest.adaptation, manifest = release(program, { memberships: false });
  const binding0 = new host.WeightBinding(manifest, { executor: base });
  const names = Object.fromEntries(manifest.bindings.map(entry => [entry.component.endsWith('inner.nl::instructions') ? 'inner' : 'outer',
    host.stackServingName(binding0.resolve(entry.definitionId ?? '', entry.component, null).effective)]));
  const directory = mkdtempSync(join(tmpdir(), 'natlang-vllm-'));
  mkdirSync(join(directory, 'vllm'));
  writeFileSync(join(directory, 'vllm', 'stack-exports.json'), JSON.stringify(Object.entries(names).map(([who, identity]) => ({ identity, directory: identity,
    artifacts: binding0.resolve('', manifest.bindings.find(b => b.component.endsWith(who === 'inner' ? 'inner.nl::instructions' : 'outer.nl::instructions')).component, null).effective.map(e => e.id) }))));
  const loads = [], models = [];
  const server = createServer((request, response) => {
    let text = ''; request.on('data', chunk => { text += chunk; });
    request.on('end', () => {
      response.setHeader('content-type', 'application/json');
      const body = text ? JSON.parse(text) : {};
      if (request.url.endsWith('/load_lora_adapter')) { loads.push(body); return response.end('{}'); }
      if (request.url.endsWith('/models')) return response.end(JSON.stringify({ data: [{ id: 'base', max_model_len: 8192 }] }));
      models.push(body.model);
      const tool = body.messages.at(-1).role === 'tool';
      const opening = String(body.messages[1].content);
      const code = opening.includes('Pass value') ? 'return await inner(value)' : 'return value.toUpperCase()';
      response.end(JSON.stringify({ choices: [{ index: 0, finish_reason: tool ? 'stop' : 'tool_calls', message: tool ? { role: 'assistant', content: 'done' }
        : { role: 'assistant', content: '', tool_calls: [{ id: 'c', type: 'function', function: { name: 'eval', arguments: JSON.stringify({ code }) } }] } }] }));
    });
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  try {
    const endpoint = `http://127.0.0.1:${server.address().port}/v1`;
    const loaded = await loadVllmStacks(endpoint, directory);
    assert.deepEqual(loads.map(item => item.lora_name).sort(), Object.values(names).sort());
    const session = createResolvedModelSession({ kind: 'external', endpoint, model: 'base', apiKeyEnv: 'NONE', vllm: true });
    session.vllmStacks(loaded.names);
    const driver = Object.assign((request, signal, options) => session.turn(request, signal, options), { decide: (r, s) => session.decide(r, s), contextWindow: () => session.contextWindow() });
    const runtime = host.createNatlangRuntime({ program, model: driver, weights: [new host.WeightBinding(manifest, { executor: base, available: loaded.available })] });
    assert.equal(await runtime.run(() => app.require('main.ts').default('hi')), 'HI');
    assert.ok(models.includes(names.outer) && models.includes(names.inner), 'each call used its own whole-stack LoRA');
    await session.close();
  } finally { server.close(); }
});
