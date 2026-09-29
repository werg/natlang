import assert from 'node:assert/strict';
import { test } from 'node:test';
import * as host from '../dist/index.js';
import { candidateArtifact } from '../dist/evaluation/runner.js';
import { scriptedModel } from './support/natlang.mjs';

const identity = { id: 'scripted-executor', configuration: { revision: 1 } };
const named = body => `---\nargs: { value: string }\nreturns: string\n---\n${body}\n`;
function compile(files) {
  const app = host.compileVirtualProject({ files }, host, { target: 'node' });
  assert.equal(app.ok, true, host.formatDiagnostics(app.diagnostics));
  return app;
}
function overlay(program, values) {
  const components = Object.keys(values);
  return candidateArtifact({ program, components, suite: { executorIdentity: identity }, suiteHash: host.fingerprint('suite') }, values);
}

test('artifact parsing rejects duplicate escaped keys, tampering, stale contracts and executor identities', () => {
  const app = compile({ 'echo.nl': named('Return value.'), 'main.ts': "import echo from './echo.nl'; export default echo;" });
  const program = app.manifest.adaptation;
  const component = program.components.find(c => c.origin === 'named');
  const values = { [component.key]: { kind: 'lambda.instructions', template: { segments: ['Return the first word of value.'], slotIds: [] } } };
  const artifact = overlay(program, values);
  const loaded = host.parseAdaptation(JSON.stringify(artifact));
  assert.equal(host.bindAdaptation(loaded, program, identity).artifact.digest, artifact.digest);
  assert.throws(() => host.parseAdaptation(JSON.stringify(artifact).replace('"schema":', '"schema":"x","\\u0073chema":')), /duplicate JSON key/);
  assert.throws(() => host.validateAdaptation({ ...artifact, digest: 'tampered' }), /digest mismatch/);
  assert.throws(() => host.bindAdaptation(artifact, { ...program, buildHash: 'other' }, identity), /build metadata/);
  assert.throws(() => host.bindAdaptation(artifact, program, { ...identity, id: 'other' }), /executor/);
  assert.equal(Object.isFrozen(loaded.components[0].value.template.segments), true);
});

test('preexisting named callables resolve concurrent task bindings and explicit baseline rollback', async () => {
  const app = compile({ 'echo.nl': named('Return value.'), 'main.ts': "import echo from './echo.nl'; export default echo;" });
  const callable = app.require('main.ts').default;
  const program = app.manifest.adaptation;
  const component = program.components.find(c => c.origin === 'named');
  const make = body => host.bindAdaptation(overlay(program, { [component.key]: { kind: 'lambda.instructions', template: { segments: [body], slotIds: [] } } }), program, identity);
  const first = make('Return the first word of value.'), upper = make('Return value in uppercase.');
  const traces = [];
  const model = scriptedModel(opening => opening.includes('first word') ? 'return value.split(" ")[0]' : opening.includes('uppercase') ? 'return value.toUpperCase()' : 'return value');
  const runtime = host.createNatlangRuntime({ program, adaptation: first, executorIdentity: identity, model: model.driver, trace: trace => traces.push(trace) });
  assert.deepEqual(await Promise.all([
    runtime.run(() => callable('hello world')),
    runtime.run(() => callable('hello world'), { adaptation: upper }),
    runtime.run(() => callable('hello world'), { adaptation: null }),
  ]), ['hello', 'HELLO WORLD', 'hello world']);
  assert.deepEqual(new Set(traces.map(t => t.adaptation.artifact)), new Set([first.artifact.digest, upper.artifact.digest, null]));
  assert.equal(await runtime.run(() => callable('again now')), 'again');
});

test('foreign module bindings are revalidated and artifact settings cannot silently drift', async () => {
  const app = compile({ 'echo.nl': named('Return value.'), 'main.ts': "import echo from './echo.nl'; export default echo;" });
  const program = app.manifest.adaptation, component = program.components.find(c => c.origin === 'named');
  const artifact = overlay(program, { [component.key]: component.baseline });
  const foreign = { ...host.bindAdaptation(artifact, program, identity), candidate: { [component.key]:
    { kind: 'lambda.instructions', template: { segments: ['Tampered.'], slotIds: [] } } } };
  const runtime = host.createNatlangRuntime({ program, executorIdentity: identity, adaptation: foreign,
    agent: session => { assert.ok(!session.lam.body.includes('Tampered')); session.lam.return = 'ok'; } });
  assert.equal(await runtime.run(() => app.require('main.ts').default('hello')), 'ok');
  const drifted = host.createNatlangRuntime({ program, executorIdentity: identity, adaptation: foreign,
    model: { driver: () => ({}), temperature: 0.2 } });
  await assert.rejects(() => drifted.run(() => 'unused'), /inference settings differ/);
});

test('program guidance follows generated descendants through the depth limit and baseline rollback', async () => {
  const app = compile({ 'root.nl': named('Outer stage.'), 'root/helper.nl': named('Helper stage.'),
    'main.ts': "import root from './root.nl'; export default root;" });
  const program = app.manifest.adaptation, component = program.components.find(c => c.kind === 'program.guidance');
  const guidance = 'Keep the application objective; use nl exactly as the runtime permits.';
  const binding = host.bindAdaptation(overlay(program, { [component.key]: { kind: 'program.guidance', text: guidance } }), program, identity);
  const systems = [], traces = [];
  const scripted = scriptedModel(opening => opening.includes('Outer stage') ? 'return await nl<string>`One stage.`()' :
    opening.includes('One stage') ? 'return await nl<string>`Two stage.`()' :
    opening.includes('Two stage') ? 'return await nl<string>`Three stage.`()' : 'return "ok"');
  const runtime = host.createNatlangRuntime({ program, executorIdentity: identity, adaptation: binding, trace: trace => traces.push(trace),
    model: request => { systems.push(String(request.messages[0].content)); return scripted.driver(request); } });
  assert.equal(await runtime.run(() => app.require('main.ts').default('input')), 'ok');
  assert.ok(systems.every(system => system.endsWith(guidance + '\n</natlang_program_guidance>\n')));
  assert.ok(systems.some(system => system.includes('third and final layer')));
  assert.ok(traces.every(trace => trace.adaptation.guidanceApplied));
  systems.length = 0;
  assert.equal(await runtime.run(() => app.require('main.ts').default('input'), { adaptation: null }), 'ok');
  assert.ok(systems.every(system => !system.includes('<natlang_program_guidance>')));
});

test('authored inline templates preserve slots, live captures, original expression evaluation and escaping', async () => {
  const files = { 'main.ts': `import { nl } from '@natlang/node';
let tally = 2;
let conversions = 0;
const slot = { toJSON() { conversions++; return 'marker'; } };
export const judge = /* @natlangSite urgency */ nl<number>\`Return tally with \${slot}.\`;
export function change(value: number) { tally = value; }
export function count() { return conversions; }
` };
  const app = compile(files), module = app.require('main.ts');
  const program = app.manifest.adaptation;
  const component = program.components.find(c => c.origin === 'authored-inline');
  assert.match(component.key, /site%3Aurgency/);
  assert.equal(component.contract.slots.length, 1);
  const replacement = { kind: 'lambda.instructions', template: { segments: ['Double tally with literal ` and ${ignored} and ', '.'], slotIds: component.contract.slots } };
  const binding = host.bindAdaptation(overlay(program, { [component.key]: replacement }), program, identity);
  const model = scriptedModel(opening => opening.includes('Double tally') ? 'return tally * 2' : 'return tally');
  const runtime = host.createNatlangRuntime({ program, executorIdentity: identity, model: model.driver });
  assert.equal(module.count(), 1);
  module.change(7);
  assert.equal(await runtime.run(() => module.judge(), { adaptation: binding }), 14);
  assert.equal(await runtime.run(() => module.judge()), 7);
  assert.equal(module.count(), 1, 'slot coercion is not repeated at invocation');
  const dropped = { ...replacement, template: { ...replacement.template, segments: ['Use ', '.'] } };
  assert.throws(() => overlay(program, { [component.key]: dropped }), /capture contract changed/);
  const changedSlot = { ...replacement, template: { ...replacement.template, slotIds: ['different'] } };
  assert.throws(() => overlay(program, { [component.key]: changedSlot }), /slot identity/);
});

test('inventory includes helper modules, duplicate site labels fail, helper behavior changes build fingerprint', () => {
  const files = { 'root.nl': named('Return value.'), 'root/nested.nl': named('Return value.'),
    'root/helper.ts': "import { nl } from '@natlang/node'; export default async function helper(value: string): Promise<string> { return await /* @natlangSite inside */ nl<string>`Return value.`(value); }",
    'main.ts': "import root from './root.nl'; export default root;" };
  const app = compile(files);
  assert.equal(app.manifest.adaptation.components.filter(c => c.origin === 'named').length, 2);
  assert.equal(app.manifest.adaptation.components.filter(c => c.origin === 'authored-inline').length, 1);
  const changed = compile({ ...files, 'root/helper.ts': files['root/helper.ts'].replace('Return value.', 'Return value unchanged.') });
  assert.notEqual(app.manifest.adaptation.buildHash, changed.manifest.adaptation.buildHash);
  const duplicate = host.compileVirtualProject({ files: { 'main.ts': "import { nl } from '@natlang/node'; const a = /* @natlangSite same */ nl<string>`Hi.`; const b = /* @natlangSite same */ nl<string>`Bye.`;" } }, host);
  assert.equal(duplicate.ok, false);
  assert.ok(duplicate.diagnostics.some(d => d.code === 'duplicate-site'));
});

test('read_code, runtime edits, and previously constructed child references share one task revision', async () => {
  const app = compile({ 'root.nl': named('Use child to return value.'), 'root/child.nl': named('Return value.'),
    'main.ts': "import root from './root.nl'; export default root;" });
  const root = app.require('main.ts').default, child = root.child, program = app.manifest.adaptation;
  const descriptor = program.components.find(c => c.source?.path === 'root/child.nl');
  const binding = host.bindAdaptation(overlay(program, { [descriptor.key]: { kind: 'lambda.instructions', template: { segments: ['Return uppercase value.'], slotIds: [] } } }), program, identity);
  let seenSource;
  const runtime = host.createNatlangRuntime({ program, adaptation: binding, executorIdentity: identity,
    agent: async session => {
      if (session.lam.functionName === 'root') {
        seenSource = session.apply('read_code', { name: 'child' }).value;
        assert.match(seenSource, /Return uppercase value\./);
        const edited = session.apply('edit_code', { name: 'child', find: 'Return uppercase value.', replace_with: 'Return first word of value.' });
        assert.equal(edited.kind, 'ok', edited.text);
        const conflict = session.apply('edit_code', { name: 'child', find: 'Return first word of value.', replace_with: 'Return value.', expected_revision: 0 });
        assert.equal(conflict.kind, 'rejected');
        assert.equal((await child('hello world')), 'hello');
        const evalResult = await session.applyAsync('eval', { code: 'return await child(value)' });
        assert.equal(evalResult.kind, 'ok', evalResult.text);
      } else {
        assert.equal((await session.applyAsync('eval', { code: session.lam.body.includes('first word') ? 'return value.split(" ")[0]' : session.lam.body.includes('uppercase') ? 'return value.toUpperCase()' : 'return value' })).kind, 'ok');
      }
    } });
  // The helper creates deny-policy artifacts by default; export an explicitly evaluated allow policy.
  const allowArtifact = { ...binding.artifact, policy: { codeEdits: 'allow', settings: {} }, digest: '' };
  allowArtifact.digest = host.artifactDigest(allowArtifact);
  const allow = host.bindAdaptation(allowArtifact, program, identity);
  assert.equal(await runtime.run(() => root('hello world'), { adaptation: allow }), 'hello');
  assert.equal(await runtime.run(() => child('hello world')), 'HELLO WORLD', 'task edit does not change later tasks');
  assert.equal(await runtime.run(() => child('hello world'), { adaptation: null }), 'hello world');
});

test('source export escapes inline text, preserves expressions, and produces an applicable compiling patch', async () => {
  const { mkdtempSync, writeFileSync, mkdirSync } = await import('node:fs');
  const { tmpdir } = await import('node:os');
  const { join } = await import('node:path');
  const { spawnSync } = await import('node:child_process');
  const { buildProject } = host;
  const { exportAdaptationPatch } = await import('../dist/optimization/index.js');
  const root = mkdtempSync(join(tmpdir(), 'natlang-export-'));
  const source = "import { nl } from '@natlang/node';\nexport function main(value: string) { return /* @natlangSite first */ nl<string>`Return ${value}.`(); }\n";
  writeFileSync(join(root, 'package.json'), '{"type":"module"}'); writeFileSync(join(root, 'main.ts'), source);
  const result = buildProject({ project: root, programId: 'patch', write: false, surfaceSpecifiers: ['@natlang/node'] });
  assert.equal(result.ok, true, host.formatDiagnostics(result.diagnostics));
  const program = result.manifest.adaptation, descriptor = program.components.find(c => c.origin === 'authored-inline');
  const replacement = { kind: 'lambda.instructions', template: { segments: ['Return literal ` and ${text} before ', '.'], slotIds: descriptor.contract.slots } };
  const artifact = overlay(program, { [descriptor.key]: replacement });
  // Export uses the installed declaration path; link the workspace package for this temp project.
  const { symlinkSync } = await import('node:fs'); symlinkSync(new URL('../../node_modules', import.meta.url).pathname, join(root, 'node_modules'));
  const exported = exportAdaptationPatch(artifact, program, identity, root);
  const patch = join(root, 'export.patch'); writeFileSync(patch, exported.patch);
  const check = spawnSync('git', ['apply', '--check', patch], { cwd: root, encoding: 'utf8' });
  assert.equal(check.status, 0, check.stderr);
  assert.match(exported.patch, /\\`/); assert.match(exported.patch, /\\\$\{text\}/);
  writeFileSync(join(root, 'main.ts'), source + '// Source changed after inventory.\n');
  assert.throws(() => exportAdaptationPatch(artifact, program, identity, root), /program source changed; rebuild\/revalidate/);
});

test('application system prompt is snapshotted per task and checked against the artifact hash', async () => {
  const app = compile({ 'echo.nl': named('Return value.'), 'main.ts': "import echo from './echo.nl'; export default echo;" });
  const program = app.manifest.adaptation, component = program.components.find(c => c.origin === 'named');
  const artifact = overlay(program, { [component.key]: component.baseline });
  const originalPrompt = 'Application instructions revision one.';
  artifact.policy = { ...artifact.policy, systemPromptHash: host.fingerprint(originalPrompt, 'natlang.system-prompt/v1') };
  artifact.digest = host.artifactDigest(artifact);
  const binding = host.bindAdaptation(artifact, program, identity);
  let currentPrompt = originalPrompt;
  const seen = [], scripted = scriptedModel(() => 'return value');
  const runtime = host.createNatlangRuntime({ program, executorIdentity: identity, adaptation: binding,
    systemPrompt: () => currentPrompt,
    model: request => {
      seen.push(String(request.messages[0].content));
      if (seen.length === 1) currentPrompt = 'Application instructions revision two.';
      return scripted.driver(request);
    } });
  assert.equal(await runtime.run(() => app.require('main.ts').default('snapshot')), 'snapshot');
  assert.ok(seen.length >= 2);
  assert.ok(seen.every(prompt => prompt.includes(originalPrompt)));
  assert.ok(seen.every(prompt => !prompt.includes('revision two')));
  await assert.rejects(() => runtime.run(() => 'unused'), /application system prompt differs from evaluated artifact policy/);
});

test('an adaptation artifact rejects inference limits that differ from its evaluated policy', async () => {
  const app = compile({ 'echo.nl': named('Return value.'), 'main.ts': "import echo from './echo.nl'; export default echo;" });
  const program = app.manifest.adaptation, component = program.components.find(c => c.origin === 'named');
  const binding = host.bindAdaptation(overlay(program, { [component.key]: component.baseline }), program, identity);
  const runtime = host.createNatlangRuntime({ program, executorIdentity: identity, adaptation: binding,
    model: () => ({}), limits: { maxDepth: 2 } });
  await assert.rejects(() => runtime.run(() => 'unused'), /inference limits differ from evaluated artifact policy/);
});

test('authored program guidance contributes to the full program build fingerprint', async () => {
  const { mkdtempSync, rmSync, writeFileSync } = await import('node:fs');
  const { tmpdir } = await import('node:os');
  const { join } = await import('node:path');
  const directory = mkdtempSync(join(tmpdir(), 'natlang-guidance-fingerprint-'));
  try {
    writeFileSync(join(directory, 'echo.nl'), named('Return value.'));
    writeFileSync(join(directory, 'main.ts'), "import echo from './echo.nl'; export default echo;\n");
    const base = { project: directory, programId: 'guidance-fingerprint', emit: false, write: false };
    const first = host.buildProject({ ...base, guidance: 'Use concise answers.' });
    const second = host.buildProject({ ...base, guidance: 'Use detailed answers.' });
    assert.equal(first.ok, true, host.formatDiagnostics(first.diagnostics));
    assert.equal(second.ok, true, host.formatDiagnostics(second.diagnostics));
    assert.notEqual(first.manifest.adaptation.buildHash, second.manifest.adaptation.buildHash);
  } finally { rmSync(directory, { recursive: true, force: true }); }
});

test('service declaration and scope changes alter component service contracts', async () => {
  const { mkdtempSync, rmSync, writeFileSync } = await import('node:fs');
  const { tmpdir } = await import('node:os');
  const { join } = await import('node:path');
  const directory = mkdtempSync(join(tmpdir(), 'natlang-service-contract-'));
  try {
    writeFileSync(join(directory, 'echo.nl'), named('Return value.'));
    writeFileSync(join(directory, 'main.ts'), "import echo from './echo.nl'; export default echo;\n");
    const base = { project: directory, programId: 'service-contract', emit: false, write: false };
    const declarations = { clock: 'declare namespace clock { function now(): string; }' };
    const baseline = host.buildProject({ ...base, services: { declarations, scopes: { 'echo.nl': ['clock'] } } });
    const declarationDrift = host.buildProject({ ...base, services: {
      declarations: { clock: 'declare namespace clock { function now(): number; }' }, scopes: { 'echo.nl': ['clock'] } } });
    const scopeDrift = host.buildProject({ ...base, services: { declarations, scopes: { 'echo.nl': [] } } });
    for (const result of [baseline, declarationDrift, scopeDrift]) assert.equal(result.ok, true, host.formatDiagnostics(result.diagnostics));
    const get = result => result.manifest.adaptation.components.find(c => c.origin === 'named');
    assert.notEqual(get(baseline).contract.servicesHash, get(declarationDrift).contract.servicesHash);
    assert.notEqual(get(baseline).contract.servicesHash, get(scopeDrift).contract.servicesHash);
    assert.notEqual(get(baseline).contractHash, get(declarationDrift).contractHash);
    assert.notEqual(get(baseline).contractHash, get(scopeDrift).contractHash);
  } finally { rmSync(directory, { recursive: true, force: true }); }
});

test('duplicate authored interpolation expressions preserve distinct ordered slots through artifact round trip', () => {
  const app = compile({ 'main.ts': `import { nl } from '@natlang/node';
const marker = 'same';
export const judge = /* @natlangSite duplicateSlots */ nl<string>\`Use \${marker} and \${marker}.\`;
` });
  const program = app.manifest.adaptation, component = program.components.find(c => c.origin === 'authored-inline');
  assert.equal(component.source.expressions.length, 2);
  assert.equal(component.contract.slots.length, 2);
  assert.equal(component.contract.slots[0], component.contract.slots[1]);
  const artifact = overlay(program, { [component.key]: component.baseline });
  const parsed = host.parseAdaptation(JSON.stringify(artifact));
  const value = parsed.components[0].value;
  assert.deepEqual(value.template.slotIds, component.contract.slots);
  assert.equal(value.template.segments.length, 3);
  assert.ok(host.bindAdaptation(parsed, program, identity));
});

test('a retained callable-folder inline reference follows successive static edits and preserves captures per task', async () => {
  const app = compile({
    'root.nl': named('Use helper.judge and return its result.'),
    'root/helper.ts': `import { nl } from '@natlang/node';
let tally = 2;
const label = 'anchor';
export const judge = /* @natlangSite judge */ nl<number>\`Read tally for \${label}.\`;
`,
    'main.ts': "import root from './root.nl'; export default root;",
  });
  const retained = [];
  const seen = [];
  let rootCalls = 0, freshResult;
  const runtime = host.createNatlangRuntime({ program: app.manifest.adaptation, executorIdentity: identity,
    agent: async session => {
      if (session.lam.functionName === 'root') {
        const judge = session.callables().helper.judge;
        if (rootCalls++ > 0) {
          freshResult = await judge();
          session.apply('return_result', { status: 'success', value: 'done' });
          return;
        }
        retained.push(judge);
        const first = await judge();
        assert.equal(first, 3);
        assert.match(session.lam.body, /helper\.judge/);
        const firstEdit = session.apply('edit_code', { name: 'helper', find: 'Read tally for ', replace_with: 'Measure tally for ' });
        assert.equal(firstEdit.kind, 'ok', firstEdit.text);
        assert.equal(await judge(), 4, 'the retained reference resolves the first committed source revision');
        const secondEdit = session.apply('edit_code', { name: 'helper', find: 'Measure tally for ', replace_with: 'Inspect tally for ' });
        assert.equal(secondEdit.kind, 'ok', secondEdit.text);
        assert.equal(await judge(), 5, 'the same reference resolves a second committed source revision');
        assert.equal(session.apply('edit_code', { name: 'helper', find: 'let tally = 2;', replace_with: 'let total = 2;' }).kind, 'ok');
        await assert.rejects(() => judge(), /changed an existing inline callable contract/);
        const done = session.apply('return_result', { status: 'success', value: 'done' });
        assert.equal(done.kind, 'completed');
      } else {
        seen.push(session.lam.body);
        assert.match(session.lam.body, /anchor/);
        const result = await session.applyAsync('eval', { code: 'tally += 1; return tally' });
        assert.equal(result.kind, 'ok', result.text);
        const returned = session.apply('return_result', { status: 'success', value: result.value });
        assert.equal(returned.kind, 'completed', returned.text);
      }
    } });
  const root = app.require('main.ts').default;
  assert.equal(await runtime.run(() => root('ignored')), 'done');
  assert.deepEqual(seen.map(text => text.match(/(?:Read|Measure|Inspect) tally/)?.[0]),
    ['Read tally', 'Measure tally', 'Inspect tally']);
  assert.equal(retained.length, 1);
  assert.equal(await runtime.run(() => root('ignored')), 'done');
  assert.equal(freshResult, 6);
  assert.equal(seen.at(-1), 'Read tally for anchor.\n', 'a later task sees baseline inline text, not the prior task edit');
});

test('program guidance includes only explicitly imported program owners', async () => {
  const files = { 'echo.nl': named('Return value.'), 'main.ts': "import echo from './echo.nl'; export default echo;" };
  const make = (id, imports = []) => host.compileVirtualProject({ files }, host,
    { target: 'node', programId: id, importedGuidancePrograms: imports });
  const application = make('application', ['included-library']);
  const included = make('included-library'), excluded = make('excluded-library');
  for (const app of [application, included, excluded]) assert.equal(app.ok, true, host.formatDiagnostics(app.diagnostics));
  const program = application.manifest.adaptation;
  const descriptor = program.components.find(component => component.kind === 'program.guidance');
  const binding = host.bindAdaptation(overlay(program, { [descriptor.key]: { kind: 'program.guidance', text: 'Application guidance.' } }), program, identity);
  const systems = [], scripted = scriptedModel(() => 'return value');
  const runtime = host.createNatlangRuntime({ program, executorIdentity: identity, adaptation: binding,
    model: request => { systems.push(String(request.messages[0].content)); return scripted.driver(request); } });
  for (const [app, guided] of [[application, true], [included, true], [excluded, false]]) {
    systems.length = 0;
    assert.equal(await runtime.run(() => app.require('main.ts').default('ok')), 'ok');
    assert.ok(systems.length > 0);
    assert.ok(systems.every(system => system.includes('<natlang_program_guidance>') === guided));
  }
});

test('nested programs with identical relative definition names have separate recursion identities', async () => {
  const files = { 'echo.nl': named('Return value.'), 'main.ts': "import echo from './echo.nl'; export default echo;" };
  const application = host.compileVirtualProject({ files }, host, { target: 'node', programId: 'application' });
  const library = host.compileVirtualProject({ files }, host, { target: 'node', programId: 'library' });
  assert.equal(application.ok, true); assert.equal(library.ok, true);
  const caller = application.require('main.ts').default, callee = library.require('main.ts').default;
  let ownerSeen;
  const runtime = host.createNatlangRuntime({ program: application.manifest.adaptation,
    agent: async session => {
      if (session.runtime.frame.programOwner === 'application') {
        session.lam.return = await callee('from library');
      } else {
        ownerSeen = session.runtime.frame.programOwner;
        const result = await session.applyAsync('eval', { code: 'return value' });
        assert.equal(result.kind, 'ok', result.text);
      }
    } });
  assert.equal(await runtime.run(() => caller('input')), 'from library');
  assert.equal(ownerSeen, 'library');
});
