import assert from 'node:assert/strict';
import test from 'node:test';
import * as core from '../dist/index.js';
import { workerdPlatform } from '../dist/core/workerd.js';
import { checkEvalFields } from '../dist/compiler/eval-check.js';
import { toPiContext } from '../dist/model/pi-translation.js';

const scope = { types: {}, inputs: [], locals: [], captures: [], imports: [] };
const nl = body => `---\nargs: {}\nreturns: string\n---\n${body}`;

test('unknown runtime names reject, while erased types and safe typeof probes remain legal', () => {
  for (const code of ['missingHelper();', 'const record = { missingHelper };', 'const known = 1; console.log(knwon);',
    'class Child extends MissingBase {}', 'type OnlyType = string; console.log(OnlyType);'])
    assert.ok(checkEvalFields(code, scope).some(d => d.code === 'unbound-name'), code);
  for (const code of ['const value: MissingType = 1; console.log(value);',
    'type T = typeof MissingType; console.log(1);', 'interface Child extends MissingInterface {}', 'console.log(typeof missingHelper);',
    'console.log(typeof (missingHelper as unknown));', 'const missingHelper = () => 1; console.log(missingHelper());'])
    assert.deepEqual(checkEvalFields(code, scope), [], code);
  assert.ok(checkEvalFields('console.log(typeof missingHelper.field);', scope).some(d => d.code === 'unbound-name'));
  assert.deepEqual(checkEvalFields('missingHelper();', { ...scope, checkUnknownBindings: false }), []);
  assert.ok(checkEvalFields('secret.read();', { ...scope, unavailableServices: ['secret'], checkUnknownBindings: false })
    .some(d => d.code === 'callable-scope'));
});

for (const mode of ['node', 'workerd']) {
  const platformFor = () => mode === 'node' ? core.nodePlatform() : workerdPlatform({
    realm: { call(body, params, args) { return Function(...params, body)(...args); } },
  });
  for (const checkEvalBindings of [true, false]) test(`${mode} preflight rejects before effects; explicit legacy optout ${checkEvalBindings}`, async () => {
    let writes = 0;
    const platform = platformFor();
    const rt = core.createNatlangRuntime({ platform, calls: false, checkEvalBindings,
      services: { svc: { write() { writes++; } } },
      agent: async session => {
        const result = await session.applyAsync('eval', { code: 'console.log(inputs.token, read_inputs().token); svc.write(); missingHelper();', finish: false });
        assert.equal(result.kind, checkEvalBindings ? 'rejected' : 'error', result.text);
        assert.match(result.text, /missingHelper/);
        if (checkEvalBindings) assert.ok(result.codes.includes('unbound-name'), result.text);
        const finish = await session.applyAsync('eval', { code: 'return "recovered";', finish: true });
        assert.equal(finish.kind, 'completed', finish.text);
      } });
    try {
      const project = core.compileVirtualProject({ root: '/preflight', files: {
        'owner.ts': 'import judge from "./judge.nl"; export default async function owner():Promise<string>{return await judge("bound");}',
        'judge.nl': '---\nargs:\n  token: string\nreturns: string\n---\nReturn the result.',
      } }, core, { platform });
      assert.equal(project.ok, true, JSON.stringify(project.diagnostics));
      assert.equal(await rt.run(() => project.require('owner.ts').default()), 'recovered');
      assert.equal(writes, checkEvalBindings ? 0 : 1);
    } finally { rt.close(); }
  });

  test(`${mode} inline calls and persistent eval helpers retain real injected bindings`, async () => {
    const platform = platformFor(); let seen = 0;
    const rt = core.createNatlangRuntime({ platform, calls: false, agent: async session => {
      seen++;
      if (session.lam.functionName === 'owner') {
        const first = await session.applyAsync('eval', { code: 'function suffix(v:string):string { return v + "!"; }', finish: false });
        assert.equal(first.kind, 'ok', first.text);
        const second = await session.applyAsync('eval', { code: 'const judge = nl.with<string>({ label: suffix("bound") })`Return the supplied label.`; return await judge();', finish: true });
        assert.equal(second.kind, 'completed', second.text);
      } else {
        const bad = await session.applyAsync('eval', { code: 'missingInlineHelper();', finish: false });
        assert.ok(bad.codes.includes('unbound-name'), bad.text);
        const result = await session.applyAsync('eval', { code: 'console.log(read_code("nl")); console.log(transcript); console.log(NatlangContext.empty()); return label;', finish: true });
        assert.equal(result.kind, 'completed', result.text);
      }
    } });
    try {
      const project = core.compileVirtualProject({ root: '/inline', files: { 'entry.ts': 'import owner from "./owner.nl"; export default async function entry():Promise<string>{return await owner();}', 'owner.nl': nl('Ask a child for the label.') } }, core, { platform });
      assert.equal(project.ok, true, JSON.stringify(project.diagnostics));
      assert.equal(await rt.run(() => project.require('entry.ts').default()), 'bound!');
      assert.equal(seen, 2);
    } finally { rt.close(); }
  });
}

test('Luna/Pi opening advertises admitted callables and ports without unavailable helper names', async () => {
  let opening;
  const platform = core.nodePlatform();
  const rt = core.createNatlangRuntime({ platform, calls: false,
    services: { allowedPort: { read() {} }, hiddenPort: { read() {} } },
    serviceScopes: { hiddenPort: ['unavailableHelper.nl'] },
    model: async request => {
      opening = request;
      return { calls: [['return_result', { status: 'success', value: 'done' }]] };
    } });
  try {
    const project = core.compileVirtualProject({ root: '/opening', files: {
      'entry.ts': 'import owner from "./owner.nl"; export default async function entry():Promise<string>{return await owner();}',
      'owner.nl': nl('Return the text done.'), 'owner/availableHelper.nl': nl('Return supplied text.'),
    } }, core, { platform });
    assert.equal(project.ok, true, JSON.stringify(project.diagnostics));
    assert.equal(await rt.run(() => project.require('entry.ts').default()), 'done');
    const priming = opening.messages.filter(m => m.natlang_host_generated);
    assert.ok(priming.length);
    const actual = JSON.stringify(toPiContext(opening, { api: 'openai-codex-responses', provider: 'openai-codex', id: 'gpt-6-luna' }));
    assert.doesNotMatch(actual, /unavailableHelper|hiddenPort/);
    assert.match(actual, /availableHelper/);
    assert.match(actual, /allowedPort/);
  } finally { rt.close(); }
});
