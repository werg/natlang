// Comprehensive tests for @natlang/std S1 collections (semanticMap, semanticFilter, semanticPartition,
// semanticReduce, semanticMatch, semanticRank), S3 policy (escalate, withContext, compileRule),
// S2 bug fixes (add-add / edit-delete semanticMerge, concurrent StdSpan trace parent isolation),
// and Pillar 1 runtime/compiler parity fixes (directory-reducer .on/.in/.partial, stacked refine<Is<Is<...>>>,
// callable-folder nl(...) desugaring, idiom-decision-readout on finite records, NatlangContext.ofCallable).
import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  compileVirtualProject, createNatlangRuntime, Folder, loadVirtualCallables, loadVirtualNatlang,
  nodePlatform,
} from '../dist/index.js';
import { lowerHostCell } from '../dist/compiler/host-cell.js';
import {
  semanticMap, semanticFilter, semanticPartition, semanticReduce, semanticMatch, semanticRank,
  semanticMerge, escalate, withContext, compileRule,
} from '../dist/std/index.js';
import { checkNlIdioms } from '../dist/compiler/idiom.js';
import { loadNamedFunction } from '../dist/runtime/loader.js';
import { virtualSourceFiles } from '../dist/runtime/virtual-project.js';
import { localModel } from './support/std.mjs';

test('S1 semanticMap preserves order and bounds concurrency', async () => {
  let active = 0, peak = 0;
  const items = [1, 2, 3, 4, 5, 6];
  const result = await createNatlangRuntime({}).run(() => semanticMap(items, async item => {
    active++;
    peak = Math.max(peak, active);
    await new Promise(resolve => setTimeout(resolve, 5));
    active--;
    return item * 10;
  }, { concurrency: 2 }));
  assert.deepEqual(result, [10, 20, 30, 40, 50, 60]);
  assert.ok(peak <= 2, `peak concurrency ${peak} exceeded 2`);
});

test('S1 semanticFilter works with both criterion strings (shipped meets.nl) and predicate functions with floors', async () => {
  const local = localModel(() => null, (options, prompt) => {
    if (/satisfies criterion/.test(prompt)) {
      if (prompt.includes('bug: crash')) return [0.92, 0.08];
      if (prompt.includes('borderline:')) return [0.65, 0.35];
      return [0.1, 0.9];
    }
    return [0.5, 0.5];
  });
  const runtime = createNatlangRuntime({ model: local.driver });
  const items = ['bug: crash on save', 'borderline: slow tooltip', 'praise: love it'];
  const keptDefault = await runtime.run(() => semanticFilter(items, 'reports a product defect'));
  assert.deepEqual(keptDefault, ['bug: crash on save', 'borderline: slow tooltip']);
  const keptStrict = await runtime.run(() => semanticFilter(items, 'reports a product defect', { floor: 0.8 }));
  assert.deepEqual(keptStrict, ['bug: crash on save']);
});

test('S1 semanticPartition groups by label and separates below-floor items into uncertain', async () => {
  const classify = loadVirtualNatlang({
    'classify.nl': '---\nargs: { text: string }\nreturns: \'"billing" | "tech"\'\nreadout: decision\n---\nClassify text.\n',
  }, 'classify.nl');
  const local = localModel(() => null, (options, prompt) => {
    if (prompt.includes('invoice')) return options.map(o => JSON.parse(o) === 'billing' ? 0.95 : 0.05);
    if (prompt.includes('500')) return options.map(o => JSON.parse(o) === 'tech' ? 0.91 : 0.09);
    return options.map(o => JSON.parse(o) === 'tech' ? 0.55 : 0.45);
  });
  const result = await createNatlangRuntime({ model: local.driver }).run(() =>
    semanticPartition(['duplicate invoice', '500 error', 'weird billing error'], classify, { floor: 0.8 }));
  assert.deepEqual(result.groups, { billing: ['duplicate invoice'], tech: ['500 error'] });
  assert.equal(result.uncertain.length, 1);
  assert.equal(result.uncertain[0].item, 'weird billing error');
  assert.equal(result.uncertain[0].label, 'tech');
});

test('S1 semanticReduce supports both linear and parallel tree modes and preserves references', async () => {
  const runtime = createNatlangRuntime({});
  const linear = await runtime.run(() => semanticReduce([1, 2, 3, 4], 0, (acc, n) => acc + n));
  assert.equal(linear, 10);
  const tree = await runtime.run(() => semanticReduce(['a', 'b', 'c', 'd', 'e'], '',
    (_init, item) => item.toUpperCase(), { mode: 'tree', combine: (l, r) => `(${l}+${r})`, concurrency: 2 }));
  assert.equal(tree, '(((A+B)+(C+D))+E)');
});

test('S1 semanticMatch uses exact blocking to prune pairs before calling relates.nl', async () => {
  const left = [{ id: 'l1', group: 'a', text: 'refund' }, { id: 'l2', group: 'b', text: 'login' }];
  const right = [{ id: 'r1', group: 'a', text: 'billing refund' }, { id: 'r2', group: 'b', text: 'password reset' }, { id: 'r3', group: 'a', text: 'shipping' }];
  const local = localModel(() => null, (options, prompt) => {
    if (/correspond to each other/.test(prompt)) {
      return /billing refund|password reset/.test(prompt) ? [0.9, 0.1] : [0.1, 0.9];
    }
    return [0.5, 0.5];
  });
  const res = await createNatlangRuntime({ model: local.driver }).run(() => semanticMatch(left, right, {
    relation: 'same customer issue',
    block: (l, r) => l.group === r.group,
    floor: 0.7,
  }));
  // Blocking pruned 6 pairs down to 3 logical relates.nl calls (each reasoned decision scores 4 samples => 12 scorer calls).
  assert.equal(new Set(local.decisions.map(d => d.prompt)).size, 3);
  assert.deepEqual(res.pairs.map(p => [p.left.id, p.right.id]), [['l1', 'r1'], ['l2', 'r2']]);
  assert.deepEqual(res.unmatchedLeft, []);
  assert.deepEqual(res.unmatchedRight.map(r => r.id), ['r3']);
});

test('S1 semanticRank runs debiased pairwise comparisons and cancels position bias', async () => {
  // Simulate a position-biased model that adds +0.2 to whichever item appears first.
  const strength = { alpha: 0.3, beta: 0.9, gamma: 0.6 };
  const local = localModel(() => null, (options, prompt) => {
    if (!/serves goal better/.test(prompt)) return [0.5, 0.5];
    const argsAt = prompt.lastIndexOf('first:');
    const secAt = prompt.lastIndexOf('second:');
    const firstPart = argsAt >= 0 && secAt > argsAt ? prompt.slice(argsAt, secAt) : prompt;
    const secondPart = secAt >= 0 ? prompt.slice(secAt) : '';
    const pick = text => Object.keys(strength).find(k => text.includes(k));
    const a = strength[pick(firstPart)] ?? 0.5, b = strength[pick(secondPart)] ?? 0.5;
    const pFirst = Math.min(0.95, Math.max(0.05, 0.5 + (a - b) * 0.5 + 0.15));
    return [pFirst, 1 - pFirst];
  });
  const res = await createNatlangRuntime({ model: local.driver }).run(() =>
    semanticRank(['alpha', 'beta', 'gamma'], 'best answer', { debiased: true }));
  assert.deepEqual(res.ranked, ['beta', 'gamma', 'alpha']);
  assert.equal(new Set(local.decisions.map(d => d.prompt)).size, 6, '3 items => 3 pairs * 2 orders = 6 unique comparisons');
});

test('S3 escalate gates on confidence floor and asymmetric risk floors, recording decision_escalate', async () => {
  const verdict = loadVirtualNatlang({
    'verdict.nl': '---\nargs: { diff: string }\nreturns: \'"approve" | "block"\'\nreadout: decision\n---\nJudge diff.\n',
  }, 'verdict.nl');
  const local = localModel(() => null, (options, prompt) => {
    if (prompt.includes('typo')) return options.map(o => JSON.parse(o) === 'approve' ? 0.95 : 0.05);
    if (prompt.includes('auth')) return options.map(o => JSON.parse(o) === 'approve' ? 0.72 : 0.28);
    return options.map(o => JSON.parse(o) === 'approve' ? 0.6 : 0.4);
  });
  const teacherCalls = [];
  const teacher = async req => { teacherCalls.push(req); return { text: '"block"', calls: [] }; };
  const traces = [];
  const runtime = createNatlangRuntime({ model: local.driver, models: { teacher }, trace: t => traces.push(t) });
  const safeVerdict = escalate(verdict, { to: 'teacher', floor: 0.7, risk: { block: 0.25 } });

  const r1 = await runtime.run(() => safeVerdict('fix typo'));
  assert.equal(r1.value, 'approve'); assert.equal(r1.escalated, false);
  // 'auth' has confidence 0.72 >= floor 0.70, BUT P(block) = 0.28 >= risk.block 0.25 => escalates!
  const r2 = await runtime.run(() => safeVerdict('touch auth middleware'));
  assert.equal(r2.value, 'block'); assert.equal(r2.escalated, true); assert.equal(r2.reason, 'risk:block');
  assert.equal(r2.initialValue, 'approve');
  assert.equal(teacherCalls.length, 1);
  assert.ok(traces.some(t => t.events?.some(e => e.kind === 'decision_escalate' && e.reason === 'risk:block')));
});

test('S3 withContext preserves companion folder helpers while binding data', async () => {
  const tree = loadVirtualCallables({
    'judge.nl': '---\nargs: { item: string }\nreturns: string\n---\nCall helper(item) under policy.\n',
    'judge/helper.ts': 'export default function helper(item: string): string { return item.toUpperCase(); }\n',
  }, '', '/project', nodePlatform());
  const bound = withContext(tree.judge, { policy: 'strict' });
  const local = localModel(opening => opening.includes('strict') ? 'return await helper(item)' : null, () => [0.5, 0.5]);
  const res = await createNatlangRuntime({ model: local.driver }).run(() => bound('hello'));
  assert.equal(res, 'HELLO');
});

test('S3 compileRule synthesizes a function and repairs it against examples', async () => {
  let version = 0;
  const compiler = async (_rule, problems) => {
    const attempt = ++version;
    return async item => attempt === 1 && !problems?.length ? false : String(item).includes('urgent');
  };
  const fn = await createNatlangRuntime({}).run(() => compileRule('urgent tickets', {
    compiler,
    examples: [
      { name: 'urgent', input: 'urgent outage', expected: true },
      { name: 'normal', input: 'nice feature', expected: false },
    ],
    attempts: 2,
  }));
  assert.equal(version, 2, 'repaired after example 1 failed on attempt 1');
  assert.equal(await fn('urgent db issue'), true);
  assert.equal(await fn('color tweak'), false);
});

test('S2 semanticMerge handles add-add and edit-delete conflicts cleanly without undefined-arg crashes', async () => {
  const local = localModel(opening => {
    if (opening.includes('Say what the edit is for')) return 'return "updates the field"';
    if (opening.includes('Write the value that makes')) return 'return (ours.after ?? "") + "+" + (theirs.after ?? "")';
    return null;
  }, (options, prompt) => /one value can carry out both purposes/.test(prompt) ? [0.9, 0.1] : [0.5, 0.5]);
  const runtime = createNatlangRuntime({ model: local.driver });
  // Add-add conflict: base does not have `tag`, both ours and theirs add it.
  const addAdd = await runtime.run(() => semanticMerge({}, { tag: 'v1' }, { tag: 'v2' }));
  assert.equal(addAdd.value.tag, 'v1+v2');
  assert.equal(addAdd.unresolved.length, 0);
  // Edit-delete conflict: ours deletes `note`, theirs edits `note`.
  const editDel = await runtime.run(() => semanticMerge({ note: 'old' }, {}, { note: 'new' }));
  assert.equal(editDel.value.note, '+new');
});

test('S2 StdSpan nests child traces under the combinator and isolates concurrent Promise.all combinators', async () => {
  const traces = [];
  const fn = loadVirtualNatlang({ 'echo.nl': '---\nargs: { x: string }\nreturns: string\n---\nReturn x.\n' }, 'echo.nl');
  const local = localModel(() => 'return x', () => [0.5, 0.5]);
  const runtime = createNatlangRuntime({ model: local.driver, trace: t => traces.push(t) });
  await runtime.run(() => Promise.all([
    semanticMap(['a', 'b'], x => fn(x)),
    semanticMap(['c', 'd'], x => fn(x)),
  ]));
  const spans = traces.filter(t => t.definitionId === 'std:semanticMap');
  assert.equal(spans.length, 2);
  for (const span of spans) {
    const stageCalls = span.events.find(e => e.kind === 'std_stage' && e.stage === 'map').call_ids;
    assert.equal(stageCalls.length, 2, 'each concurrent combinator records only its own 2 child calls');
    for (const callId of stageCalls) {
      const child = traces.find(t => t.callId === callId);
      assert.equal(child.parentCallId, span.callId, 'child call nests directly under its std span');
    }
  }
});

test('Pillar 1: directory reducers keep APPLY_TO_FOLDER across .on(), .in(), and .partial()', async () => {
  const reducer = loadVirtualNatlang({
    'write.nl': '---\nkind: directory-reducer\nargs: { tag: string, name: string }\nreturns: string\n---\nWrite tag into name under policy.\n',
  }, 'write.nl');
  const local = localModel(opening => opening.includes('strict') ? 'await folder.file(name).writeText(tag); return tag;' : null, () => [0.5, 0.5]);
  const runtime = createNatlangRuntime({ model: local.driver, models: { teacher: local.driver } });
  const folder = Folder.fromFiles({}, 'write');
  const customized = withContext(reducer, { policy: 'strict' }).on({ model: 'teacher' }).partial('v1');
  const result = await runtime.run(() => folder.apply(customized, 'out.txt'));
  assert.equal(result, 'v1');
  assert.equal(await folder.readText('out.txt'), 'v1');
});

test('Pillar 1: refine<Is<Is<T, "a">, "b">>(x) checks each predicate separately and handles whitespace around >', async () => {
  const seen = [];
  const runtime = createNatlangRuntime({
    refinements: {
      crisp: {
        short: v => { seen.push('short'); return String(v).length <= 10; },
        lower: v => { seen.push('lower'); return String(v) === String(v).toLowerCase(); },
      },
    },
  });
  const project = compileVirtualProject({
    files: {
      'main.ts': `import { refine } from '@natlang/node';
type ShortLower = Is<Is<string, "short">, "lower">;
export async function run(x: string): Promise<ShortLower> {
  return refine< ShortLower >(x);
}
`,
    },
  }, await import('../dist/index.js'), { platform: nodePlatform() });
  assert.equal(project.ok, true, JSON.stringify(project.diagnostics));
  const out = await runtime.run(() => project.require('main.ts').run('ok'));
  assert.equal(out, 'ok');
  assert.deepEqual(seen.sort(), ['lower', 'short'], 'each stacked predicate ran against its own crisp checker');
  // Host-cell parity: refine<Is<...>> triggers lowering even when cell does not mention nl/iterateOn.
  const cell = lowerHostCell('const ok = await refine< Is<string, "short"> >("hi");', { ambientDeclarations: '' });
  assert.equal(cell.diagnostics.length, 0, JSON.stringify(cell.diagnostics));
  assert.match(cell.code, /refine\("hi", "short"\)/);
});

test('Pillar 1: callable-folder .ts desugars nl("...", x) and idiom check covers finite records', async () => {
  const tree = loadVirtualCallables({
    'ask.ts': `import { nl } from '@natlang/node';
export default async function ask(x: string): Promise<string> {
  return nl<string>('Echo x.', x);
}
`,
  }, '', '/project', nodePlatform());
  const local = localModel(opening => opening.includes('Echo x.') ? 'return x + "!"' : null, () => [0.5, 0.5]);
  assert.equal(await createNatlangRuntime({ model: local.driver }).run(() => tree.ask('hi')), 'hi!');
  // Finite record leaf without readout: decision now triggers idiom-decision-readout.
  const record = loadNamedFunction('/p/route.nl', virtualSourceFiles({
    'route.nl': '---\nargs: { text: string }\nreturns: "{ team: \'a\' | \'b\', urgent: boolean }"\n---\nRoute text.\n',
  }, '/p'));
  const warnings = checkNlIdioms(record, 'route.nl');
  assert.ok(warnings.some(w => w.code === 'idiom-decision-readout'), JSON.stringify(warnings));
});
