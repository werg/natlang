import assert from 'node:assert/strict';
import { test } from 'node:test';
import { analyzeEvalSnippet } from '../dist/compiler/eval-check.js';
import { createNatlangRuntime, loadVirtualNatlang } from '../dist/index.js';

const scope = { types: { Ticket: '{ id: string, text: string, priority: number }' },
  inputs: [{ name: 'tickets', type: 'Ticket[]' }, { name: 'note', type: 'string' }, { name: 'plan', type: 'string' }],
  locals: [], captures: [], imports: [], returns: 'string[]' };

function signatures(code) {
  const { plans, diagnostics } = analyzeEvalSnippet(code, scope);
  assert.deepEqual(diagnostics.filter(item => item.severity === 'error').map(item => item.message), []);
  return plans.map(plan => `(${plan.parameters.map(item => `${item.name}: ${item.type.natlang}`).join(', ')}) => ${plan.returns.natlang}`);
}

test('an unannotated nl result gets its type from how the snippet uses it', () => {
  const cases = [
    ['if (await nl`Is note urgent?`(note)) {}', '(note: string) => boolean'],
    ["const label = !(await nl`Is note spam?`(note)) ? 'keep' : 'drop';", '(note: string) => boolean'],
    ['const verdicts = await Promise.all(tickets.map(t => nl`Does t report an outage?`(t)));\nreturn tickets.filter((t, i) => verdicts[i]).map(t => t.id);',
      '(t: Ticket) => boolean'],
    ['const verdicts: boolean[] = await Promise.all(tickets.map(t => nl`Does t report an outage?`(t)));', '(t: Ticket) => boolean'],
    ['const labels = await Promise.all(tickets.map(t => nl`Summarize t in five words.`(t)));\nreturn labels;', '(t: Ticket) => string'],
    ['const judge = nl`Does t report an outage?`;\nconst hits: string[] = [];\nfor (const t of tickets) if (await judge(t)) hits.push(t.id);',
      '(t: Ticket) => boolean'],
    ['const score = await nl`Score note from 1 to 10.`(note);\nconst doubled = score * 2;', '(note: string) => number'],
    ["const r = await nl`Rate note.`(note);\nconst label: 'low' | 'high' = r;\nconsole.log(r.toUpperCase());", '(note: string) => "low" | "high"'],
    ["const r: { risk: 'low' | 'high' } = { risk: await nl`Rate note.`(note) };", '(note: string) => "low" | "high"'],
    ['const s = await nl`Summarize note.`(note);\ns', '(note: string) => string'],
  ];
  for (const [code, expected] of cases) assert.deepEqual(signatures(code), [expected], code);
});

test('an nl step of iterateOn takes the initial state\'s type, and its stopping check gets it from context', () => {
  assert.deepEqual(signatures('const final = await nl`Make plan more evil.`.iterateOn(plan).until(nl`The plan is evil enough.`);'),
    ['(plan: string) => string', '(state: string) => boolean']);
  assert.deepEqual(signatures('const final = await iterateOn(nl`Improve plan.`, plan).until(d => d.length > 100);'), ['(plan: string) => string']);
});

test('uses that disagree, or say nothing about fields, produce diagnostics that propose the annotation', () => {
  const conflict = analyzeEvalSnippet('const v = await nl`Judge note.`(note);\nif (v) {}\nconst n: number = v;', scope).diagnostics;
  assert.equal(conflict[0].code, 'nl-ambiguous-signature');
  assert.match(conflict[0].message, /boolean \(tested as a condition in `if \(v\) \{\}`\); number \(expected here in `const n: number = v;`\)/);
  const fields = analyzeEvalSnippet('const a = await nl`Assess note.`(note);\nif (a.severity > 3) console.log(a.reason);', scope).diagnostics;
  assert.equal(fields[0].code, 'nl-unknown-return');
  assert.match(fields[0].message, /write `nl<\{ severity: …; reason: … \}>`/);
});

test('an inferred inline judgment runs per item with a boolean result', async () => {
  const seen = [];
  let started = false, staged = '';
  const model = async request => {
    const text = JSON.stringify(request.messages);
    if (!text.includes('Keep the tickets')) {
      // An inline child: its opening states the inferred signature.
      seen.push(String(request.messages[1].content));
      return { calls: [['return_result', { value: text.includes('total outage') }]] };
    }
    if (!started && (started = true)) return { calls: [['eval', { code:
      'const flags = await Promise.all(tickets.map(t => nl`Does t report an outage?`(t)));\nreturn tickets.filter((t, i) => flags[i]).map(t => t.id);' }]] };
    staged = String(request.messages.at(-1).content);
    return { calls: [['return_result', { value: ['a'] }]] };
  };
  const runtime = createNatlangRuntime({ model, seed: { mode: 'backend' } });
  const fn = loadVirtualNatlang({ 'root.nl': '---\nargs: { tickets: "{ id: string, text: string }[]" }\nreturns: string[]\n---\nKeep the tickets that report an outage.\n' }, 'root.nl');
  const value = await runtime.run(() => fn([{ id: 'a', text: 'total outage in eu' }, { id: 'b', text: 'typo on page' }]));
  assert.deepEqual(value, ['a']);
  assert.match(staged, /\["a"\]/);
  assert.deepEqual(seen.map(text => /^You are inside this call: nl@eval:1\(t: \{ id: string, text: string \}\): boolean$/m.test(text)), [true, true]);
});
