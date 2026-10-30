import assert from 'node:assert/strict';
import { test } from 'node:test';
import ts from 'typescript';
import { declarationsForValues, lowerHostCell } from '../dist/compiler/host-cell.js';
import { createNatlangRuntime, nl } from '../dist/index.js';
import { scriptedModel } from './support/natlang.mjs';

const TICKETS = [{ id: 'a', text: 'site is down' }, { id: 'b', text: 'a typo' }];
const ambient = declarationsForValues({ scope: { tickets: TICKETS, threshold: 3, log: () => {}, opened: new Date(0) }, help: () => {} });
const lower = (source, options = {}) => lowerHostCell(source, { ambientDeclarations: ambient, ...options });
const signature = plan => `(${plan.parameters.map(item => `${item.name}: ${item.type.natlang}`).join(', ')}) => ${plan.returns.natlang}`;
const lines = text => text.split('\n').length;

test('a cell that never mentions the ambient nl is not analyzed: no program is built, even over broken declarations', () => {
  const source = 'const rows = scope.tickets.map(t => t.id);\nreturn rows;';
  const result = lowerHostCell(source, { ambientDeclarations: 'declare const (((' });
  assert.deepEqual(result, { code: source, plans: [], diagnostics: [], usesNl: false });
  // The word alone is not enough once the checker has looked: a property or a string is not the ambient binding.
  const named = lower('const note = "nl is a tag"; const row = { nl: 1 }; return row.nl;');
  assert.equal(named.usesNl, false);
  assert.equal(named.code, 'const note = "nl is a tag"; const row = { nl: 1 }; return row.nl;');
  assert.deepEqual(named.diagnostics, []);
});

test('a local declaration named nl shadows the ambient one and is not lowered', () => {
  const parameter = 'function f(nl) { return nl`not natlang`(1); }\nreturn f((strings) => () => strings[0]);';
  const local = lower(parameter);
  assert.equal(local.usesNl, false);
  assert.equal(local.code, parameter);
  assert.deepEqual(local.diagnostics, []);
  const moduleLevel = 'const nl = (s: TemplateStringsArray) => s[0];\nreturn nl`plain`;';
  assert.equal(lower(moduleLevel).code, moduleLevel);
  assert.equal(lower(moduleLevel).usesNl, false);
  // The shadow is a scope, not the file: the ambient nl outside it is still natlang's, and a shadowed one-shot call stays a call.
  const mixed = lower([
    'function f(nl: (text: string) => string) { return nl("kept as a call"); }',
    'const verdict: boolean = await nl`Is the sky blue?`();',
    'return [f(String), verdict];'].join('\n'));
  assert.deepEqual(mixed.diagnostics, []);
  assert.equal(mixed.usesNl, true);
  assert.equal(mixed.plans.length, 1);
  assert.match(mixed.code, /return nl\("kept as a call"\)/);
  assert.match(mixed.code, /await nl\.__inline\(/);
});

test('TSX cells keep types and JSX and the line of every statement', () => {
  const source = [
    'import { Badge } from "@workspace/ui";',
    'type Row = { id: string; text: string };',
    'const rows: Row[] = scope.tickets;',
    'const urgent = await Promise.all(rows.map(async (row: Row) => {',
    '  const flag: boolean = await nl`Is row an outage?`(row);',
    '  return flag;',
    '}));',
    'const Panel = () => <section data-count={urgent.length}>{rows.map(row => <Badge key={row.id}>{row.id}</Badge>)}</section>;',
    'export default Panel;'].join('\n');
  const result = lower(source, { jsx: true, fileName: 'panel.tsx' });
  assert.deepEqual(result.diagnostics, []);
  assert.equal(result.plans.length, 1);
  assert.equal(result.plans[0].sourceSpan.file, 'panel.tsx');
  assert.equal(signature(result.plans[0]), '(row: Row) => boolean');
  assert.equal(result.plans[0].parameters[0].type.aliases.Row, '{ id: string, text: string }');
  assert.match(result.code, /type Row = \{ id: string; text: string \};/);
  assert.match(result.code, /async \(row: Row\) =>/);
  assert.match(result.code, /<Badge key=\{row\.id\}>\{row\.id\}<\/Badge>/);
  assert.equal(lines(result.code), lines(source), 'sucrase positions and runtime stack lines stay those of the cell');
  const line = result.code.split('\n')[4];
  assert.match(line, /^  const flag: boolean = await nl\.__inline\(/);
  assert.match(line, /\)\(row\);$/);
  // The result is still TSX: the host's transform (sucrase) runs next. TypeScript's transpiler stands in for it here.
  const emitted = ts.transpileModule(result.code, { fileName: 'panel.tsx',
    compilerOptions: { jsx: ts.JsxEmit.ReactJSX, module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 }, reportDiagnostics: true });
  assert.deepEqual(emitted.diagnostics, []);
});

test('the signature comes from contextual type, immediate call arguments, result context and typed holes', () => {
  const plans = (source) => { const result = lower(source); assert.deepEqual(result.diagnostics, [], source); return result.plans.map(signature); };
  assert.deepEqual(plans('const summarize: (text: string) => Promise<string> = nl`Summarize text in one sentence.`;\nreturn summarize("x");'),
    ['(text: string) => string']);
  assert.deepEqual(plans('const t = scope.tickets[0]!;\nconst priority: "urgent" | "normal" = await nl`Decide whether the ticket is urgent.`(t);'),
    ['(t: { id: string, text: string }) => "urgent" | "normal"']);
  assert.deepEqual(plans('const n = await nl<number>`How many tickets are open?`(scope.tickets);'),
    ['(input: ({ id: string, text: string })[]) => number']);
  assert.deepEqual(plans('const flags = await Promise.all(scope.tickets.map(async t => { const ok: boolean = await nl`Is t an outage?`(t); return ok; }));'),
    ['(t: { id: string, text: string }) => boolean']);
  // Nothing says what the result is, but the code reads fields from it: those shape it.
  assert.deepEqual(plans('const a = await nl`Assess the note.`("note");\nif (a.severity > 3) console.log(a.reason);'),
    ['(input: string) => { severity: unknown, reason: unknown }']);
});

test('uses that disagree report the annotation to write, at the position in the cell as written', () => {
  const source = '// a cell\nconst note = "x";\nconst v = await nl`Judge note.`(note);\nif (v) {}\nconst n: number = v;';
  const result = lower(source);
  const [diagnostic] = result.diagnostics;
  assert.equal(diagnostic.code, 'nl-ambiguous-signature');
  assert.equal(diagnostic.severity, 'error');
  assert.match(diagnostic.message, /boolean \(tested as a condition in `if \(v\) \{\}`\); number \(expected here in `const n: number = v;`\)/);
  assert.match(diagnostic.message, /Write `nl<T>`/);
  assert.equal(diagnostic.file, 'cell.ts');
  assert.deepEqual([diagnostic.line, diagnostic.column], [3, 17]);
  assert.equal(source.slice(diagnostic.start, diagnostic.end).startsWith('nl`Judge note.`'), true);
  assert.equal(result.code, source, 'a cell with an error is returned as written');
  // A tag nobody calls gets the teaching error with its place too.
  const uncalled = lower('const ids = scope.tickets.map(t => nl<string>`Name the ticket.`);');
  assert.equal(uncalled.diagnostics[0].code, 'nl-not-called');
  assert.deepEqual([uncalled.diagnostics[0].line, uncalled.diagnostics[0].column], [1, 36]);
});

test('the one-shot call form is the call it reads as, and only for the built-in nl', () => {
  const source = 'const big: boolean = await nl(`Is ${scope.threshold} large?`);\nconst word = await nl<string>("Name a color.");\nreturn [big, word];';
  const result = lower(source);
  assert.deepEqual(result.diagnostics, []);
  assert.deepEqual(result.plans.map(signature), ['() => boolean', '() => string']);
  assert.equal(result.plans[0].instructions, 'Is ${…} large?');
  assert.equal(result.plans[0].interpolations[0].expression, 'scope.threshold');
  assert.equal(lines(result.code), lines(source));
  assert.match(result.code, /await nl\.__inline\(.*, \[scope\.threshold\], \{\}, undefined\)\(\);\nconst word = await nl\.__inline\(/s);
  // Instructions built at run time are an error that shows the template form, at the call.
  const built = lower('const text = "Is it " + "big?";\nawait nl(text);');
  assert.equal(built.diagnostics[0].code, 'nl-not-tag');
  assert.deepEqual([built.diagnostics[0].line, built.diagnostics[0].column], [2, 7]);
  assert.match(built.diagnostics[0].message, /nl is a template tag/);
});

test('declared types drive inference: scope contents come from the values', () => {
  const declarations = declarationsForValues({
    scope: { tickets: TICKETS, owner: { name: 'ada' }, byId: Object.fromEntries(Array.from({ length: 20 }, (_, i) => [`k${i}`, i])),
      empty: [], handle: new Map(), run: () => 1, opened: new Date(0), opaque: new (class Thing { x = 1; })() },
    count: 3, label: 'tickets', 'not-an-identifier': 1, class: 2 });
  assert.match(declarations, /^declare const scope: \{ tickets: \(\{ id: string, text: string \}\)\[\]; owner: \{ name: string \}; byId: Record<string, number>; empty: unknown\[\]; handle: Map<any, any>; run: \(\.\.\.args: any\[\]\) => any; opened: Date; opaque: any \};$/m);
  assert.match(declarations, /^declare const count: number;$/m);
  assert.doesNotMatch(declarations, /not-an-identifier|class/);
  const result = lowerHostCell('for (const t of scope.tickets) { const u: "urgent" | "normal" = await nl`Decide urgency.`(t); }\n' +
    'const who: string = await nl`Name the person.`(scope.owner);', { ambientDeclarations: declarations });
  assert.deepEqual(result.diagnostics, []);
  assert.deepEqual(result.plans.map(signature), ['(t: { id: string, text: string }) => "urgent" | "normal"', '(input: { name: string }) => string']);
  // Declarations are cut off, not read whole: a huge list is sampled and a cyclic value terminates.
  const cyclic = { name: 'loop' }; cyclic.self = cyclic;
  const big = declarationsForValues({ big: Array.from({ length: 100000 }, (_, i) => ({ i })), cyclic, deep: { a: { b: { c: { d: [1] } } } } },
    { maxDepth: 3 });
  assert.match(big, /^declare const big: \(\{ i: number \}\)\[\];$/m);
  assert.match(big, /^declare const cyclic: \{ name: string, self: Record<string, unknown> \};$/m);
  assert.match(big, /^declare const deep: \{ a: \{ b: \{ c: Record<string, unknown> \} \} \};$/m);
  // Typed from the declarations only: a name nothing declares is open (unknown), not guessed.
  const open = lowerHostCell('const n: number = await nl`Count.`(mystery);', { ambientDeclarations: '' });
  assert.deepEqual(open.plans.map(signature), ['(mystery: unknown) => number']);
});

test('ambient bindings are explicit authority: an instruction naming one does not capture it, nl.with does', () => {
  const implicit = lower('const answer: string = await nl`Summarize the scope for help.`();');
  assert.deepEqual(implicit.plans[0].captures, []);
  const explicit = lower('const answer: string = await nl.with({ scope })`Summarize the scope.`();');
  assert.deepEqual(explicit.diagnostics, []);
  assert.deepEqual(explicit.plans[0].captures.map(item => [item.name, item.mode]), [['scope', 'snapshot']]);
  assert.match(explicit.code, /nl\.__inline\(.*\{ scope: \[\(\) => scope\] \}, undefined\)\(\)/s);
});

test('let captures are lowered to live accessors, nl.with listings to snapshot and live ones, suffix .with to the same', () => {
  const source = [
    'let attempts = 0;',
    'const policy = "outage";',
    'const a: boolean = await nl`Is it a ${policy} after attempts? `(scope.tickets[0]);',
    'const b: string = await nl.with<{ attempts: number; label: string }, string>({ attempts: live(attempts), label: policy.toUpperCase() })`Name it.`();',
    'const c: string = await nl<string>`Name it again.`.with({ policy })();'].join('\n');
  const result = lower(source);
  assert.deepEqual(result.diagnostics, []);
  assert.deepEqual(result.plans[0].captures.map(item => [item.name, item.mutable]), [['policy', false], ['attempts', true]]);
  assert.match(result.code, /\{ policy: \[\(\) => policy\], attempts: \[\(\) => attempts, __natlang_value => \{ attempts = __natlang_value; \}\] \}/);
  // A snapshot expression is evaluated where the cell wrote it; the getter hands its value over.
  assert.match(result.code, /\(__natlang_snapshot_1 => \(\{ attempts: \[\(\) => attempts, __natlang_value => \{ attempts = __natlang_value; \}\], label: \[\(\) => __natlang_snapshot_1\] \}\)\)\(policy\.toUpperCase\(\)\)/);
  assert.match(result.code, /const c: string = await nl\.__inline\(.*\{ policy: \[\(\) => policy\] \}, undefined\)\(\);$/);
  assert.equal(lines(result.code), lines(source));
});

test('a callable folder is the context of every site, as in project builds', () => {
  const files = { '/app/natlang.d/note.ts': 'export function note(text: string): string { return text.trim(); }\n' };
  const sourceFiles = {
    join: (...parts) => parts.join('/').replace(/\/+/g, '/'), dirname: path => path.slice(0, path.lastIndexOf('/')) || '/',
    basename: (path, extension) => { const base = path.slice(path.lastIndexOf('/') + 1); return extension && base.endsWith(extension) ? base.slice(0, -extension.length) : base; },
    extname: path => { const base = path.slice(path.lastIndexOf('/') + 1); const at = base.lastIndexOf('.'); return at > 0 ? base.slice(at) : ''; },
    isFile: path => path in files, isDirectory: path => path === '/app/natlang.d',
    read: path => files[path], list: () => ['note.ts'] };
  const result = lower('const s: string = await nl`Tidy the text.`("x");', { context: { directory: '/app/natlang.d', files: sourceFiles } });
  assert.deepEqual(result.diagnostics, []);
  assert.match(result.code, /, undefined\)|, \{"note":\{/);
  assert.match(result.code, /\{"note":\{"kind":"module"/);
});

test('syntax errors are reported with their position, not thrown', () => {
  const source = 'const ok = 1;\nconst v: boolean = await nl`Is it?`(;\n';
  const result = lower(source);
  assert.equal(result.code, source);
  assert.equal(result.usesNl, true);
  assert.ok(result.diagnostics.length > 0);
  assert.equal(result.diagnostics[0].severity, 'error');
  assert.equal(result.diagnostics[0].code, 'typescript');
  assert.equal(result.diagnostics[0].line, 2);
  const brokenAmbient = lowerHostCell('await nl<string>`x`();', { ambientDeclarations: 'declare const (((' });
  assert.equal(brokenAmbient.diagnostics[0].file, 'ambient declarations');
});

test('a lowered cell runs against a scripted agent: captures read live, a let is written back, values interpolate', async () => {
  const source = [
    'let seen = 0;',
    'const verdicts: ("urgent" | "normal")[] = [];',
    'for (const t of scope.tickets) {',
    '  const verdict: "urgent" | "normal" = await nl`Decide whether the ticket is urgent. Tickets seen so far: ${seen}.`(t);',
    '  verdicts.push(verdict);',
    '  seen = seen + 1;',
    '}',
    'const word = await nl<string>("Name a color for the first ticket: " + 1);',
    'const spelled: string = await nl.with({ extra: await Promise.resolve("blue") })`Spell extra.`();',
    'return { verdicts, seen };'].join('\n');
  // The last line is a run-time-built instruction: an error. The runnable cell uses a literal one-shot call instead.
  assert.equal(lower(source).diagnostics[0].code, 'nl-not-tag');
  const runnable = source.replace('nl<string>("Name a color for the first ticket: " + 1)', 'nl<string>(`Name a color for ticket ${scope.tickets[0].id}.`)')
    .replace('return { verdicts, seen };', 'return { verdicts, seen, word, spelled };');
  const lowered = lower(runnable);
  assert.deepEqual(lowered.diagnostics, []);
  assert.equal(lowered.plans.length, 3);

  const js = ts.transpileModule(lowered.code, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext } }).outputText;
  const AsyncFunction = Object.getPrototypeOf(async () => {}).constructor;
  const cell = new AsyncFunction('nl', 'scope', js);

  const model = scriptedModel(opening => {
    if (opening.includes('Decide whether the ticket is urgent')) {
      // The child reads the cell's `seen` as it is now, and its write is the cell's value afterwards.
      return 'const before = seen; seen = seen + 100; return t.text.includes("down") ? "urgent" : "normal"';
    }
    if (opening.includes('Name a color')) return 'return "teal"';
    // A snapshot that awaits is evaluated in the cell, where await is legal, and handed over as a value.
    if (opening.includes('Spell extra')) return 'return extra';
    return null;
  });
  const result = await createNatlangRuntime({ model: model.driver }).run(() => cell(nl, { tickets: TICKETS }));
  assert.deepEqual(result, { verdicts: ['urgent', 'normal'], seen: 202, word: 'teal', spelled: 'blue' });
  const decisions = model.openings.filter(opening => opening.includes('Decide whether the ticket is urgent'));
  assert.equal(decisions.length, 2);
  assert.match(decisions[0], /Tickets seen so far: 0\./);
  assert.match(decisions[0], /let seen: number = 0;.*written back/);
  assert.match(decisions[1], /Tickets seen so far: 101\./, 'interpolations are evaluated in the cell, at the call');
  assert.match(decisions[1], /let seen: number = 101;.*written back/, 'the capture is read live at each call');
});
