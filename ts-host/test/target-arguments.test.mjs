import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { argumentSignature, argumentUsage, describeArguments, parseTargetArguments, readTargetArguments } from '../dist/runtime/target-arguments.js';

const TYPES = `export type Ticket = {
  /** One line summary */
  title: string;
  priority?: 1 | 2 | 3;
  tags: string[];
};\n`;

function program() {
  const root = mkdtempSync(join(tmpdir(), 'natlang-target-arguments-'));
  writeFileSync(join(root, 'types.ts'), TYPES);
  writeFileSync(join(root, 'tickets.json'), '[]');
  return root;
}
const declared = parseTargetArguments([
  { name: 'mode', type: '"fast" | "careful"', default: 'fast' },
  { name: 'count', type: 'number', required: true },
  { name: 'tickets', type: 'Ticket[]' },
  { name: 'file', type: 'string', path: 'file' },
  { name: 'note', type: 'string | number' },
], 'main');

test('command-line words are read by type: text as written, other types as JSON, defaults and paths applied', () => {
  const root = program();
  try {
    assert.equal(argumentUsage(declared), '[mode: "fast" | "careful"] <count: number> [tickets: Ticket[]] [file: string] [note: string | number]');
    assert.equal(argumentSignature(declared), 'mode?: "fast" | "careful" = "fast", count: number, tickets?: Ticket[], file?: string, note?: string | number');
    const read = readTargetArguments(declared, ['', '3', '[{"title":"a","tags":[]}]', 'tickets.json', 'hello'], root, root);
    assert.deepEqual(read.problems, []);
    assert.deepEqual(read.values, { mode: 'fast', count: 3, tickets: [{ title: 'a', tags: [] }], file: join(root, 'tickets.json'), note: 'hello' });
    assert.deepEqual(readTargetArguments(declared, ['careful', '2', '', '', '42'], root, root).values, { mode: 'careful', count: 2, note: 42 });
  } finally { rmSync(root, { recursive: true, force: true }); }
});

test('every wrong argument is reported at the path the runtime check names', () => {
  const root = program();
  try {
    const messages = readTargetArguments(declared, ['slow', 'x', '[{"title":"a","priority":5}]', 'nope.json'], root, root).problems.map(problem => problem.message);
    assert.deepEqual(messages, ['mode: expected "fast" | "careful", got "slow"', 'count: expected number, got "x"', 'tickets/0/priority: expected 1 | 2 | 3, got 5',
      `file: no file nope.json in ${root}`]);
    assert.match(readTargetArguments(declared, ['', '1', '[{"title": }]'], root, root).problems[0].message, /^tickets is written as JSON \(Ticket\[\]\): /);
    assert.deepEqual(readTargetArguments(declared, ['', '1', '[{"title":"a"}]'], root, root).problems.map(problem => problem.path), ['tickets/0/tags']);
    assert.deepEqual(readTargetArguments(declared, [], root, root).problems.map(problem => problem.message), ['count is required (number)']);
    assert.match(readTargetArguments(declared, ['', '1', '', '', '', 'extra'], root, root).problems[0].message, /^takes 5 arguments/);
  } finally { rmSync(root, { recursive: true, force: true }); }
});

test('arguments can be named: --name VALUE or --name=VALUE, and the other words fill the rest in order', () => {
  const root = program();
  try {
    assert.deepEqual(readTargetArguments(declared, ['--count', '3', 'careful', '--note=x=y'], root, root),
      { values: { mode: 'careful', count: 3, note: 'x=y' }, problems: [] });
    assert.deepEqual(readTargetArguments(declared, ['--count=4', '--mode='], root, root).values, { mode: 'fast', count: 4 });
    assert.deepEqual(readTargetArguments(declared, ['--colour=red', '--count'], root, root).problems.map(problem => problem.message),
      ['--colour is not an argument of this target; its arguments are --mode, --count, --tickets, --file, --note',
        '--count needs a value: --count VALUE or --count=VALUE', 'count is required (number)']);
    assert.match(readTargetArguments(declared, ['--count=1', '--count=2'], root, root).problems[0].message, /--count is given twice/);
  } finally { rmSync(root, { recursive: true, force: true }); }
});

test('the dashboard description expands aliases with field docs and reports a default of the wrong type', () => {
  const root = program();
  try {
    const [, , tickets] = describeArguments(declared, root);
    assert.equal(tickets.text, 'Ticket[]');
    assert.equal(tickets.spelling, 'json');
    assert.equal(tickets.tree.element.alias, 'Ticket');
    assert.deepEqual(tickets.tree.element.fields.map(field => [field.name, field.optional, field.doc]),
      [['title', false, 'One line summary'], ['priority', true, undefined], ['tags', false, undefined]]);
    const [wrong] = describeArguments(parseTargetArguments([{ name: 'n', type: 'number', default: 'three' }], 'main'), root);
    assert.equal(wrong.problem, 'n default: expected number, got "three"');
    const [unknown] = describeArguments(parseTargetArguments([{ name: 'x', type: 'Missing' }], 'main'), root);
    assert.equal(unknown.problem, 'unknown type name Missing');
  } finally { rmSync(root, { recursive: true, force: true }); }
});

test('the manifest rejects arguments that are not data or not natlang types, with the fix', () => {
  const reject = (argument, pattern) => assert.throws(() => parseTargetArguments([argument], 'main'), pattern);
  reject({ name: 'x', type: 'json' }, /json is not a natlang type; use unknown/);
  reject({ name: 'x', type: '(a: string) => string' }, /lambda types cannot be passed/);
  reject({ name: 'x', type: 'string', required: true, default: 'a' }, /an argument with a default is optional; remove required/);
  reject({ name: 'x', type: 'number', path: 'file' }, /a path argument has type string/);
  reject({ name: 'x', type: 'string', label: 'X' }, /unknown field label/);
  assert.throws(() => parseTargetArguments([{ name: 'x', type: 'string' }, { name: 'x', type: 'string' }], 'main'), /duplicate argument name x/);
});
