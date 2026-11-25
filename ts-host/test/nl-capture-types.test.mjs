import assert from 'node:assert/strict';
import { test } from 'node:test';
import { analyzeEvalSnippet } from '../dist/compiler/eval-check.js';
import { eagerTyping } from '../dist/compiler/data-rewrites.js';

const scope = { types: {}, inputs: [{ name: 'kind', type: 'string' }, { name: 'level', type: '"low" | "high"' }],
  locals: [], captures: [], imports: [], returns: 'string' };
const captureTypes = code => Object.fromEntries(analyzeEvalSnippet(code, scope).plans[0].captures
  .map(capture => [capture.name, capture.type.natlang]));

// authored-w1 agent-run-blame-tree: `const aim = kind === 'a' ? '...' : '...'` interpolated into an nl literal was
// captured with the type "<text a>" | "<text b>", so every call's opening listed every kind's instructions.
test('a local inferred from literals is captured as string, not the union of every branch text', () => {
  const code = "const aim = kind === 'a' ? 'restate the lost requirement' : 'copy the figures as given';\n" +
    'return nl<(x: string) => string>`Write ${aim} for x.`;';
  assert.deepEqual(captureTypes(code), { aim: 'string' });
  assert.deepEqual(captureTypes("const n = kind === 'a' ? 1 : 2;\nreturn nl<(x: string) => string>`Repeat x n times.`;"), { n: 'number' });
  // A declared type is the author's and stays.
  assert.deepEqual(captureTypes("const aim: 'a' | 'b' = kind === 'a' ? 'a' : 'b';\nreturn nl<(x: string) => string>`Write ${aim} for x.`;"),
    { aim: '"a" | "b"' });
  assert.deepEqual(captureTypes('const tier = level;\nreturn nl<(x: string) => string>`Rate x at tier.`;'), { tier: '"low" | "high"' });
  // Explicit captures alike.
  const explicit = analyzeEvalSnippet("const aim = kind === 'a' ? 'one' : 'two';\n" +
    'return nl.with<(x: string) => string>({ aim, other: kind === \'a\' ? \'x\' : \'y\' })`Write aim and other for x.`;', scope);
  assert.deepEqual(Object.fromEntries(explicit.plans[0].captures.map(c => [c.name, c.type.natlang])), { aim: 'string', other: 'string' });
});

test('NATLANG_LITERAL_CAPTURE_TYPES=1 keeps the literal types', () => {
  process.env.NATLANG_LITERAL_CAPTURE_TYPES = '1';
  try {
    assert.deepEqual(captureTypes("const aim = kind === 'a' ? 'one' : 'two';\nreturn nl<(x: string) => string>`Write ${aim} for x.`;"),
      { aim: '"one" | "two"' });
  } finally { delete process.env.NATLANG_LITERAL_CAPTURE_TYPES; }
});

test('eager typing annotates a conditional of literals with the base type', () => {
  const result = eagerTyping({ code: "const aim = kind === 'a' ? 'one' : 'two';\nconst n = 3;", scope });
  assert.match(result.code, /const aim: string = /);
  assert.match(result.code, /const n: number = /);
});

// authored-w1 annotated-game-claim-checker: "could not be played" captured an array named `played`.
test('a capture made only by an ordinary word in verb position is a warning naming the phrase and the fixes', () => {
  const warn = code => analyzeEvalSnippet(code, scope).diagnostics.filter(item => item.code === 'nl-word-capture');
  const played = warn("const played = ['Kd8', 'Qe2'];\nreturn nl<(move: string) => boolean>`False when move could not be played.`;");
  assert.equal(played.length, 1);
  assert.equal(played[0].severity, 'warning');
  assert.match(played[0].message, /"played" \("not be played"\)/);
  assert.match(played[0].message, /nl\.with/);
  // Still captured: the language rule is unchanged.
  assert.deepEqual(Object.keys(captureTypes("const played = ['Kd8'];\nreturn nl<(move: string) => boolean>`False when move could not be played.`;")), ['played']);
  // A noun use, a quoted name, a property access or an interpolation is a name.
  for (const code of [
    "const rubric = 'clear';\nreturn nl<(x: string) => string>`Grade x against rubric.`;",
    "const played = ['Kd8'];\nreturn nl<(m: string) => boolean>`Is m one of \\`played\\`? It could not be played otherwise.`;",
    "const played = ['Kd8'];\nreturn nl<(m: string) => boolean>`Is m in played.length moves? It could not be played.`;",
    "const played = 'Kd8';\nreturn nl<(m: string) => boolean>`Was ${played} the move m? It could not be played.`;",
  ]) assert.deepEqual(warn(code), [], code);
  // A warning never fails an eval.
  assert.ok(!analyzeEvalSnippet("const played = ['Kd8'];\nreturn nl<(m: string) => boolean>`m could not be played.`;", scope)
    .diagnostics.some(item => item.severity === 'error'));
});

test('NATLANG_WORD_CAPTURE_LINT=0 turns the warning off', () => {
  process.env.NATLANG_WORD_CAPTURE_LINT = '0';
  try {
    assert.deepEqual(analyzeEvalSnippet("const played = ['Kd8'];\nreturn nl<(m: string) => boolean>`m could not be played.`;", scope)
      .diagnostics.filter(item => item.code === 'nl-word-capture'), []);
  } finally { delete process.env.NATLANG_WORD_CAPTURE_LINT; }
});
