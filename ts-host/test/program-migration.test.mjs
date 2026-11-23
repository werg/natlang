// applications/migration `program` target (semantic-FP WS7 M2): the exact parts of the program migrator. The
// judgments run on a model; these are the crisp steps around them: splitting a body into steps, assembling a file
// with its front matter kept, the role of a flagged file, and the exact check of a candidate with `natlang check`.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, mkdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { assemble, roleOf, splitSource, stepsOf, settle } from '../../applications/dist/migration/program/host.js';

test('stepsOf splits numbered items, keeping a preamble and fenced blocks with their step', () => {
  const body = 'Decide the verdict.\n1. Weigh the evidence.\n   More of step one.\n2. Return it:\n```ts\nconst a = 1;\n3. not a step\n```\n';
  assert.deepEqual(stepsOf(body), ['Decide the verdict.', '1. Weigh the evidence.\n   More of step one.', '2. Return it:\n```ts\nconst a = 1;\n3. not a step\n```']);
});

test('stepsOf splits paragraphs when nothing is numbered', () => {
  assert.deepEqual(stepsOf('First paragraph\ncontinues.\n\nSecond one.\n'), ['First paragraph\ncontinues.', 'Second one.']);
});

test('assemble keeps the front matter and adds readout: decision once', () => {
  const { front, body } = splitSource('---\nargs: { a: A }\nreturns: V\n---\nOld instructions.\n');
  assert.equal(body, 'Old instructions.\n');
  assert.equal(assemble(front, 'New.', true), '---\nargs: { a: A }\nreturns: V\nreadout: decision\n---\nNew.\n');
  assert.equal(assemble(`${front}\nreadout: decision`, 'New.', true), '---\nargs: { a: A }\nreturns: V\nreadout: decision\n---\nNew.\n');
  assert.equal(assemble(front, 'New.', false), '---\nargs: { a: A }\nreturns: V\n---\nNew.\n');
  assert.throws(() => splitSource('no front matter'), /front matter/);
});

test('roleOf: prose code in a file with a callable folder is a driver; otherwise a leaf', () => {
  const root = mkdtempSync(join(tmpdir(), 'program-migration-'));
  mkdirSync(join(root, 'run'));
  writeFileSync(join(root, 'run.nl'), '---\nreturns: string\n---\nx\n');
  writeFileSync(join(root, 'leaf.nl'), '---\nreturns: string\n---\nx\n');
  assert.equal(roleOf(root, 'run.nl', ['idiom-code-in-prose']), 'driver');
  assert.equal(roleOf(root, 'run.nl', ['idiom-decision-readout']), 'leaf');
  assert.equal(roleOf(root, 'leaf.nl', ['idiom-code-in-prose']), 'leaf');
});

test('settle accepts only what the exact rule accepted and gives up without attempts, with no model call', async () => {
  assert.equal(await settle({ accepted: true, problems: [], expectedPassingFirst: 3, expectedPassingSecond: 3 }, 0, []), 'accept');
  assert.equal(await settle({ accepted: false, problems: ['x'], expectedPassingFirst: 3, expectedPassingSecond: 1 }, 0, []), 'give-up');
});

test('problemsOf runs natlang check: a leaf with readout passes; a changed declaration or a prose-code body does not', async () => {
  process.env.NATLANG_BIN ??= fileURLToPath(new URL('../../scripts/natlang', import.meta.url));
  const { checkProgram, problemsOf } = await import('../../applications/dist/migration/program/host.js');
  const root = mkdtempSync(join(tmpdir(), 'program-migration-check-'));
  writeFileSync(join(root, 'natlang.json'), '{}');
  writeFileSync(join(root, 'types.ts'), 'export type Verdict = "pass" | "fail";\n');
  writeFileSync(join(root, 'gate.nl'), '---\nargs: { text: string }\nreturns: Verdict\n---\nReturn "pass" when text states a reason.\n');
  const original = await checkProgram(root);
  assert.ok(original.diagnostics.some(d => d.code === 'idiom-decision-readout'));
  assert.deepEqual(await problemsOf(root, original, { 'gate.nl': '---\nargs: { text: string }\nreturns: Verdict\nreadout: decision\n---\nAnswer "pass" when text gives a reason, else "fail".\n' }), []);
  const changed = await problemsOf(root, original, { 'gate.nl': '---\nargs: { words: string }\nreturns: Verdict\nreadout: decision\n---\nAnswer.\n' });
  assert.ok(changed.some(p => /front matter changed/.test(p)), changed.join('\n'));
  const stillFlagged = await problemsOf(root, original, { 'gate.nl': '---\nargs: { text: string }\nreturns: Verdict\n---\nAnswer "pass" when text gives a reason.\n' });
  assert.ok(stillFlagged.some(p => /readout: decision/.test(p)), stillFlagged.join('\n'));
});
