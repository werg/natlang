import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, mkdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { findRefinementsModule, loadProgramRefinements } from '../dist/cli/program-refinements.js';

function build(files) {
  const root = mkdtempSync(join(tmpdir(), 'natlang-crisp-'));
  for (const [path, text] of Object.entries(files)) {
    mkdirSync(join(root, path, '..'), { recursive: true });
    writeFileSync(join(root, path), text);
  }
  return root;
}

test('a program refinements module beside or above the entry supplies crisp checkers by normalized predicate', async () => {
  const root = build({
    'src/app.js': 'export default 1;\n',
    'src/refinements.js': 'export const refinements = { "one   line": value => typeof value === "string" && !value.includes("\\n"), "defer": () => undefined };\n',
  });
  const entry = join(root, 'src/app.js');
  assert.equal(findRefinementsModule(entry, root), join(root, 'src/refinements.js'));
  const crisp = await loadProgramRefinements(entry, root);
  assert.equal(crisp['one line']('a'), true);
  assert.equal(crisp['one line']('a\nb'), false);
  assert.equal(crisp.defer('x'), undefined);
});

test('no refinements module means no crisp checkers; a malformed one is an error', async () => {
  const bare = build({ 'src/app.js': 'export default 1;\n' });
  assert.equal(await loadProgramRefinements(join(bare, 'src/app.js'), bare), undefined);
  const bad = build({ 'src/app.js': '', 'src/refinements.js': 'export const refinements = { p: 3 };\n' });
  await assert.rejects(loadProgramRefinements(join(bad, 'src/app.js'), bad), /must be a function/);
  const wrong = build({ 'src/app.js': '', 'src/refinements.js': 'export const other = 1;\n' });
  await assert.rejects(loadProgramRefinements(join(wrong, 'src/app.js'), wrong), /must export `refinements`/);
});
