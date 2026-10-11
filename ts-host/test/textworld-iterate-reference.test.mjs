import assert from 'node:assert/strict';
import { test } from 'node:test';
import { pathToFileURL } from 'node:url';
import { textworldIterate } from '../scripts/inline-curriculum/textworld.mjs';
import { missingDatasets } from './support/datasets.mjs';

const curriculumUrl = process.env.NATLANG_CURRICULUM_JS ?
  pathToFileURL(process.env.NATLANG_CURRICULUM_JS).href :
  new URL('../dist/teacher/curriculum.js', import.meta.url).href;
const { verifyCases } = await import(curriculumUrl);
const { TOOLS_PROMPT } = await import(new URL('../dist/native/prompt.js', import.meta.url).href);

const skip = missingDatasets('textworld');

// Regression (2026-10-11): the object_missing twin's loop is stopped by the progress review's divergent verdict, which
// fails that eval by design; the reference did not declare that failure, so every object_missing case failed
// verification and the family silently trained only on playable games.
test('TextWorld iterate references verify in both variants, the divergent stop as the one expected failure', { skip }, async () => {
  const records = textworldIterate(17, 0);
  const variants = records.map(record => record.curriculum.variant).sort();
  assert.deepEqual(variants, ['object_missing', 'playable']);
  const checks = await verifyCases(records, TOOLS_PROMPT);
  for (const check of checks) assert.equal(check.ok, true, `${check.id}: ${JSON.stringify(check.problems)}`);
  const missing = records.find(record => record.curriculum.variant === 'object_missing');
  assert.equal(missing.curriculum.reference.failures, 1);
});
