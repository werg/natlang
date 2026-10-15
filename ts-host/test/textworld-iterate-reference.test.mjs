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

// A finite exploration horizon returns normally before the missing-object twin
// reports its blocker; it no longer depends on a semantic divergence verdict.
test('TextWorld iterate references verify in both variants with structural termination', { skip }, async () => {
  const records = textworldIterate(17, 0);
  const variants = records.map(record => record.curriculum.variant).sort();
  assert.deepEqual(variants, ['object_missing', 'playable']);
  const checks = await verifyCases(records, TOOLS_PROMPT);
  for (const check of checks) assert.equal(check.ok, true, `${check.id}: ${JSON.stringify(check.problems)}`);
  const missing = records.find(record => record.curriculum.variant === 'object_missing');
  assert.equal(missing.curriculum.reference.failures, 0);
});
