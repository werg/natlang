import assert from 'node:assert/strict';
import { test } from 'node:test';
import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { mkdtemp, readFile, readlink, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { basename, dirname, extname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createNatlangRuntime, openFolder } from '../dist/index.js';
import { readTypesFile } from '../dist/runtime/loader.js';
import { EvidenceCollection } from '../../applications/dist/evidence/index.js';
import { DocumentPublisher, numbersIn, outlineProblems, publishBrief, unsourcedNumbers } from '../../applications/dist/publisher/index.js';
import { scriptedModel } from './support/natlang.mjs';

const sourceFiles = {
  join, dirname, basename, extname,
  isFile: path => existsSync(path) && statSync(path).isFile(),
  isDirectory: path => existsSync(path) && statSync(path).isDirectory(),
  read: path => readFileSync(path, 'utf8'),
  list: path => readdirSync(path),
  relative: path => relative(process.cwd(), path),
};

test('natlang composes pinned evidence and prepares identical-source Markdown and HTML', async () => {
  const root = await mkdtemp(join(tmpdir(), 'natlang-publish-'));
  await writeFile(join(root, 'editorial.md'), 'Keep the summary concise.');
  const evidence = new EvidenceCollection([{ id: 'study', text: 'The count is 12.\n\nFurther work is needed.' }]);
  const publisher = new DocumentPublisher(root, evidence, {
    tables: { counts: { columns: ['Kind', 'Count'], rows: [['Samples', 12]] } }, assets: { graph: './graph.png' } });
  const model = scriptedModel(opening => {
    if (opening.includes('Return the text of the editorial context file')) return 'return await files.file("editorial.md").readText();';
    if (opening.includes('Plan the document that brief asks for')) return 'return { title: "Study <results>", sections: [' +
      '{ heading: note.includes("concise") ? "Summary" : "Details", purpose: "State the count", passage_ids: ["study#p0"], table_id: "counts", asset_ids: ["graph"] }] }';
    return 'return { body: "12 samples & follow-up.", claims: [{ text: "The count is 12.", span_id: "study#p0", quote: "count is 12" }] }';
  });
  try {
    const result = await createNatlangRuntime({ model: model.driver }).run(() => publishBrief(publisher, {
      brief: 'Summarize, following editorial.md', span_ids: ['study#p0'], collection_revision: evidence.revision(),
      table_ids: ['counts'], asset_ids: ['graph'], target: 'report', files: openFolder(root).root() }));
    assert.equal(result.status, 'prepared', result.detail);
    const html = await readFile(join(root, 'report', 'document.html'), 'utf8');
    const md = await readFile(join(root, 'report', 'document.md'), 'utf8');
    assert.match(html, /Study &lt;results&gt;/);
    assert.match(html, /<h2>Summary<\/h2>/);
    assert.match(html, /12 samples &amp; follow-up/);
    assert.match(html, /<td>12<\/td>/);
    assert.match(html, /<img alt="graph"/);
    assert.match(md, /count is 12/);
    assert.match(await readlink(join(root, 'report')), /^\.versions\//);
    // The model never wrote the evidence revision or the claim revision: the host did.
    assert.ok(html.includes(`data-revision="${evidence.docs.get('study').revision}"`));
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('sections are written at once and only a failing section is written again', async () => {
  const root = await mkdtemp(join(tmpdir(), 'natlang-publish-'));
  const evidence = new EvidenceCollection([{ id: 'study', text: 'The count is 12.\n\nThe ratio is 3.5 overall.' }]);
  const publisher = new DocumentPublisher(root, evidence);
  const written = [];
  const model = scriptedModel(async opening => {
    if (opening.includes('Plan the document that brief asks for')) return 'return { title: "Two parts", sections: [' +
      '{ heading: "Count", purpose: "PURPOSE-COUNT", passage_ids: ["study#p0"], table_id: null, asset_ids: [] },' +
      '{ heading: "Ratio", purpose: "PURPOSE-RATIO", passage_ids: ["study#p1"], table_id: null, asset_ids: [] }] }';
    const name = opening.includes('PURPOSE-COUNT') ? 'count' : 'ratio';
    written.push(name + (opening.includes('neither the passages nor the table') ? '-retry' : ''));
    await new Promise(resolve => setTimeout(resolve, 20));
    if (name === 'count') return 'return { body: "There are 12 samples.", claims: [{ text: "The count is 12.", span_id: "study#p0", quote: "count is 12" }] }';
    return opening.includes('neither the passages nor the table') ?
      'return { body: "The ratio is 3.5.", claims: [{ text: "The ratio is 3.5.", span_id: "study#p1", quote: "ratio is 3.5" }] }' :
      'return { body: "The ratio is 99.", claims: [{ text: "The ratio is 3.5.", span_id: "study#p1", quote: "ratio is 3.5" }] }';
  });
  try {
    const result = await createNatlangRuntime({ model: model.driver }).run(() => publishBrief(publisher, {
      brief: 'Report both', span_ids: ['study#p0', 'study#p1'], collection_revision: evidence.revision(),
      table_ids: [], asset_ids: [], target: 'report' }));
    assert.equal(result.status, 'prepared', result.detail);
    assert.deepEqual(written.slice(0, 2).sort(), ['count', 'ratio'], 'both sections start before either is retried');
    assert.deepEqual(written.slice(2), ['ratio-retry'], 'only the failing section is written again');
    assert.match(await readFile(join(root, 'report', 'document.md'), 'utf8'), /The ratio is 3\.5\./);
    const retry = model.openings.find(opening => opening.includes('neither the passages nor the table'));
    assert.match(retry, /99/);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('a section that fails twice rejects the brief and publishes nothing', async () => {
  const root = await mkdtemp(join(tmpdir(), 'natlang-publish-'));
  const evidence = new EvidenceCollection([{ id: 'study', text: 'The count is 12.' }]);
  const publisher = new DocumentPublisher(root, evidence);
  const model = scriptedModel(opening => opening.includes('Plan the document that brief asks for') ?
    'return { title: "T", sections: [{ heading: "Count", purpose: "p", passage_ids: ["study#p0"], table_id: null, asset_ids: [] }] }' :
    'return { body: "Twelve.", claims: [{ text: "The count is 12.", span_id: "study#p0", quote: "not in the text" }] }');
  try {
    const result = await createNatlangRuntime({ model: model.driver }).run(() => publishBrief(publisher, {
      brief: 'b', span_ids: ['study#p0'], collection_revision: evidence.revision(), table_ids: [], asset_ids: [], target: 'report' }));
    assert.equal(result.status, 'rejected');
    assert.match(result.detail, /section "Count": .*does not occur in passage study#p0/);
    assert.equal(model.openings.filter(opening => opening.includes('Write the section headed')).length, 2);
    assert.equal(existsSync(join(root, 'report')), false);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('an outline with ids that were not offered is repaired once, then rejected', async () => {
  const root = await mkdtemp(join(tmpdir(), 'natlang-publish-'));
  const evidence = new EvidenceCollection([{ id: 'study', text: 'The count is 12.' }]);
  const publisher = new DocumentPublisher(root, evidence, { tables: { counts: { columns: ['K'], rows: [['v']] } } });
  const bad = 'return { title: "T", sections: [{ heading: "A", purpose: "p", passage_ids: ["study#p7"], table_id: "other", asset_ids: [] }] }';
  const good = 'return { title: "T", sections: [{ heading: "A", purpose: "p", passage_ids: ["study#p0"], table_id: "counts", asset_ids: [] }] }';
  let outlines = 0;
  const run = (outline, request) => createNatlangRuntime({ model: scriptedModel(opening => {
    if (opening.includes('Plan the document that brief asks for')) { outlines++; return outline(outlines, opening); }
    return 'return { body: "The count is 12.", claims: [{ text: "The count is 12.", span_id: "study#p0", quote: "count is 12" }] }';
  }).driver }).run(() => publishBrief(publisher, { brief: 'b', span_ids: ['study#p0'], collection_revision: evidence.revision(),
    table_ids: ['counts'], asset_ids: [], target: 'report', ...request }));
  try {
    const repaired = await run((n, opening) => n === 1 ? bad : (assert.match(opening, /names passage ids that were not offered: study#p7/), good), {});
    assert.equal(repaired.status, 'prepared', repaired.detail);
    outlines = 0;
    const refused = await run(() => bad, { target: 'again' });
    assert.equal(refused.status, 'rejected');
    assert.match(refused.detail, /^invalid outline: .*study#p7.*table "other"/);
    assert.equal(outlines, 2);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('the crisp verifiers: numbers in prose, outline ids', () => {
  assert.deepEqual(numbersIn('1,200 and 3.5 of 12.'), ['1200', '3.5', '12']);
  const passages = [{ id: 'a#p0', text: 'Revenue was 1,200 in 2024.' }];
  const table = { columns: ['Kind', 'Count'], rows: [['Samples', 12]] };
  assert.deepEqual(unsourcedNumbers('Revenue 1200 in 2024; 12 samples.', passages, table), []);
  assert.deepEqual(unsourcedNumbers('Revenue 1300 in 2024; 13 samples; 13 again.', passages, table), ['1300', '13']);
  assert.deepEqual(unsourcedNumbers('No digits at all.', [], null), []);
  const offered = { passage_ids: ['a#p0'], table_ids: ['t'], asset_ids: ['x'] };
  const section = (extra = {}) => ({ heading: 'H', purpose: 'p', passage_ids: ['a#p0'], table_id: null, asset_ids: [], ...extra });
  assert.deepEqual(outlineProblems({ title: 'T', sections: [section({ table_id: 't', asset_ids: ['x'] })] }, offered), []);
  assert.match(outlineProblems({ title: 'T', sections: [] }, offered)[0], /at least one section/);
  assert.match(outlineProblems({ title: 'T', sections: [section({ asset_ids: ['y'] })] }, offered)[0], /asset "y"/);
  assert.match(outlineProblems({ title: 'T', sections: [section({ table_id: 't' }), section({ table_id: 't' })] }, offered)[0], /table t is assigned to 2 sections/);
  assert.match(outlineProblems({ title: ' ', sections: [section()] }, offered)[0], /needs a title/);
});

test('Passage and Claim have one definition: the publisher re-exports the evidence types', () => {
  const applications = fileURLToPath(new URL('../../applications', import.meta.url));
  const evidence = readTypesFile(join(applications, 'evidence', 'types.ts'), sourceFiles);
  const publisher = readTypesFile(join(applications, 'publisher', 'types.ts'), sourceFiles);
  for (const name of ['Passage', 'Claim', 'ClaimDraft']) {
    assert.ok(evidence[name], name);
    assert.equal(publisher[name], evidence[name], `${name} is the evidence definition`);
  }
  assert.match(publisher.Passage, /text: Untrusted<string>/);
  assert.match(publisher.Section, /claims: Claim\[\]/);
  assert.equal(readFileSync(join(applications, 'publisher', 'types.ts'), 'utf8').includes('type Passage ='), false);
});

test('publication rejects fabricated claims, stale source, missing assets and unsafe URLs', async () => {
  const root = await mkdtemp(join(tmpdir(), 'natlang-publish-'));
  const evidence = new EvidenceCollection([{ id: 'doc', text: 'Verified result.' }]);
  const publisher = new DocumentPublisher(root, evidence, { assets: { evil: 'javascript:alert(1)' } });
  const doc = { title: 'Report', evidence_revision: evidence.revision(), assets: [],
    sections: [{ heading: 'Result', body: 'Verified', claims: [{ text: 'Verified',
      span_id: 'doc#p0', revision: evidence.docs.get('doc').revision,
      quote: 'not present' }] }] };
  try {
    assert.equal(publisher.check(doc).ok, false);
    doc.sections[0].claims[0].quote = 'Verified result';
    assert.equal(publisher.check(doc).ok, true);
    doc.assets = ['evil'];
    assert.equal(publisher.check(doc).ok, false);
    doc.assets = ['missing'];
    assert.equal(publisher.check(doc).ok, false);
    doc.assets = [];
    evidence.update('doc', 'Updated result.');
    assert.equal((await publisher.publish(doc, 'report')).status, 'rejected');
    assert.equal((await publisher.publish(doc, '../escape')).status, 'rejected');
  } finally { await rm(root, { recursive: true, force: true }); }
});
