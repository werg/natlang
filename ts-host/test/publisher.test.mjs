import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtemp, readFile, readlink, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createNatlangRuntime, openFolder } from '../dist/index.js';
import { EvidenceCollection } from '../../applications/dist/evidence/index.js';
import { DocumentPublisher, publishBrief } from '../../applications/dist/publisher/index.js';
import { scriptedModel } from './support/natlang.mjs';

test('natlang composes pinned evidence and prepares identical-source Markdown and HTML', async () => {
  const root = await mkdtemp(join(tmpdir(), 'natlang-publish-'));
  await writeFile(join(root, 'editorial.md'), 'Keep the summary concise.');
  const evidence = new EvidenceCollection([{ id: 'study', text: 'The count is 12.\n\nFurther work is needed.' }]);
  const publisher = new DocumentPublisher(root, evidence, {
    tables: { counts: { columns: ['Kind', 'Count'], rows: [['Samples', 12]] } }, assets: { graph: './graph.png' } });
  const revision = evidence.docs.get('study').revision;
  const model = scriptedModel(opening => {
    if (opening.includes('Plan a short document')) return 'const note = await files.file("editorial.md").readText();\n' +
      'return { title: "Study <results>", headings: [note.includes("concise") ? "Summary" : "Details"], selected_tables: ["counts"], selected_assets: ["graph"] }';
    return `return { title: outline.title, evidence_revision: collection_revision, assets: ["graph"],
      sections: [{ heading: outline.headings[0], body: "12 samples & follow-up.", table_id: "counts",
        claims: [{ text: "The count is 12.", span_id: "study#p0", revision: ${JSON.stringify(revision)}, quote: "count is 12" }] }] }`;
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
    assert.match(md, /count is 12/);
    assert.match(await readlink(join(root, 'report')), /^\.versions\//);
  } finally { await rm(root, { recursive: true, force: true }); }
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
