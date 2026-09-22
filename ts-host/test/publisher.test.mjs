import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtemp, readFile, readlink, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { NatlangHost, NodeFileTree } from '../dist/index.js';
import { EvidenceCollection } from '../../applications/evidence_atlas.mjs';
import { DocumentPublisher } from '../../applications/publisher.mjs';
import { evalTurn } from './support/eval-turn.mjs';

const path = fileURLToPath(new URL('../../codebases/publisher/publish.nl', import.meta.url));

test('natlang composes pinned evidence and prepares identical-source Markdown and HTML', async () => {
  const root = await mkdtemp(join(tmpdir(), 'natlang-publish-'));
  await writeFile(join(root, 'editorial.md'), 'Keep the summary concise.');
  const evidence = new EvidenceCollection([{ id: 'study', text: 'The count is 12.\n\nFurther work is needed.' }]);
  const publisher = new DocumentPublisher(root, { evidence,
    tables: { counts: { columns: ['Kind', 'Count'], rows: [['Samples', 12]] } },
    assets: { graph: './graph.png' } });
  const revision = evidence.docs.get('study').revision;
  const modelTurn = turn => {
    const prompt = String(turn.messages.find(m => m.role === 'user')?.content ?? '');
    if (prompt.includes('function publish(')) return evalTurn(turn,
      'const passages = await read(span_ids, collection_revision); const outline = await plan(brief, passages, table_ids, asset_ids, files); const document = await compose(brief, outline, passages, collection_revision, table_ids, asset_ids, files); const checked = await check(document); if (checked.ok) await prepare(document, target); else await reject(target, checked)');
    if (prompt.includes('Plan a short document')) return evalTurn(turn, `(${JSON.stringify({
      title: 'Study <results>', headings: ['Summary'],
      selected_tables: ['counts'], selected_assets: ['graph'],
    })})`);
    return evalTurn(turn, `(${JSON.stringify({
      title: 'Study <results>', evidence_revision: evidence.revision(), assets: ['graph'],
      sections: [{ heading: 'Summary', body: '12 samples & follow-up.', table_id: 'counts',
        claims: [{ text: 'The count is 12.', span_id: 'study#p0', revision,
          quote: 'count is 12' }] }],
    })})`);
  };
  const host = new NatlangHost({ host: { publisher,
    drainEvents: () => publisher.drainEvents() } });
  try {
    const result = await host.run({ source: { kind: 'file', path },
      inputs: { brief: 'Summarize', span_ids: ['study#p0'],
        collection_revision: evidence.revision(), table_ids: ['counts'],
        asset_ids: ['graph'], target: 'report', files: new NodeFileTree(root) }, modelTurn });
    assert.equal(result.outcome.kind, 'done');
    assert.equal(result.value.status, 'prepared');
    const html = await readFile(join(root, 'report', 'document.html'), 'utf8');
    const md = await readFile(join(root, 'report', 'document.md'), 'utf8');
    assert.match(html, /Study &lt;results&gt;/);
    assert.match(html, /12 samples &amp; follow-up/);
    assert.match(html, /<td>12<\/td>/);
    assert.match(md, /count is 12/);
    assert.match(await readlink(join(root, 'report')), /^\.versions\//);
  } finally { host.close(); await rm(root, { recursive: true, force: true }); }
});

test('publication rejects fabricated claims, stale source, missing assets and unsafe URLs', async () => {
  const root = await mkdtemp(join(tmpdir(), 'natlang-publish-'));
  const evidence = new EvidenceCollection([{ id: 'doc', text: 'Verified result.' }]);
  const publisher = new DocumentPublisher(root, { evidence,
    assets: { evil: 'javascript:alert(1)' } });
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
