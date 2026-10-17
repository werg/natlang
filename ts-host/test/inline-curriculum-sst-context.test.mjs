import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import test from 'node:test';
import { sstFullSentenceIndex, sstRecordContext } from '../scripts/inline-curriculum/source-context.mjs';
import { SOURCES } from '../scripts/inline-curriculum/acquire.mjs';

function fixture(sentences = 'sentence_index\tsentence\n1\tThe film is undone by anachronistic quick edits and occasional jarring glimpses of a modern theater audience watching the events unfold .\n2\tA short real review .\n3\tA held review .\n', splits = 'sentence_index,splitset_label\n1,1\n2,1\n3,2\n') {
  const bytes = [Buffer.from(sentences), Buffer.from(splits)];
  const bind = raw => ({ sha256: createHash('sha256').update(raw).digest('hex'), bytes: raw.length });
  const authority = { sentences: bind(bytes[0]), splits: bind(bytes[1]), source_url: 'https://source.invalid/pinned', archive_sha256: 'a'.repeat(64) };
  return { bytes, authority, index: sstFullSentenceIndex(...bytes, authority) };
}

test('long source phrase is not a full sentence; full parent and short sentence are eligible', () => {
  const { index, authority } = fixture();
  const fragment = { id: 'original-fragment', text: 'anachronistic quick edits and occasional jarring glimpses of a modern theater audience watching the events unfold', label: 'positive' };
  assert.ok(fragment.text.length > 80);
  assert.equal(sstRecordContext(fragment, index, authority), null);
  const full = { id: 'full-parent', text: 'the film is undone by anachronistic quick edits and occasional jarring glimpses of a modern theater audience watching the events unfold .', label: 'negative' };
  const proof = sstRecordContext(full, index, authority);
  assert.equal(proof.members[0].sentence_index, 1);
  assert.equal(proof.record_label, 'negative');
  assert.equal(proof.matching, 'source-whitespace-and-case-fold-v1');
  assert.equal(sstRecordContext({ id: 'short', text: 'a short real review .', label: 'positive' }, index, authority).members[0].sentence_index, 2);
  assert.deepEqual(fragment, { id: 'original-fragment', text: fragment.text, label: 'positive' });
});

test('official test/dev membership cannot become train context', () => {
  const { index, authority } = fixture();
  assert.equal(sstRecordContext({ id: 'held', text: 'A held review .', label: 'negative' }, index, authority), null);
});

test('exact member hashes are mandatory even if a mutation still parses', () => {
  const { bytes, authority } = fixture();
  assert.throws(() => sstFullSentenceIndex(Buffer.from(bytes[0].toString().replace('short', 'other')), bytes[1], authority), /pinned context authority/);
  assert.throws(() => sstFullSentenceIndex(bytes[0], Buffer.from(bytes[1].toString().replace('3,2', '3,1')), authority), /pinned context authority/);
});

test('malformed or incomplete sentence/split joins fail closed', () => {
  for (const [sentences, splits] of [
    ['sentence_index\tsentence\n1\tReview\n1\tOther\n', 'sentence_index,splitset_label\n1,1\n'],
    ['sentence_index\tsentence\n1\tReview\n', 'sentence_index,splitset_label\n1,1\n2,1\n'],
    ['sentence_index\tsentence\n2\tReview\n', 'sentence_index,splitset_label\n1,1\n'],
    ['sentence_index\tsentence\n1\tReview\n', 'sentence_index,splitset_label\n1,4\n'],
  ]) assert.throws(() => fixture(sentences, splits), /SST/);
});

test('source acquisition binds official archive and exact sentence/split members', () => {
  const context = SOURCES.sst2.files.find(file => file.role === 'full-sentence-membership');
  assert.equal(context.url, 'https://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip');
  assert.match(context.sha256, /^[0-9a-f]{64}$/);
  assert.deepEqual(context.extract, Object.values(context.members).map(member => member.path));
  for (const member of Object.values(context.members)) {
    assert.match(member.sha256, /^[0-9a-f]{64}$/);
    assert.ok(Number.isSafeInteger(member.bytes) && member.bytes > 0);
  }
});

test('both actual SST2 loaders exclude context-free phrases and retain exact record provenance', async () => {
  const { mkdtempSync, mkdirSync, writeFileSync, rmSync } = await import('node:fs');
  const { tmpdir } = await import('node:os');
  const { join, dirname } = await import('node:path');
  const root = mkdtempSync(join(tmpdir(), 'natlang-sst-context-'));
  const old = process.env.NATLANG_DATASETS;
  const original = SOURCES.sst2.files;
  try {
    const f = fixture();
    const context = structuredClone(original.find(file => file.role === 'full-sentence-membership'));
    for (const [key, raw] of [['sentences', f.bytes[0]], ['splits', f.bytes[1]]]) {
      context.members[key] = { ...context.members[key], sha256: createHash('sha256').update(raw).digest('hex'), bytes: raw.length };
      const file = join(root, 'sst2', SOURCES.sst2.revision, 'stanfordSentimentTreebank', context.members[key].path);
      mkdirSync(dirname(file), { recursive: true }); writeFileSync(file, raw);
    }
    SOURCES.sst2.files = [original[0], context];
    const full = 'the film is undone by anachronistic quick edits and occasional jarring glimpses of a modern theater audience watching the events unfold .';
    const fragment = 'anachronistic quick edits and occasional jarring glimpses of a modern theater audience watching the events unfold';
    const file = join(root, 'sst2', SOURCES.sst2.revision, original[0].path + '.jsonl');
    mkdirSync(dirname(file), { recursive: true });
    writeFileSync(file, [{ sentence: full, label: 'negative' }, { sentence: fragment, label: 'positive' }].map(JSON.stringify).join('\n') + '\n');
    process.env.NATLANG_DATASETS = root;
    const { labeledRows } = await import('../scripts/inline-curriculum/folder-data.mjs');
    const { rowsOf } = await import('../scripts/inline-curriculum/labeled.mjs');
    for (const loader of [labeledRows, rowsOf]) {
      const rows = [...loader('sst2', 'train'), ...loader('sst2', 'test')];
      assert.equal(rows.length, 1);
      assert.equal(rows[0].text, full);
      assert.equal(rows[0].label, 'negative');
      assert.equal(rows[0].source_context.members[0].sentence_index, 1);
      assert.equal(rows[0].source_context.record_label, 'negative');
    }
    // Exercise the real unrelated CoEdIT generator: SST provenance must never
    // introduce a free dataset variable or leak into another source family.
    const coedit = join(root, 'coedit', SOURCES.coedit.revision, SOURCES.coedit.files[0].path);
    mkdirSync(dirname(coedit), { recursive: true });
    writeFileSync(coedit, Array.from({ length: 64 }, (_, i) => ({ task: 'gec',
      src: `Fix grammar: This are the complete draft number ${i} for the source fixture.`,
      tgt: `This is the complete draft number ${i} for the source fixture.` })).map(JSON.stringify).join('\n') + '\n');
    const { folderEdit } = await import('../scripts/inline-curriculum/folder-families.mjs');
    const [edit] = folderEdit(7001, 0, 'train', 'gec');
    assert.equal(edit.dataset, 'coedit');
    assert.ok(edit.dataset_records.length >= 20);
    assert.equal(edit.generation.source_context, undefined);
    assert.equal(edit.semantics.expected, edit.dataset_records.length);
  } finally {
    SOURCES.sst2.files = original;
    if (old === undefined) delete process.env.NATLANG_DATASETS; else process.env.NATLANG_DATASETS = old;
    rmSync(root, { recursive: true, force: true });
  }
});
