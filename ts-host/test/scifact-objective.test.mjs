import test from 'node:test';
import assert from 'node:assert/strict';
import { buildScifactEpisode, serviceSource } from '../scripts/skills/build-scifact-episodes.mjs';
import { scoreScifactObjective } from '../src/skills/scifact-objective.ts';
import { createHash } from 'node:crypto';

const sha = text => createHash('sha256').update(text).digest('hex');
const makeRow = (n, label = 'SUPPORT', component = `cmp-${n}`) => {
  const docId = 100 + n;
  const sentences = [{ sentence_id: 0, text: `Document ${n} reports measured result.` }, { sentence_id: 1, text: `Independent detail for claim ${n}.` }];
  return { schema: 'natlang.scifact-research-candidate/1', candidate_id: `claim-${n}`, collection_status: 'candidate_not_admitted', role: 'train',
    task: { instruction: 'Evaluate the claim.', claim: `Claim ${n}.`, documents: [{ doc_id: docId, title: `Paper ${n}`, abstract_sentences: sentences }] },
    host_only_oracle: { label, accepted_evidence_sets: label === 'NOT_ENOUGH_INFO' ? [] : [{ doc_id: docId, rationale_index: 0, label, sentence_ids: [1] }] },
    source_groups: [`claim:${n}`, `scifact:document:${docId}`, `scifact:component:${component}`],
    provenance: { component_id: component, claim_license: 'CC-BY-4.0', abstract_license: 'ODC-By-1.0', source_line_sha256: sha(`row${n}`), source_revision: 'abc' } };
};
const build = rows => {
  const text = rows.map(row => JSON.stringify(row)).join('\n') + '\n';
  const manifestText = JSON.stringify({ schema: 'natlang.scifact-research-preparation/1', source_revision: 'abc', source_archive_sha256: 'archive',
    dataset_citation: 'SciFact citation', schema_reference: 'schema', license_reference: 'license', outputs: { 'candidates.jsonl': { sha256: sha(text) } }, counts: { candidate_rows: rows.length } });
  return buildScifactEpisode({ candidateText: text, sourceManifest: JSON.parse(manifestText), candidateSha256: sha(text), sourceManifestSha256: sha(manifestText) });
};

test('SciFact scorer accepts exact label and any exact sufficient evidence alternative', () => {
  const packet = { allowedLabels: ['SUPPORT', 'CONTRADICT', 'NOT_ENOUGH_INFO'], catalog: [{ id: '9', title: 'Paper' }] };
  const expected = { kind: 'scifact-claim-evidence', label: 'SUPPORT', scope: 'supplied-cited-documents-only',
    available_documents: [{ doc_id: 9, sentence_ids: [0, 1] }],
    accepted_evidence_sets: [[{ doc_id: 9, sentence_ids: [1] }], [{ doc_id: '9', sentence_ids: ['0'] }]] };
  assert.equal(scoreScifactObjective(packet, { label: 'support', citations: [{ doc_id: '9', sentence_ids: ['1'] }] }, expected).quality, 1);
  assert.equal(scoreScifactObjective(packet, { label: 'SUPPORT', citations: [{ doc_id: 9, sentence_ids: [0] }] }, expected).quality, 1);
  assert.equal(scoreScifactObjective(packet, { label: 'CONTRADICT', citations: [{ doc_id: 9, sentence_ids: [1] }] }, expected).quality, 0);
  assert.equal(scoreScifactObjective(packet, { label: 'SUPPORT', citations: [{ doc_id: 9, sentence_ids: [99] }] }, expected).quality, 0);
  assert.equal(scoreScifactObjective(packet, { label: 'SUPPORT', citations: [{ doc_id: 404, sentence_ids: [1] }] }, expected).quality, 0);
});

test('NOT_ENOUGH_INFO is correct only with no citation and remains scoped to supplied documents', () => {
  const packet = { allowedLabels: ['SUPPORT', 'CONTRADICT', 'NOT_ENOUGH_INFO'], catalog: [{ id: '7', title: 'Cited paper' }] };
  const expected = { kind: 'scifact-claim-evidence', label: 'NOT_ENOUGH_INFO', scope: 'supplied-cited-documents-only',
    available_documents: [{ doc_id: 7, sentence_ids: [0] }], accepted_evidence_sets: [] };
  assert.equal(scoreScifactObjective(packet, { label: 'not enough info', citations: [] }, expected).quality, 1);
  assert.equal(scoreScifactObjective(packet, { label: 'NOT_ENOUGH_INFO', citations: [{ doc_id: 7, sentence_ids: [0] }] }, expected).quality, 0);
});

test('episode builder keeps whole connected components on one side and keeps labels out of services', () => {
  const rows = Array.from({ length: 30 }, (_, i) => makeRow(i, ['SUPPORT','CONTRADICT','NOT_ENOUGH_INFO'][i % 3]));
  const { episode, audit, lineage } = build(rows);
  assert.ok(audit.support_components >= 2 && audit.query_components >= 2);
  const support = new Set(episode.support.cases.map(row => row.group));
  assert.equal(episode.query.cases.some(row => support.has(row.group)), false);
  assert.equal(episode.support.cases.length + episode.query.cases.length, rows.length);
  assert.equal(lineage.length, rows.length);
  assert.ok(episode.support.cases.every(row => row.expected.kind === 'scifact-claim-evidence' && !row.args[0].includes('host_only_oracle')));
  for (const row of [...episode.support.cases, ...episode.query.cases]) {
    const service = row.services.research;
    assert.match(service, /export function search\(query: string\)/u);
    assert.match(service, /export function read\(sourceId: string\)/u);
    assert.doesNotMatch(service, /host_only_oracle|accepted_evidence_sets|rationale_index/u);
    assert.doesNotMatch(service, /NOT_ENOUGH_INFO|SUPPORT|CONTRADICT/u);
  }
  assert.equal(episode.provenance.admission, 'candidate skill episodes only; not admitted or training-ready');
});

test('episode builder fails closed on unlinked rationale sentence or candidate bytes drift', () => {
  const row = makeRow(1);
  row.host_only_oracle.accepted_evidence_sets[0].sentence_ids = [8];
  assert.throws(() => build([row]), /missing sentence/u);
  const good = makeRow(2);
  const text = JSON.stringify(good) + '\n';
  const manifest = { schema: 'natlang.scifact-research-preparation/1', outputs: { 'candidates.jsonl': { sha256: sha(text) } }, counts: { candidate_rows: 1 } };
  assert.throws(() => buildScifactEpisode({ candidateText: text, sourceManifest: manifest, candidateSha256: '0'.repeat(64), sourceManifestSha256: sha('{}') }), /hash mismatch/u);
});
