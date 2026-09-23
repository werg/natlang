import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createNatlangRuntime } from '../dist/index.js';
import { EvidenceCollection, answer } from '../../applications/dist/evidence/index.js';
import { scriptedModel } from './support/natlang.mjs';

test('natlang plans search, reads versioned spans and returns citation-checked claims', async () => {
  const evidence = new EvidenceCollection([
    { id: 'spec', text: 'A reducer consumes events in the order supplied by its source.\n\nAn empty open stream waits for more events.' },
    { id: 'notes', text: 'Independent slots can run in parallel.\n\nThis text says the opposite of nothing.' },
  ]);
  const revision = evidence.docs.get('spec').revision;
  const model = scriptedModel(opening => {
    if (opening.includes('two or three focused search phrases')) return 'result = ["reducer consumes events", "source order"]';
    if (opening.includes('Choose IDs of the offered hits')) return 'result = [found.hits[0].id]';
    return `result = { answer: "A reducer consumes events in source order.", claims: [{ text: "Events arrive in source order",
      span_id: passages[0].id, revision: passages[0].revision, quote: "in the order supplied by its source" }], gaps: [] }`;
  });
  const result = await createNatlangRuntime({ model: model.driver }).run(() => answer(evidence, 'How does a reducer consume events?'));
  assert.equal(result.status, 'citation-checked', result.detail);
  assert.equal(result.claims[0].span_id, 'spec#p0');
  assert.equal(result.claims[0].revision, revision);
  assert.equal(model.openings.length, 3);
});

test('fabricated quotes, changed passages and stale collection revisions are rejected', () => {
  const evidence = new EvidenceCollection([{ id: 'doc', text: 'The answer is limited.\n\nA second paragraph.' }]);
  const found = evidence.search(['answer limited']);
  const passage = evidence.read(['doc#p0'], found.collection_revision)[0];
  const draft = { answer: 'Limited answer', claims: [{ text: 'limited', span_id: passage.id,
    revision: passage.revision, quote: 'not present' }], gaps: [] };
  assert.equal(evidence.verify([passage], draft, found.collection_revision).status, 'invalid-citation');
  draft.claims[0].quote = 'answer is limited';
  assert.equal(evidence.verify([{ ...passage, text: 'answer is limited but changed' }],
    draft, found.collection_revision).status, 'invalid-citation');
  evidence.update('doc', 'The answer changed.');
  assert.throws(() => evidence.read(['doc#p0'], found.collection_revision), /collection changed/);
});
