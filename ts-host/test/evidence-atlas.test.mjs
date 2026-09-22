import assert from 'node:assert/strict';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { NatlangHost } from '../dist/index.js';
import { EvidenceCollection } from '../../applications/evidence_atlas.mjs';
import { evalTurn } from './support/eval-turn.mjs';

const path = fileURLToPath(new URL('../../codebases/evidence_atlas/answer.nl', import.meta.url));

test('natlang plans search, reads versioned spans and returns citation-checked claims', async () => {
  const evidence = new EvidenceCollection([
    { id: 'spec', text: 'A Fold consumes events in the order supplied by its source.\n\nAn empty open stream waits for more events.' },
    { id: 'notes', text: 'A Map can process independent slots in parallel.\n\nThis text says the opposite of nothing.' },
  ]);
  const host = new NatlangHost({ host: { evidence,
    drainEvents: () => evidence.drainEvents() } });
  const revision = evidence.docs.get('spec').revision;
  const modelTurn = turn => {
    const prompt = String(turn.messages.find(m => m.role === 'user')?.content ?? '');
    if (prompt.includes('function answer(')) return evalTurn(turn,
      'const queries = await plan_search(question); const found = await search(queries); const selected = await select(question, found); const passages = await read(selected, found.collection_revision); const draft = await compose(question, passages, found.truncated); await verify(passages, draft, found.collection_revision)');
    if (prompt.includes('two or three focused search phrases')) return evalTurn(turn,
      `JSON.stringify(${JSON.stringify(['Fold consumes events', 'source order'])})`);
    if (prompt.includes('Choose IDs of the offered hits')) return evalTurn(turn,
      `JSON.stringify(${JSON.stringify(['spec#p0'])})`);
    return evalTurn(turn, `(${JSON.stringify({
      answer: 'Fold consumes events in source order.',
      claims: [{ text: 'Fold consumes events in source order', span_id: 'spec#p0',
        revision, quote: 'in the order supplied by its source' }], gaps: [],
    })})`);
  };
  try {
    const result = await host.run({ source: { kind: 'file', path },
      inputs: { question: 'How does Fold consume events?' }, modelTurn,
      });
    assert.equal(result.outcome.kind, 'done');
    assert.equal(result.value.status, 'citation-checked');
    assert.equal(result.value.claims[0].span_id, 'spec#p0');
  } finally { host.close(); }
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
