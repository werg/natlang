import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createNatlangRuntime } from '../dist/index.js';
import { EvidenceCollection, answer, truncationGap } from '../../applications/dist/evidence/index.js';
import { appCrisp, scriptedModel } from './support/natlang.mjs';

const documents = [
  { id: 'spec', text: 'A reducer consumes events in the order supplied by its source.\n\nAn empty open stream waits for more events.' },
  { id: 'notes', text: 'Independent slots can run in parallel.\n\nThis text says the opposite of nothing.' },
];

/**
 * A scripted model for the five stages. `stages` replaces the code a stage returns; a stage that is a function of the
 * opening text may return different code for the first call and for a repair.
 */
function stagesModel(stages = {}) {
  const calls = { plan: 0, select: 0, claims: 0, gaps: 0, write: 0 };
  const model = scriptedModel(opening => {
    const stage = opening.includes('Write search phrases for question') ? 'plan' :
      opening.includes('Choose the hits of found') ? 'select' :
      opening.includes('Write the claims that passages support') ? 'claims' :
      opening.includes('Write the gaps in what the claims establish') ? 'gaps' : 'write';
    calls[stage]++;
    const defaults = {
      plan: () => 'return ["reducer consumes events", "source order"]',
      select: () => 'return [found.hits[0].id]',
      claims: () => 'return [{ text: "Events arrive in source order", span_id: passages[0].id, quote: "in the order supplied by its source" }]',
      gaps: () => 'return []',
      write: () => 'return "A reducer consumes events in source order."',
    };
    return (stages[stage] ?? defaults[stage])(opening, calls[stage]);
  });
  return { ...model, calls };
}

const ask = (model, evidence, question = 'How does a reducer consume events?') =>
  createNatlangRuntime({ model: model.driver }).run(() => answer(evidence, question));

test('natlang plans search, reads versioned spans and returns citation-checked claims', async () => {
  const evidence = new EvidenceCollection(documents);
  const revision = evidence.docs.get('spec').revision;
  const model = stagesModel();
  const result = await ask(model, evidence);
  assert.equal(result.status, 'citation-checked', result.detail);
  assert.equal(result.answer, 'A reducer consumes events in source order.');
  assert.equal(result.claims[0].span_id, 'spec#p0');
  assert.equal(result.claims[0].revision, revision, 'the host fills the revision; the model never writes it');
  assert.equal(result.collection_revision, evidence.revision());
  assert.deepEqual(model.calls, { plan: 1, select: 1, claims: 1, gaps: 1, write: 1 });
});

test('the host adds the truncation gap and the answer is partial', async () => {
  const evidence = new EvidenceCollection([{ id: 'many', text: 'events one.\n\nevents two.\n\nevents three.' }], { maxHits: 2 });
  const model = stagesModel({
    claims: () => 'return [{ text: "events one", span_id: passages[0].id, quote: "events one" }]',
    gaps: () => 'return ["The second question part is not addressed."]' });
  const result = await ask(model, evidence);
  const found = evidence.search(['events']);
  assert.equal(found.truncated, true);
  assert.deepEqual(result.gaps, ['The second question part is not addressed.', truncationGap(found)]);
  assert.match(result.gaps[1], /2 of 3 matching passages/);
  assert.equal(result.status, 'partial');
});

test('a selected id that is not a hit gets one repair attempt with the exact problem', async () => {
  const evidence = new EvidenceCollection(documents);
  const model = stagesModel({
    select: (opening, n) => n === 1 ? 'return ["spec#p9"]' : 'return [found.hits[0].id]' });
  const result = await ask(model, evidence);
  assert.equal(result.status, 'citation-checked', result.detail);
  assert.equal(model.calls.select, 2);
  const repair = model.openings.filter(opening => opening.includes('Choose the hits of found')).at(1);
  assert.match(repair, /These ids are not hits of found: spec#p9/);
});

test('two invalid selections end as an unresolved value, not a throw', async () => {
  const evidence = new EvidenceCollection(documents);
  const model = stagesModel({ select: () => 'return ["spec#p9"]' });
  const result = await ask(model, evidence);
  assert.equal(result.status, 'unresolved');
  assert.match(result.detail, /Invalid selection: .*spec#p9/);
  assert.equal(model.calls.select, 2);
  assert.equal(model.calls.claims, 0);
});

test('an invalid quote gets one repair attempt, then reports invalid-citation', async () => {
  const evidence = new EvidenceCollection(documents);
  const repaired = stagesModel({
    claims: (opening, n) => n === 1 ?
      'return [{ text: "Events arrive in order", span_id: passages[0].id, quote: "no such words" }]' :
      'return [{ text: "Events arrive in order", span_id: passages[0].id, quote: "consumes events in the order" }]' });
  const first = await ask(repaired, evidence);
  assert.equal(first.status, 'citation-checked', first.detail);
  assert.equal(repaired.calls.claims, 2);
  assert.match(repaired.openings.filter(opening => opening.includes('Write the claims that passages support')).at(1),
    /no such words.{1,3} does not occur in passage spec#p0; copy a short stretch/);

  const stubborn = stagesModel({ claims: () => 'return [{ text: "x", span_id: "ghost#p0", quote: "no such words" }]' });
  const second = await ask(stubborn, evidence);
  assert.equal(second.status, 'invalid-citation');
  assert.equal(second.detail, 'Invalid citation: ghost#p0');
  assert.equal(stubborn.calls.claims, 2);
  assert.equal(stubborn.calls.write, 1, 'the answer is still written, and the status says the citation failed');
});

test('crisp checkers for the refined results', async () => {
  const crisp = await appCrisp('evidence');
  const phrases = crisp['two or three search phrases'];
  assert.equal(phrases(['a b', 'c d']), true);
  assert.equal(phrases(['a b', 'c d', 'e f']), true);
  assert.equal(phrases(['a b']), false);
  assert.equal(phrases(['a', 'b', 'c', 'd']), false);
  assert.equal(phrases(['a', ' ']), false);
  const once = crisp['each id listed once'];
  assert.equal(once([]), true);
  assert.equal(once(['a', 'b']), true);
  assert.equal(once(['a', 'a']), false);
});

test('fabricated quotes, changed passages and stale collection revisions are rejected', () => {
  const evidence = new EvidenceCollection([{ id: 'doc', text: 'The answer is limited.\n\nA second paragraph.' }]);
  const found = evidence.search(['answer limited']);
  const passage = evidence.read(['doc#p0'], found.collection_revision)[0];
  const draft = { answer: 'Limited answer', claims: [{ text: 'limited', span_id: passage.id,
    revision: passage.revision, quote: 'not present' }], gaps: [] };
  assert.equal(evidence.verify([passage], draft, found.collection_revision).status, 'invalid-citation');
  assert.match(evidence.citationProblems([passage], draft.claims)[0].problem, /does not occur in passage doc#p0/);
  draft.claims[0].quote = 'answer is limited';
  assert.equal(evidence.citationProblems([passage], draft.claims).length, 0);
  assert.equal(evidence.verify([{ ...passage, text: 'answer is limited but changed' }],
    draft, found.collection_revision).status, 'invalid-citation');
  evidence.update('doc', 'The answer changed.');
  assert.throws(() => evidence.read(['doc#p0'], found.collection_revision), /collection changed/);
});
