import test from 'node:test';
import assert from 'node:assert/strict';
import { folioEntailment } from '../scripts/inline-curriculum/sources.mjs';
import { missingDatasets } from './support/datasets.mjs';

test('FOLIO story 133 decisive metadata names only facts needed by the exact conclusion', { skip: missingDatasets('folio') }, () => {
  const item = folioEntailment(0, 10)[0];
  assert.equal(item.id, 'inline-curriculum:folio_entailment:story133:ex393');
  assert.deepEqual(item.curriculum.decisive.map(entry => entry.marker), [
    'The Emmet Building is a five-story building in Portland, Oregon.',
    'The Emmet Building was built in 1915.',
  ]);
  assert.deepEqual(item.semantics.expected, { verdict: 'entailed' });
  assert.deepEqual(item.curriculum.evidence.world, [
    'The Blake McFall Company Building is a commercial warehouse listed on the National Register of Historic Places.',
    'The Blake McFall Company Building was added to the National Register of Historic Places in 1990.',
    'The Emmet Building is a five-story building in Portland, Oregon.',
    'The Emmet Building was built in 1915.',
    'The Emmet Building is another name for the Blake McFall Company Building.',
    'John works at the Emmet Building.',
  ]);
});
