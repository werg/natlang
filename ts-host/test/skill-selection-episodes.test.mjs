import test from 'node:test';
import assert from 'node:assert/strict';
import { buildSelectionEpisodes } from '../scripts/skills/build-selection-episodes.mjs';
import { authorView, validateEpisode } from '../dist/skills/episode.js';

const taskFamilies = ['active-urgency', 'explicit-consent', 'completed-delivery', 'active-license', 'final-cancellation',
  'overall-recommendation', 'authorized-access', 'resolved-support', 'exact-multi-edit', 'identity-join', 'latest-revision', 'ranked-selection'];
function base(family, split, suffix) {
  const group = `${family}:${split}:${suffix}`;
  const target = { kind: 'improvement-case', entry: 'solve.nl', exportName: 'default',
    source: { schema: 'fixture', id: `opaque-target-${suffix}` },
    files: { 'solve.nl': '---\nargs: { input: string }\nreturns: string\n---\nResolve the requested case using the available evidence.\n' } };
  const one = role => [{ id: `case-${role}-${suffix}`, group: `${group}:${role}`, args: ['sample input'], expected: role }];
  return { version: 'natlang.skill-episode/1', id: `private-base-${group}`, family, split,
    source_groups: [`opaque-${group}`], license: 'project-generated', target,
    library: { kind: 'empty', skills: {} }, support: { cases: one('support') }, query: { cases: one('query') },
    transfer: { family: 'related-' + family, target: { ...target, source: { schema: 'fixture', id: `opaque-transfer-${suffix}` } }, cases: one('transfer') },
    operations: ['create', 'select', 'revise', 'test'], limits: { maxSteps: 4 },
    provenance: { private_source_id: group } };
}

test('selection packet has train and held-out counterparts with role-closed cases', () => {
  const bases = taskFamilies.flatMap(family => [base(family, 'train', 'tr'), base(family, 'validation', 'va')]);
  bases.push(...['optimization-knapsack', 'optimization-bin-packing', 'optimization-weighted-tardiness'].map(f => base(f, 'train', 'opt')));
  const { episodes, uniqueGroups } = buildSelectionEpisodes(bases);
  assert.equal(episodes.length, 54);
  assert.equal(episodes.filter(e => e.split === 'train').length, 30);
  assert.equal(episodes.filter(e => e.split === 'validation').length, 24);
  const assigned = new Map();
  for (const ep of episodes) {
    assert.deepEqual(validateEpisode(ep), [], ep.id);
    const view = authorView(ep), visible = JSON.stringify(view);
    assert.ok(!visible.includes('private_source_id') && !visible.includes('private-base-'));
    for (const [role, cases] of [['support', ep.support.cases], ['query', ep.query.cases], ['transfer', ep.transfer?.cases ?? []]])
      for (const item of cases) {
        const roleSplit = `${ep.split}/${role}`;
        assert.ok(!assigned.has(item.group) || assigned.get(item.group) === roleSplit);
        assigned.set(item.group, roleSplit);
      }
  }
  assert.equal(assigned.size, uniqueGroups);
});

test('library variants cover applicable, irrelevant, redundant and metadata-tuning candidates', () => {
  const { episodes } = buildSelectionEpisodes([base('exact-multi-edit', 'train', 'one')]);
  const [selection, tuning] = episodes;
  assert.equal(Object.keys(selection.library.skills).length, 3);
  assert.equal(Object.keys(tuning.library.skills).length, 3);
  const metadata = Object.values(tuning.library.skills).map(x => x['SKILL.md']).join('\n');
  assert.match(metadata, /families: \[\]/, 'metadata-only variant includes a missing applicability edge');
  assert.match(metadata, /families: \[exact-multi-edit, identity-join\]/, 'a distractor has an overbroad edge to tune');
  assert.deepEqual(Object.keys(selection.library.skills).sort(), Object.keys(tuning.library.skills).sort());
  for (const name of Object.keys(selection.library.skills)) {
    const stripMetadata = text => text.replace(/^---\n[\s\S]*?\n---\n\s*/, '');
    assert.equal(stripMetadata(selection.library.skills[name]['SKILL.md']), stripMetadata(tuning.library.skills[name]['SKILL.md']));
  }
  assert.equal(selection.target.files['solve.nl'], tuning.target.files['solve.nl']);
  assert.deepEqual(selection.query, tuning.query);
  assert.deepEqual(selection.transfer, tuning.transfer);
  assert.ok(!JSON.stringify(authorView(tuning)).includes('selection_design'));
  assert.ok(!JSON.stringify(authorView(tuning)).includes('applicability_metadata'));
});
