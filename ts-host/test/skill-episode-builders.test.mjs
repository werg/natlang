import test from 'node:test';
import assert from 'node:assert/strict';
import { slateEpisodes, programEpisodes } from '../scripts/skills/build-episodes.mjs';
import { repairEpisodes } from '../scripts/skills/build-repair-episodes.mjs';
import { authorView, validateEpisode } from '../dist/skills/episode.js';

function slate() {
  const families = ['active-urgency', 'explicit-consent', 'completed-delivery'];
  return families.flatMap(family => Array.from({ length: 12 }, (_, variant) => ({
    version: 'natlang.improvement-case/1', id: `private-task-${family}-${variant}`, family,
    provenance: { variant }, sourceGroups: [`private-source-group-${family}-${variant}`],
    contract: { entry: 'solve.nl', exportName: 'default' },
    files: { 'solve.nl': '---\nargs: {x: number}\nreturns: number\n---\nReturn x.\n' },
    cases: (variant < 8 ? ['train', 'train', 'train'] : ['validation', 'validation']).map((split, i) => ({
      id: `private-case-${family}-${variant}-${i}`, group: `private-case-group-${family}-${variant}-${i}`,
      split, args: [i], expected: i + 1,
    })),
  })));
}

test('slate generation keeps source groups in one split and one role across the corpus', () => {
  const episodes = slateEpisodes(slate());
  assert.equal(episodes.length, 4);
  const seen = new Map();
  for (const episode of episodes) {
    assert.deepEqual(validateEpisode(episode), []);
    for (const [role, cases] of [['support', episode.support.cases], ['query', episode.query.cases], ['transfer', episode.transfer.cases]]) {
      for (const row of cases) {
        assert.ok(!row.id.startsWith('private-'));
        assert.ok(!row.group.includes('private-'));
        const prior = seen.get(row.group);
        assert.ok(!prior || prior === `${episode.split}/${role}`, `group crossed role/split: ${row.group}`);
        seen.set(row.group, `${episode.split}/${role}`);
      }
    }
    const visible = JSON.stringify(authorView(episode));
    assert.ok(!visible.includes('private-task-'));
    assert.ok(!visible.includes('private-case-'));
  }
});

test('program records do not merge train and validation or split one source group', () => {
  const records = ['train-a', 'train-b', 'validation-c', 'validation-d'].map((id, i) => ({
    version: 'natlang.program/2', id: `private-${id}`, family: 'demo', split: i < 2 ? 'train' : 'validation',
    source_groups: [`private-group-${id}`], license: 'project-generated',
    semantics: { root: 'solve.nl', files: { 'solve.nl': '---\nargs: {}\nreturns: number\n---\nReturn 1.\n' }, expected: i,
      inputs: {} },
  }));
  const { episodes } = programEpisodes(records);
  assert.deepEqual(episodes.map(e => e.split).sort(), ['train', 'validation']);
  assert.ok(episodes.every(e => validateEpisode(e).length === 0));
  assert.ok(episodes.every(e => e.support.cases.every(c => !c.id.startsWith('private-'))));
});

test('repair episode IDs and author-visible provenance do not disclose repair labels or base identity', () => {
  const base = { version: 'natlang.skill-episode/1', id: 'private-base-id', family: 'f', split: 'train',
    source_groups: ['commitment'], license: 'project-generated', target: { kind: 'improvement-case', entry: 'solve.nl', source: { schema: 'x', id: 'opaque' }, files: { 'solve.nl': '---\nargs: {}\nreturns: number\n---\nReturn 1.\n' } },
    library: { kind: 'empty', skills: {} }, support: { cases: [{ id: 's1', group: 'sg1', expected: 1 }, { id: 's2', group: 'sg2', expected: 2 }] },
    query: { cases: [{ id: 'q', group: 'qg', expected: 3 }] }, operations: ['create'], limits: { maxSteps: 2 }, provenance: { raw_task_ids: ['private-base-id'] } };
  const seed = { name: 'demo', families: ['f'], files: { 'SKILL.md': '---\nname: demo\ndescription: Demo.\n---\n1. First.\n2. Then.\n' } };
  const repairs = repairEpisodes(base, [seed]).episodes;
  assert.ok(repairs.length >= 2);
  for (const episode of repairs) {
    assert.ok(!episode.id.includes('missing') && !episode.id.includes('incorrect') && !episode.id.includes('irrelevant'));
    assert.deepEqual(episode.provenance, { generator: 'natlang.skill-episodes/repair-2', reference_hidden: true });
    assert.ok(!JSON.stringify(authorView(episode)).includes('private-base-id'));
  }
});
