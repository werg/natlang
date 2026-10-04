import test from 'node:test';
import assert from 'node:assert/strict';
import { buildTimeSeriesEpisodes, families, metric, variants } from '../scripts/skills/build-timeseries-episodes.mjs';
import { authorView, validateEpisode } from '../dist/skills/episode.js';
import { scoreGraded } from '../dist/skills/graded.js';

test('time-series episodes have six contextual families, three starting libraries, and template-held-out queries', () => {
  const episodes = buildTimeSeriesEpisodes();
  assert.equal(families.length, 6);
  assert.deepEqual(Object.keys(variants), ['empty', 'general-guide', 'metadata-tuned']);
  assert.equal(episodes.length, 18);
  for (const family of families) {
    const familyEpisodes = episodes.filter(row => row.family === `timeseries:${family.id}`);
    assert.equal(familyEpisodes.length, 3);
    for (const episode of familyEpisodes) {
      assert.deepEqual(validateEpisode(episode), [], episode.id);
      assert.deepEqual(episode.source_groups,
        family.templates.map(template => `timeseries-template/${family.id}/${template.id}`));
      assert.deepEqual(episode.support.cases.map(row => row.group), episode.provenance.template_holdout.support
        .map(template => `timeseries-template/${family.id}/${template}`));
      assert.deepEqual(episode.query.cases.map(row => row.group), episode.provenance.template_holdout.query
        .map(template => `timeseries-template/${family.id}/${template}`));
      assert.ok(episode.support.cases.every(row => !episode.query.cases.some(query => query.group === row.group)));
      assert.equal(episode.provenance.metric.kind, metric.kind);
      for (const row of [...episode.support.cases, ...episode.query.cases]) {
        const scenario = JSON.parse(row.args[0]);
        assert.ok(scenario.story.length > 20);
        assert.ok(scenario.rule.length > 50);
        assert.ok(Array.isArray(scenario.observations) && scenario.observations.length >= 3);
        assert.equal(row.expected.kind, 'assignment');
        assert.ok(scenario.allowedCategories.length >= 2);
        for (const label of Object.values(row.expected.value)) assert.ok(scenario.allowedCategories.includes(label), `${episode.id}: category invisible to executor`);
        const scored = scoreGraded(metric, row.expected.value, row.expected);
        assert.equal(scored.quality, 1, `${episode.id} ${row.id}`);
        assert.equal(scored.gates.all_correct, true);
      }
      const visible = JSON.stringify(authorView(episode));
      assert.equal(authorView(episode).query, undefined);
      assert.ok(!visible.includes('template_holdout'));
      assert.ok(!visible.includes('skill-graded/1'));
    }
    const guides = familyEpisodes.filter(row => row.library.kind === 'existing');
    const plain = guides.find(row => row.provenance.variant === 'general-guide').library.skills['series-reading']['SKILL.md'];
    const tuned = guides.find(row => row.provenance.variant === 'metadata-tuned').library.skills['series-reading']['SKILL.md'];
    assert.equal(plain.split('---')[2], tuned.split('---')[2], 'only the skill description metadata changes');
    assert.notEqual(plain.split('---')[1], tuned.split('---')[1]);
  }
});

test('time-series packet construction is deterministic and includes visible operational context', () => {
  const first = buildTimeSeriesEpisodes(), second = buildTimeSeriesEpisodes();
  assert.deepEqual(first, second);
  const all = JSON.stringify(first);
  for (const phrase of ['calendarClass', 'workOrders', 'resetLog', 'calibrationFactor', 'interventions', 'backlogIn'])
    assert.ok(all.includes(phrase), `missing contextual signal ${phrase}`);
});

test('delayed intervention explicitly names both exact category IDs in visible rules', () => {
  const episode = buildTimeSeriesEpisodes().find(row => row.id === 'timeseries-delayed-intervention-empty');
  for (const row of [...episode.support.cases, ...episode.query.cases]) {
    const scenario = JSON.parse(row.args[0]);
    assert.deepEqual(scenario.allowedCategories, ['model-consistent', 'deviation']);
    for (const label of scenario.allowedCategories) assert.ok(scenario.rule.includes(label));
  }
});
