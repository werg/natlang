import test from 'node:test';
import assert from 'node:assert/strict';
import { createEpisodes } from '../scripts/skills/build-adversarial-episodes.mjs';
import { authorView, validateEpisode } from '../dist/skills/episode.js';
import { wordGames, wordScenarioCases } from '../dist/self-play/word-games.js';
import { semanticGames, semanticScenarioCases } from '../dist/self-play/semantic-games.js';
import { GAME_POLICY_SOURCE } from '../dist/self-play/policy.js';
import { arenaTicket } from '../dist/self-play/evaluation.js';

test('adversarial packets cover each semantic game seat with group-held-out queries and closed opponent snapshots', () => {
  const executor = 'reviewed-opponent-model';
  const episodes = createEpisodes(executor);
  const allGames = [...semanticGames, ...wordGames], allCases = [...semanticScenarioCases(), ...wordScenarioCases()];
  const expectedCount = allGames.reduce((total, game) => total + game.seats.length, 0);
  assert.equal(episodes.length, expectedCount);
  const groups = new Map(allCases.map(item => [item.group, item]));
  const seenRoles = new Set();
  for (const episode of episodes) {
    assert.deepEqual(validateEpisode(episode), [], episode.id);
    const parts = episode.family.split('-');
    assert.equal(parts.shift(), 'adversarial');
    const seat = parts.pop();
    const gameId = parts.join('-');
    const game = allGames.find(candidate => candidate.id === gameId);
    assert.ok(game?.seats.includes(seat));
    assert.ok(!seenRoles.has(`${gameId}:${seat}`));
    seenRoles.add(`${gameId}:${seat}`);
    assert.deepEqual(episode.target.files, { 'play.nl': GAME_POLICY_SOURCE });
    assert.equal(episode.target.entry, 'play.nl');
    assert.deepEqual(episode.library, { kind: 'empty', skills: {} });
    assert.equal(episode.provenance.arena.games[gameId].revision, game.revision);
    assert.deepEqual(episode.support.cases.map(row => row.group),
      allCases.filter(row => row.family === gameId).slice(0, 2).map(row => row.group));
    assert.deepEqual(episode.query.cases.map(row => row.group),
      allCases.filter(row => row.family === gameId).slice(2).map(row => row.group));
    const rows = [...episode.support.cases, ...episode.query.cases];
    assert.equal(Object.keys(episode.provenance.arena.cases).length, rows.length);
    for (const row of rows) {
      const config = episode.provenance.arena.cases[row.id];
      assert.deepEqual(row.args, [arenaTicket(row.id, config)]);
      assert.equal(row.expected, null);
      assert.equal(config.game, gameId);
      assert.equal(config.seat, seat);
      assert.deepEqual(Object.keys(config.opponents).sort(), game.seats.filter(s => s !== seat).sort());
      for (const opponent of Object.values(config.opponents)) {
        assert.equal(opponent.model, executor);
        assert.equal(opponent.entry, 'play.nl');
        assert.deepEqual(opponent.files, { 'play.nl': GAME_POLICY_SOURCE });
      }
      assert.deepEqual(config.scenario, groups.get(row.group).scenario);
    }
    const visible = JSON.stringify(authorView(episode));
    assert.ok(!visible.includes('adversarial-arena/1'));
    assert.ok(!visible.includes('rationale'));
    assert.ok(!visible.includes('privateVerdict'));
    assert.ok(!visible.includes('utilities'));
    assert.ok(!visible.includes('opponents'));
    assert.equal(authorView(episode).support.cases.length, episode.support.cases.length);
    assert.equal(authorView(episode).query, undefined);
  }
  assert.equal(seenRoles.size, 11);
});

test('builder requires a frozen opponent executor identity', () => {
  assert.throws(() => createEpisodes(''), /non-empty/);
});
