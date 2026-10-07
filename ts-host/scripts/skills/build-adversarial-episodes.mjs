#!/usr/bin/env node
import { mkdirSync, openSync, writeFileSync, closeSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { fingerprint } from '../../dist/adaptation/identity.js';
import { wordGames, wordScenarioCases } from '../../dist/self-play/word-games.js';
import { semanticGames, semanticScenarioCases } from '../../dist/self-play/semantic-games.js';
import { GAME_POLICY_SOURCE } from '../../dist/self-play/policy.js';
import { arenaTicket } from '../../dist/self-play/evaluation.js';

const SCHEMA = 'natlang.adversarial-arena/1';
const CASE_EXPECTATION = null; // Game outcomes are scored by the host; no literal answer is claimed.

function parseArgs(argv) {
  const options = {};
  for (let index = 0; index < argv.length; index++) {
    const key = argv[index];
    if (!['--out', '--executor-id', '--chesst-source', '--arena-root'].includes(key) || !argv[index + 1] || argv[index + 1].startsWith('--'))
      throw new Error('usage: build-adversarial-episodes.mjs --out DIR --executor-id ENDPOINT:MODEL [--chesst-source RULES.js --arena-root DIR]');
    options[key.slice(2)] = argv[++index];
  }
  if (!options.out || !options['executor-id'])
    throw new Error('usage: build-adversarial-episodes.mjs --out DIR --executor-id ENDPOINT:MODEL');
  // The collector checks frozen opponents against `${executorEndpoint}:${executorModel}` (collect-episodes.mjs); a bare
  // model id makes every arena case invalid at collection time.
  if (!/^https?:\/\/[^\s]+:[^\s/][^\s]*$/.test(options['executor-id']) || !/:\/\/[^/]+:\d+:/.test(options['executor-id']))
    throw new Error('--executor-id must be the collector executor id ENDPOINT:MODEL, e.g. http://127.0.0.1:8082:nvidia/Qwen3.6-35B-A3B-NVFP4');
  if (Boolean(options['chesst-source']) !== Boolean(options['arena-root']))
    throw new Error('--chesst-source and --arena-root go together (the collector loads ChessT from the arena root)');
  return { out: resolve(options.out), executorId: options['executor-id'], chesstSource: options['chesst-source'],
    arenaRoot: options['arena-root'] };
}

function caseId(gameId, seat, group) {
  return `arena_${fingerprint({ schema: 'natlang.adversarial-case/1', gameId, seat, group }).slice(0, 32)}`;
}

/** ``extra``: further games with their own scenarios, arena descriptor and decision cap (ChessT: pinned rules source
 * loaded from the collector's arena root). */
function createEpisodes(executorId, extra = []) {
  if (typeof executorId !== 'string' || !executorId.trim()) throw new Error('executor id must be non-empty');
  const casesByGame = new Map();
  for (const item of [...semanticScenarioCases(), ...wordScenarioCases(), ...extra.flatMap(row => row.scenarios)]) {
    const rows = casesByGame.get(item.family) ?? [];
    rows.push(item);
    casesByGame.set(item.family, rows);
  }
  const episodes = [];
  const descriptors = new Map(extra.map(row => [row.game.id, row]));
  for (const game of [...semanticGames, ...wordGames, ...extra.map(row => row.game)]) {
    const scenarios = casesByGame.get(game.id) ?? [];
    if (scenarios.length < 3 || new Set(scenarios.map(item => item.group)).size !== scenarios.length)
      throw new Error(`game ${game.id} needs at least three uniquely grouped scenarios`);
    const supportScenarios = scenarios.slice(0, 2);
    const queryScenarios = scenarios.slice(2);
    for (const seat of game.seats) {
      const id = `arena_${fingerprint({ schema: 'natlang.adversarial-episode/1', game: game.id, seat }).slice(0, 32)}`;
      const special = descriptors.get(game.id);
      const arena = { schema: SCHEMA, maxDecisions: special?.maxDecisions ?? 24,
        games: { [game.id]: special?.descriptor ?? { revision: game.revision } }, cases: {} };
      const makeCase = (item) => {
        const id = caseId(game.id, seat, item.group);
        const opponents = Object.fromEntries(game.seats.filter(other => other !== seat).map(other => [other,
          { files: { 'play.nl': GAME_POLICY_SOURCE }, entry: 'play.nl', model: executorId }]));
        const config = { game: game.id, scenario: structuredClone(item.scenario), seed: 1, seat, opponents };
        arena.cases[id] = config;
        return { id, group: item.group, args: [arenaTicket(id, config)], expected: CASE_EXPECTATION };
      };
      const support = supportScenarios.map(makeCase);
      const query = queryScenarios.map(makeCase);
      episodes.push({
        version: 'natlang.skill-episode/1', id, family: `adversarial-${game.id}-${seat}`, split: 'train',
        source_groups: scenarios.map(item => item.group), license: 'project-generated',
        target: { kind: 'improvement-case', entry: 'play.nl',
          files: { 'play.nl': GAME_POLICY_SOURCE },
          source: { schema: 'natlang.adversarial-game/1', id: `${game.id}:${seat}:${game.revision}` } },
        library: { kind: 'empty', skills: {} },
        support: { cases: support }, query: { cases: query },
        operations: ['create', 'revise'], limits: { maxSteps: 6 },
        provenance: { generator: 'natlang.adversarial-episodes/1', expected_semantics: 'host-scored-game-outcome', arena },
      });
    }
  }
  return episodes;
}

export { createEpisodes };

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    const { out, executorId, chesstSource, arenaRoot } = parseArgs(process.argv.slice(2));
    const extra = [];
    if (chesstSource) {
      const { chesstScenarios } = await import('./chesst-scenarios.mjs');
      const { game, scenarios } = await chesstScenarios(resolve(chesstSource), { count: 6, seed: 7, cap: 40 });
      const file = relative(resolve(arenaRoot), resolve(chesstSource));
      if (file.startsWith('..')) throw new Error('ChessT source must lie under the arena root');
      extra.push({ game, scenarios, maxDecisions: 40,
        descriptor: { revision: game.revision, source: { file, sha256: game.source.sha256 } } });
    }
    const episodes = createEpisodes(executorId, extra);
    mkdirSync(out, { recursive: true });
    const path = join(out, 'adversarial-episodes.jsonl');
    const fd = openSync(path, 'wx');
    try { writeFileSync(fd, episodes.map(row => JSON.stringify(row)).join('\n') + '\n'); }
    finally { closeSync(fd); }
    process.stdout.write(`${JSON.stringify({ path, episodes: episodes.length })}\n`);
  } catch (error) {
    process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  }
}
