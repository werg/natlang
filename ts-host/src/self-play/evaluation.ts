/** Host-owned repeated-game execution for the existing skill-authoring pipeline. */
import { resolve, relative, isAbsolute, sep } from 'node:path';
import { fingerprint } from '../adaptation/identity.js';
import type { SkillEpisode } from '../skills/episode.js';
import type { ModelDriver } from '../runtime/runtime.js';
import { sourceFiles, type SourceCaseExecution } from '../improvement/host.js';
import { playMatch, replayMatch } from './arena.js';
import { createChesstGame } from './chesst.js';
import { wordGames } from './word-games.js';
import { semanticGames } from './semantic-games.js';
import { natlangGamePolicy, type PolicySnapshot, type PolicyDecisionAudit } from './policy.js';
import type { GameMatch, GameSpec, GamePolicy } from './types.js';

export type ArenaCase = { game: string; scenario: unknown; seed: number; seat: string;
  opponents: Record<string, PolicySnapshot> };
export type ArenaProfile = { schema: 'natlang.adversarial-arena/1'; maxDecisions: number;
  games: Record<string, { revision: string; source?: { file: string; sha256: string } }>;
  cases: Record<string, ArenaCase> };
export const ARENA_CODE_FILES = ['self-play/types.js', 'self-play/arena.js', 'self-play/policy.js',
  'self-play/evaluation.js', 'self-play/chesst.js', 'self-play/semantic-games.js', 'self-play/word-games.js'] as const;

export function arenaTicket(caseId: string, row: ArenaCase): string {
  return JSON.stringify({ schema: 'natlang.arena-case-ticket/1', case: caseId, game: row.game, seat: row.seat });
}

export async function arenaEpisodeExecutions(episode: SkillEpisode, executor: ModelDriver,
    options: { pins: Record<string, string>; executorId: string; arenaRoot?: string; signal?: AbortSignal;
      onMatch?: (event: { caseId: string; source: string; match: GameMatch;
        audits: Record<string, PolicyDecisionAudit[]> }) => void | Promise<void> }) {
  const supplied = episode.provenance.arena;
  if (!supplied) return {};
  const profile = structuredClone(supplied) as ArenaProfile;
  if (profile.schema !== 'natlang.adversarial-arena/1' || !Number.isSafeInteger(profile.maxDecisions) ||
      profile.maxDecisions < 1 || !profile.games || !profile.cases) throw Error('invalid adversarial arena profile');
  const pins = Object.fromEntries(ARENA_CODE_FILES.map(file => {
    if (!options.pins[file]) throw Error('arena runtime code pin missing: ' + file);
    return [file, options.pins[file]];
  }));
  const games = new Map<string, GameSpec>();
  for (const [id, descriptor] of Object.entries(profile.games)) {
    let game = [...semanticGames, ...wordGames].find(game => game.id === id);
    if (id === 'chesst') {
      if (!descriptor.source || !options.arenaRoot) throw Error('Chesst requires a pinned source and arena root');
      const root = resolve(options.arenaRoot), path = resolve(root, descriptor.source.file), rel = relative(root, path);
      if (!rel || rel === '..' || rel.startsWith('..' + sep) || isAbsolute(rel)) throw Error('arena source escapes root');
      const chesst = await createChesstGame(path);
      if (chesst.source.sha256 !== descriptor.source.sha256) throw Error('Chesst source pin differs');
      game = chesst;
    }
    if (!game || game.revision !== descriptor.revision) throw Error('arena game revision differs: ' + id);
    games.set(id, game);
  }
  const allCases = [...episode.support.cases, ...episode.query.cases, ...(episode.transfer?.cases ?? [])];
  if (new Set(allCases.map(row => row.id)).size !== allCases.length ||
      Object.keys(profile.cases).sort().join('\0') !== allCases.map(row => row.id).sort().join('\0'))
    throw Error('arena case table must close over exactly the episode cases');
  for (const row of allCases) {
    const config = profile.cases[row.id]!, game = games.get(config.game);
    if (!game || !game.seats.includes(config.seat) || !Number.isSafeInteger(config.seed) ||
        row.args?.length !== 1 || row.args[0] !== arenaTicket(row.id, config) ||
        Object.keys(config.opponents).sort().join('\0') !== game.seats.filter(seat => seat !== config.seat).sort().join('\0') ||
        Object.values(config.opponents).some(policy => policy.model !== options.executorId))
      throw Error('invalid arena case, ticket or frozen opponent roster: ' + row.id);
    // Validate fixtures before spending model calls or exposing training evidence.
    game.initialize(config.scenario, config.seed);
  }
  const identity = fingerprint({ profile, pins, executor: options.executorId });
  const makeExecution = (entry: string): SourceCaseExecution => Object.assign(async (...[folder, row, _seed, gateway]: Parameters<SourceCaseExecution>) => {
    const config = profile.cases[row.id];
    if (!config || row.args[0] !== arenaTicket(row.id, config)) throw Error('arena case is outside frozen profile');
    const game = games.get(config.game)!, audits: Record<string, PolicyDecisionAudit[]> = {};
    const before = gateway.ledger.usage.modelCalls, policies: Record<string, GamePolicy> = {};
    for (const seat of game.seats) {
      audits[seat] = [];
      policies[seat] = natlangGamePolicy({ snapshot: seat === config.seat
        ? { files: sourceFiles(folder), entry, model: options.executorId } : config.opponents[seat]!,
        driver: (request, signal, turnOptions) => gateway.request(executor, request, signal ?? options.signal, 'executor', turnOptions),
        seed: config.seed, onDecision: audit => { audits[seat]!.push(audit); } });
    }
    const match = await playMatch({ game, scenario: config.scenario, seed: config.seed, policies,
      maxDecisions: profile.maxDecisions, signal: options.signal });
    await options.onMatch?.({ caseId: row.id, source: folder.digest, match, audits });
    if (match.disposition === 'incomplete' || match.disposition === 'illegal-action' && match.rejected?.seat !== config.seat)
      throw Error('arena execution incomplete; not a scored candidate loss: ' + (match.error ?? match.disposition));
    if (match.disposition === 'completed') await replayMatch(game, match);
    const quality = match.outcome?.scores[config.seat] ?? 0;
    const candidateAudits = audits[config.seat]!;
    const skillUseTrace = candidateAudits.flatMap(audit => audit.traces.flatMap(trace =>
      trace.events.filter(event => event.kind === 'skill_use')));
    return { value: { schema: 'natlang.arena-case-result/1', seat: config.seat, quality,
      reason: match.outcome?.reason ?? match.error,
      // The improver receives only the candidate's information sets, never hidden opponent state.
      decisions: candidateAudits.map(audit => ({ view: audit.view, action: audit.action, error: audit.error })) },
      modelCalls: gateway.ledger.usage.modelCalls - before,
      skillUseTrace: skillUseTrace as import('../skills/observability.js').SkillUseEvent[],
      score: { quality, gates: { completed: match.disposition === 'completed', legal: match.disposition === 'completed' } } };
  }, { identity: identity + ':' + entry, evaluationLevel: 1 as const });
  return { executeCase: makeExecution(episode.target.entry),
    ...(episode.transfer ? { transferExecuteCase: makeExecution(episode.transfer.target.entry) } : {}),
    executionDescriptor: { schema: profile.schema, identity, pins } };
}
