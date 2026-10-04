import { fingerprint } from '../adaptation/identity.js';
import { IllegalGameAction, type GameSpec, type GamePolicy, type GameMatch, type GameFrame, type GameOutcome } from './types.js';

function checkOutcome(game: GameSpec, outcome: GameOutcome): void {
  if (!outcome || typeof outcome.reason !== 'string' || !outcome.reason ||
      Object.keys(outcome.scores).sort().join('\0') !== [...game.seats].sort().join('\0') ||
      Object.values(outcome.scores).some(value => !Number.isFinite(value) || value < 0 || value > 1))
    throw Error('game outcome must include one bounded score per seat and a reason');
}

/** Resource exhaustion never becomes a draw or a scored defeat. Only game rules award outcomes. */
export async function playMatch(options: { game: GameSpec; scenario: unknown; seed: number;
  policies: Record<string, GamePolicy>; maxDecisions: number; signal?: AbortSignal;
  onFrame?: (frame: GameFrame) => void | Promise<void> }): Promise<GameMatch> {
  const { game, policies } = options;
  if (!Number.isSafeInteger(options.seed) || !Number.isSafeInteger(options.maxDecisions) || options.maxDecisions < 1 ||
      !game.seats.length || new Set(game.seats).size !== game.seats.length ||
      game.seats.some(seat => !policies[seat]?.id || typeof policies[seat].decide !== 'function'))
    throw Error('match requires an integer seed, finite decision allocation and one identified policy per seat');
  const identity = fingerprint({ version: 'natlang.adversarial-match/1', game: game.id, revision: game.revision,
    scenario: options.scenario, seed: options.seed, allocation: options.maxDecisions,
    policies: Object.fromEntries(game.seats.map(seat => [seat, policies[seat]!.id])) });
  let state = game.initialize(structuredClone(options.scenario), options.seed);
  const initial = structuredClone(state), frames: GameFrame[] = [];
  const finish = (disposition: GameMatch['disposition'], outcome: GameOutcome | null, error?: string): GameMatch => ({
    schema: 'natlang.adversarial-match/1', game: game.id, revision: game.revision, identity,
    policies: Object.fromEntries(game.seats.map(seat => [seat, policies[seat]!.id])), maxDecisions: options.maxDecisions,
    seed: options.seed, scenario: structuredClone(options.scenario), initial, frames,
    disposition, outcome, ...(error ? { error } : {}), final: structuredClone(state),
  });
  try {
    for (let index = 0; index <= options.maxDecisions; index++) {
      if (options.signal?.aborted) throw options.signal.reason ?? Error('match cancelled');
      const outcome = game.outcome(state);
      if (outcome) { checkOutcome(game, outcome); return finish('completed', outcome); }
      if (index === options.maxDecisions) return finish('incomplete', null, 'decision allocation exhausted');
      const seat = game.actor(state);
      if (!game.seats.includes(seat)) throw Error('game selected an unknown actor');
      const before = fingerprint(state), policy = policies[seat]!;
      const view = structuredClone({ game: game.id, rules: game.rules, seat, decision: index,
        observation: game.observe(state, seat),
        ...(game.legalActions ? { legalActions: game.legalActions(state, seat) } : {}) });
      if (before !== fingerprint(state)) throw Error('observation mutated authoritative game state');
      // JSON is the policy boundary and persisted replay format; reject undefined or non-finite engine values early.
      fingerprint(view);
      const savedView = structuredClone(view);
      const action = await policy.decide(view, options.signal);
      let next;
      try { next = game.apply(state, seat, structuredClone(action)); }
      catch (error) {
        if (!(error instanceof IllegalGameAction)) throw error;
        // Preserve the rejected move, but do not invent rewards for allies or opponents.
        return { ...finish('illegal-action', null, error.message),
          rejected: { seat, view: savedView, action: structuredClone(action), reason: error.message } };
      }
      if (before !== fingerprint(state)) throw Error('transition mutated its input state');
      state = next;
      const frame: GameFrame = { index, seat, policy: policy.id, view: savedView, action: structuredClone(action),
        before, after: fingerprint(state) };
      frames.push(frame); await options.onFrame?.(structuredClone(frame));
    }
    throw Error('unreachable match state');
  } catch (error) { return finish('incomplete', null, String(error)); }
}

/** Replay without model calls, checking every private view, transition and final reward. */
export async function replayMatch(game: GameSpec, match: GameMatch): Promise<GameMatch> {
  if (game.id !== match.game || game.revision !== match.revision) throw Error('replay game revision differs');
  let at = 0;
  const policies = Object.fromEntries(game.seats.map(seat => [seat, { id: match.policies[seat]!,
    decide: async (view: unknown) => {
      const frame = match.frames[at++];
      if (!frame || fingerprint(view) !== fingerprint(frame.view)) throw Error('replay private observation differs');
      return structuredClone(frame.action);
    } }]));
  if (match.disposition !== 'completed') throw Error('only complete legal matches can establish replay admission');
  const replayed = await playMatch({ game, scenario: match.scenario, seed: match.seed, policies,
    maxDecisions: match.maxDecisions });
  if (replayed.disposition !== 'completed' || at !== match.frames.length ||
      fingerprint(replayed.initial) !== fingerprint(match.initial) ||
      fingerprint(replayed.final) !== fingerprint(match.final) ||
      replayed.identity !== match.identity ||
      fingerprint(replayed.outcome) !== fingerprint(match.outcome) ||
      fingerprint(replayed.frames) !== fingerprint(match.frames)) throw Error('game match did not replay exactly');
  return replayed;
}
