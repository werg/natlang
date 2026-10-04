import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { resolve } from 'node:path';
import { IllegalGameAction, type GameSpec } from './types.js';

type JsonRecord = Record<string, unknown>;
type ChesstRules = {
  PHASES: Record<string, string>;
  createState(input: unknown): JsonRecord;
  getObservation(state: JsonRecord, viewer: string): JsonRecord;
  getLegalActions(state: JsonRecord): unknown[];
  applyAction(state: JsonRecord, action: unknown): JsonRecord;
  getOutcome(state: JsonRecord): { winner: string | null; type: string; reason: string } | null;
  validateState(state: JsonRecord): string[];
};

function record(value: unknown, label: string): JsonRecord {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError(`${label} must be an object`);
  }
  return value as JsonRecord;
}

function stableJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(stableJson).join(',')}]`;
  if (value && typeof value === 'object') {
    const entries = Object.entries(value as JsonRecord).sort(([a], [b]) => a.localeCompare(b));
    return `{${entries.map(([key, item]) => `${JSON.stringify(key)}:${stableJson(item)}`).join(',')}}`;
  }
  return JSON.stringify(value);
}

function assertValidState(rules: ChesstRules, state: JsonRecord): JsonRecord {
  const normalized = rules.createState(state);
  const errors = rules.validateState(normalized);
  if (errors.length) throw new Error(`Invalid ChessT state: ${errors.join('; ')}`);
  return normalized;
}

function findExactAction(actions: unknown[], candidate: unknown): unknown | undefined {
  const key = stableJson(candidate);
  return actions.find((action) => stableJson(action) === key);
}

/**
 * Load the local ChessT rules kernel without copying it into Natlang.
 * The returned spec carries a digest of the exact rules source used.
 */
export async function createChesstGame(rulesPath: string): Promise<GameSpec & {
  source: { path: string; sha256: string };
}> {
  const absolutePath = resolve(rulesPath);
  const source = await readFile(absolutePath);
  const sourceSha256 = createHash('sha256').update(source).digest('hex');
  const require = createRequire(import.meta.url);
  const resolvedModule = require.resolve(absolutePath);
  delete require.cache[resolvedModule];
  const rules = require(absolutePath) as ChesstRules;
  const afterLoad = await readFile(absolutePath);
  const afterLoadSha256 = createHash('sha256').update(afterLoad).digest('hex');
  if (sourceSha256 !== afterLoadSha256) {
    throw new Error('ChessT rules source changed while loading; retry with a stable file');
  }
  if (!rules || typeof rules.getObservation !== 'function' ||
      typeof rules.getLegalActions !== 'function' ||
      typeof rules.applyAction !== 'function' || typeof rules.getOutcome !== 'function') {
    throw new TypeError('The supplied module is not a compatible ChessT rules kernel');
  }

  const seats = ['gold', 'blue'];
  const spec: GameSpec & { source: { path: string; sha256: string } } = {
    id: 'chesst',
    revision: sourceSha256,
    rules: 'ChessT canonical JavaScript rules; hidden items are visible only to their owner until revealed.',
    seats,
    source: { path: absolutePath, sha256: sourceSha256 },
    initialize(scenario: unknown, _seed: number): unknown {
      const supplied = scenario === undefined || scenario === null
        ? createChesstTacticFixture(rules.PHASES)
        : record(scenario, 'ChessT scenario');
      const initial = 'state' in supplied ? supplied.state : supplied;
      return assertValidState(rules, record(initial, 'ChessT scenario state'));
    },
    actor(state: unknown): string {
      const value = record(state, 'ChessT state').decisionPlayer;
      if (value !== 'gold' && value !== 'blue') throw new Error('ChessT state has no valid decisionPlayer');
      return value;
    },
    observe(state: unknown, seat: string): unknown {
      if (!seats.includes(seat)) throw new Error(`Unknown ChessT seat: ${seat}`);
      return rules.getObservation(record(state, 'ChessT state'), seat);
    },
    legalActions(state: unknown, seat: string): unknown[] {
      const observation = rules.getObservation(record(state, 'ChessT state'), seat);
      return rules.getLegalActions(observation);
    },
    apply(state: unknown, seat: string, action: unknown): unknown {
      const current = record(state, 'ChessT state');
      if (seat !== current.decisionPlayer) throw new IllegalGameAction('ChessT action submitted for the wrong decision seat');
      const publicLegal = rules.getLegalActions(rules.getObservation(current, seat));
      const exact = findExactAction(publicLegal, action);
      if (exact === undefined) throw new IllegalGameAction('Illegal ChessT action for the current public observation');
      return assertValidState(rules, rules.applyAction(current, exact));
    },
    outcome(state: unknown): { scores: Record<string, number>; reason: string } | null {
      const result = rules.getOutcome(record(state, 'ChessT state'));
      if (!result) return null;
      if (result.winner === null) return { scores: { gold: 0.5, blue: 0.5 }, reason: `${result.type}:${result.reason}` };
      if (!seats.includes(result.winner)) throw new Error(`Unknown ChessT winner: ${result.winner}`);
      const loser = result.winner === 'gold' ? 'blue' : 'gold';
      return { scores: { [result.winner]: 1, [loser]: 0 }, reason: `${result.type}:${result.reason}` };
    },
  };
  return spec;
}

/** Small deterministic legal positions for adapter and scenario-level smoke tests. */
export function createChesstTacticFixture(phases: Record<string, string> = {
  CHOOSE_ACTION: 'CHOOSE_ACTION',
}): JsonRecord {
  const board: (JsonRecord | null)[][] = Array.from({ length: 8 }, () => Array(8).fill(null));
  board[7]![4] = { type: 'king', color: 'gold', hasMoved: false };
  board[0]![4] = { type: 'king', color: 'blue', hasMoved: false };
  board[6]![3] = { type: 'pawn', color: 'gold', file: 'd', item: { id: 'swiftness', cost: 1, revealed: false } };
  board[1]![0] = { type: 'pawn', color: 'blue', file: 'a', item: { id: 'dagger', cost: 1, revealed: false } };
  return {
    board,
    activePlayer: 'gold',
    nominalPlayer: 'gold',
    decisionPlayer: 'gold',
    phase: phases.CHOOSE_ACTION ?? 'CHOOSE_ACTION',
    pending: null,
    captured: { gold: [], blue: [] },
    sorcererCooldown: { gold: 0, blue: 0 },
    tripleThreat: { active: false, player: null, turnsLeft: 0 },
  };
}
