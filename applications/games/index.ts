/**
 * Game worlds whose rules are natural language. A world is a value; a turn (games.nl) observes a snapshot of it,
 * decides in natural-language stages, and commits once, with crisp code checking what the stages produced
 * (conservation of money and goods, the seeded order, simultaneous combat, one-time observations). This file is the
 * host: the initial states, and a session that holds the current state and moves it only when a turn commits.
 */
import games from './games.nl';
import { totalsOf } from './games/economy/ledger.js';
import type { CombatState, EconomyState, Fighter, GameEvent, Merchant, NpcEvent, NpcState, Scene, Settings, TurnReport, World } from './types.js';

export type * from './types.js';

/** Every pluggable part in natural language. */
export const defaultSettings = { validate: 'nl', settle: 'nl', resolve: 'nl', remember: 'nl', narrate: 'nl' } as Settings;
/** Every pluggable part crisp: only the actors' own choices (choose, tactic, respond) are left to a model. */
export const crispSettings = { validate: 'crisp', settle: 'crisp', resolve: 'crisp', remember: 'crisp', narrate: 'crisp' } as Settings;

const copy = <T>(value: T): T => structuredClone(value);
const valid = (value: unknown): value is string => typeof value === 'string' && /^[A-Za-z][A-Za-z0-9_-]*$/.test(value);
const isCount = (value: unknown) => Number.isSafeInteger(value) && (value as number) >= 0;

/** A market. Merchants hold whole non-negative cash, goods and offers; the totals at the start are what every tick keeps. */
export function createEconomy(merchants: Merchant[], { seed = 0 } = {}): EconomyState {
  if (!Number.isSafeInteger(seed) || !Array.isArray(merchants) || merchants.length < 2) throw new Error('invalid economy');
  if (new Set(merchants.map(row => row.id)).size !== merchants.length) throw new Error('duplicate merchant');
  for (const row of merchants)
    if (!valid(row.id) || !isCount(row.cash) || !row.goods || !Object.values(row.goods).every(isCount) ||
        !row.offers || !Object.values(row.offers).every(isCount)) throw new Error('invalid merchant');
  return { kind: 'economy', tick: 0, seed, merchants: copy(merchants), initial: totalsOf(merchants) };
}

/** An arena of `width` cells. */
export function createArena(fighters: { id: string, x: number, hp: number }[], { width = 8 } = {}): CombatState {
  if (!Number.isSafeInteger(width) || width < 2) throw new Error('invalid arena');
  if (!Array.isArray(fighters) || fighters.length < 2 || new Set(fighters.map(row => row.id)).size !== fighters.length ||
      fighters.some(row => !valid(row.id) || !Number.isSafeInteger(row.x) || row.x < 0 || row.x >= width ||
        !Number.isSafeInteger(row.hp) || row.hp <= 0)) throw new Error('invalid fighters');
  return { kind: 'combat', round: 0, width, fighters: fighters.map(row => ({ ...copy(row), cooldown: 0 })) as Fighter[] };
}

/** A village of NPCs with an inventory each and empty memories. */
export function createVillage(actors: { id: string, inventory: Record<string, number> }[]): NpcState {
  if (!Array.isArray(actors) || new Set(actors.map(row => row.id)).size !== actors.length || actors.some(row => !valid(row.id)))
    throw new Error('invalid NPCs');
  return { kind: 'npc', version: 0, nextEvent: 1, applied: [], actors: actors.map(row => ({ ...copy(row), memory: [], commitments: [] })) };
}

/** Play one turn of a scene (a model call per stage that is natural language). */
export function playTurn(scene: Scene, settings: Settings = defaultSettings): Promise<TurnReport> {
  return games(scene, settings);
}

/**
 * The current state of one world. A turn is decided on the state as it is; the state moves only when the turn
 * commits, and a turn decided on a state another turn has since moved is refused (compare-and-set).
 */
export class Session<W extends World = World> {
  readonly events: GameEvent[] = [];
  constructor(public state: W, readonly settings: Settings = defaultSettings, private readonly play = playTurn) {}

  /** One turn. An NPC world needs who is addressed and what happened. */
  async turn(address?: { actor: string, event: NpcEvent }): Promise<TurnReport> {
    const before = this.state;
    const scene: Scene = before.kind === 'npc'
      ? (address ? { kind: 'npc', state: before, ...address } : (() => { throw new Error('an NPC turn needs an actor and an event'); })())
      : { kind: before.kind, state: before } as Scene;
    const report = await this.play(scene, this.settings);
    if (this.state !== before) throw new Error('stale turn: the state moved while the turn was decided');
    if (report.ok) { this.state = report.state as W; this.events.push(...report.events); }
    return report;
  }
}
