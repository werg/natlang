/**
 * Headless game worlds with exact rules; natlang decides actor policy at decision boundaries. The
 * economy settles intents in a seeded order and asserts conservation of money and goods; combat
 * resolves simultaneous movement and attacks; NPCs act from their own memory, and each observation
 * can be applied exactly once.
 */
import { createHash } from 'node:crypto';
import { nl } from '@natlang/node';

export type Merchant = { id: string, cash: number, goods: Record<string, number>, offers: Record<string, number> };
export type Offer = { seller: string, good: string, price: number };
export type MarketObservation = { actor: string, tick: number, cash: number, goods: Record<string, number>, offers: Offer[] };
export type TradeIntent = { kind: 'buy' | 'pass', seller?: string, good?: string, quantity?: number };
export type TradeReceipt = { status: 'accepted', tick: number, actor: string };
export type TradeOutcome = { actor: string, status: 'pass' | 'rejected' | 'traded', seller?: string, good?: string, quantity?: number, total?: number };

export type Fighter = { id: string, x: number, hp: number, cooldown: number };
export type Opponent = { id: string, x: number, hp: number };
export type CombatObservation = { actor: string, round: number, width: number, self: Fighter, others: Opponent[] };
export type CombatPlan = { move: 'left' | 'stay' | 'right', action: 'attack' | 'guard' | 'rest', target?: string };
export type CombatReceipt = { status: 'accepted', actor: string, round: number };

export type NpcEvent = { from: string, text: string };
export type Memory = { id: string, from: string, text: string };
export type Commitment = { to: string, detail: string, evidence_id: string };
export type NpcObservation = { actor: string, event_id: string, event: NpcEvent, inventory: Record<string, number>,
  memory: Memory[], commitments: Commitment[] };
export type NpcPlan = { say: string, action: 'none' | 'give' | 'promise', item?: string, target?: string, detail?: string };
export type NpcResult = { actor: string, said: string, action: string, evidence_id: string, inventory: Record<string, number>,
  commitments: Commitment[] };

const hash = (value: unknown) => createHash('sha256').update(String(value)).digest('hex');
const copy = <T>(value: T): T => structuredClone(value);
const valid = (value: unknown): value is string => typeof value === 'string' && /^[A-Za-z][A-Za-z0-9_-]*$/.test(value);
const isCount = (value: unknown) => Number.isSafeInteger(value) && (value as number) >= 0;

export class EconomyWorld {
  tick = 0;
  private readonly seed: number;
  private readonly pending = new Map<string, TradeIntent>();
  private readonly events: Record<string, unknown>[] = [];
  private readonly merchants: Map<string, Merchant>;
  private readonly initial: ReturnType<EconomyWorld['totals']>;

  constructor(merchants: Merchant[], { seed = 0 } = {}) {
    if (!Number.isSafeInteger(seed) || !Array.isArray(merchants) || merchants.length < 2) throw new Error('invalid economy');
    this.seed = seed;
    this.merchants = new Map(merchants.map(row => [row.id, copy(row)]));
    if (this.merchants.size !== merchants.length) throw new Error('duplicate merchant');
    for (const row of this.merchants.values())
      if (!valid(row.id) || !isCount(row.cash) || !row.goods || !Object.values(row.goods).every(isCount) ||
          !row.offers || !Object.values(row.offers).every(isCount)) throw new Error('invalid merchant');
    this.initial = this.totals();
  }

  private totals(): { cash: number, goods: Record<string, number> } {
    const goods: Record<string, number> = {};
    for (const row of this.merchants.values())
      for (const [name, quantity] of Object.entries(row.goods)) goods[name] = (goods[name] ?? 0) + quantity;
    return { cash: [...this.merchants.values()].reduce((sum, row) => sum + row.cash, 0),
      goods: Object.fromEntries(Object.entries(goods).sort(([a], [b]) => a.localeCompare(b))) };
  }

  observe(actor: string): MarketObservation {
    const mine = this.merchants.get(actor);
    if (!mine) throw new Error('unknown actor');
    return { actor, tick: this.tick, cash: mine.cash, goods: copy(mine.goods),
      offers: [...this.merchants.values()].filter(row => row.id !== actor)
        .flatMap(row => Object.entries(row.offers).map(([good, price]) => ({ seller: row.id, good, price }))) };
  }

  submit(actor: string, tick: number, intent: TradeIntent): TradeReceipt {
    if (tick !== this.tick || !this.merchants.has(actor) || this.pending.has(actor)) throw new Error('stale or duplicate intent');
    if (!intent || intent.kind !== 'buy' && intent.kind !== 'pass' || intent.kind === 'buy' &&
        (!valid(intent.seller) || !valid(intent.good) || !Number.isSafeInteger(intent.quantity) || intent.quantity! < 1))
      throw new Error('invalid trade intent');
    this.pending.set(actor, copy(intent));
    this.events.push({ operation: 'economy.intent', tick, actor, intent: copy(intent) });
    return { status: 'accepted', tick, actor };
  }

  settle(): { tick: number, outcomes: TradeOutcome[], merchants: Merchant[] } {
    const order = [...this.pending.keys()].sort((a, b) =>
      hash(`${this.seed}:${this.tick}:${a}`).localeCompare(hash(`${this.seed}:${this.tick}:${b}`)));
    const outcomes: TradeOutcome[] = [];
    for (const actor of order) {
      const intent = this.pending.get(actor)!, buyer = this.merchants.get(actor)!;
      if (intent.kind === 'pass') { outcomes.push({ actor, status: 'pass' }); continue; }
      const good = intent.good!, quantity = intent.quantity!, seller = this.merchants.get(intent.seller!);
      const price = seller?.offers[good];
      if (!seller || seller.id === actor || !Number.isSafeInteger(price) || !Number.isSafeInteger(quantity * price!) ||
          (seller.goods[good] ?? 0) < quantity || buyer.cash < quantity * price!) {
        outcomes.push({ actor, status: 'rejected' }); continue;
      }
      seller.goods[good]! -= quantity;
      buyer.goods[good] = (buyer.goods[good] ?? 0) + quantity;
      buyer.cash -= quantity * price!; seller.cash += quantity * price!;
      outcomes.push({ actor, status: 'traded', seller: seller.id, good, quantity, total: quantity * price! });
    }
    if (JSON.stringify(this.totals()) !== JSON.stringify(this.initial)) throw new Error('economy conservation failure');
    this.events.push({ operation: 'economy.settle', tick: this.tick, outcomes: copy(outcomes) });
    this.pending.clear(); this.tick++;
    return { tick: this.tick, outcomes, merchants: copy([...this.merchants.values()]) };
  }

  drainEvents(): Record<string, unknown>[] { return this.events.splice(0); }
}

export class CombatWorld {
  round = 0;
  private readonly width: number;
  private readonly pending = new Map<string, CombatPlan>();
  private readonly events: Record<string, unknown>[] = [];
  private readonly fighters: Map<string, Fighter>;

  constructor(fighters: Omit<Fighter, 'cooldown'>[], { width = 8 } = {}) {
    if (!Number.isSafeInteger(width) || width < 2) throw new Error('invalid arena');
    this.width = width;
    this.fighters = new Map(fighters.map(row => [row.id, { ...copy(row), cooldown: 0 }]));
    if (this.fighters.size !== fighters.length || fighters.length < 2 || fighters.some(row => !valid(row.id) ||
        !Number.isSafeInteger(row.x) || row.x < 0 || row.x >= width || !Number.isSafeInteger(row.hp) || row.hp <= 0))
      throw new Error('invalid fighters');
  }

  observe(actor: string): CombatObservation {
    const self = this.fighters.get(actor);
    if (!self) throw new Error('unknown fighter');
    return { actor, round: this.round, width: this.width, self: copy(self),
      others: [...this.fighters.values()].filter(row => row.id !== actor).map(row => ({ id: row.id, x: row.x, hp: row.hp })) };
  }

  submit(actor: string, round: number, plan: CombatPlan): CombatReceipt {
    if (round !== this.round || this.pending.has(actor) || !this.fighters.has(actor) || !plan ||
        !['left', 'stay', 'right'].includes(plan.move) || !['attack', 'guard', 'rest'].includes(plan.action) ||
        plan.action === 'attack' && !this.fighters.has(plan.target!))
      throw new Error('stale or illegal combat plan');
    this.pending.set(actor, copy(plan));
    return { status: 'accepted', actor, round };
  }

  resolve(): { round: number, fighters: Fighter[], damage: Record<string, number> } {
    const before = new Map([...this.fighters].map(([id, row]) => [id, copy(row)]));
    for (const [id, plan] of this.pending) {
      const fighter = this.fighters.get(id)!;
      if (fighter.hp <= 0) continue;
      fighter.x = Math.max(0, Math.min(this.width - 1, fighter.x + (plan.move === 'left' ? -1 : plan.move === 'right' ? 1 : 0)));
    }
    const damage = new Map<string, number>();
    for (const [id, plan] of this.pending) {
      const attacker = this.fighters.get(id)!, target = plan.target ? this.fighters.get(plan.target) : undefined;
      if (plan.action !== 'attack' || attacker.hp <= 0 || attacker.cooldown > 0 || !target || target.hp <= 0 ||
          Math.abs(attacker.x - target.x) > 1) continue;
      damage.set(target.id, (damage.get(target.id) ?? 0) + (this.pending.get(target.id)?.action === 'guard' ? 1 : 2));
      attacker.cooldown = 1;
    }
    for (const [id, amount] of damage) this.fighters.get(id)!.hp = Math.max(0, this.fighters.get(id)!.hp - amount);
    for (const [id, fighter] of this.fighters) if (before.get(id)!.cooldown > 0) fighter.cooldown = Math.max(0, fighter.cooldown - 1);
    const report = { round: ++this.round, fighters: copy([...this.fighters.values()]), damage: Object.fromEntries(damage) };
    this.events.push({ operation: 'combat.resolve', ...copy(report) });
    this.pending.clear();
    return report;
  }

  drainEvents(): Record<string, unknown>[] { return this.events.splice(0); }
}

type Npc = { id: string, inventory: Record<string, number>, memory: Memory[], commitments: Commitment[] };

export class NpcWorld {
  private readonly actors: Map<string, Npc>;
  private readonly events: Record<string, unknown>[] = [];
  private nextEvent = 1;
  private readonly observations = new Map<string, { identity: string, used: boolean }>();

  constructor(actors: { id: string, inventory: Record<string, number> }[]) {
    this.actors = new Map(actors.map(row => [row.id, { ...copy(row), memory: [], commitments: [] }]));
    if (this.actors.size !== actors.length || actors.some(row => !valid(row.id))) throw new Error('invalid NPCs');
  }

  observe(actor: string, event: NpcEvent): NpcObservation {
    const npc = this.actors.get(actor);
    if (!npc || !event?.text || !event?.from) throw new Error('invalid NPC event');
    const eventId = `event-${this.nextEvent++}`;
    npc.memory.push({ id: eventId, from: event.from, text: event.text });
    const observation = { actor, event_id: eventId, event: copy(event), inventory: copy(npc.inventory),
      memory: copy(npc.memory), commitments: copy(npc.commitments) };
    this.observations.set(eventId, { identity: hash(JSON.stringify(observation)), used: false });
    return observation;
  }

  apply(observation: NpcObservation, plan: NpcPlan): NpcResult {
    const npc = this.actors.get(observation.actor);
    const issued = this.observations.get(observation.event_id);
    if (!npc || !issued || issued.used || issued.identity !== hash(JSON.stringify(observation)) ||
        !npc.memory.some(row => row.id === observation.event_id) || !plan || typeof plan.say !== 'string' ||
        !['none', 'give', 'promise'].includes(plan.action))
      throw new Error('invalid NPC plan');
    if (plan.action === 'give') {
      if (!valid(plan.item) || !valid(plan.target) || (npc.inventory[plan.item] ?? 0) < 1) throw new Error('unavailable item');
      npc.inventory[plan.item]!--;
    }
    if (plan.action === 'promise') {
      if (!valid(plan.target) || !plan.detail) throw new Error('invalid promise');
      npc.commitments.push({ to: plan.target, detail: plan.detail, evidence_id: observation.event_id });
    }
    issued.used = true;
    const result = { actor: npc.id, said: plan.say, action: plan.action, evidence_id: observation.event_id,
      inventory: copy(npc.inventory), commitments: copy(npc.commitments) };
    this.events.push({ operation: 'npc.act', ...copy(result) });
    return result;
  }

  drainEvents(): Record<string, unknown>[] { return this.events.splice(0); }
}

/** One merchant's turn: observe privately, decide in natural language, submit the intent. */
export async function tradeTurn(world: EconomyWorld, actor: string): Promise<TradeReceipt> {
  const observation = world.observe(actor);
  const intent: TradeIntent = await nl`Choose buy or pass for this merchant from its observation: only its own cash and goods
plus the public offers. A buy names an offered seller and good and a positive integer quantity. Weigh price against likely
future use; do not assume other merchants' private inventories. Settlement rejects unavailable stock or insufficient cash.`(observation);
  return world.submit(actor, observation.tick, intent);
}

/** One fighter's tactic for the current round. */
export async function combatTurn(world: CombatWorld, actor: string): Promise<CombatReceipt> {
  const perception = world.observe(actor);
  const plan: CombatPlan = await nl`Choose a move (left, stay or right) and an action (attack, guard or rest) for the fighter
in perception; an attack targets a visible opponent. Movement and attacks resolve together after all fighters submit.
Consider health, distance and cooldown; the simulation decides collisions, damage and death.`(perception);
  return world.submit(actor, perception.round, plan);
}

/** An NPC responds in character to an event and takes one exact world action. */
export async function npcReact(world: NpcWorld, actor: string, event: NpcEvent): Promise<NpcResult> {
  const observation = world.observe(actor, event);
  const plan: NpcPlan = await nl`Respond in character to the event in observation, using only its memory and commitments.
Choose none, give or promise as the world action. Give only an available item; a promise names its recipient and a concrete
detail. Do not claim another NPC's private knowledge.`(observation);
  return world.apply(observation, plan);
}
