/** Exact headless worlds; natlang owns actor policy at decision boundaries. */
import { createHash } from 'node:crypto';

const hash = value => createHash('sha256').update(String(value)).digest('hex');
const copy = value => structuredClone(value);
const valid = value => typeof value === 'string' && /^[A-Za-z][A-Za-z0-9_-]*$/.test(value);

export class EconomyWorld {
  constructor(merchants, { seed = 0 } = {}) {
    if (!Number.isSafeInteger(seed) || !Array.isArray(merchants) || merchants.length < 2)
      throw new Error('invalid economy');
    this.seed = seed; this.tick = 0; this.pending = new Map(); this.events = [];
    this.merchants = new Map(merchants.map(row => [row.id, copy(row)]));
    if (this.merchants.size !== merchants.length) throw new Error('duplicate merchant');
    for (const row of this.merchants.values()) {
      if (!valid(row.id) || !Number.isSafeInteger(row.cash) || row.cash < 0 ||
          !row.goods || Object.values(row.goods).some(value =>
            !Number.isSafeInteger(value) || value < 0) ||
          !row.offers || Object.values(row.offers).some(value =>
            !Number.isSafeInteger(value) || value < 0)) throw new Error('invalid merchant');
    }
    this.initial = this.#totals();
  }

  #totals() {
    const goods = {};
    for (const row of this.merchants.values())
      for (const [name, quantity] of Object.entries(row.goods))
        goods[name] = (goods[name] ?? 0) + quantity;
    return { cash: [...this.merchants.values()].reduce((sum, row) => sum + row.cash, 0),
      goods: Object.fromEntries(Object.entries(goods).sort(([a], [b]) => a.localeCompare(b))) };
  }

  observe(actor) {
    const mine = this.merchants.get(actor);
    if (!mine) throw new Error('unknown actor');
    return { actor, tick: this.tick, cash: mine.cash, goods: copy(mine.goods),
      offers: [...this.merchants.values()].filter(row => row.id !== actor)
        .flatMap(row => Object.entries(row.offers).map(([good, price]) => ({
          seller: row.id, good, price }))) };
  }

  submit(actor, tick, intent) {
    if (tick !== this.tick || !this.merchants.has(actor) || this.pending.has(actor))
      throw new Error('stale or duplicate intent');
    if (!intent || intent.kind !== 'buy' && intent.kind !== 'pass' ||
        intent.kind === 'buy' && (!valid(intent.seller) || !valid(intent.good) ||
          !Number.isSafeInteger(intent.quantity) || intent.quantity < 1))
      throw new Error('invalid trade intent');
    this.pending.set(actor, copy(intent));
    this.events.push({ operation: 'economy.intent', tick, actor, intent: copy(intent) });
    return { status: 'accepted', tick, actor };
  }

  settle() {
    const order = [...this.pending.keys()].sort((a, b) =>
      hash(`${this.seed}:${this.tick}:${a}`).localeCompare(hash(`${this.seed}:${this.tick}:${b}`)));
    const outcomes = [];
    for (const actor of order) {
      const intent = this.pending.get(actor), buyer = this.merchants.get(actor);
      if (intent.kind === 'pass') { outcomes.push({ actor, status: 'pass' }); continue; }
      const seller = this.merchants.get(intent.seller);
      const price = seller?.offers[intent.good], quantity = intent.quantity;
      if (!seller || seller.id === actor || !Number.isSafeInteger(price) ||
          !Number.isSafeInteger(quantity * price) ||
          (seller.goods[intent.good] ?? 0) < quantity || buyer.cash < quantity * price) {
        outcomes.push({ actor, status: 'rejected' }); continue;
      }
      seller.goods[intent.good] -= quantity; buyer.goods[intent.good] =
        (buyer.goods[intent.good] ?? 0) + quantity;
      buyer.cash -= quantity * price; seller.cash += quantity * price;
      outcomes.push({ actor, status: 'traded', seller: seller.id,
        good: intent.good, quantity, total: quantity * price });
    }
    if (JSON.stringify(this.#totals()) !== JSON.stringify(this.initial))
      throw new Error('economy conservation failure');
    this.events.push({ operation: 'economy.settle', tick: this.tick, outcomes: copy(outcomes) });
    this.pending.clear(); this.tick++;
    return { tick: this.tick, outcomes, merchants: copy([...this.merchants.values()]) };
  }

  drainEvents() { return this.events.splice(0); }
}

export class CombatWorld {
  constructor(fighters, { width = 8 } = {}) {
    if (!Number.isSafeInteger(width) || width < 2) throw new Error('invalid arena');
    this.width = width; this.round = 0; this.pending = new Map(); this.events = [];
    this.fighters = new Map(fighters.map(row => [row.id, { ...copy(row), cooldown: 0 }]));
    if (this.fighters.size !== fighters.length || fighters.length < 2 ||
        fighters.some(row => !valid(row.id) || !Number.isSafeInteger(row.x) ||
          row.x < 0 || row.x >= width || !Number.isSafeInteger(row.hp) || row.hp <= 0))
      throw new Error('invalid fighters');
  }

  observe(actor) {
    const self = this.fighters.get(actor);
    if (!self) throw new Error('unknown fighter');
    return { actor, round: this.round, width: this.width, self: copy(self),
      others: [...this.fighters.values()].filter(row => row.id !== actor)
        .map(row => ({ id: row.id, x: row.x, hp: row.hp })) };
  }

  submit(actor, round, plan) {
    if (round !== this.round || this.pending.has(actor) ||
        !this.fighters.has(actor) || !plan ||
        !['left', 'stay', 'right'].includes(plan.move) ||
        !['attack', 'guard', 'rest'].includes(plan.action) ||
        plan.action === 'attack' && !this.fighters.has(plan.target))
      throw new Error('stale or illegal combat plan');
    this.pending.set(actor, copy(plan));
    return { status: 'accepted', actor, round };
  }

  resolve() {
    const before = new Map([...this.fighters].map(([id, row]) => [id, copy(row)]));
    for (const [id, plan] of this.pending) {
      const fighter = this.fighters.get(id);
      if (fighter.hp <= 0) continue;
      fighter.x = Math.max(0, Math.min(this.width - 1,
        fighter.x + (plan.move === 'left' ? -1 : plan.move === 'right' ? 1 : 0)));
    }
    const damage = new Map();
    for (const [id, plan] of this.pending) {
      const attacker = this.fighters.get(id), target = this.fighters.get(plan.target);
      if (plan.action !== 'attack' || attacker.hp <= 0 || attacker.cooldown > 0 ||
          !target || target.hp <= 0 || Math.abs(attacker.x - target.x) > 1) continue;
      damage.set(target.id, (damage.get(target.id) ?? 0) +
        (this.pending.get(target.id)?.action === 'guard' ? 1 : 2));
      attacker.cooldown = 1;
    }
    for (const [id, amount] of damage) this.fighters.get(id).hp =
      Math.max(0, this.fighters.get(id).hp - amount);
    for (const [id, fighter] of this.fighters)
      if (before.get(id).cooldown > 0) fighter.cooldown = Math.max(0, fighter.cooldown - 1);
    const report = { round: ++this.round, fighters: copy([...this.fighters.values()]),
      damage: Object.fromEntries(damage) };
    this.events.push({ operation: 'combat.resolve', ...copy(report) });
    this.pending.clear();
    return report;
  }

  drainEvents() { return this.events.splice(0); }
}

export class NpcWorld {
  constructor(actors) {
    this.actors = new Map(actors.map(row => [row.id, { ...copy(row), memory: [], commitments: [] }]));
    if (this.actors.size !== actors.length || actors.some(row => !valid(row.id)))
      throw new Error('invalid NPCs');
    this.events = []; this.nextEvent = 1; this.observations = new Map();
  }

  observe(actor, event) {
    const npc = this.actors.get(actor);
    if (!npc || !event?.text || !event?.from) throw new Error('invalid NPC event');
    const eventId = `event-${this.nextEvent++}`;
    npc.memory.push({ id: eventId, from: event.from, text: event.text });
    const observation = { actor, event_id: eventId, event: copy(event), inventory: copy(npc.inventory),
      memory: copy(npc.memory), commitments: copy(npc.commitments) };
    this.observations.set(eventId, { identity: hash(JSON.stringify(observation)), used: false });
    return observation;
  }

  apply(observation, plan) {
    const npc = this.actors.get(observation.actor);
    const issued = this.observations.get(observation.event_id);
    if (!npc || !issued || issued.used ||
        issued.identity !== hash(JSON.stringify(observation)) ||
        !npc.memory.some(row => row.id === observation.event_id) ||
        !plan || typeof plan.say !== 'string' ||
        !['none', 'give', 'promise'].includes(plan.action))
      throw new Error('invalid NPC plan');
    if (plan.action === 'give') {
      if (!valid(plan.item) || !valid(plan.target) ||
          (npc.inventory[plan.item] ?? 0) < 1) throw new Error('unavailable item');
      npc.inventory[plan.item]--;
    }
    if (plan.action === 'promise') {
      if (!valid(plan.target) || !plan.detail) throw new Error('invalid promise');
      npc.commitments.push({ to: plan.target, detail: plan.detail,
        evidence_id: observation.event_id });
    }
    issued.used = true;
    const result = { actor: npc.id, said: plan.say, action: plan.action,
      evidence_id: observation.event_id, inventory: copy(npc.inventory),
      commitments: copy(npc.commitments) };
    this.events.push({ operation: 'npc.act', ...copy(result) });
    return result;
  }

  drainEvents() { return this.events.splice(0); }
}
