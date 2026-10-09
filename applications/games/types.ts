/** Which implementation runs a pluggable part: its natural-language function, or its crisp reference. */
export type Engine = 'nl' | 'crisp' | 'shadow';

/**
 * Which implementation each pluggable part uses. The parts that run for every actor of every turn have two
 * implementations behind one interface: validate (intent, plan and tactic legality), settle (the economy's trades),
 * resolve (the combat round), remember (an NPC's notes) and narrate (the turn's story).
 */
export type Settings = Is<{ validate: Engine, settle: Engine, resolve: Engine, remember: Engine, narrate: Engine },
  "settings in which validate, settle and resolve are each nl, crisp or shadow, and remember and narrate are each nl or crisp">;

// ---------------------------------------------------------------- refined slots
// The slots a model writes carry a predicate; the state a host builds and the commit arithmetic keep plain types (their
// invariants are established by the creators and by the commits). Every predicate below has an exact crisp checker in
// refinements.ts. (NpcPlan.detail would take a judged predicate; it is proposed in DECOMPOSITION.md and not wired.)

/** The shape of every id a model names: merchants, goods, items, recipients and targets. */
export type Name = Is<string, "a name: a letter followed by letters, digits, '_' or '-'">;
/** Money and prices a model writes. */
export type Count = Is<number, "a non-negative safe integer">;
/** A quantity a model chooses. */
export type Quantity = Is<number, "a positive safe integer">;
/** The id of an event in a village. */
export type EventId = Is<string, "an event id: 'event-' followed by digits">;
/** The id of a memory entry: an event, or a note on an event. */
export type EntryId = Is<string, "a memory entry id: 'event-' followed by digits, optionally followed by '.note-' and digits">;


/** A fact the world recorded as data: `operation` names it (economy.intent, economy.settle, combat.resolve, npc.act). */
export type GameEvent = Record<string, unknown>;

// ---------------------------------------------------------------- shared

/** The answer of validation for one actor's proposal: ok, or not ok with the reason in one sentence. */
export type Verdict = { actor: string, ok: boolean, reason: string };

// ---------------------------------------------------------------- economy

/** A merchant: cash and goods are non-negative integers; offers maps a good to the price at which it sells one unit. */
export type Merchant = { id: string, cash: number, goods: Record<string, number>, offers: Record<string, number> };

/** The money and the goods of the whole economy; settlement keeps both unchanged. */
export type Totals = { cash: number, goods: Record<string, number> };

/** The economy at a tick. `initial` are the totals at the start; seed fixes the order in which a tick settles. */
export type EconomyState = { kind: 'economy', tick: number, seed: number, merchants: Merchant[], initial: Totals };

export type Offer = { seller: string, good: string, price: number };

/** What one merchant knows: its own cash and goods, and the public offers of the others. */
export type MarketObservation = { actor: string, tick: number, cash: number, goods: Record<string, number>, offers: Offer[] };

/** A buy names an offered seller, a good and a positive integer quantity; a pass names nothing. */
export type TradeIntent = { kind: 'buy' | 'pass', seller?: Name, good?: Name, quantity?: Quantity };

/** One merchant's intent for the tick. */
export type Submission = { actor: string, intent: TradeIntent };

/**
 * How one intent settled. pass: the merchant passed. traded: seller, good, quantity and total (quantity times the
 * seller's price) are set. rejected: the trade could not happen; reason says which condition failed.
 */
export type TradeOutcome = { actor: string, status: 'pass' | 'rejected' | 'traded', seller?: Name, good?: Name,
  quantity?: Quantity, total?: Count, reason?: string };

/**
 * One movement of money or goods between two merchants: kind cash moves `amount` units of money, kind good moves
 * `amount` units of `good`. A traded outcome is exactly two entries, the goods from seller to buyer, then the money
 * from buyer to seller.
 */
export type Entry = { kind: 'cash' | 'good', from: string, to: string, good?: string, amount: number };

/** The result of settling a tick: the outcomes in settlement order, and the entries they amount to, in the same order. */
export type Settlement = Is<{ outcomes: TradeOutcome[], entries: Entry[] },
  "entries that are, for each traded outcome in order, the goods from seller to buyer and then the money from buyer to seller, and no others">;

/** What the economy decided for a tick, as data for the commit. */
export type EconomyEffects = { basis: number, order: string[], submissions: Submission[], settlement: Settlement };

// ---------------------------------------------------------------- combat

/** A fighter on a line of cells 0..width-1. cooldown is a timer: while above 0 the fighter cannot attack. hp 0 is dead. */
export type Fighter = { id: string, x: Count, hp: Count, cooldown: Is<number, "0 or 1"> };

export type Opponent = { id: string, x: number, hp: number };

/** The arena at a round. */
export type CombatState = { kind: 'combat', round: number, width: number, fighters: Fighter[] };

/** What one fighter sees: itself and the position and health of every other fighter. */
export type CombatObservation = { actor: string, round: number, width: number, self: Fighter, others: Opponent[] };

/** One fighter's tactic: a step (left lowers x, right raises it), and an action; an attack names another fighter as target. */
export type CombatPlan = { move: 'left' | 'stay' | 'right', action: 'attack' | 'guard' | 'rest', target?: Name };

export type CombatSubmission = { actor: string, plan: CombatPlan };

/** One landed attack: amount is 2, or 1 when the target guarded this round. */
export type Hit = { attacker: string, target: string, amount: Is<number, "1 or 2"> };

/** The result of a round: every fighter afterwards (same order), the hits, and the damage each target took. */
export type CombatResolution = { fighters: Fighter[], hits: Hit[], damage: Record<string, number> };

/** What the arena decided for a round, as data for the commit. */
export type CombatEffects = { basis: number, submissions: CombatSubmission[], resolution: CombatResolution };

// ---------------------------------------------------------------- NPCs

export type NpcEvent = { from: string, text: string };

/** A memory entry. An event entry quotes what happened; a note (about lists the entry ids it summarizes) is the NPC's own. */
export type Memory = { id: string, from: string, text: string, about?: string[] };

/** What the NPC concluded about an event: text, and the ids of the memory entries it rests on. */
export type Note = { text: string, about: EntryId[] };

export type Commitment = { to: string, detail: string, evidence_id: EventId };

export type NpcActor = { id: string, inventory: Record<string, number>, memory: Memory[], commitments: Commitment[] };

/**
 * The village. version counts committed turns; nextEvent numbers events (`event-N`); applied lists the events
 * already acted on, each of which can be acted on once.
 */
export type NpcState = { kind: 'npc', version: number, nextEvent: number, applied: string[], actors: NpcActor[] };

/** What an NPC knows when an event reaches it: its inventory, its memory including this event, and its commitments. */
export type NpcObservation = { actor: string, event_id: string, event: NpcEvent, inventory: Record<string, number>,
  memory: Memory[], commitments: Commitment[] };

/** A reply and one world action: none; give (item to target, one unit); promise (to target, a concrete detail). */
export type NpcPlan = { say: string, action: 'none' | 'give' | 'promise', item?: Name, target?: Name, detail?: string };

/** What the village decided for an event, as data for the commit. */
export type NpcEffects = { basis: number, actor: string, event_id: EventId, event: NpcEvent, notes: Note[], plan: NpcPlan };

// ---------------------------------------------------------------- turns

/** One scene to play a turn of: the world and, for the village, who is addressed and what happened. */
export type Scene =
  | { kind: 'economy', state: EconomyState }
  | { kind: 'combat', state: CombatState }
  | { kind: 'npc', state: NpcState, actor: string, event: NpcEvent };

export type World = EconomyState | CombatState | NpcState;

/** The result of a turn. When ok is false, state is the state the turn started from and problem says why. */
export type Turn = { ok: boolean, state: World, events: GameEvent[], log: string[], problem: string };

/** What the commit of effects returned: the next state and the events, or the state unchanged and the problem. */
export type Committed = { ok: boolean, state: World, events: GameEvent[], problem: string };

export type TurnReport = { kind: 'economy' | 'combat' | 'npc', ok: boolean, state: World, events: GameEvent[], log: string[], problem: string, narration: string };
