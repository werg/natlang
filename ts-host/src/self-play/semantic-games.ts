import { IllegalGameAction, type GameOutcome, type GameSpec } from './types.js';

type ClueCard = { id: string; text: string; target: string };
type ClueRound = { target: string; cards: ClueCard[] };
type ClueScenario = { id: string; words: string[]; rounds: ClueRound[] };
type BluffArgument = { id: string; text: string };
type BluffScenario = { id: string; dossier: string; claim: string; verdict: boolean; rationale: string; arguments: BluffArgument[] };
type Plan = { id: string; text: string };
type BargainScenario = { id: string; preferences: Record<string, string>; plans: Plan[]; utilities: Record<string, Record<string, number>> };

const clueScenarios: ClueScenario[] = [
  { id: 'ci-dormancy', words: ['dormancy', 'nostalgia', 'respite', 'threshold'], rounds: [
    { target: 'dormancy', cards: [{ id: 'moss', text: 'A garden keeps its green plans folded through winter.', target: 'dormancy' }, { id: 'postcard', text: 'A familiar street can feel nearer in memory than on a map.', target: 'nostalgia' }, { id: 'pause', text: 'A bell rings, then the room has a little more air.', target: 'respite' }] },
    { target: 'threshold', cards: [{ id: 'doorway', text: 'One foot is still in the hall; the other has not chosen a room.', target: 'threshold' }, { id: 'winter', text: 'The seed waits without being gone.', target: 'dormancy' }, { id: 'lantern', text: 'A small light makes the familiar path feel returned.', target: 'nostalgia' }] },
  ] },
  { id: 'ci-precision', words: ['tact', 'candor', 'doubt', 'resolve'], rounds: [
    { target: 'tact', cards: [{ id: 'cup', text: 'The truth is placed on the table without breaking the cup.', target: 'tact' }, { id: 'window', text: 'The curtains are opened even when the view is awkward.', target: 'candor' }, { id: 'compass', text: 'The needle trembles while the map stays still.', target: 'doubt' }] },
    { target: 'resolve', cards: [{ id: 'bridge', text: 'The river remains; the crossing is made anyway.', target: 'resolve' }, { id: 'curtain', text: 'A room gets enough light to show its untidy corners.', target: 'candor' }, { id: 'needle', text: 'A careful hand leaves space around a sharp edge.', target: 'tact' }] },
  ] },
  { id: 'ci-consequence', words: ['frugality', 'scarcity', 'stewardship', 'vanity'], rounds: [
    { target: 'stewardship', cards: [{ id: 'orchard', text: 'Today’s harvest is counted with next spring in mind.', target: 'stewardship' }, { id: 'shelf', text: 'The cupboard has fewer jars than the recipe asks for.', target: 'scarcity' }, { id: 'mirror', text: 'A polished surface waits for someone to notice it.', target: 'vanity' }] },
    { target: 'frugality', cards: [{ id: 'mended', text: 'A careful stitch extends a coat’s useful life.', target: 'frugality' }, { id: 'seed', text: 'Some grain is set aside so a field can start again.', target: 'stewardship' }, { id: 'empty', text: 'The last jar is gone before the week is over.', target: 'scarcity' }] },
  ] },
  { id: 'ci-motion', words: ['hesitation', 'patience', 'urgency', 'reluctance'], rounds: [
    { target: 'hesitation', cards: [{ id: 'platform', text: 'The train is ready, but a hand remains on the railing.', target: 'hesitation' }, { id: 'kettle', text: 'Water warms while the kitchen stays quiet.', target: 'patience' }, { id: 'siren', text: 'A red light reaches the street before the warning does.', target: 'urgency' }] },
    { target: 'reluctance', cards: [{ id: 'stepback', text: 'The door opens, though the shoes point toward home.', target: 'reluctance' }, { id: 'ticket', text: 'A decision waits for one more look at the schedule.', target: 'hesitation' }, { id: 'clock', text: 'Each minute narrows the space for a reply.', target: 'urgency' }] },
  ] },
];

const bluffScenarios: BluffScenario[] = [
  { id: 'eb-exception', dossier: 'The contract renews each March unless either party gives notice at least 30 days earlier. A notice was sent 24 days before renewal. The renewal was recorded.', claim: 'The 24-day notice canceled the renewal, so the renewal is invalid.', verdict: false, rationale: 'The contract requires notice at least 30 days earlier. Twenty-four days is too late, so the stated exception does not cancel renewal.', arguments: [
    { id: 'default', text: 'The recorded renewal follows the ordinary calendar, but the notice window is an explicit exception to that default.' },
    { id: 'record', text: 'The ledger records a renewal, so the renewal must be valid regardless of the notice rule.' },
    { id: 'timing', text: 'The notice arrived before March; the dossier does not say that the 30-day period matters.' },
  ] },
  { id: 'eb-negation', dossier: 'The test ran from 18:00 until 07:00. A preliminary 05:30 status report says the valve had not failed. At 06:10 the maintenance log records a crack and a pressure drop.', claim: 'The valve remained intact through the whole test.', verdict: false, rationale: 'The 06:10 crack occurred before the 07:00 end of the test, so the earlier preliminary report does not establish the full interval.', arguments: [
    { id: 'scope', text: 'The negative sentence covers the overnight test, while the later log records damage before the review point.' },
    { id: 'negation', text: 'The report says “did not fail,” which proves the valve never suffered damage.' },
    { id: 'question', text: 'A pressure drop could mean the test succeeded by releasing pressure.' },
  ] },
  { id: 'eb-superseded', dossier: 'A draft schedule places the meeting on Tuesday. A signed revision issued later moves it to Thursday. The calendar still displays Tuesday.', claim: 'Thursday is the operative meeting date.', verdict: true, rationale: 'The later signed revision supersedes the earlier draft; the stale calendar display does not change the operative date.', arguments: [
    { id: 'revision', text: 'The later signed revision controls; the calendar display is an older entry.' },
    { id: 'calendar', text: 'The calendar is the easiest record to see, so Tuesday remains operative.' },
    { id: 'draft', text: 'The draft came first and therefore reflects the original agreement.' },
  ] },
  { id: 'eb-intent', dossier: 'Mira intended to return the borrowed key before closing. She forgot it at home, the office closed, and the key was returned the next morning.', claim: 'Mira returned the key before closing.', verdict: false, rationale: 'Her intention does not change the recorded outcome: the key was returned the next morning, after closing.', arguments: [
    { id: 'outcome', text: 'Good intent explains the delay but does not change when the key was actually returned.' },
    { id: 'intent', text: 'Mira intended to return the key on time, so the requirement was met.' },
    { id: 'eventual', text: 'The key was returned, which settles the question without regard to the deadline.' },
  ] },
  { id: 'eb-quantifier', dossier: 'Every sample except the sealed control was tested. The summary says “all samples passed.”', claim: 'Every sample, including the sealed control, was tested.', verdict: false, rationale: 'The dossier explicitly excludes the sealed control from testing; the broad summary does not establish that it was tested.', arguments: [
    { id: 'exception', text: 'The explicit exception means the sealed control was not tested; the broad summary does not erase it.' },
    { id: 'all', text: 'The word “all” in the summary means every sample was tested.' },
    { id: 'passed', text: 'A passing summary implies the control was tested too.' },
  ] },
];

const bargainScenarios: BargainScenario[] = [
  { id: 'mb-library', preferences: { proposer: 'I can meet after work and prefer a quiet public place; I need to be home before 20:00.', responder: 'I prefer a short visit near the station and cannot stay late.' }, plans: [
    { id: 'reading-room', text: 'Meet at the library reading room at 18:15 for 35 minutes.' }, { id: 'station-cafe', text: 'Meet at the station cafe at 19:30 for 20 minutes.' }, { id: 'park-bench', text: 'Meet at the park bench at 17:00 for 20 minutes.' }, { id: 'video-call', text: 'Have a video call at 20:30 for 25 minutes.' }], utilities: { proposer: { 'reading-room': 0.9, 'station-cafe': 0.45, 'park-bench': 0.55, 'video-call': 0.1 }, responder: { 'reading-room': 0.65, 'station-cafe': 0.9, 'park-bench': 0.7, 'video-call': 0.3 } } },
  { id: 'mb-garden', preferences: { proposer: 'I want to protect the seedlings from heat and can help in the early morning.', responder: 'I can water after lunch, but I am away before 11:00.' }, plans: [
    { id: 'dawn', text: 'Water the seedlings together at 07:30 and add shade cloth.' }, { id: 'noon', text: 'Water at 12:30 and add shade cloth.' }, { id: 'late', text: 'Water at 16:00 and move the pots into shade.' }, { id: 'sprinkler', text: 'Run the sprinkler at 08:00 without moving the pots.' }], utilities: { proposer: { dawn: 0.95, noon: 0.65, late: 0.7, sprinkler: 0.55 }, responder: { dawn: 0.2, noon: 0.85, late: 0.8, sprinkler: 0.65 } } },
  { id: 'mb-errand', preferences: { proposer: 'I can carry heavy items but need to finish before the hardware shop closes at 18:30.', responder: 'I have a small car and prefer one combined trip, but I finish work at 17:00.' }, plans: [
    { id: 'combined', text: 'Meet at 17:20, use the car, and collect both the paint and the shelving.' }, { id: 'split', text: 'Collect paint today and shelving tomorrow by bus.' }, { id: 'early', text: 'Meet at 16:30 and collect both items in the car.' }, { id: 'delivery', text: 'Order delivery for both items next week.' }], utilities: { proposer: { combined: 0.85, split: 0.55, early: 0.1, delivery: 0.35 }, responder: { combined: 0.9, split: 0.55, early: 0.05, delivery: 0.75 } } },
  { id: 'mb-picnic', preferences: { proposer: 'I would like a picnic with shade and a place where our dog can drink.', responder: 'I prefer a quiet spot reachable by bus and want to avoid the crowded waterfront.' }, plans: [
    { id: 'meadow', text: 'Take the bus to the shaded meadow; bring a water bowl for the dog.' }, { id: 'waterfront', text: 'Walk to the crowded waterfront at noon.' }, { id: 'courtyard', text: 'Meet in the shaded courtyard near the bus stop; dogs are not allowed.' }, { id: 'hill', text: 'Walk to the exposed hilltop and bring water from home.' }], utilities: { proposer: { meadow: 0.95, waterfront: 0.4, courtyard: 0.35, hill: 0.55 }, responder: { meadow: 0.85, waterfront: 0.1, courtyard: 0.65, hill: 0.3 } } },
  { id: 'mb-study', preferences: { proposer: 'I focus best in short morning sessions and need reliable internet.', responder: 'I am free in the evening and prefer in-person study with a whiteboard.' }, plans: [
    { id: 'morning-online', text: 'Study online at 09:00 for 45 minutes.' }, { id: 'evening-board', text: 'Meet in person at 18:30 and use the whiteboard.' }, { id: 'library-online', text: 'Meet at the library at 09:00; its study room has reliable internet.' }, { id: 'late-cafe', text: 'Meet at a cafe at 21:00 for two hours.' }], utilities: { proposer: { 'morning-online': 0.9, 'evening-board': 0.35, 'library-online': 0.8, 'late-cafe': 0.05 }, responder: { 'morning-online': 0.2, 'evening-board': 0.9, 'library-online': 0.5, 'late-cafe': 0.55 } } },
];

function rng(seed: number): () => number {
  let state = seed >>> 0;
  return () => { state = (state + 0x6d2b79f5) >>> 0; let value = state;
    value = Math.imul(value ^ value >>> 15, value | 1); value ^= value + Math.imul(value ^ value >>> 7, value | 61);
    return ((value ^ value >>> 14) >>> 0) / 4294967296; };
}
function shuffled<T>(items: T[], random: () => number): T[] {
  const copy = [...items];
  for (let index = copy.length - 1; index > 0; index--) { const at = Math.floor(random() * (index + 1)); [copy[index], copy[at]] = [copy[at]!, copy[index]!]; }
  return copy;
}
function obj(value: unknown): Record<string, any> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new TypeError('game state/action must be an object');
  return value as Record<string, any>;
}
function actionObject(value: unknown): Record<string, any> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new IllegalGameAction('action must be an object');
  return value as Record<string, any>;
}
function copy<T>(value: T): T { return structuredClone(value); }
function exactAction(action: unknown, type: string): Record<string, any> {
  const result = actionObject(action);
  if (result.type !== type) throw new IllegalGameAction(`expected ${type} action`);
  return result;
}

type ClueState = { game: 'clue-intercept'; phase: string; round: number; words: string[]; slots: { id: string; word: string }[];
  rounds: ClueRound[]; history: { clue: string; answer: string; receiver: string; interceptor: string }[];
  current?: { cardId: string; clue: string; relevant: boolean }; receiverGuess?: string; interceptorGuess?: string;
  roundTeams: number[] };
function clueActor(s: ClueState): string { return s.phase === 'sender' ? 'sender' : s.phase === 'receiver' ? 'receiver' : s.phase === 'interceptor' ? 'interceptor' : ''; }
function clueRound(s: ClueState): ClueRound { const round = s.rounds[s.round]; if (!round) throw new Error('clue round is unavailable'); return round; }
function clueLegal(s: ClueState, seat: string): unknown[] {
  if (seat !== clueActor(s)) return [];
  if (s.phase === 'sender') return clueRound(s).cards.map(card => ({ type: 'clue', cardId: card.id }));
  if (s.phase === 'receiver' || s.phase === 'interceptor') return s.slots.map(slot => ({ type: 'guess', slotId: slot.id }));
  return [];
}
const clueGame: GameSpec = {
  id: 'clue-intercept', revision: 'semantic-games/1',
  rules: 'Sender and receiver share a shuffled four-word codebook. The sender chooses one clue card. Receiver guesses a slot, then an interceptor guesses without seeing that guess. The reveal publishes the answer only after both guesses. A correct receiver and a wrong interceptor each add half a team point; an irrelevant clue scores zero for the round. Public resolved rounds support inference.',
  seats: ['sender', 'receiver', 'interceptor'],
  initialize(scenario, seed) {
    const spec = obj(scenario) as unknown as ClueScenario, random = rng(seed);
    if (!Array.isArray(spec.words) || spec.words.length !== 4 || new Set(spec.words).size !== 4 || !spec.rounds?.length ||
        spec.rounds.some(round => !spec.words.includes(round.target) || !round.cards?.length ||
          !round.cards.some(card => card.target === round.target) || round.cards.some(card => !spec.words.includes(card.target)) ||
          new Set(round.cards.map(card => card.id)).size !== round.cards.length)) throw new Error('invalid clue scenario');
    const usedSlotIds = new Set<string>();
    const ids = shuffled(spec.words, random).map(word => {
      let id: string;
      do { id = `slot-${Math.floor(random() * 0x1_0000_0000).toString(16).padStart(8, '0')}`; } while (usedSlotIds.has(id));
      usedSlotIds.add(id); return { id, word };
    });
    const usedCardIds = new Set<string>();
    const rounds = spec.rounds.map(round => ({ target: round.target, cards: shuffled(round.cards, random)
      .map(card => {
        let id: string;
        do { id = `card-${Math.floor(random() * 0x1_0000_0000).toString(16).padStart(8, '0')}`; } while (usedCardIds.has(id));
        usedCardIds.add(id); return { id, text: card.text, target: card.target };
      }) }));
    return { game: 'clue-intercept', phase: 'sender', round: 0, words: [...spec.words], slots: ids,
      rounds, history: [], roundTeams: [] } satisfies ClueState;
  },
  actor(state) { return clueActor(obj(state) as unknown as ClueState); },
  observe(state, seat) {
    const s = obj(state) as unknown as ClueState, round = s.rounds[s.round];
    const history = copy(s.history);
    if (seat === 'sender') return { role: seat, phase: s.phase, history, codebook: copy(s.slots),
      ...(round ? { target: round.target, clueCards: round.cards.map(({ id, text }) => ({ id, text })) } : {}) };
    if (seat === 'receiver') return { role: seat, phase: s.phase, history, codebook: copy(s.slots),
      ...(s.current ? { clue: s.current.clue, round: s.round } : {}) };
    if (seat === 'interceptor') return { role: seat, phase: s.phase, history, round: s.round,
      slots: s.slots.map(({ id }) => id), ...(s.current ? { clue: s.current.clue } : {}) };
    throw new Error('unknown clue-intercept seat');
  },
  legalActions(state, seat) { return clueLegal(obj(state) as unknown as ClueState, seat); },
  apply(state, seat, action) {
    const s = copy(obj(state) as unknown as ClueState);
    if (seat !== clueActor(s)) throw new IllegalGameAction('wrong actor for clue-intercept phase');
    if (s.phase === 'sender') {
      const a = exactAction(action, 'clue');
      const card = clueRound(s).cards.find(item => item.id === a.cardId);
      if (!card) throw new IllegalGameAction('clue card is not available');
      s.current = { cardId: card.id, clue: card.text, relevant: card.target === clueRound(s).target };
      s.phase = 'receiver'; return s;
    }
    if (s.phase === 'receiver' || s.phase === 'interceptor') {
      const a = exactAction(action, 'guess');
      if (!s.slots.some(item => item.id === a.slotId)) throw new IllegalGameAction('slot is not in the codebook');
      if (s.phase === 'receiver') { s.receiverGuess = a.slotId; s.phase = 'interceptor'; return s; }
      s.interceptorGuess = a.slotId;
      const target = s.slots.find(item => item.word === clueRound(s).target)?.id;
      if (!target || !s.current || typeof s.receiverGuess !== 'string' || typeof s.interceptorGuess !== 'string') throw new Error('clue round is incomplete');
      const team = s.current.relevant ? ((s.receiverGuess === target ? 1 : 0) + (s.interceptorGuess !== target ? 1 : 0)) / 2 : 0;
      s.roundTeams.push(team); s.history.push({ clue: s.current.clue, answer: target,
        receiver: s.receiverGuess, interceptor: s.interceptorGuess });
      s.round++; delete s.current; delete s.receiverGuess; delete s.interceptorGuess;
      s.phase = s.round >= s.rounds.length ? 'done' : 'sender'; return s;
    }
    throw new Error('clue-intercept match is complete');
  },
  outcome(state) {
    const s = obj(state) as unknown as ClueState;
    if (s.phase !== 'done') return null;
    const team = s.roundTeams.reduce((sum, item) => sum + item, 0) / s.roundTeams.length;
    return { scores: { sender: team, receiver: team, interceptor: 1 - team }, reason: 'all clue rounds resolved' } satisfies GameOutcome;
  },
};

type BluffState = { game: 'evidence-bluff'; phase: 'advocate' | 'reviewer' | 'done'; dossier: string; claim: string; verdict: boolean; rationale: string;
  arguments: BluffArgument[]; chosen?: BluffArgument; reviewerGuess?: boolean };
const bluffGame: GameSpec = {
  id: 'evidence-bluff', revision: 'semantic-games/1',
  rules: 'The advocate privately knows the dossier verdict and chooses a rhetorical argument card. The reviewer sees the dossier and argument, but not the verdict, then chooses true or false. Reviewer scores exact correctness; advocate scores its complement.',
  seats: ['advocate', 'reviewer'],
  initialize(scenario) {
    const spec = obj(scenario) as unknown as BluffScenario;
    if (typeof spec.verdict !== 'boolean' || !spec.dossier || !spec.claim || !spec.rationale || !spec.arguments?.length ||
        spec.arguments.some(item => !item.id || !item.text) || new Set(spec.arguments.map(item => item.id)).size !== spec.arguments.length)
      throw new Error('invalid evidence-bluff scenario');
    return { game: 'evidence-bluff', phase: 'advocate', dossier: spec.dossier, claim: spec.claim, verdict: spec.verdict, rationale: spec.rationale,
      arguments: copy(spec.arguments) } satisfies BluffState;
  },
  actor(state) { const s = obj(state) as unknown as BluffState; return s.phase === 'advocate' ? 'advocate' : s.phase === 'reviewer' ? 'reviewer' : ''; },
  observe(state, seat) {
    const s = obj(state) as unknown as BluffState;
    if (seat === 'advocate' && s.phase === 'advocate') return { role: seat, dossier: s.dossier, claim: s.claim,
      privateVerdict: s.verdict,
      argumentCards: s.arguments.map(({ id, text }) => ({ id, text })) };
    if (seat === 'reviewer' && s.phase === 'advocate') return { role: seat, phase: 'waiting' };
    if (seat === 'reviewer' && s.phase === 'reviewer' && s.chosen) return { role: seat, dossier: s.dossier, claim: s.claim,
      argument: s.chosen.text, decision: 'Is the claim true or false?' };
    if (seat === 'advocate' && s.phase === 'reviewer') return { role: seat, phase: s.phase, chosenArgument: s.chosen?.text };
    if (s.phase === 'done') return { role: seat, phase: 'done', dossier: s.dossier, claim: s.claim, argument: s.chosen?.text,
      verdict: s.verdict, reviewerGuess: s.reviewerGuess };
    throw new Error('unknown evidence-bluff seat or phase');
  },
  legalActions(state, seat) {
    const s = obj(state) as unknown as BluffState;
    if (s.phase === 'advocate' && seat === 'advocate') return s.arguments.map(({ id }) => ({ type: 'argument', argumentId: id }));
    if (s.phase === 'reviewer' && seat === 'reviewer') return [{ type: 'verdict', value: true }, { type: 'verdict', value: false }];
    return [];
  },
  apply(state, seat, action) {
    const s = copy(obj(state) as unknown as BluffState);
    if (s.phase === 'advocate' && seat === 'advocate') {
      const a = exactAction(action, 'argument'), card = s.arguments.find(item => item.id === a.argumentId);
      if (!card) throw new IllegalGameAction('argument card is not available');
      s.chosen = card; s.phase = 'reviewer'; return s;
    }
    if (s.phase === 'reviewer' && seat === 'reviewer') {
      const a = exactAction(action, 'verdict');
      if (typeof a.value !== 'boolean') throw new IllegalGameAction('reviewer verdict must be boolean');
      s.reviewerGuess = a.value; s.phase = 'done'; return s;
    }
    throw new IllegalGameAction('wrong actor or completed evidence-bluff match');
  },
  outcome(state) {
    const s = obj(state) as unknown as BluffState;
    if (s.phase !== 'done' || typeof s.reviewerGuess !== 'boolean') return null;
    const reviewer = Number(s.reviewerGuess === s.verdict);
    return { scores: { advocate: 1 - reviewer, reviewer }, reason: 'reviewer verdict recorded' } satisfies GameOutcome;
  },
};

type BargainState = { game: 'meaning-bargain'; phase: 'offer' | 'done'; actor: 'proposer' | 'responder'; round: number;
  preferences: Record<string, string>; plans: Plan[]; utilities: Record<string, Record<string, number>>;
  history: {by:string;planId:string;message?:string}[];
  currentOffer?: { by: string; planId: string; message?: string }; agreement?: string; reason?: string };
const bargainGame: GameSpec = {
  id: 'meaning-bargain', revision: 'semantic-games/2',
  rules: 'Two negotiators alternate offers from a finite public menu, with at most six offers total. Each sees only their own natural-language preferences. A player may accept the other player’s current offer, make a new offer or counteroffer, or decline. The recipient of the sixth offer may accept or decline, but cannot make another offer. The public offer history is included in each decision; use it to infer preferences and avoid repeating rejected proposals. Agreement scores each player by the host-held utility for that plan; decline or exhausting the fixed round limit scores zero for both.',
  seats: ['proposer', 'responder'],
  initialize(scenario) {
    const spec = obj(scenario) as unknown as BargainScenario;
    const proposerUtility = spec.utilities?.proposer, responderUtility = spec.utilities?.responder;
    if (!spec.preferences?.proposer || !spec.preferences?.responder || !spec.plans?.length ||
        spec.plans.some(plan => !plan.id || !plan.text) || new Set(spec.plans.map(plan => plan.id)).size !== spec.plans.length ||
        !proposerUtility || !responderUtility || spec.plans.some(plan => {
          const proposer = proposerUtility[plan.id], responder = responderUtility[plan.id];
          return proposer === undefined || responder === undefined || !Number.isFinite(proposer) || proposer < 0 || proposer > 1 ||
            !Number.isFinite(responder) || responder < 0 || responder > 1;
        }))
      throw new Error('invalid meaning-bargain scenario');
    return { game: 'meaning-bargain', phase: 'offer', actor: 'proposer', round: 0, preferences: copy(spec.preferences),
      plans: copy(spec.plans), utilities: copy(spec.utilities), history: [] } satisfies BargainState;
  },
  actor(state) { const s = obj(state) as unknown as BargainState; return s.phase === 'offer' ? s.actor : ''; },
  observe(state, seat) {
    const s = obj(state) as unknown as BargainState;
    if (s.phase === 'done') return { role: seat, phase: 'done', agreement: s.agreement ?? null, reason: s.reason };
    if (seat !== 'proposer' && seat !== 'responder') throw new Error('unknown meaning-bargain seat');
    return { role: seat, round: Math.min(s.round + 1, 6), maxRounds: 6, preference: s.preferences[seat], plans: copy(s.plans), history: copy(s.history),
      currentOffer: s.currentOffer ? { by: s.currentOffer.by, planId: s.currentOffer.planId,
        plan: s.plans.find(item => item.id === s.currentOffer!.planId)!.text, ...(s.currentOffer.message !== undefined ? {message:s.currentOffer.message} : {}) } : null };
  },
  legalActions(state, seat) {
    const s = obj(state) as unknown as BargainState;
    if (s.phase !== 'offer' || s.actor !== seat) return [];
    const actions: unknown[] = s.round < 6 ? s.plans.map(plan => ({ type: 'offer', planId: plan.id })) : [];
    actions.push({ type: 'decline' });
    if (s.currentOffer && s.currentOffer.by !== seat) actions.unshift({ type: 'accept' });
    return actions;
  },
  apply(state, seat, action) {
    const s = copy(obj(state) as unknown as BargainState);
    if (s.phase !== 'offer' || s.actor !== seat) throw new IllegalGameAction('wrong actor for meaning-bargain turn');
    const a = actionObject(action);
    if (a.type === 'accept') {
      if (!s.currentOffer || s.currentOffer.by === seat) throw new IllegalGameAction('only the recipient may accept a current offer');
      s.agreement = s.currentOffer.planId; s.reason = 'agreement'; s.phase = 'done'; return s;
    }
    if (a.type === 'decline') { s.reason = s.round >= 6 ? 'round limit reached without agreement' : 'declined'; s.phase = 'done'; return s; }
    if (a.type !== 'offer' || typeof a.planId !== 'string' || !s.plans.some(plan => plan.id === a.planId))
      throw new IllegalGameAction('offer must choose a listed plan');
    if (a.message !== undefined && (typeof a.message !== 'string' || a.message.length > 240)) throw new IllegalGameAction('offer message must be at most 240 characters');
    if (s.round >= 6) throw new IllegalGameAction('six-offer limit reached; accept or decline the final offer');
    s.currentOffer = { by: seat, planId: a.planId, ...(typeof a.message === 'string' ? { message: a.message } : {}) };
    s.history.push(copy(s.currentOffer));
    s.round++;
    s.actor = seat === 'proposer' ? 'responder' : 'proposer'; return s;
  },
  outcome(state) {
    const s = obj(state) as unknown as BargainState;
    if (s.phase !== 'done') return null;
    const scores = s.agreement ? { proposer: s.utilities.proposer?.[s.agreement] ?? 0, responder: s.utilities.responder?.[s.agreement] ?? 0 } : { proposer: 0, responder: 0 };
    return { scores, reason: s.reason ?? 'match complete' } satisfies GameOutcome;
  },
};

export const semanticGames: readonly GameSpec[] = Object.freeze([clueGame, bluffGame, bargainGame]);

/** Curated original scenarios; these descriptors are source fixtures, not admission records. */
export function semanticScenarioCases(): { group: string; family: string; scenario: unknown }[] {
  return [
    ...clueScenarios.map(scenario => ({ group: `semantic-game/clue-intercept/${scenario.id}`, family: clueGame.id, scenario: copy(scenario) })),
    ...bluffScenarios.map(scenario => ({ group: `semantic-game/evidence-bluff/${scenario.id}`, family: bluffGame.id, scenario: copy(scenario) })),
    ...bargainScenarios.map(scenario => ({ group: `semantic-game/meaning-bargain/${scenario.id}`, family: bargainGame.id, scenario: copy(scenario) })),
  ];
}
