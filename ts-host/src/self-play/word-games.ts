import { IllegalGameAction, type GameOutcome, type GameSpec } from './types.js';

type Term = { word: string; meaning: string };
type WordScenario = { id: string; context: string; terms: Record<string, Term>; clues: { text: string; target: string }[]; taboo?: string[] };
type Slot = { id: string; word: string; meaning: string };
type Challenge = { id: string; target: string; clue: string };
type WordState = { game: string; scenario: string; phase: 'set' | 'guess' | 'done'; actor: string;
  context: string; slots: Slot[]; challenges: Challenge[]; taboo: string[]; secret?: string; selectedClue?: string;
  guessed?: string; correct?: boolean };

function copy<T>(value: T): T { return structuredClone(value); }
function object(value: unknown, label: string): Record<string, any> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error(`${label} must be an object`);
  return value as Record<string, any>;
}
function actionObject(value: unknown): Record<string, any> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new IllegalGameAction('action must be an object');
  return value as Record<string, any>;
}
function randomFor(seed: number): () => number {
  let value = seed >>> 0;
  return () => {
    value = (value + 0x6d2b79f5) >>> 0;
    let next = value;
    next = Math.imul(next ^ next >>> 15, next | 1);
    next ^= next + Math.imul(next ^ next >>> 7, next | 61);
    return ((next ^ next >>> 14) >>> 0) / 4294967296;
  };
}
function shuffle<T>(items: T[], random: () => number): T[] {
  const result = [...items];
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [result[i], result[j]] = [result[j]!, result[i]!];
  }
  return result;
}
function validateScenario(value: unknown, requiresTaboo: boolean): WordScenario {
  const s = object(value, 'scenario') as unknown as WordScenario;
  if (typeof s.id !== 'string' || !s.id || typeof s.context !== 'string' || !s.context.trim() ||
      !s.terms || typeof s.terms !== 'object' || Array.isArray(s.terms) || Object.keys(s.terms).length < 3 ||
      Object.entries(s.terms).some(([key, term]) => !key || !term || typeof term.word !== 'string' ||
        !term.word.trim() || term.word !== key || typeof term.meaning !== 'string' || !term.meaning.trim()) ||
      !Array.isArray(s.clues) || s.clues.length < 2 || s.clues.some(clue => !clue ||
        typeof clue.text !== 'string' || !clue.text.trim() || typeof clue.target !== 'string' || !s.terms[clue.target])) {
    throw new Error('invalid word-game scenario');
  }
  if (requiresTaboo) {
    if (!Array.isArray(s.taboo) || !s.taboo.length || s.taboo.some(word => typeof word !== 'string' || !word.trim()))
      throw new Error('taboo scenarios require a nonempty forbidden-word list');
    const forbidden = new Set(s.taboo.map(word => word.toLocaleLowerCase('en').trim()));
    const containsForbidden = (text: string) => text.toLocaleLowerCase('en').match(/[\p{L}\p{N}’'-]+/gu)?.some(token => forbidden.has(token)) ?? false;
    if (s.clues.some(clue => containsForbidden(clue.text))) throw new Error('curated taboo clue contains a forbidden word');
  } else if (s.taboo !== undefined) {
    throw new Error('near-synonym scenarios do not use taboo constraints');
  }
  return copy(s);
}

function makeWordGame(id: 'near-synonym' | 'taboo-clue', requiresTaboo: boolean): GameSpec {
  const setter = requiresTaboo ? 'cluegiver' : 'challenger';
  const solver = requiresTaboo ? 'interpreter' : 'interpreter';
  const label = requiresTaboo ? 'Taboo clue challenge' : 'Contextual near-synonym challenge';
  return {
    id,
    revision: 'word-games/1',
    rules: requiresTaboo
      ? 'The cluegiver chooses a hidden target term and one curated clue. The interpreter sees the context, clue, candidate meanings, and forbidden words, then guesses. Clues containing a forbidden word invalidate the scenario. Exact host-held target labels determine a bounded zero-sum score.'
      : 'The challenger chooses a hidden target term and one curated contextual clue. The interpreter sees the context, clue, and shuffled near-synonym choices, then guesses. Exact host-held target labels determine a bounded zero-sum score.',
    seats: [setter, solver],
    initialize(scenario, seed) {
      const spec = validateScenario(scenario, requiresTaboo), random = randomFor(seed);
      const slots = shuffle(Object.entries(spec.terms), random).map(([word, term], index) => ({
        id: `option-${index}`,
        word,
        meaning: term.meaning,
      }));
      const pairs = shuffle(spec.clues, random).map(clue => ({
        target: slots.find(slot => slot.word === clue.target)!.id,
        clue: clue.text,
      }))
        .map((pair, index) => ({ id: `challenge-${index}`, ...pair }));
      return {
        game: id, scenario: spec.id, phase: 'set', actor: setter, context: spec.context,
        slots, challenges: pairs, taboo: copy(spec.taboo ?? []),
      } satisfies WordState;
    },
    actor(state) { return object(state, 'state').actor as string; },
    observe(state, seat) {
      const s = object(state, 'state') as unknown as WordState;
      if (seat !== setter && seat !== solver) throw new Error(`unknown ${id} seat: ${seat}`);
      const shared = { game: id, role: seat, phase: s.phase, context: s.context };
      if (requiresTaboo) Object.assign(shared, { forbiddenWords: copy(s.taboo) });
      if (seat === setter) {
        return s.phase === 'set'
          ? { ...shared, challengeChoices: s.challenges.map(c => ({
            challengeId: c.id,
            targetWord: s.slots.find(slot => slot.id === c.target)?.word,
            clue: c.clue,
          })) }
          : { ...shared, waiting: true };
      }
      if (s.phase === 'guess') {
        return { ...shared, clue: s.selectedClue, choices: copy(s.slots).map(({ id: optionId, word, meaning }) => ({ optionId, word, meaning })) };
      }
      if (s.phase === 'done') return { ...shared, result: s.correct === true ? 'correct' : 'incorrect' };
      return { ...shared, choices: copy(s.slots).map(({ id: optionId, word, meaning }) => ({ optionId, word, meaning })) };
    },
    legalActions(state, seat) {
      const s = object(state, 'state') as unknown as WordState;
      if (seat !== s.actor) return [];
      if (s.phase === 'set') return s.challenges.map(({ id }) => ({ type: 'challenge', challengeId: id }));
      if (s.phase === 'guess') return s.slots.map(({ id }) => ({ type: 'guess', optionId: id }));
      return [];
    },
    apply(state, seat, action) {
      const s = copy(object(state, 'state') as unknown as WordState);
      if (seat !== s.actor) throw new IllegalGameAction(`wrong actor for ${id}`);
      const a = actionObject(action);
      if (s.phase === 'set' && a.type === 'challenge' && typeof a.challengeId === 'string') {
        const challenge = s.challenges.find(item => item.id === a.challengeId);
        if (!challenge) throw new IllegalGameAction('challenge is not available');
        s.secret = challenge.target;
        s.selectedClue = challenge.clue;
        s.phase = 'guess'; s.actor = solver;
        return s;
      }
      if (s.phase === 'guess' && a.type === 'guess' && typeof a.optionId === 'string' && s.slots.some(slot => slot.id === a.optionId)) {
        s.guessed = a.optionId;
        s.correct = a.optionId === s.secret;
        s.phase = 'done'; s.actor = '';
        return s;
      }
      throw new IllegalGameAction(`invalid ${id} action for phase ${s.phase}`);
    },
    outcome(state): GameOutcome | null {
      const s = object(state, 'state') as unknown as WordState;
      if (s.phase !== 'done' || typeof s.correct !== 'boolean') return null;
      return { scores: { [setter]: s.correct ? 0 : 1, [solver]: s.correct ? 1 : 0 }, reason: s.correct ? 'target interpreted correctly' : 'target interpretation missed' };
    },
  };
}

const nearSynonymCases: WordScenario[] = [
  { id: 'near-weather', context: 'Choose the word that best fits the sentence, not merely a related feeling.', terms: {
    apprehensive: { word: 'apprehensive', meaning: 'uneasy because something possibly unpleasant may happen' },
    skeptical: { word: 'skeptical', meaning: 'doubtful that a claim or expectation is true' },
    reluctant: { word: 'reluctant', meaning: 'unwilling to act despite pressure or expectation' },
  }, clues: [
    { text: 'She checked the forecast twice because the roof might leak tonight.', target: 'apprehensive' },
    { text: 'She asked for evidence before accepting the forecast.', target: 'skeptical' },
    { text: 'She agreed to climb the ladder only after being asked again.', target: 'reluctant' },
  ] },
  { id: 'near-economy', context: 'Distinguish careful use from unwillingness to spend.', terms: {
    frugal: { word: 'frugal', meaning: 'careful with resources to avoid waste' },
    stingy: { word: 'stingy', meaning: 'unwilling to give or spend even when appropriate' },
    extravagant: { word: 'extravagant', meaning: 'spending or using far more resources than needed' },
  }, clues: [
    { text: 'He repaired the coat so the cloth would serve another winter.', target: 'frugal' },
    { text: 'She refused to contribute to a shared meal despite having plenty.', target: 'stingy' },
    { text: 'She spent half the travel budget on a gold-plated suitcase for a weekend trip.', target: 'extravagant' },
  ] },
  { id: 'near-speech', context: 'Choose the attitude conveyed by what the person says.', terms: {
    candid: { word: 'candid', meaning: 'direct and honest, including about uncomfortable facts' },
    tactful: { word: 'tactful', meaning: 'careful to communicate truth without needless hurt' },
    evasive: { word: 'evasive', meaning: 'avoiding a direct answer or commitment' },
  }, clues: [
    { text: '“The figures are wrong; I should have checked them before sending.”', target: 'candid' },
    { text: '“The total needs one correction; let us fix it together.”', target: 'tactful' },
    { text: '“There are several ways to look at the figures, so let us move on.”', target: 'evasive' },
  ] },
  { id: 'near-attention', context: 'Separate noticing from deliberate examination.', terms: {
    notice: { word: 'notice', meaning: 'become aware of something, often without seeking it' },
    inspect: { word: 'inspect', meaning: 'look at something carefully to assess its condition' },
    observe: { word: 'observe', meaning: 'watch or take note of something, often over time' },
  }, clues: [
    { text: 'She saw a crack in the cup while reaching for it.', target: 'notice' },
    { text: 'The technician checked each seam for damage before approval.', target: 'inspect' },
    { text: 'For a week, the botanist watched when the flower opened.', target: 'observe' },
  ] },
];

const tabooCases: WordScenario[] = [
  { id: 'taboo-repair', context: 'Identify the action described by the clue.', terms: {
    mend: { word: 'mend', meaning: 'repair damage, especially by joining or stitching' },
    restore: { word: 'restore', meaning: 'return something to an earlier or better condition' },
    replace: { word: 'replace', meaning: 'put a different item in the place of the old one' },
  }, taboo: ['repair', 'fix', 'new'], clues: [
    { text: 'She joined the torn cuff with a needle and matching thread.', target: 'mend' },
    { text: 'The conservator returned the faded portrait to its former condition.', target: 'restore' },
    { text: 'They removed the broken lamp and installed another one.', target: 'replace' },
  ] },
  { id: 'taboo-motion', context: 'The clue avoids common forms of the answer words.', terms: {
    borrow: { word: 'borrow', meaning: 'receive temporary use of something from another person' },
    lend: { word: 'lend', meaning: 'allow another person temporary use of something' },
    rent: { word: 'rent', meaning: 'pay for temporary use of property or goods' },
  }, taboo: ['borrow', 'lend', 'rent'], clues: [
    { text: 'Mara took Jo’s umbrella for the afternoon and promised to return it.', target: 'borrow' },
    { text: 'Jo let Mara use the umbrella until the rain stopped.', target: 'lend' },
    { text: 'Mara paid the shop to use a bicycle for the day.', target: 'rent' },
  ] },
  { id: 'taboo-secrecy', context: 'Infer the precise verb from the situation.', terms: {
    conceal: { word: 'conceal', meaning: 'keep something from being seen or discovered' },
    disguise: { word: 'disguise', meaning: 'change appearance so identity or nature is mistaken' },
    withhold: { word: 'withhold', meaning: 'keep back information or an item that could be given' },
  }, taboo: ['hide', 'secret', 'cover'], clues: [
    { text: 'He placed the letter beneath a loose floorboard.', target: 'conceal' },
    { text: 'The actor changed voice and clothing to pass as a traveler.', target: 'disguise' },
    { text: 'The witness kept one relevant detail out of the report.', target: 'withhold' },
  ] },
  { id: 'taboo-evidence', context: 'Select the word whose meaning best matches the clue.', terms: {
    infer: { word: 'infer', meaning: 'reach a conclusion from evidence and reasoning' },
    imply: { word: 'imply', meaning: 'suggest something indirectly without stating it' },
    assume: { word: 'assume', meaning: 'accept something as true without enough confirmation' },
  }, taboo: ['conclude', 'suggest', 'guess'], clues: [
    { text: 'From the wet footprints and open window, the guard reasoned that someone entered.', target: 'infer' },
    { text: 'Her raised eyebrow communicated that she disagreed without saying so.', target: 'imply' },
    { text: 'He treated the unverified rumor as true before checking it.', target: 'assume' },
  ] },
];

export const wordGames: readonly GameSpec[] = Object.freeze([
  makeWordGame('near-synonym', false),
  makeWordGame('taboo-clue', true),
]);

/** Original curated scenario fixtures, each with a stable source-group identity. */
export function wordScenarioCases(): { group: string; family: string; scenario: unknown }[] {
  return [
    ...nearSynonymCases.map(scenario => ({ group: `word-game/near-synonym/${scenario.id}`, family: 'near-synonym', scenario: copy(scenario) })),
    ...tabooCases.map(scenario => ({ group: `word-game/taboo-clue/${scenario.id}`, family: 'taboo-clue', scenario: copy(scenario) })),
  ];
}
