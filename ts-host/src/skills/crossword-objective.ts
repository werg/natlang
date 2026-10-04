/** Host-only scoring for small, project-authored crossword and word-constraint episodes. */
import type { SkillObjectiveScore } from './objective.js';

export const CROSSWORD_SCHEMA = 'natlang.crossword-csp/1' as const;
export type CrosswordFamily = 'crossword' | 'mini-cryptic' | 'word-lattice';
export type CrosswordSlot = {
  id: string;
  clue: string;
  length: number;
  /** Grid positions, used by crossword-family crossing constraints. */
  cells?: Array<{ row: number; col: number }>;
};
export type CrosswordConstraint =
  | { kind: 'cross'; a: { slot: string; offset: number }; b: { slot: string; offset: number } }
  | { kind: 'all-different'; slots: string[] }
  | { kind: 'same-category'; slots: string[]; category: string };
export type CrosswordInstance = {
  schema: typeof CROSSWORD_SCHEMA;
  family: CrosswordFamily;
  rows: number;
  cols: number;
  instructions: string;
  slots: CrosswordSlot[];
  constraints: CrosswordConstraint[];
};
export type CrosswordReference = {
  schema: 'natlang.crossword-reference/1';
  solutionCount: number;
  accepted: Array<Record<string, string>>;
};

type ClueRecord = { family: CrosswordFamily; clue: string; answers: string[]; category?: string; rationale: string; wordplay?: { op: 'anagram' | 'reverse' | 'prepend'; input: string; added?: string } };

// This small lexicon is authored with the puzzle corpus. Rationales are host-side audit metadata.
const clues: ClueRecord[] = [
  { family: 'crossword', clue: 'A round object used in many sports', answers: ['BALL'], rationale: 'A ball is a round object used in sports.' },
  { family: 'crossword', clue: 'A measured region or amount of space', answers: ['AREA'], rationale: 'Area names a region or a measure of surface.' },
  { family: 'crossword', clue: 'Guide or direct a person', answers: ['LEAD'], rationale: 'To lead is to guide or direct.' },
  { family: 'crossword', clue: 'A woman, in a traditional form of address', answers: ['LADY'], rationale: 'Lady is a traditional noun and form of address for a woman.' },
  { family: 'crossword', clue: 'A domesticated canine', answers: ['DOG'], rationale: 'A dog is a domesticated canine.' },
  { family: 'crossword', clue: 'A metal-bearing rock', answers: ['ORE'], rationale: 'Ore is rock containing useful minerals or metals.' },
  { family: 'crossword', clue: 'A precious stone', answers: ['GEM'], rationale: 'A gem is a precious or ornamental stone.' },
  { family: 'crossword', clue: 'The household canine that commonly barks', answers: ['DOG'], rationale: 'A dog is the familiar domesticated canine.' },
  { family: 'crossword', clue: 'Rock from which useful metal can be extracted', answers: ['ORE'], rationale: 'Ore is a metal-bearing rock.' },
  { family: 'crossword', clue: 'A precious stone suitable for jewelry', answers: ['GEM'], rationale: 'A gem is a precious stone used in jewelry.' },
  { family: 'mini-cryptic', clue: 'Rearrange NIGHT to make something that can be described as an object or idea', answers: ['THING'], rationale: 'THING is an anagram of NIGHT; the definition is “something”.', wordplay: { op: 'anagram', input: 'NIGHT' } },
  { family: 'mini-cryptic', clue: 'Reverse FLOW to name an animal', answers: ['WOLF'], rationale: 'Reversing FLOW gives WOLF, an animal.', wordplay: { op: 'reverse', input: 'FLOW' } },
  { family: 'mini-cryptic', clue: 'Rearrange EARN to mean close by', answers: ['NEAR'], rationale: 'NEAR is an anagram of EARN and means close by.', wordplay: { op: 'anagram', input: 'EARN' } },
  { family: 'mini-cryptic', clue: 'Add C before AT to name a feline', answers: ['CAT'], rationale: 'Prepending C to AT gives CAT, a feline.', wordplay: { op: 'prepend', input: 'AT', added: 'C' } },
  { family: 'word-lattice', clue: 'An adult sheep', answers: ['RAM', 'EWE'], category: 'sheep', rationale: 'An adult sheep may be a ram or a ewe.' },
  { family: 'word-lattice', clue: 'A female sheep', answers: ['EWE'], category: 'sheep', rationale: 'Ewe means an adult female sheep.' },
  { family: 'word-lattice', clue: 'A young sheep', answers: ['LAMB'], category: 'sheep', rationale: 'A lamb is a young sheep.' },
  { family: 'word-lattice', clue: 'A young dog', answers: ['PUP', 'PUPPY'], category: 'dog', rationale: 'Pup and puppy both mean a young dog.' },
  { family: 'word-lattice', clue: 'A domesticated canine', answers: ['DOG'], category: 'dog', rationale: 'Dog is the ordinary name for a domesticated canine.' },
  { family: 'word-lattice', clue: 'A female dog', answers: ['BITCH'], category: 'dog', rationale: 'Bitch is the precise zoological term for a female dog.' },
];

const clueKey = (family: CrosswordFamily, clue: string) => `${family}\0${clue}`;
const wordplayResult = (c: ClueRecord): string | null => {
  if (!c.wordplay) return null;
  if (c.wordplay.op === 'reverse') return [...c.wordplay.input].reverse().join('');
  if (c.wordplay.op === 'prepend') return `${c.wordplay.added ?? ''}${c.wordplay.input}`;
  return [...c.wordplay.input].sort().join('');
};
const clueMap = new Map(clues.filter(c => {
  if (!c.answers.length) return false;
  if (c.family !== 'mini-cryptic') return true;
  const transformed = wordplayResult(c);
  if (!c.wordplay || !transformed) return false;
  return c.wordplay.op === 'anagram'
    ? c.answers.every(a => [...a].sort().join('') === [...transformed].sort().join(''))
    : c.answers.every(a => a === transformed);
}).map(c => [clueKey(c.family, c.clue), c]));
const termCategory = new Map<string, string>();
for (const c of clues) for (const answer of c.answers) if (c.category) termCategory.set(answer, c.category);
const upperWord = (x: unknown): x is string => typeof x === 'string' && /^[A-Z]+$/.test(x);

function fail(reason: string): SkillObjectiveScore { return { quality: 0, gates: { feasible: false, [reason]: false } }; }
function parseObject(value: unknown): any {
  if (typeof value !== 'string') return value;
  try { return JSON.parse(value); } catch { return null; }
}
function validateInstance(instance: unknown): CrosswordInstance | null {
  const x = parseObject(instance) as CrosswordInstance | null;
  if (!x || x.schema !== CROSSWORD_SCHEMA || !['crossword', 'mini-cryptic', 'word-lattice'].includes(x.family) ||
      !Number.isSafeInteger(x.rows) || !Number.isSafeInteger(x.cols) || x.rows < 1 || x.cols < 1 ||
      typeof x.instructions !== 'string' || !Array.isArray(x.slots) || x.slots.length < 1 || x.slots.length > 16 || !Array.isArray(x.constraints)) return null;
  const ids = new Set<string>();
  for (const s of x.slots) {
    const c = clueMap.get(clueKey(x.family, s.clue));
    if (!s || typeof s.id !== 'string' || !s.id || ids.has(s.id) || !c || !Number.isSafeInteger(s.length) || s.length < 2 || !c.answers.some(a => a.length === s.length)) return null;
    if (c.wordplay) {
      const transformed = wordplayResult(c);
      const isValid = c.wordplay.op === 'anagram'
        ? c.answers.every(a => [...a].sort().join('') === [...transformed!].sort().join(''))
        : c.answers.every(a => a === transformed);
      if (!isValid) return null;
    }
    ids.add(s.id);
    if (s.cells !== undefined) {
      if (!Array.isArray(s.cells) || s.cells.length !== s.length || s.cells.some(p => !Number.isSafeInteger(p.row) || !Number.isSafeInteger(p.col) || p.row < 0 || p.row >= x.rows || p.col < 0 || p.col >= x.cols)) return null;
      const horizontal = s.cells.every((p, i) => p.row === s.cells![0]!.row && p.col === s.cells![0]!.col + i);
      const vertical = s.cells.every((p, i) => p.col === s.cells![0]!.col && p.row === s.cells![0]!.row + i);
      if (!horizontal && !vertical) return null;
    } else if (x.family === 'crossword') return null;
  }
  for (const constraint of x.constraints) {
    if (constraint.kind === 'cross') {
      const a = x.slots.find(s => s.id === constraint.a.slot), b = x.slots.find(s => s.id === constraint.b.slot);
      if (!a || !b || a.id === b.id || !Number.isSafeInteger(constraint.a.offset) || !Number.isSafeInteger(constraint.b.offset) || constraint.a.offset < 0 || constraint.a.offset >= a.length || constraint.b.offset < 0 || constraint.b.offset >= b.length || !a.cells || !b.cells) return null;
      const pa = a.cells[constraint.a.offset]!, pb = b.cells[constraint.b.offset]!;
      if (pa.row !== pb.row || pa.col !== pb.col) return null;
    } else if (constraint.kind === 'all-different' || constraint.kind === 'same-category') {
      if (!Array.isArray(constraint.slots) || constraint.slots.length < 2 || new Set(constraint.slots).size !== constraint.slots.length || constraint.slots.some(id => !ids.has(id))) return null;
      if (constraint.kind === 'same-category' && typeof constraint.category !== 'string') return null;
    } else return null;
  }
  if (x.family === 'crossword') {
    const occupied = new Map<string, Array<{ id: string; offset: number; direction: 'across' | 'down' }>>();
    for (const slot of x.slots) for (const [offset, p] of slot.cells!.entries()) {
      const key = `${p.row},${p.col}`;
      const direction = slot.cells![0]!.row === slot.cells![1]!.row ? 'across' : 'down';
      occupied.set(key, [...(occupied.get(key) ?? []), { id: slot.id, offset, direction }]);
    }
    const crosses = x.constraints.filter((c): c is Extract<CrosswordConstraint, { kind: 'cross' }> => c.kind === 'cross');
    const actualCrosses = new Set(crosses.map(c => [c.a.slot, c.a.offset, c.b.slot, c.b.offset].join(':')));
    if (actualCrosses.size !== crosses.length) return null;
    for (const cell of occupied.values()) if (cell.length > 2) return null;
    for (const cell of occupied.values()) if (cell.length === 2) {
      const [a, b] = cell;
      if (a!.direction === b!.direction) return null;
      if (!actualCrosses.has([a!.id, a!.offset, b!.id, b!.offset].join(':')) && !actualCrosses.has([b!.id, b!.offset, a!.id, a!.offset].join(':'))) return null;
    }
    const requiredCrosses = [...occupied.values()].filter(cell => cell.length === 2).length;
    if (crosses.length !== requiredCrosses) return null;
  }
  return x;
}

function assignmentsValid(x: CrosswordInstance, values: Record<string, string>, partial: boolean): boolean {
  const ids = new Set(x.slots.map(s => s.id));
  if (Object.keys(values).some(id => !ids.has(id)) || (!partial && Object.keys(values).length !== x.slots.length)) return false;
  for (const [id, answer] of Object.entries(values)) {
    const s = x.slots.find(slot => slot.id === id)!;
    const c = clueMap.get(clueKey(x.family, s.clue))!;
    if (!upperWord(answer) || answer.length !== s.length || !c.answers.includes(answer)) return false;
  }
  for (const con of x.constraints) {
    if (con.kind === 'cross') {
      const a = values[con.a.slot], b = values[con.b.slot];
      if (a !== undefined && b !== undefined && a[con.a.offset] !== b[con.b.offset]) return false;
    } else {
      const chosen = con.slots.map(id => values[id]).filter((v): v is string => v !== undefined);
      if (con.kind === 'all-different' && new Set(chosen).size !== chosen.length) return false;
      if (con.kind === 'same-category' && chosen.some(word => termCategory.get(word) !== con.category)) return false;
    }
  }
  return true;
}

/** Enumerate the authored finite answer domains and constraints. Results are host-only. */
export function enumerateCrosswordSolutions(instance: unknown, limit = 10000): Array<Record<string, string>> {
  const x = validateInstance(instance);
  if (!x || !Number.isSafeInteger(limit) || limit < 1 || limit > 100000) throw Error('invalid crossword instance or enumeration limit');
  const solutions: Array<Record<string, string>> = [];
  const current: Record<string, string> = {};
  const visit = (index: number) => {
    if (solutions.length >= limit) return;
    if (index === x.slots.length) { if (assignmentsValid(x, current, false)) solutions.push({ ...current }); return; }
    const slot = x.slots[index]!, domain = clueMap.get(clueKey(x.family, slot.clue))!.answers.filter(answer => answer.length === slot.length);
    for (const answer of domain) {
      current[slot.id] = answer;
      if (assignmentsValid(x, current, true)) visit(index + 1);
      delete current[slot.id];
    }
  };
  visit(0);
  return solutions;
}

/** Create a reference only after enumerating the full bounded solution space. */
export function verifyCrosswordReference(instance: unknown): CrosswordReference {
  const solutions = enumerateCrosswordSolutions(instance, 10000);
  if (!solutions.length || solutions.length >= 10000) throw Error('puzzle must have a finite, explicitly enumerable solution set');
  return { schema: 'natlang.crossword-reference/1', solutionCount: solutions.length, accepted: solutions };
}

/** Host-trusted partial/full fill scoring. Any bad word or contradiction gates the whole proposal to zero. */
export function scoreCrosswordObjective(instance: unknown, value: unknown, expected: unknown): SkillObjectiveScore {
  const x = validateInstance(instance);
  if (!x) return fail('valid_instance');
  const ref = parseObject(expected) as CrosswordReference | null;
  if (!ref || ref.schema !== 'natlang.crossword-reference/1' || !Array.isArray(ref.accepted)) return fail('valid_reference');
  let exact: CrosswordReference;
  try { exact = verifyCrosswordReference(x); } catch { return fail('enumerable_reference'); }
  const normalize = (rows: Array<Record<string, string>>) => rows.map(row => JSON.stringify(Object.fromEntries(Object.entries(row).sort(([a], [b]) => a.localeCompare(b))))).sort();
  if (ref.solutionCount !== exact.solutionCount || JSON.stringify(normalize(ref.accepted)) !== JSON.stringify(normalize(exact.accepted))) return fail('reference_matches_constraints');
  const answer = parseObject(value);
  if (!answer || typeof answer !== 'object' || Array.isArray(answer) || !answer.fills || typeof answer.fills !== 'object' || Array.isArray(answer.fills)) return fail('solution_shape');
  const fills = answer.fills as Record<string, string>;
  if (!Object.keys(fills).length || !assignmentsValid(x, fills, true)) return fail('clue_domain_and_constraints');
  if(!exact.accepted.some(solution=>Object.entries(fills).every(([id,word])=>solution[id]===word)))return fail('extendable_completion');
  const complete = Object.keys(fills).length === x.slots.length;
  if (complete && !exact.accepted.some(s => Object.entries(s).every(([id, word]) => fills[id] === word))) return fail('accepted_completion');
  const quality = complete ? 1 : 0.75 * Object.keys(fills).length / x.slots.length;
  return { quality, gates: { feasible: true, clue_domain: true, constraints: true, complete }, objective: Object.keys(fills).length };
}

export const CROSSWORD_CLUE_AUDIT = clues.map(({ family, clue, answers, category, rationale, wordplay }) => ({ family, clue, answers: [...answers], ...(category ? { category } : {}), rationale, ...(wordplay ? { wordplay: { ...wordplay } } : {}) }));
