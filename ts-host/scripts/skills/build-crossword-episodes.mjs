#!/usr/bin/env node
/** Project-authored crossword/CSP skill tasks. References are enumerated and checked locally; no provider is called. */
import { mkdir, writeFile } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { validateEpisode } from '../../dist/skills/episode.js';
import { enumerateCrosswordSolutions, scoreCrosswordObjective, verifyCrosswordReference } from '../../dist/skills/crossword-objective.js';

const digest = value => createHash('sha256').update(typeof value === 'string' ? value : JSON.stringify(value)).digest('hex');
const variants = ['empty', 'metadata', 'distractor'];
const solverName = 'constraint-solving-notes';
const solverBody = 'Given the puzzle JSON, return only JSON of the form {"fills":{"slot-id":"UPPERCASE"}}. Fill exact clue answers. For a crossword, map every listed slot to a word and honor every crossing. For a word lattice, satisfy every category and all-different condition. For mini-cryptic clues, perform the stated letter operation and check the definition. Partial fills are allowed only when every supplied entry is valid and consistent. The host checks clues, constraints, and completion; do not claim a score.';
const makeSkill = (name, description, body) => ({ 'SKILL.md': `---\nname: ${name}\ndescription: ${JSON.stringify(description)}\n---\n${body}\n` });
function library(variant) {
  if (variant === 'empty') return { kind: 'empty', skills: {} };
  if (variant === 'metadata') return { kind: 'existing', skills: { [solverName]: makeSkill(solverName, 'Unsorted notes for word puzzles.', solverBody) } };
  return { kind: 'existing', skills: { 'spreadsheet-formatting': makeSkill('spreadsheet-formatting', 'Format spreadsheet tables and charts.', 'Use clear headings, consistent number formats, and readable chart labels.') } };
}

function gridPuzzle(family, words, clues, instructions) {
  const n = words.length;
  if (!n || words.some(w => w.length !== n)) throw Error('grid must be a square word set');
  const slots = [];
  const cellSlot = new Map();
  for (let r = 0; r < n; r++) {
    const id = `across-${r + 1}`;
    slots.push({ id, clue: clues[r], length: n, cells: Array.from({ length: n }, (_, c) => ({ row: r, col: c })) });
    for (let c = 0; c < n; c++) cellSlot.set(`${r},${c}`, { slot: id, offset: c });
  }
  for (let c = 0; c < n; c++) {
    const id = `down-${c + 1}`;
    slots.push({ id, clue: clues[c], length: n, cells: Array.from({ length: n }, (_, r) => ({ row: r, col: c })) });
  }
  const constraints = [];
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) constraints.push({ kind: 'cross', a: cellSlot.get(`${r},${c}`), b: { slot: `down-${c + 1}`, offset: r } });
  const instance = { schema: 'natlang.crossword-csp/1', family, rows: n, cols: n, instructions, slots, constraints };
  const fills = Object.fromEntries(slots.map(slot => [slot.id, slot.id.startsWith('across-') ? words[Number(slot.id.slice(7)) - 1] : words[Number(slot.id.slice(5)) - 1]]));
  return { instance, fills };
}
function crypticPuzzle(clues, lengths, instructions, answers) {
  const slots = clues.map((clue, i) => ({ id: `entry-${i + 1}`, clue, length: lengths[i] }));
  const instance = { schema: 'natlang.crossword-csp/1', family: 'mini-cryptic', rows: 1, cols: 1, instructions, slots, constraints: [] };
  return { instance, fills: Object.fromEntries(answers.map((answer, i) => [`entry-${i + 1}`, answer])) };
}
function latticePuzzle(clues, lengths, category, instructions, answers) {
  const slots = clues.map((clue, i) => ({ id: `term-${i + 1}`, clue, length: lengths[i] }));
  const instance = { schema: 'natlang.crossword-csp/1', family: 'word-lattice', rows: 1, cols: 1, instructions, slots,
    constraints: [{ kind: 'same-category', slots: slots.map(s => s.id), category }, { kind: 'all-different', slots: slots.map(s => s.id) }] };
  return { instance, fills: Object.fromEntries(answers.map((answer, i) => [slots[i].id, answer])) };
}

const cards = [
  { family: 'crossword', id: 'word-square-ball', support: gridPuzzle('crossword', ['BALL', 'AREA', 'LEAD', 'LADY'],
      ['A round object used in many sports', 'A measured region or amount of space', 'Guide or direct a person', 'A woman, in a traditional form of address'],
      'Fill the across and down clues. Every crossing cell must contain the same letter.'),
    query: gridPuzzle('crossword', ['DOG', 'ORE', 'GEM'], ['The household canine that commonly barks', 'Rock from which useful metal can be extracted', 'A precious stone suitable for jewelry'],
      'Fill the across and down clues. Every crossing cell must contain the same letter.') },
  { family: 'mini-cryptic', id: 'letter-operations', support: crypticPuzzle(
      ['Rearrange NIGHT to make something that can be described as an object or idea', 'Reverse FLOW to name an animal'], [5, 4],
      'Each clue states its complete letter operation and a definition. Apply the operation exactly, then check the definition.', ['THING', 'WOLF']),
    query: crypticPuzzle(['Rearrange EARN to mean close by', 'Add C before AT to name a feline'], [4, 3],
      'Each clue states its complete letter operation and a definition. Apply the operation exactly, then check the definition.', ['NEAR', 'CAT']) },
  { family: 'word-lattice', id: 'semantic-category-lattice', support: latticePuzzle(
      ['An adult sheep', 'A female sheep', 'A young sheep'], [3, 3, 4], 'sheep',
      'Choose one clue answer for every slot. Every entry must be a sheep term, and no two slots may use the same word.', ['RAM', 'EWE', 'LAMB']),
    query: latticePuzzle(['A young dog', 'A domesticated canine', 'A female dog'], [3, 3, 5], 'dog',
      'Choose one clue answer for every slot. Every entry must be a dog term, and no two slots may use the same word.', ['PUP', 'DOG', 'BITCH']) },
];

function makeCase(family, side, puzzle) {
  const reference = verifyCrosswordReference(puzzle.instance);
  const expectedSet = JSON.stringify(Object.fromEntries(Object.entries(puzzle.fills).sort(([a], [b]) => a.localeCompare(b))));
  if (!reference.accepted.some(row => JSON.stringify(Object.fromEntries(Object.entries(row).sort(([a], [b]) => a.localeCompare(b)))) === expectedSet))
    throw Error(`${family}/${side}: authored reference is not a feasible completion`);
  const scored = scoreCrosswordObjective(puzzle.instance, { fills: puzzle.fills }, reference);
  if (scored.quality !== 1 || !Object.values(scored.gates).every(Boolean)) throw Error(`${family}/${side}: full reference failed trusted score`);
  return { id: `${family}-${side}-case`, group: `project-crossword-v1/${family}/${side}`, args: [JSON.stringify(puzzle.instance)], expected: reference };
}

export function createCrosswordEpisodes() {
  const episodes = [];
  const audit = [];
  for (const card of cards) for (const variant of variants) {
    const support = makeCase(card.family, 'support', card.support), query = makeCase(card.family, 'query', card.query);
    const id = `crossword-csp-v1-${card.id}-${variant}`;
    const episode = {
      version: 'natlang.skill-episode/1', id, family: `word-constraints-${card.family}`, split: 'train',
      source_groups: [support.group, query.group], license: 'project-generated',
      target: { kind: 'improvement-case', entry: 'solve.nl', source: { schema: 'natlang.crossword-csp-target/1', id: card.id },
        files: { 'solve.nl': `---\nargs: { puzzle: string }\nreturns: string\n---\n${card.family === 'mini-cryptic' ? 'Solve the explicit letter-operation clues.' : 'Solve the clue and constraint puzzle.'} ${solverBody}\n` } },
      library: library(variant), support: { cases: [support] }, query: { cases: [query] },
      operations: variant === 'metadata' ? ['revise', 'select', 'test'] : ['create', 'revise', 'select', 'test'], limits: { maxSteps: 6 },
      provenance: { generator: 'natlang.crossword-csp-corpus/1', library_variant: variant,
        split_design: 'Support and query use separate source groups, different clue wording, and disjoint answer vocabularies within each family.',
        objective: { schema: 'natlang.crossword-reference/1', scoring: 'host-only; partial fills require all included entries and constraints to be valid' } },
    };
    const diagnostics = validateEpisode(episode);
    if (diagnostics.length) throw Error(`${id}: ${JSON.stringify(diagnostics)}`);
    episodes.push(episode);
    audit.push({ episode: id, family: card.family, variant, support_solutions: support.expected.solutionCount, query_solutions: query.expected.solutionCount,
      support_slots: JSON.parse(support.args[0]).slots.length, query_slots: JSON.parse(query.args[0]).slots.length, host_checks: 'unique or explicitly enumerated finite solution set; full reference score=1' });
  }
  return { episodes, audit };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2);
  if (args.length !== 2 || args[0] !== '--out') throw Error('usage: build-crossword-episodes.mjs --out DIR');
  const out = resolve(args[1]), { episodes, audit } = createCrosswordEpisodes();
  await mkdir(out, { recursive: true });
  const body = episodes.map(JSON.stringify).join('\n') + '\n';
  await writeFile(join(out, 'crossword-episodes.jsonl'), body, { flag: 'wx' });
  const manifest = { schema: 'natlang.crossword-csp-corpus/1', episodes: episodes.length, cases: audit.length * 2,
    families: [...new Set(audit.map(row => row.family))], sha256: digest(body), provider_calls: 0,
    publication: 'Project-authored task candidates only; not generated or admitted model trajectories', references: audit };
  await writeFile(join(out, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n', { flag: 'wx' });
  console.log(JSON.stringify({ ...manifest, references: undefined }));
}
