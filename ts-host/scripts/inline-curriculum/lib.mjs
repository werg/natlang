// Shared construction helpers for the inline-natlang curriculum families.
import { createHash } from 'node:crypto';
import { PROGRAM_VERSION } from '../../dist/teacher/program.js';
import { CURRICULUM_VERSION } from '../../dist/teacher/curriculum.js';

export const GENERATOR_VERSION = 'natlang.inline_curriculum_generator/1';

/** Deterministic PRNG derived from (seed, key). */
export class Random {
  constructor(seed, key) {
    const digest = createHash('sha256').update(`${seed}:${key}:${GENERATOR_VERSION}`).digest();
    this.state = digest.readUInt32LE(0) || 0x9e3779b9;
  }
  next() {
    let x = this.state;
    x ^= x << 13; x ^= x >>> 17; x ^= x << 5;
    this.state = x >>> 0;
    return this.state / 0x1_0000_0000;
  }
  int(min, max) { return min + Math.floor(this.next() * (max - min + 1)); }
  pick(values) { return values[this.int(0, values.length - 1)]; }
  shuffle(values) {
    const out = [...values];
    for (let i = out.length - 1; i > 0; i--) { const j = this.int(0, i); [out[i], out[j]] = [out[j], out[i]]; }
    return out;
  }
  sample(values, count) { return this.shuffle(values).slice(0, count); }
}

/** Pronounceable nonce words, so a world's local rules cannot be answered from real-world knowledge. */
export function nonceWords(rng, count, used = new Set()) {
  const onsets = ['b', 'bl', 'd', 'dr', 'f', 'g', 'gl', 'k', 'kr', 'l', 'm', 'n', 'p', 'pr', 's', 'sk', 't', 'tr', 'v', 'z'];
  const vowels = ['a', 'e', 'i', 'o', 'u', 'ai', 'oo'];
  const codas = ['b', 'd', 'g', 'k', 'l', 'm', 'n', 'p', 'r', 's', 't', 'x', 'sh', 'ng'];
  const out = [];
  while (out.length < count) {
    const word = rng.pick(onsets) + rng.pick(vowels) + rng.pick(codas) + (rng.next() < 0.4 ? rng.pick(vowels) + rng.pick(codas) : '');
    if (!used.has(word)) { used.add(word); out.push(word); }
  }
  return out;
}
export const capitalize = word => word[0].toUpperCase() + word.slice(1);

/** A `.nl` source file. */
export function nlFile({ args = {}, returns, instructions, kind, description }) {
  const argLines = Object.entries(args).map(([name, type]) => `  ${name}: ${JSON.stringify(type)}`);
  return `---\n${description ? `description: ${JSON.stringify(description)}\n` : ''}` +
    `args:${argLines.length ? '\n' + argLines.join('\n') : ' {}'}\nreturns: ${JSON.stringify(returns)}\n` +
    `${kind ? `kind: ${kind}\n` : ''}---\n${instructions.trim()}\n`;
}

/** A JSON literal for embedding data in a TypeScript module. */
export const literal = value => JSON.stringify(value, null, 1).replace(/\n\s*/g, ' ');

/**
 * One curriculum case. `root` is `{ name, args, returns, instructions, kind? }`; `files` are the other
 * project files (callable folder items and `types.ts`).
 */
export function curriculumCase({ family, familyVersion = 1, shape, variant, pairGroup = null, splitGroup, slice, domain,
  mode, inline = 'optional', edits, named, iterate, worldSemantics, evidence = {}, assumptions = [], decisive = [], plausibleActions = [],
  minimumSequence = [], reference, root, files = {}, inputs = {}, expected = null, operation, folderFiles, expectedFiles,
  failureSeed, split = 'train', sketch }) {
  const id = `inline-curriculum:${family}:${shape}:${variant}`;
  const semantics = { root: `${root.name}.nl`, files: { [`${root.name}.nl`]: nlFile(root), ...files }, inputs, expected,
    ...(operation ? { operation } : {}), ...(folderFiles ? { folder_files: folderFiles } : {}),
    ...(expectedFiles ? { expected_files: expectedFiles } : {}), ...(failureSeed ? { failure_seed: failureSeed } : {}) };
  externalize(semantics, family);
  return { version: PROGRAM_VERSION, id, kind: 'lambda_source', family: `curriculum_${family}`,
    source: 'natlang-inline-curriculum', split, source_ids: [id], source_groups: [splitGroup ?? `${family}:${shape}`],
    source_revisions: [GENERATOR_VERSION], license: 'project-generated', gold_sources: ['constructed-world-oracle'],
    generation: { generator: GENERATOR_VERSION },
    curriculum: { version: CURRICULUM_VERSION, family, family_version: familyVersion, shape, variant,
      pair_group: pairGroup, split_group: splitGroup ?? `${family}:${shape}`, slice, domain, mode, inline,
      ...(edits ? { edits } : {}), ...(named ? { named } : {}), ...(iterate ? { iterate } : {}), ...(sketch ? { sketch } : {}),
      ...(worldSemantics ? { world_semantics: worldSemantics } : {}),
      evidence: { world: evidence.world ?? [], retrieved: evidence.retrieved ?? [], background: evidence.background ?? [] },
      assumptions, decisive, plausible_actions: plausibleActions, minimum_sequence: minimumSequence, reference },
    semantics };
}

/**
 * Modules a program owns and may change: a helper it was given, code it is asked to repair, or modules whose scoping
 * the task is about. Every other module a case calls stands for something outside the program (a board, a world, a
 * store, a checker) and becomes an external service: the model reads its declaration and calls it, but cannot read or
 * edit its implementation (native/external.ts). A module in a function's own directory is scoped to that function:
 * only its calls, and the calls they make, can use it. A module that imports another, or is itself called (a default
 * export), stays a file.
 */
const PROGRAM_MODULES = new Set(['review_each', 'line_total']);
const PROGRAM_FAMILIES = new Set(['scoped_module_discovery']);
export function externalize(semantics, family) {
  if (PROGRAM_FAMILIES.has(family)) return semantics;
  const modules = Object.keys(semantics.files).filter(path => path.endsWith('.ts') && !path.endsWith('types.ts'));
  const name = path => path.split('/').pop().replace(/\.ts$/, '');
  for (const path of modules) {
    const source = semantics.files[path];
    if (PROGRAM_MODULES.has(name(path)) || /^import |^export default /m.test(source) || modules.filter(other => name(other) === name(path)).length > 1) continue;
    (semantics.services ??= {})[name(path)] = source;
    delete semantics.files[path];
    // A module inside a function's own directory (answer_question/table_expert/tables.ts) is that function's to use.
    const owner = `${path.split('/').slice(0, -1).join('/')}.nl`;
    if (owner !== `${semantics.root.replace(/\.nl$/, '')}.nl` && semantics.files[owner]) (semantics.service_scopes ??= {})[name(path)] = [owner];
  }
  return semantics;
}

/** Reference-solution call shorthands. */
export const evalCall = code => ['eval', { code }];
export const returnCall = value => ['return_result', { status: 'success', value }];
export const blockedCall = reason => ['return_result', { status: 'blocked', reason }];
export const failedCall = reason => ['return_result', { status: 'failed', reason }];

export const ITERATE_HINT = 'Required: solve this with iterateOn, not with evals or loops that you step by hand. Write a step function (it may be an nl function) from the current state to the next state, and run the whole process in one eval with await step.iterateOn(initial).withLimit({maxSteps: 128}).until(done).';
export const INLINE_HINT = 'Required: make each judgment about an item with a natural-language function called on that item (await nl`...`(item)), not with keyword or regular-expression matching.';
/** The hint for a case that requires a technique: the technique's requirement, then the family's sketch if it has one. */
export function hintFor(curriculum) {
  const parts = [...(curriculum.iterate === 'required' ? [ITERATE_HINT] : []), ...(curriculum.inline === 'required' ? [INLINE_HINT] : [])];
  if (!parts.length) return null;
  return [...parts, ...(curriculum.sketch ? [`Sketch: ${curriculum.sketch}`] : [])].join(' ');
}
export function hinted(record, hint) {
  const twin = structuredClone(record);
  twin.id = `${record.id}:hinted`;
  twin.source_ids = [twin.id];
  const root = twin.semantics.root;
  twin.semantics.files[root] = twin.semantics.files[root].replace(/\n$/, '') + `\n\n${hint}\n`;
  twin.curriculum.hint = hint;
  twin.curriculum.hinted_of = record.id;
  if (twin.curriculum.pair_group) twin.curriculum.pair_group = `${twin.curriculum.pair_group}:hinted`;
  return twin;
}
