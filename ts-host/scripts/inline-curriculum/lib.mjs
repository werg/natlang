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
  failureSeed, split = 'train' }) {
  const id = `inline-curriculum:${family}:${shape}:${variant}`;
  const semantics = { root: `${root.name}.nl`, files: { [`${root.name}.nl`]: nlFile(root), ...files }, inputs, expected,
    ...(operation ? { operation } : {}), ...(folderFiles ? { folder_files: folderFiles } : {}),
    ...(expectedFiles ? { expected_files: expectedFiles } : {}), ...(failureSeed ? { failure_seed: failureSeed } : {}) };
  return { version: PROGRAM_VERSION, id, kind: 'lambda_source', family: `curriculum_${family}`,
    source: 'natlang-inline-curriculum', split, source_ids: [id], source_groups: [splitGroup ?? `${family}:${shape}`],
    source_revisions: [GENERATOR_VERSION], license: 'project-generated', gold_sources: ['constructed-world-oracle'],
    generation: { generator: GENERATOR_VERSION },
    curriculum: { version: CURRICULUM_VERSION, family, family_version: familyVersion, shape, variant,
      pair_group: pairGroup, split_group: splitGroup ?? `${family}:${shape}`, slice, domain, mode, inline,
      ...(edits ? { edits } : {}), ...(named ? { named } : {}), ...(iterate ? { iterate } : {}),
      ...(worldSemantics ? { world_semantics: worldSemantics } : {}),
      evidence: { world: evidence.world ?? [], retrieved: evidence.retrieved ?? [], background: evidence.background ?? [] },
      assumptions, decisive, plausible_actions: plausibleActions, minimum_sequence: minimumSequence, reference },
    semantics };
}

/** Reference-solution call shorthands. */
export const evalCall = code => ['eval', { code }];
export const returnCall = value => ['return_result', { value }];
export const blockedCall = missing => ['blocked', { missing }];
export const failedCall = message => ['failed', { message }];
