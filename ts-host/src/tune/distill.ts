/**
 * Distillation exports (plans/HIERARCHICAL_LORA_TUNING.md §5.7): verified training-split outputs of a tuned stack,
 * together with the teacher's training rows, as one deduplicated corpus for the student's shared production SFT
 * recipe. Only training roots with a root oracle are re-run, only oracle-correct runs of the tuned stack are kept, and
 * development, confirmation and protected roots never enter. The export is conversion, not admission: registering and
 * admitting it (`training/neuralese_corpora.json`) is an explicit operations decision. After the base is retrained,
 * rerunning `natlang tune` against the new exact base rebuilds every adapter; no tensor coordinates carry over.
 *
 *   <program>/.natlang/tune/distill/<release>/turns.jsonl      natlang.tune-turn/v1, partition "train" only
 *   <program>/.natlang/tune/distill/<release>/manifest.json    hashes, counts by source, split receipt and identities
 *   then `python -m natlang_neuralese.tune.distill --export <dir> --checkpoint <student>` writes native-graphs.jsonl
 */
import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { tuneDirectory } from '../adaptation/weights/release.js';
import { WeightBinding } from '../adaptation/weights/binding.js';
import type { WeightManifest } from '../adaptation/weights/manifest.js';
import { collect, type CollectedTurn } from './collect.js';
import type { TuneData, TuneOptions } from './pipeline.js';

const sha = (value: unknown) => createHash('sha256').update(typeof value === 'string' ? value : JSON.stringify(value)).digest('hex');

export type DistillManifest = { readonly schema: 'natlang.tune-distill/v1'; readonly program: { readonly id: string; readonly buildHash: string };
  readonly release: string; readonly base: WeightManifest['base']; readonly student: string; readonly split: string;
  readonly turns: { readonly path: 'turns.jsonl'; readonly sha256: string; readonly rows: number;
    readonly bySource: { readonly teacher: number; readonly stack: number }; readonly duplicates: number };
  readonly roots: { readonly rerun: number; readonly correct: number };
  readonly admission: 'not-admitted' };

export async function exportDistillation(options: TuneOptions, data: TuneData, release: WeightManifest,
  log: (line: string) => void = () => {}): Promise<{ directory: string; manifest: DistillManifest }> {
  const roots = data.examples.filter(example => example.partition === 'train' && !example.protected && 'expected' in example);
  if (!roots.length) throw new Error('distillation re-runs training roots with a root oracle; this program has none');
  const directory = join(tuneDirectory(options.root), 'releases', release.digest);
  const served = await options.serve({ manifest: release, directory });
  let stack: CollectedTurn[];
  let correct = 0;
  try {
    const weights = new WeightBinding(release, { program: options.program, available: served.loaded, nativeState: !!served.nativeState, strict: true });
    log(`natlang tune: distillation re-runs ${roots.length} training roots with release ${release.digest.slice(0, 12)}`);
    const collected = await collect({ root: options.root, program: options.program, scaffold: release, examples: roots, partitions: ['train'], services: options.services, execution: options.settings.execution,
      teacher: served.driver, teacherId: `student-stack:${release.base.checkpoint_sha256}:${release.digest}`, weights });
    correct = collected.roots.filter(root => root.oracle === true).length;
    // Reverified: the tuned stack's turns count only when its whole root run met the oracle.
    stack = collected.turns.filter(turn => turn.evidence.rootOracle === true && turn.evidence.actionDisposition === 'admissible' &&
      turn.binding && !turn.reply.truncated);
  } finally { await served.close(); }
  const teacher = data.trainRows.filter(turn => turn.evidence.rootOracle !== false && turn.evidence.actionDisposition === 'admissible');
  const seen = new Set<string>(), rows: CollectedTurn[] = [];
  let duplicates = 0;
  for (const turn of [...teacher, ...stack]) {
    const key = sha([turn.binding, turn.messages, turn.tools, turn.reply]);
    if (seen.has(key)) { duplicates++; continue; }
    seen.add(key); rows.push(turn);
  }
  if (rows.some(turn => turn.partition !== 'train')) throw new Error('distillation admits training-split rows only');
  const out = join(tuneDirectory(options.root), 'distill', release.digest);
  mkdirSync(out, { recursive: true });
  const text = rows.map(turn => JSON.stringify(turn)).join('\n') + '\n';
  writeFileSync(join(out, 'turns.jsonl.tmp'), text); renameSync(join(out, 'turns.jsonl.tmp'), join(out, 'turns.jsonl'));
  const manifest: DistillManifest = { schema: 'natlang.tune-distill/v1', program: { id: options.program.id, buildHash: options.program.buildHash },
    release: release.digest, base: release.base, student: options.student.name,
    split: sha(readFileSync(join(tuneDirectory(options.root), 'splits.json'), 'utf8')),
    turns: { path: 'turns.jsonl', sha256: sha(text), rows: rows.length, duplicates,
      bySource: { teacher: rows.filter(turn => !turn.teacher.startsWith('student-stack:')).length, stack: rows.filter(turn => turn.teacher.startsWith('student-stack:')).length } },
    roots: { rerun: roots.length, correct }, admission: 'not-admitted' };
  writeFileSync(join(out, 'manifest.json'), JSON.stringify(manifest, null, 1) + '\n');
  return { directory: out, manifest };
}
