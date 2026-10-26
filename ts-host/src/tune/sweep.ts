/**
 * Rank/penalty/capacity sweeps (plans/HIERARCHICAL_LORA_TUNING.md §5.6). A spec crosses structural arms, trainer
 * recipes (fields of natlang_neuralese.tune.trainer.Recipe), seeds and training-data sizes counted in independent
 * root invocations. Each cell trains through the normal train stage (its own receipt), is measured on DEVELOPMENT roots
 * only, and is journaled; rerunning resumes after the last finished cell. Selection reads development results only.
 *
 * Structural arms map onto recipe ranks (a level with rank 0 gets no adapter):
 *   hierarchy        the recipe as given (tree, folder and lambda nodes)
 *   flat-lambda      independent per-lambda adapters only (tree and folder rank 0)
 *   program-single   one program adapter only (folder and lambda rank 0)
 *   full-finetune    per-lambda full fine-tuning: recorded as unsupported until the trainer implements it
 */
import { createHash } from 'node:crypto';
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { tuneDirectory } from '../adaptation/weights/release.js';
import { armSummary } from './evaluate.js';
import { evaluateRelease, saveArm, trainRelease, type TuneData, type TuneOptions } from './pipeline.js';
import { servingCost } from './placement.js';
import { DEFAULT_PLACEMENT } from './settings.js';

export type SweepArm = 'hierarchy' | 'flat-lambda' | 'program-single' | 'full-finetune';
export type SweepSpec = { readonly name: string; readonly arms?: readonly SweepArm[];
  readonly recipes: readonly { readonly name: string; readonly recipe: Readonly<Record<string, unknown>> }[];
  readonly sizes?: readonly (number | 'full')[]; readonly seeds?: readonly number[] };
export type SweepCell = { readonly key: string; readonly arm: SweepArm; readonly recipe: string; readonly size: number | 'full'; readonly seed: number;
  readonly status: 'done' | 'skipped' | 'unsupported' | 'failed'; readonly reason?: string; readonly release?: string;
  readonly roots?: number; readonly components?: number; readonly rows?: number;
  readonly development?: { readonly success: number; readonly roots: number; readonly completed: number };
  readonly cost?: number; readonly ms?: number };

export const DEFAULT_SIZES: readonly (number | 'full')[] = [8, 32, 128, 512, 'full'];

type Ranks = Record<string, number>;
function armRecipe(arm: SweepArm, recipe: Readonly<Record<string, unknown>>, seed: number): Record<string, unknown> {
  const ranks: Ranks = { tree: 8, folder: 4, lambda: 2, ...(recipe.ranks as Ranks | undefined) };
  const shaped = arm === 'flat-lambda' ? { ...ranks, tree: 0, folder: 0 } : arm === 'program-single' ? { ...ranks, folder: 0, lambda: 0 } : ranks;
  return { ...recipe, ranks: shaped, seed };
}

export function readJournal(path: string): SweepCell[] {
  return existsSync(path) ? readFileSync(path, 'utf8').split('\n').filter(Boolean).map(line => JSON.parse(line) as SweepCell) : [];
}

export async function runSweep(options: TuneOptions, data: TuneData, spec: SweepSpec, log: (line: string) => void = () => {}):
  Promise<{ journal: string; cells: SweepCell[]; selected: SweepCell | null }> {
  if (!spec.name || /[^\w.-]/.test(spec.name)) throw new Error('a sweep spec needs a name of letters, digits, dot, dash or underscore');
  if (!spec.recipes?.length) throw new Error('a sweep spec needs at least one recipe');
  const directory = join(tuneDirectory(options.root), 'sweeps', spec.name);
  mkdirSync(directory, { recursive: true });
  const journal = join(directory, 'journal.jsonl');
  const finished = new Map(readJournal(journal).filter(cell => cell.status !== 'failed').map(cell => [cell.key, cell]));
  const development = data.examples.filter(example => example.partition === 'development');
  if (!development.length) throw new Error('a sweep needs development roots for selection');
  // Training roots in a fixed order of their component, so a smaller size is a prefix of a larger one.
  const order = (value: string) => createHash('sha256').update('natlang.tune-sweep/v1\0' + value).digest('hex');
  const trainRoots = data.examples.filter(example => example.partition === 'train')
    .sort((a, b) => order(a.component).localeCompare(order(b.component)) || a.id.localeCompare(b.id));
  const cells: SweepCell[] = [];
  for (const arm of spec.arms ?? ['hierarchy', 'flat-lambda', 'program-single', 'full-finetune'] as SweepArm[])
    for (const { name, recipe } of spec.recipes) for (const seed of spec.seeds ?? [0]) for (const size of spec.sizes ?? DEFAULT_SIZES) {
      const key = JSON.stringify([arm, name, seed, size, data.dataKey]);
      const done = finished.get(key);
      if (done) { cells.push(done); continue; }
      const write = (cell: SweepCell) => { appendFileSync(journal, JSON.stringify(cell) + '\n'); cells.push(cell); };
      const base = { key, arm, recipe: name, size, seed };
      if (arm === 'full-finetune') { write({ ...base, status: 'unsupported', reason: 'the shared trainer has no full per-lambda fine-tuning arm yet' }); continue; }
      const chosen = size === 'full' ? trainRoots : trainRoots.slice(0, size);
      if (size !== 'full' && chosen.length < size) { write({ ...base, status: 'skipped', reason: `only ${trainRoots.length} independent training roots` }); continue; }
      const ids = new Set(chosen.map(example => example.id));
      const rows = data.trainRows.filter(row => ids.has(row.root));
      const started = Date.now();
      try {
        log(`natlang tune: sweep ${arm} ${name} seed ${seed} size ${size}`);
        const { manifest } = await trainRelease(options, data, { manifest: data.scaffold, recipe: armRecipe(arm, recipe, seed),
          rows, label: `sweep:${spec.name}:${arm}:${name}:${seed}:${size}` }, [], log);
        const result = await evaluateRelease(options, data, manifest, development, `sweep-${spec.name}-${arm}-${name}-${seed}-${size}`, 'development');
        saveArm(options.root, result);
        const summary = armSummary(result);
        write({ ...base, status: 'done', release: manifest.digest, roots: chosen.length, components: new Set(chosen.map(example => example.component)).size,
          rows: rows.length, development: { success: summary.success, roots: summary.roots, completed: summary.completed },
          cost: servingCost(options.root, manifest, result, (options.settings.placement ?? DEFAULT_PLACEMENT).costWeights).total, ms: Date.now() - started });
      } catch (error) {
        write({ ...base, status: 'failed', reason: (error instanceof Error ? error.message : String(error)).slice(0, 1000), ms: Date.now() - started });
      }
    }
  const ranked = cells.filter(cell => cell.status === 'done')
    .sort((a, b) => b.development!.success - a.development!.success || (a.cost ?? 0) - (b.cost ?? 0));
  const selected = ranked[0] ?? null;
  writeFileSync(join(directory, 'selection.json'), JSON.stringify({ schema: 'natlang.tune-sweep-selection/v1', selected,
    rule: 'highest development success, then lowest serving cost', cells: cells.length }, null, 2));
  return { journal, cells, selected };
}
