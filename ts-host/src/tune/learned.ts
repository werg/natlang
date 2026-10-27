/**
 * `natlang tune <program> --learned <spec.json>`: the trained joint learned optimizer proposes the program's native
 * soft prompts, soft skills and LoRA mixtures (plans/neuralese/JOINT_LEARNED_OPTIMIZER.md). It runs
 * `natlang_neuralese.tune.learned.apply` with the ordinary `tune.build` plan/result contract, so the ordinary pipeline
 * validates, activates and rolls back its release exactly as for a joint fit.
 *
 * Spec: `{ "bank": "<bank.json>", "state": "<optimizer state.json>", "store": "<operation artifact store>",
 *          "scopes": { "tree": 24, "folder": 24, "lambda": 32 }, "steps": 4, "patience": 2, "seed": 0,
 *          "incoming"?: "<tune build directory>" }`.
 * `incoming` names a joint fit on the bank's basis whose tuned bundle the optimizer revises (and releases unchanged
 * when nothing improves on it). The job key binds the bank, the optimizer state and the incoming fit's state contents:
 * a retrained optimizer or a continued fit never reuses an older job.
 */
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { trainRelease, type ReleaseSelector, type TuneOptions } from './pipeline.js';

export type LearnedSettings = {
  readonly bank: string; readonly state: string; readonly store: string;
  readonly scopes: Readonly<Record<'tree' | 'folder' | 'lambda', number>>;
  readonly steps?: number; readonly patience?: number; readonly seed?: number; readonly incoming?: string;
};

const LEVELS = ['tree', 'folder', 'lambda'] as const;
const fileSha = (path: string) => createHash('sha256').update(readFileSync(path)).digest('hex');

export function readLearnedSettings(path: string): LearnedSettings {
  const spec = JSON.parse(readFileSync(path, 'utf8')) as Partial<LearnedSettings>;
  for (const key of ['bank', 'state', 'store'] as const)
    if (typeof spec[key] !== 'string' || !spec[key]) throw new Error(`learned optimizer spec needs a "${key}" path`);
  const scopes = spec.scopes ?? {} as LearnedSettings['scopes'];
  if (!Object.keys(scopes).length || Object.entries(scopes).some(([level, size]) =>
    !(LEVELS as readonly string[]).includes(level) || !Number.isSafeInteger(size) || (size as number) < 1))
    throw new Error('learned optimizer spec "scopes" maps tree/folder/lambda to positive bank-view sizes');
  const base = resolve(path, '..');
  if (spec.incoming !== undefined && (typeof spec.incoming !== 'string' || !spec.incoming))
    throw new Error('learned optimizer spec "incoming" is a tune build directory');
  return { ...spec, bank: resolve(base, spec.bank!), state: resolve(base, spec.state!), store: resolve(base, spec.store!), scopes,
    ...(spec.incoming ? { incoming: resolve(base, spec.incoming) } : {}) } as LearnedSettings;
}

export function learnedSelector(options: TuneOptions, settings: LearnedSettings): ReleaseSelector {
  return async (data, timings, log) => {
    const learned = { ...settings, bindings: { bankSha256: fileSha(settings.bank), stateSha256: fileSha(settings.state),
      ...(settings.incoming ? { incomingSha256: fileSha(resolve(settings.incoming, 'state.json')) } : {}) } };
    const { manifest } = await trainRelease(options, data, { manifest: data.scaffold, label: 'learned-optimizer',
      module: 'natlang_neuralese.tune.learned.apply', learned, receipts: { build: 'learned-optimizer', learned: learned.bindings } },
    timings, log);
    log(`natlang tune: learned optimizer proposed release ${manifest.digest.slice(0, 12)}`);
    return manifest;
  };
}
