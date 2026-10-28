/**
 * `natlang tune <program> --learned <spec.json>`: the trained joint learned optimizer proposes the program's native
 * soft prompts, soft skills and LoRA mixtures (plans/neuralese/JOINT_LEARNED_OPTIMIZER.md). It runs
 * the earned Natlang controller with the ordinary `tune.build` plan/result contract, so the ordinary pipeline
 * validates, activates and rolls back its release exactly as for a joint fit.
 *
 * Spec: `{ "bank": "<bank.json>", "state": "<optimizer state.json>", "store": "<operation artifact store>",
 *          "scopes": { "tree": 24, "folder": 24, "lambda": 32 }, "steps": 4, "patience": 2, "seed": 0,
 *          "incoming"?: "<tune build directory>", "controller": { "root": "<controller program>",
 *          "release": "<earned release digest>" } }`. The explicit `policy: "crisp-baseline"` selects the Python baseline.
 * `incoming` names a joint fit on the bank's basis whose tuned bundle the optimizer revises (and releases unchanged
 * when nothing improves on it). The job key binds the bank, the optimizer state and the incoming fit's state contents:
 * a retrained optimizer or a continued fit never reuses an older job.
 *
 * `natlang tune <program> --score-proposals <request.json>`: real task outcomes for the optimizer's outer meta-RL.
 * `python -m natlang_neuralese.tune.learned proposals` writes each sampled proposal, and each target's incoming bundle,
 * as a release fragment, and one request per program; this seals each fragment as a release of the program (stored,
 * not activated), runs it on the target's query roots (training roots outside its support, else the development
 * roots) with the program's own scoring (tune/reward.ts), and writes `{ bundle: { before, after, roots, partition } }`
 * beside the request (`<request>.outcomes.json`), which `collect` takes as `outcomes`.
 */
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { readRelease, releaseFile } from '../adaptation/weights/release.js';
import { WeightBinding } from '../adaptation/weights/binding.js';
import { createNatlangRuntime } from '../runtime/runtime.js';
import { loadNatlang } from '../runtime/node-files.js';
import { tuneRuntime } from './execution.js';
import { armSummary } from './evaluate.js';
import { evaluateRelease, exportVllmStacks, prepareData, releaseFromBuild, trainRelease, type ReleaseSelector, type TuneOptions } from './pipeline.js';

export type LearnedSettings = {
  readonly bank: string; readonly state: string; readonly store: string;
  readonly scopes: Readonly<Record<'tree' | 'folder' | 'lambda', number>>;
  readonly steps?: number; readonly patience?: number; readonly seed?: number; readonly incoming?: string;
  readonly controller?: { readonly root: string; readonly release: string };
  readonly policy?: 'controller' | 'crisp-baseline';
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
  if (spec.policy !== undefined && !['controller', 'crisp-baseline'].includes(spec.policy))
    throw new Error('learned optimizer policy must be controller or crisp-baseline');
  if (spec.policy !== 'crisp-baseline' && (!spec.controller || typeof spec.controller.root !== 'string' ||
      typeof spec.controller.release !== 'string' || !spec.controller.release))
    throw new Error('learned optimizer needs an earned controller root and release; crisp-baseline is an explicit baseline');
  if (spec.incoming !== undefined && (typeof spec.incoming !== 'string' || !spec.incoming))
    throw new Error('learned optimizer spec "incoming" is a tune build directory');
  return { ...spec, bank: resolve(base, spec.bank!), state: resolve(base, spec.state!), store: resolve(base, spec.store!), scopes,
    ...(spec.incoming ? { incoming: resolve(base, spec.incoming) } : {}),
    ...(spec.controller ? { controller: { ...spec.controller, root: resolve(base, spec.controller.root) } } : {}) } as LearnedSettings;
}

export function learnedSelector(options: TuneOptions, settings: LearnedSettings, controllerOptions?: TuneOptions): ReleaseSelector {
  return async (data, timings, log) => {
    const learned = { ...settings, bindings: { bankSha256: fileSha(settings.bank), stateSha256: fileSha(settings.state),
      ...(settings.incoming ? { incomingSha256: fileSha(resolve(settings.incoming, 'state.json')) } : {}) } };
    const execute = settings.policy === 'crisp-baseline' ? undefined : async (plan: string, out: string) => {
      if (!controllerOptions || !settings.controller) throw new Error('controller execution requires its own program configuration');
      const controller = settings.controller, manifest = readRelease(controller.root, controller.release);
      if (manifest.program.id !== controllerOptions.program.id ||
          !(manifest.artifacts.length || manifest.families.length || manifest.softStates.length))
        throw new Error('controller release must contain trained adaptation for the named controller program');
      const directory = dirname(releaseFile(controller.root, controller.release));
      const served = await controllerOptions.serve({ manifest, directory });
      const entry = options.student.trainer, moduleIndex = entry.indexOf('-m');
      if (moduleIndex < 0) throw new Error('controller service requires a Python -m trainer');
      const module = await import(pathToFileURL(resolve(controller.root, 'python.mjs')).href);
      const python = new module.PythonService({ argv: entry.slice(0, moduleIndex + 1), cwd: options.student.workdir,
        env: options.student.env });
      try {
        const services = new module.AdaptationServices(python, plan, out);
        const weights = [new WeightBinding(manifest, { program: controllerOptions.program, executor: controllerOptions.student.base, available: served.loaded,
          nativeState: !!served.nativeState, strict: true })];
        const runtime = createNatlangRuntime({ program: controllerOptions.program,
          ...await tuneRuntime(served.driver, controllerOptions.settings.execution), weights,
          services: { optimizer: services.optimizer }, serviceDeclarations: { optimizer: module.optimizerDeclaration } });
        try {
          const adapt = loadNatlang(resolve(controller.root, 'adapt.nl'), controller.root);
          const state = await runtime.run(() => adapt(settings.steps ?? 4)) as { current?: { id: string } | null; history?: unknown[] };
          await services.finish(state.current?.id ?? null, state.history ?? [], { root: controller.root, release: controller.release });
        } finally { runtime.close(); }
      } finally { await python.close(); await served.close(); }
    };
    const { manifest } = await trainRelease(options, data, { manifest: data.scaffold, label: 'learned-optimizer', execute,
      module: 'natlang_neuralese.tune.learned.apply', learned, receipts: { build: 'learned-optimizer', learned: learned.bindings } },
    timings, log);
    log(`natlang tune: learned optimizer proposed release ${manifest.digest.slice(0, 12)}`);
    return manifest;
  };
}

type ScoreRequest = { schema: 'natlang.lopt.score-request/1'; root: string; targets: {
  target: string; support: string[]; incoming: string; scaffold?: string | null; proposals: { bundle: string; release: string }[] }[] };

export async function scoreProposals(options: TuneOptions, requestPath: string, log: (line: string) => void = () => {}): Promise<string> {
  const request = JSON.parse(readFileSync(requestPath, 'utf8')) as ScoreRequest;
  if (request.schema !== 'natlang.lopt.score-request/1') throw new Error(`${requestPath} is not a learned-optimizer score request`);
  if (resolve(request.root) !== resolve(options.root)) throw new Error(`${requestPath} scores ${request.root}, not ${options.root}`);
  const data = await prepareData(options, [], log);
  const quality = new Map<string, { value: number; roots: number }>();
  const score = async (release: string, examples: Parameters<typeof evaluateRelease>[3], partition: string, target: string) => {
    const key = `${release}\n${examples.map(example => example.id).join(',')}`;
    if (!quality.has(key)) {
      const manifest = releaseFromBuild(options, data, { manifest: data.scaffold, label: 'learned-proposal',
        receipts: { build: 'learned-proposal', target } }, release, fileSha(resolve(release, 'result.json')).slice(0, 24), {});
      if (options.settings.backend === 'vllm') await exportVllmStacks(options, manifest);
      const summary = armSummary(await evaluateRelease(options, data, manifest, examples, 'learned-proposal', partition, { save: false }));
      quality.set(key, { value: summary.meanScore ?? summary.success, roots: summary.roots });
    }
    return quality.get(key)!;
  };
  const outcomes: Record<string, { before: number; after: number; roots: number; partition: string }> = {};
  for (const entry of request.targets) {
    if (entry.scaffold && entry.scaffold !== data.scaffold.digest)
      throw new Error(`target ${entry.target} was built for another scaffold of ${options.root}; rebuild the optimizer's targets`);
    const support = new Set(entry.support);
    let partition = 'train', examples = data.examples.filter(example => example.partition === 'train' && !support.has(example.id));
    if (!examples.length) { partition = 'development'; examples = data.examples.filter(example => example.partition === 'development'); }
    if (!examples.length) { log(`natlang tune: no query roots for ${entry.target}; its task outcome stays absent`); continue; }
    const before = await score(entry.incoming, examples, partition, entry.target);
    for (const proposal of entry.proposals) {
      const after = await score(proposal.release, examples, partition, entry.target);
      outcomes[proposal.bundle] = { before: before.value, after: after.value, roots: after.roots, partition };
      log(`natlang tune: proposal ${proposal.bundle.slice(0, 16)} scores ${after.value.toFixed(3)} (incoming ${before.value.toFixed(3)})`);
    }
  }
  const out = `${requestPath}.outcomes.json`;
  writeFileSync(out, JSON.stringify(outcomes, null, 1));
  return out;
}
