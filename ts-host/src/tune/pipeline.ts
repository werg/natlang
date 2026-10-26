/**
 * `natlang tune <program>` (plans/HIERARCHICAL_LORA_TUNING.md §1.1, §8): discover, collect/reuse data, train, export,
 * load on the backend, validate through the normal engine and activate atomically. Success means ready to call.
 *
 * Every stage is keyed by its inputs and leaves a receipt under `.natlang/tune/jobs/<key>/`; rerunning the same command
 * resumes and reuses valid work. Activation compares the active pointer first, so an older job finishing late cannot
 * overwrite a newer release. A failed activation leaves the previous release active and says why.
 */
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { copyFileSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import type { ProgramDescriptor } from '../adaptation/types.js';
import { sealManifest, type WeightArtifact, type WeightManifest, type FamilyVersion, type FamilyMembership, type SoftState } from '../adaptation/weights/manifest.js';
import { stackServingName, WeightBinding } from '../adaptation/weights/binding.js';
import { activateRelease, readActive, readRelease, releaseFile, storeRelease, tuneDirectory, tuneJobLock } from '../adaptation/weights/release.js';
import type { ModelDriver, Services } from '../runtime/runtime.js';
import { sharedSourceHash, rootDataValid, trainingScaffoldKey } from './keys.js';
import { programSites } from './sites.js';
import { collect, readRootExamples, scaffoldManifest, splitExamples, type CollectedTurn, type SplitExample, type SplitReceipt } from './collect.js';
import { evaluateRoots, type ArmResult } from './evaluate.js';
import type { ScopeOptions } from '../adaptation/weights/scopes.js';
import type { StudentSettings, TuneSettings } from './settings.js';

export type StageTiming = { stage: string; ms: number; reused?: boolean };
export type TuneReport = {
  schema: 'natlang.tune-report/v1'; program: string; student: string; backend: string;
  release: string | null; activated: boolean; reason?: string;
  data: { examples: number; protected: number; components: Record<string, number>; turns: number; trainTurns: number };
  validation: { tuned: { roots: number; ok: number; oracle: number }; base: { roots: number; ok: number; oracle: number } } | null;
  timings: StageTiming[]; totalMs: number; manualActions: 0;
  /** What kind of build produced the release (§8): a full joint fit, an incremental candidate or a shared refit. */
  build?: string;
};

export type TuneOptions = {
  root: string; program: ProgramDescriptor; settings: TuneSettings; student: StudentSettings & { name: string };
  /** The authorized teacher's driver and identity. */
  teacher: ModelDriver; teacherId: string;
  /** Serves a release for validation: returns a driver that applies the release's adapters, and its loaded set. */
  /** `capture` asks the backend for exact sampling receipts on every turn (RL rollouts; native backend only). */
  serve: (release: { manifest: WeightManifest; directory: string; capture?: boolean }) => Promise<{ driver: ModelDriver; loaded: ReadonlySet<string>;
    /** Whether the backend imports stored native state (soft instruction/skill states). */
    nativeState?: boolean; close(): Promise<void> }>;
  examplesPath?: string; log?: (line: string) => void;
  /** Host services for one root invocation, fresh per root: a test or sandbox world (SDK hosts; the CLI passes none).
   * Collection journals every call so repairs replay the same world (world.ts). */
  services?: (example: SplitExample) => Services;
};

/** Persist an evaluated arm for reports: `.natlang/tune/evaluations/<partition>/<arm>.json`. */
export function saveArm(root: string, arm: ArmResult): void {
  const directory = join(tuneDirectory(root), 'evaluations', arm.partition);
  mkdirSync(directory, { recursive: true });
  writeFileSync(join(directory, `${arm.arm.replace(/[^\w.-]+/g, '_')}.json`), JSON.stringify(arm, null, 2));
}

const sha = (value: unknown) => createHash('sha256').update(typeof value === 'string' ? value : JSON.stringify(value)).digest('hex');

async function timed<T>(timings: StageTiming[], stage: string, log: (line: string) => void, fn: () => Promise<T> | T, reused = false): Promise<T> {
  const started = Date.now();
  log(`natlang tune: ${stage}${reused ? ' (reusing earlier work)' : ''}`);
  try { return await fn(); } finally { timings.push({ stage, ms: Date.now() - started, ...(reused ? { reused } : {}) }); }
}

function runTrainer(student: StudentSettings, plan: string, out: string, log: (line: string) => void): Promise<void> {
  return new Promise((resolveRun, reject) => {
    const [command, ...args] = student.trainer;
    const child = spawn(command!, [...args, '--plan', plan, '--out', out], { cwd: student.workdir ?? process.cwd(),
      env: { ...process.env, ...student.env }, stdio: ['ignore', 'pipe', 'pipe'] });
    let tail = '';
    const keep = (chunk: Buffer) => { tail = (tail + chunk.toString()).slice(-8000); for (const line of chunk.toString().split('\n')) if (line.startsWith('progress:')) log('  ' + line); };
    child.stdout.on('data', keep); child.stderr.on('data', keep);
    child.once('error', reject);
    child.once('exit', code => code === 0 ? resolveRun() : reject(new Error(`trainer exited with ${code}\n${tail.trim()}`)));
  });
}

/** What the trainer writes to `<out>/result.json`: trained node artifacts with exports relative to `<out>`. */
type TrainerResult = { artifacts: WeightArtifact[]; families?: FamilyVersion[]; memberships?: FamilyMembership[]; softStates?: SoftState[];
  validatedReductions?: string[][]; receipt: Record<string, unknown> };

/** Collected, split data of one program: everything a training variant needs. */
export type TuneData = { readonly scaffold: WeightManifest; readonly sites: Readonly<Record<string, string>>; readonly examples: readonly SplitExample[]; readonly dataKey: string;
  readonly turns: readonly CollectedTurn[]; readonly trainRows: readonly CollectedTurn[]; readonly protectedCount: number; readonly total: number };

/** The program's root examples split by component; the persisted receipt keeps every component on the side it was
 * first assigned (plan §4.1). */
export function currentSplit(root: string, examplesPath?: string): ReturnType<typeof splitExamples> {
  const splitPath = join(tuneDirectory(root), 'splits.json');
  const previous = existsSync(splitPath) ? JSON.parse(readFileSync(splitPath, 'utf8')) as SplitReceipt : undefined;
  const examples = splitExamples(readRootExamples(root, examplesPath), previous);
  mkdirSync(tuneDirectory(root), { recursive: true });
  writeFileSync(splitPath, JSON.stringify(examples.receipt, null, 1) + '\n');
  return examples;
}

/** One root's collected trajectories under one teacher identity (`.natlang/tune/data/roots/<key>.json`). */
export type RootRecord = { schema: 'natlang.tune-root/v1'; root: string; teacher: string; sharedSources: string;
  bindings: Record<string, string>; outcome: { ok: boolean; oracle: boolean | null }; turns: CollectedTurn[];
  /** Where trajectories collected outside `natlang tune` came from (tune/pool.ts). */
  imported?: Record<string, unknown> };
export const rootRecordPath = (root: string, example: Pick<SplitExample, 'function' | 'input' | 'expected' | 'group'>, teacherId: string) =>
  join(tuneDirectory(root), 'data', 'roots', sha(['natlang.tune-root/v1', example.function, example.input,
    example.expected ?? null, example.group ?? null, teacherId]).slice(0, 24) + '.json');
/** Teacher identities whose trajectories were imported from the generation pool (tune/pool.ts). */
export function importedTeachers(root: string): string[] {
  const path = join(tuneDirectory(root), 'data', 'imports.json');
  return existsSync(path) ? (JSON.parse(readFileSync(path, 'utf8')) as { teachers: string[] }).teachers : [];
}
/** A root's still-valid record: the configured teacher's first, else one imported from the pool. */
export function validRootRecord(root: string, example: SplitExample, scaffold: WeightManifest, shared: string, teacherId: string): RootRecord | undefined {
  for (const teacher of [teacherId, ...importedTeachers(root).filter(id => id !== teacherId)]) {
    const path = rootRecordPath(root, example, teacher);
    if (!existsSync(path)) continue;
    const record = JSON.parse(readFileSync(path, 'utf8')) as RootRecord;
    if (rootDataValid(record, scaffold, shared)) return record;
  }
  return undefined;
}

/** Discover the scope tree, split root examples before expansion, collect or reuse teacher trajectories. */
export async function prepareData(options: TuneOptions, timings: StageTiming[] = [], log: (line: string) => void = () => {},
  scopes: ScopeOptions = {}): Promise<TuneData> {
  const { root, program, student } = options;
  const { scaffold, sites } = await timed(timings, 'discover', log, () => {
    // Persistent site IDs keep an edited or moved inline site's binding, data and lambda node (tune/sites.ts).
    const assignment = programSites(root, program);
    for (const migration of assignment.migrations) log(`natlang tune: site ${migration.site} continues as ${migration.to} (${migration.how})`);
    return { sites: assignment.sites, scaffold: scaffoldManifest(program, student.base, { ...scopes, sites: assignment.sites }) };
  });
  const all = readRootExamples(root, options.examplesPath);
  const examples = currentSplit(root, options.examplesPath);
  if (!examples.some(example => example.partition === 'train'))
    throw new Error('no training roots: add root invocations to tune.examples.jsonl (one JSON object per line: {"function", "input", "expected"?, "group"?})');
  // Per-root records: raw evidence survives edits; a root is recollected only when a binding it reached changed
  // semantics or the shared (non-instruction) sources changed (keys.ts).
  const shared = sharedSourceHash(program);
  const rootsDir = join(tuneDirectory(root), 'data', 'roots');
  const recordPath = (example: SplitExample) => rootRecordPath(root, example, options.teacherId);
  const collectable = examples.filter(example => example.partition === 'train' || example.partition === 'development');
  const records = new Map<string, RootRecord>();
  for (const example of collectable) {
    const record = validRootRecord(root, example, scaffold, shared, options.teacherId);
    if (record) records.set(example.id, record);
  }
  const missing = collectable.filter(example => !records.has(example.id));
  const reusedData = missing.length === 0;
  await timed(timings, 'data', log, async () => {
    if (!missing.length) return;
    mkdirSync(rootsDir, { recursive: true });
    const collected = await collect({ root, program, scaffold, examples: missing, teacher: options.teacher, teacherId: options.teacherId, services: options.services });
    const failures = collected.roots.filter(item => !item.ok && item.error);
    if (failures.length === collected.roots.length)
      // Nothing usable: do not cache it as reusable data; the cause is usually the teacher setup.
      throw new Error(`every root invocation failed with the teacher; first error: ${failures[0]?.error ?? 'unknown'}`);
    const semantic = new Map(scaffold.bindings.map(entry => [entry.id, entry.semanticHash]));
    for (const example of missing) {
      const outcome = collected.roots.find(item => item.id === example.id)!;
      // A root that raised (teacher or infrastructure error) is retried next time; a wrong answer is a valid outcome.
      if (!outcome.ok && outcome.error && outcome.oracle === null) continue;
      const turns = collected.turns.filter(turn => turn.root === example.id);
      const bindings = Object.fromEntries([...new Set(turns.map(turn => turn.binding).filter((id): id is string => !!id))]
        .map(id => [id, semantic.get(id) ?? '']));
      const record: RootRecord = { schema: 'natlang.tune-root/v1', root: example.id, teacher: options.teacherId, sharedSources: shared,
        bindings, outcome: { ok: outcome.ok, oracle: outcome.oracle }, turns };
      writeFileSync(recordPath(example), JSON.stringify(record));
      records.set(example.id, record);
    }
  }, reusedData);
  const turns = collectable.flatMap(example => records.get(example.id)?.turns ?? []);
  const dataKey = sha(['natlang.tune-data/v2', collectable.map(example => [example.id, example.partition, records.has(example.id)
    ? sha(records.get(example.id)) : null])]).slice(0, 24);
  // Binding IDs depend on program, component and context only, so rows stay valid for any tree-root variant.
  const trainRows = turns.filter(turn => turn.partition === 'train' && turn.binding && turn.evidence.enclosingSuccess && !turn.reply.truncated);
  return { scaffold, sites, examples, dataKey, turns, trainRows, protectedCount: all.filter(example => example.protected).length, total: all.length };
}

/** Whether this backend serves soft instruction/skill states, so the build trains and exports them (plan §3.3): the
 * native reference server always; managed llama.cpp when `tune.softStates` declares the fork's `--native-state`. */
export function softStatesServed(settings: TuneOptions['settings']): boolean {
  return settings.backend === 'native' || (settings.backend === 'managed-local' && settings.softStates === true);
}

/** One training variant: a scaffold (scales, tree roots), a recipe and optionally a row subset. */
export type TrainVariant = { readonly manifest: WeightManifest; readonly recipe?: Readonly<Record<string, unknown>>;
  readonly rows?: readonly CollectedTurn[]; readonly label?: string;
  /** Continue this earlier build (a trainer output directory): its node, gate and soft-state lineage and moments. */
  readonly parent?: string;
  /** Scored student rollouts (natlang.tune-rollout/v1 JSONL): the RL rows of this fit (§5.1a). */
  readonly rollouts?: string;
  /** Directories of published artifacts the fit imports first (frozen family versions, §5.4). */
  readonly artifactDirs?: readonly string[];
  /** Incremental builds (§8): nodes frozen at earlier artifacts, and trainable nodes seeded from earlier ones. */
  readonly pinned?: Readonly<Record<string, string>>;
  readonly warm?: Readonly<Record<string, { readonly artifact: string; readonly optimizer?: string }>>;
  /** Soft states of bindings this fit does not train, carried from an earlier release directory unchanged. */
  readonly carrySoftStates?: { readonly states: readonly SoftState[]; readonly directory: string };
  /** Extra release receipts (build kind, pinned parents and the semantics they saw). */
  readonly receipts?: Readonly<Record<string, unknown>> };

/** Train a variant (or reuse its receipt) and store its release with exports. Keyed by data, scaffold, recipe, rows. */
export async function trainRelease(options: TuneOptions, data: TuneData, variant: TrainVariant, timings: StageTiming[] = [],
  log: (line: string) => void = () => {}): Promise<{ manifest: WeightManifest; reused: boolean; build: string }> {
  const { root, student } = options;
  const rows = variant.rows ?? data.trainRows;
  if (!rows.length) throw new Error('the teacher produced no verified training trajectories; check the examples and teacher profile');
  const recipe = { seed: 0, ...options.settings.recipe, ...variant.recipe };
  const trainKey = sha(['natlang.tune-train/v3', data.dataKey, trainingScaffoldKey(variant.manifest), student, recipe,
    variant.rows ? rows.map(row => [row.root, row.call, row.turn]) : null, variant.parent ?? null,
    variant.rollouts ? sha(readFileSync(variant.rollouts, 'utf8')) : null,
    ...(variant.pinned || variant.warm ? [variant.pinned ?? {}, variant.warm ?? {}] : [])]).slice(0, 24);
  const jobDir = join(tuneDirectory(root), 'jobs', trainKey), out = join(jobDir, 'out');
  const reused = existsSync(join(out, 'result.json'));
  const result = await timed(timings, 'train', log, async () => {
    if (!reused) {
      mkdirSync(out, { recursive: true });
      const rowsPath = join(jobDir, 'train.jsonl');
      writeFileSync(rowsPath, rows.map(turn => JSON.stringify(turn)).join('\n') + '\n');
      const plan = join(jobDir, 'plan.json');
      writeFileSync(plan, JSON.stringify({ schema: 'natlang.tune-plan/v1', manifest: variant.manifest, base: student.base, rows: rowsPath,
        student: { name: student.name, moduleMap: student.moduleMap, checkpoint: student.checkpoint, engine: student.engine ?? {}, ...(student.hfConfigDir ? { hfConfigDir: student.hfConfigDir } : {}),
          ...(student.baseGgufSha256 ? { baseGgufSha256: student.baseGgufSha256 } : {}) },
        exports: options.settings.backend === 'managed-local' ? ['gguf'] : [],
        // Soft instruction/skill states only where the backend can import native state (plan §3.3).
        softStates: softStatesServed(options.settings), recipe, ...(variant.label ? { label: variant.label } : {}),
        ...(variant.parent ? { parent: variant.parent } : {}), ...(variant.rollouts ? { rollouts: variant.rollouts } : {}),
        ...(variant.artifactDirs?.length ? { artifacts: variant.artifactDirs } : {}),
        ...(variant.pinned ? { pinned: variant.pinned } : {}), ...(variant.warm ? { warm: variant.warm } : {}) }, null, 2));
      await runTrainer(student, plan, out, log);
    }
    return JSON.parse(readFileSync(join(out, 'result.json'), 'utf8')) as TrainerResult;
  }, reused);
  const manifest = await timed(timings, 'export', log, () => {
    const { digest: _digest, ...base } = variant.manifest;
    const sealed = sealManifest({ ...base, artifacts: result.artifacts, families: result.families ?? [], memberships: result.memberships ?? [],
      softStates: [...(result.softStates ?? []), ...(variant.carrySoftStates?.states ?? [])],
      validatedReductions: result.validatedReductions ?? [], receipts: { data: data.dataKey, training: result.receipt, recipe,
        ...(variant.label ? { variant: variant.label } : {}), ...variant.receipts } } as Omit<WeightManifest, 'digest'>);
    storeRelease(root, sealed);
    const directory = dirname(releaseFile(root, sealed.digest));
    // Native artifacts (coefficients, bases) and soft states, for backends that serve them exactly.
    // Carried soft states keep their exact files (and derivation parents) from the earlier release.
    const carried = variant.carrySoftStates && join(variant.carrySoftStates.directory, 'states');
    if (carried && existsSync(carried)) {
      mkdirSync(join(directory, 'states'), { recursive: true });
      for (const file of readdirSync(carried)) if (!existsSync(join(directory, 'states', file))) copyFileSync(join(carried, file), join(directory, 'states', file));
    }
    for (const folder of ['artifacts', 'states']) if (existsSync(join(out, folder))) {
      mkdirSync(join(directory, folder), { recursive: true });
      for (const file of readdirSync(join(out, folder))) if (!existsSync(join(directory, folder, file)))
        copyFileSync(join(out, folder, file), join(directory, folder, file));
    }
    for (const artifact of [...sealed.artifacts, ...sealed.families]) if (artifact.exports?.gguf) {
      const target = join(directory, artifact.exports.gguf.file);
      mkdirSync(dirname(target), { recursive: true });
      if (!existsSync(target)) copyFileSync(join(out, artifact.exports.gguf.file), target);
    }
    // Soft states exported as llama.cpp native state spans: from this build, or a carried state's earlier release.
    for (const state of sealed.softStates) if (state.exports?.gguf) {
      const target = join(directory, state.exports.gguf.file);
      mkdirSync(dirname(target), { recursive: true });
      const source = existsSync(join(out, state.exports.gguf.file)) ? join(out, state.exports.gguf.file)
        : variant.carrySoftStates ? join(variant.carrySoftStates.directory, state.exports.gguf.file) : join(out, state.exports.gguf.file);
      if (!existsSync(target)) copyFileSync(source, target);
    }
    return sealed;
  });
  if (options.settings.backend === 'vllm') await timed(timings, 'export-stacks', log, () => exportVllmStacks(options, manifest));
  return { manifest, reused, build: out };
}

/** One whole-stack PEFT LoRA per distinct effective stack (vLLM applies one adapter per request), named canonically. */
async function exportVllmStacks(options: TuneOptions, manifest: WeightManifest): Promise<void> {
  const directory = dirname(releaseFile(options.root, manifest.digest)), out = join(directory, 'vllm');
  if (existsSync(join(out, 'stack-exports.json'))) return;
  const binding = new WeightBinding(manifest, {});
  const stacks = new Map<string, { identity: string; entries: { id: string; scale: number; node: string }[] }>();
  for (const entry of manifest.bindings) {
    const effective = binding.resolve(entry.definitionId ?? '', entry.component, entry.contextId).effective;
    if (effective.length) stacks.set(stackServingName(effective), { identity: stackServingName(effective), entries: effective.map(({ id, scale, node }) => ({ id, scale, node })) });
  }
  mkdirSync(out, { recursive: true });
  const stacksPath = join(out, 'stacks.json');
  writeFileSync(stacksPath, JSON.stringify([...stacks.values()], null, 1));
  const { student } = options;
  if (!student.hfConfigDir) throw new Error('vLLM stack export needs tune.students.<name>.hfConfigDir');
  await runStudentModule(student, 'natlang_neuralese.tune.stack_export',
    ['--release', directory, '--stacks', stacksPath, '--module-map', student.moduleMap, '--hf-config', student.hfConfigDir, '--out', out]);
}

/** Run another module of the student's Python entry (the trainer argv with its `-m` module replaced). */
export async function runStudentModule(student: TuneOptions['student'], module: string, args: readonly string[]): Promise<void> {
  const index = student.trainer.indexOf('-m');
  if (index < 0) throw new Error(`${module} derives its Python entry from a \`-m\` trainer argv`);
  const argv = [...student.trainer.slice(0, index + 1), module, ...args];
  await new Promise<void>((resolveRun, reject) => {
    const child = spawn(argv[0]!, argv.slice(1), { cwd: student.workdir ?? process.cwd(), env: { ...process.env, ...student.env }, stdio: ['ignore', 'pipe', 'pipe'] });
    let tail = ''; const keep = (chunk: Buffer) => { tail = (tail + chunk.toString()).slice(-4000); };
    child.stdout.on('data', keep); child.stderr.on('data', keep);
    child.once('error', reject);
    child.once('exit', code => code === 0 ? resolveRun() : reject(new Error(`${module} exited with ${code}\n${tail.trim()}`)));
  });
}

/** Load a stored release on the backend and evaluate it (and optionally other arms) over examples, strictly. */
export async function evaluateRelease(options: TuneOptions, data: TuneData, manifest: WeightManifest | null,
  examples: readonly SplitExample[], arm: string, partition: string): Promise<ArmResult> {
  // The untuned arm serves the scaffold: no artifacts to load.
  const directory = manifest ? dirname(releaseFile(options.root, manifest.digest)) : tuneDirectory(options.root);
  const served = await options.serve({ manifest: manifest ?? data.scaffold, directory });
  try {
    const weights = manifest ? [new WeightBinding(manifest, { program: options.program, available: served.loaded,
      nativeState: !!served.nativeState, strict: true })] : [];
    return await evaluateRoots({ root: options.root, program: options.program, driver: served.driver, weights, examples, arm, services: options.services,
      release: manifest?.digest ?? null, partition });
  } finally { await served.close(); }
}

/** How a tune chooses the release it validates and activates: one joint fit by default, or SFT/RL rounds. */
export type ReleaseSelector = (data: TuneData, timings: StageTiming[], log: (line: string) => void) => Promise<WeightManifest>;

export async function tuneProgram(options: TuneOptions, select?: ReleaseSelector): Promise<TuneReport> {
  const started = Date.now(), timings: StageTiming[] = [], log = options.log ?? (() => {});
  const { root, program, student } = options;
  const report: TuneReport = { schema: 'natlang.tune-report/v1', program: program.id, student: student.name, backend: options.settings.backend,
    release: null, activated: false, data: { examples: 0, protected: 0, components: {}, turns: 0, trainTurns: 0 }, validation: null,
    timings, totalMs: 0, manualActions: 0 };
  const lock = tuneJobLock(root);
  mkdirSync(dirname(lock), { recursive: true });
  writeFileSync(lock, JSON.stringify({ pid: process.pid, started: new Date(started).toISOString() }));
  try {
    const expectedActive = readActive(root)?.release ?? null;
    // 1-2. Discovery and data.
    const data = await prepareData(options, timings, log);
    report.data.examples = data.total; report.data.protected = data.protectedCount;
    for (const example of data.examples) report.data.components[example.partition] = (report.data.components[example.partition] ?? 0) + 1;
    report.data.turns = data.turns.length; report.data.trainTurns = data.trainRows.length;
    const examples = data.examples;
    // 3-4. One joint run over all bindings against one input snapshot, then the content-addressed release.
    // A full refit keeps the active release's published families and learned gates (refit with the family frozen).
    const active = expectedActive ? { manifest: readRelease(root, expectedActive), directory: dirname(releaseFile(root, expectedActive)) } : null;
    const bindings = new Set(data.scaffold.bindings.map(entry => entry.id));
    const memberships = active?.manifest.memberships.filter(item => bindings.has(item.binding)) ?? [];
    const variant: TrainVariant = active && memberships.length
      ? { manifest: { ...data.scaffold, families: active.manifest.families, memberships }, artifactDirs: [join(active.directory, 'artifacts')] }
      : { manifest: data.scaffold };
    const manifest = select ? await select(data, timings, log) : (await trainRelease(options, data, variant, timings, log)).manifest;
    report.build = typeof manifest.receipts.build === 'string' ? manifest.receipts.build : 'full';
    report.release = manifest.digest;
    // 5. Load on the backend and validate through the normal engine, strict: nothing may silently fall back.
    const directory = dirname(releaseFile(root, manifest.digest));
    const served = await timed(timings, 'load', log, () => options.serve({ manifest, directory }));
    try {
      const development = examples.filter(example => example.partition === 'development');
      const probe = development.length ? development : examples.filter(example => example.partition === 'train').slice(0, 1);
      const tunedBinding = new WeightBinding(manifest, { program, available: served.loaded, nativeState: !!served.nativeState, strict: true });
      const validation = await timed(timings, 'validate', log, async () => {
        const tuned = await evaluateRoots({ root, program, driver: served.driver, weights: [tunedBinding], examples: probe, services: options.services,
          arm: 'hierarchy', release: manifest.digest, partition: 'development' });
        const base = await evaluateRoots({ root, program, driver: served.driver, weights: [], examples: probe, services: options.services,
          arm: 'untuned', release: null, partition: 'development' });
        return { tuned, base };
      });
      const counts = (arm: ArmResult) => ({ roots: arm.roots.length, ok: arm.roots.filter(item => item.ok).length,
        oracle: arm.roots.filter(item => item.correct === true).length });
      report.validation = { tuned: counts(validation.tuned), base: counts(validation.base) };
      saveArm(root, validation.tuned); saveArm(root, validation.base);
      const exercised = validation.tuned.roots.flatMap(item => item.stacks).find(stack => stack.effective > 0 && stack.status === 'ready');
      if (!exercised) throw new Error('no validation call ran with the tuned stack; the release was not activated');
      // 6. Activate atomically against the pointer read at the start.
      await timed(timings, 'activate', log, () => {
        activateRelease(root, manifest.digest, { backend: { kind: options.settings.backend, loaded: [...served.loaded], identity: student.base },
          readiness: { definition: probe[0]!.function, ok: true, effectiveIdentity: exercised.identity, ms: timings.at(-1)!.ms } }, expectedActive);
      });
      report.activated = true;
    } finally { await served.close(); }
  } catch (error) {
    report.reason = error instanceof Error ? error.message : String(error);
  } finally {
    rmSync(lock, { force: true });
    report.totalMs = Date.now() - started;
    const reports = join(tuneDirectory(root), 'reports');
    mkdirSync(reports, { recursive: true });
    writeFileSync(join(reports, `${new Date(started).toISOString().replace(/[:.]/g, '-')}.json`), JSON.stringify(report, null, 2));
  }
  return report;
}
