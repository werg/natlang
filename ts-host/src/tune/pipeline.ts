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
import { activateRelease, readActive, releaseFile, storeRelease, tuneDirectory, tuneJobLock } from '../adaptation/weights/release.js';
import type { ModelDriver } from '../runtime/runtime.js';
import { collect, readRootExamples, scaffoldManifest, splitExamples, type CollectedTurn, type SplitExample } from './collect.js';
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
};

export type TuneOptions = {
  root: string; program: ProgramDescriptor; settings: TuneSettings; student: StudentSettings & { name: string };
  /** The authorized teacher's driver and identity. */
  teacher: ModelDriver; teacherId: string;
  /** Serves a release for validation: returns a driver that applies the release's adapters, and its loaded set. */
  serve: (release: { manifest: WeightManifest; directory: string }) => Promise<{ driver: ModelDriver; loaded: ReadonlySet<string>;
    /** Whether the backend imports stored native state (soft instruction/skill states). */
    nativeState?: boolean; close(): Promise<void> }>;
  examplesPath?: string; log?: (line: string) => void;
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
export type TuneData = { readonly scaffold: WeightManifest; readonly examples: readonly SplitExample[]; readonly dataKey: string;
  readonly turns: readonly CollectedTurn[]; readonly trainRows: readonly CollectedTurn[]; readonly protectedCount: number; readonly total: number };

/** Discover the scope tree, split root examples before expansion, collect or reuse teacher trajectories. */
export async function prepareData(options: TuneOptions, timings: StageTiming[] = [], log: (line: string) => void = () => {},
  scopes: ScopeOptions = {}): Promise<TuneData> {
  const { root, program, student } = options;
  const scaffold = await timed(timings, 'discover', log, () => scaffoldManifest(program, student.base, scopes));
  const all = readRootExamples(root, options.examplesPath);
  const examples = splitExamples(all);
  if (!examples.some(example => example.partition === 'train'))
    throw new Error('no training roots: add root invocations to tune.examples.jsonl (one JSON object per line: {"function", "input", "expected"?, "group"?})');
  const dataKey = sha(['natlang.tune-data/v1', program.buildHash, examples, options.teacherId]).slice(0, 24);
  const dataDir = join(tuneDirectory(root), 'data', dataKey), turnsPath = join(dataDir, 'turns.jsonl');
  const reusedData = existsSync(join(dataDir, 'receipt.json'));
  const turns = await timed(timings, 'data', log, async () => {
    if (reusedData) return readFileSync(turnsPath, 'utf8').split('\n').filter(Boolean).map(line => JSON.parse(line) as CollectedTurn);
    mkdirSync(dataDir, { recursive: true });
    // Only training and development roots are executed by the teacher here; confirmation stays for evaluation.
    const collected = await collect({ root, program, scaffold, examples, teacher: options.teacher, teacherId: options.teacherId,
      partitions: ['train', 'development'] });
    const failures = collected.roots.filter(item => !item.ok);
    if (failures.length === collected.roots.length)
      // Nothing usable: do not cache it as reusable data; the cause is usually the teacher setup.
      throw new Error(`every root invocation failed with the teacher; first error: ${failures[0]?.error ?? 'unknown'}`);
    writeFileSync(turnsPath, collected.turns.map(turn => JSON.stringify(turn)).join('\n') + '\n');
    writeFileSync(join(dataDir, 'receipt.json'), JSON.stringify({ schema: 'natlang.tune-data-receipt/v1', key: dataKey, roots: collected.roots,
      turns: collected.turns.length, teacher: options.teacherId, program: program.buildHash }, null, 2));
    return collected.turns;
  }, reusedData);
  // Binding IDs depend on program, component and context only, so rows stay valid for any tree-root variant.
  const trainRows = turns.filter(turn => turn.partition === 'train' && turn.binding && turn.evidence.enclosingSuccess && !turn.reply.truncated);
  return { scaffold, examples, dataKey, turns, trainRows, protectedCount: all.filter(example => example.protected).length, total: all.length };
}

/** One training variant: a scaffold (scales, tree roots), a recipe and optionally a row subset. */
export type TrainVariant = { readonly manifest: WeightManifest; readonly recipe?: Readonly<Record<string, unknown>>;
  readonly rows?: readonly CollectedTurn[]; readonly label?: string };

/** Train a variant (or reuse its receipt) and store its release with exports. Keyed by data, scaffold, recipe, rows. */
export async function trainRelease(options: TuneOptions, data: TuneData, variant: TrainVariant, timings: StageTiming[] = [],
  log: (line: string) => void = () => {}): Promise<{ manifest: WeightManifest; reused: boolean }> {
  const { root, student } = options;
  const rows = variant.rows ?? data.trainRows;
  if (!rows.length) throw new Error('the teacher produced no verified training trajectories; check the examples and teacher profile');
  const recipe = { seed: 0, ...options.settings.recipe, ...variant.recipe };
  const trainKey = sha(['natlang.tune-train/v2', data.dataKey, variant.manifest.digest, student, recipe,
    variant.rows ? rows.map(row => [row.root, row.call, row.turn]) : null]).slice(0, 24);
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
        softStates: options.settings.backend === 'native', recipe, ...(variant.label ? { label: variant.label } : {}) }, null, 2));
      await runTrainer(student, plan, out, log);
    }
    return JSON.parse(readFileSync(join(out, 'result.json'), 'utf8')) as TrainerResult;
  }, reused);
  const manifest = await timed(timings, 'export', log, () => {
    const { digest: _digest, ...base } = variant.manifest;
    const sealed = sealManifest({ ...base, artifacts: result.artifacts, families: result.families ?? [], memberships: result.memberships ?? [], softStates: result.softStates ?? [],
      validatedReductions: result.validatedReductions ?? [], receipts: { data: data.dataKey, training: result.receipt, recipe,
        ...(variant.label ? { variant: variant.label } : {}) } } as Omit<WeightManifest, 'digest'>);
    storeRelease(root, sealed);
    const directory = dirname(releaseFile(root, sealed.digest));
    // Native artifacts (coefficients, bases) and soft states, for backends that serve them exactly.
    for (const folder of ['artifacts', 'states']) if (existsSync(join(out, folder))) {
      mkdirSync(join(directory, folder), { recursive: true });
      for (const file of readdirSync(join(out, folder))) if (!existsSync(join(directory, folder, file)))
        copyFileSync(join(out, folder, file), join(directory, folder, file));
    }
    for (const artifact of sealed.artifacts) if (artifact.exports?.gguf) {
      const target = join(directory, artifact.exports.gguf.file);
      mkdirSync(dirname(target), { recursive: true });
      if (!existsSync(target)) copyFileSync(join(out, artifact.exports.gguf.file), target);
    }
    return sealed;
  });
  if (options.settings.backend === 'vllm') await timed(timings, 'export-stacks', log, () => exportVllmStacks(options, manifest));
  return { manifest, reused };
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
  // The student's Python entry, pointed at the stack exporter instead of the trainer.
  const module = student.trainer.indexOf('-m');
  if (module < 0) throw new Error('vLLM stack export derives its Python entry from a `-m` trainer argv');
  const argv = [...student.trainer.slice(0, module + 1), 'natlang_neuralese.tune.stack_export',
    '--release', directory, '--stacks', stacksPath, '--module-map', student.moduleMap, '--hf-config', student.hfConfigDir, '--out', out];
  await new Promise<void>((resolveRun, reject) => {
    const child = spawn(argv[0]!, argv.slice(1), { cwd: student.workdir ?? process.cwd(), env: { ...process.env, ...student.env }, stdio: ['ignore', 'pipe', 'pipe'] });
    let tail = ''; const keep = (chunk: Buffer) => { tail = (tail + chunk.toString()).slice(-4000); };
    child.stdout.on('data', keep); child.stderr.on('data', keep);
    child.once('error', reject);
    child.once('exit', code => code === 0 ? resolveRun() : reject(new Error(`stack export exited with ${code}\n${tail.trim()}`)));
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
    return await evaluateRoots({ root: options.root, program: options.program, driver: served.driver, weights, examples, arm,
      release: manifest?.digest ?? null, partition });
  } finally { await served.close(); }
}

export async function tuneProgram(options: TuneOptions): Promise<TuneReport> {
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
    const { manifest } = await trainRelease(options, data, { manifest: data.scaffold }, timings, log);
    report.release = manifest.digest;
    // 5. Load on the backend and validate through the normal engine, strict: nothing may silently fall back.
    const directory = dirname(releaseFile(root, manifest.digest));
    const served = await timed(timings, 'load', log, () => options.serve({ manifest, directory }));
    try {
      const development = examples.filter(example => example.partition === 'development');
      const probe = development.length ? development : examples.filter(example => example.partition === 'train').slice(0, 1);
      const tunedBinding = new WeightBinding(manifest, { program, available: served.loaded, nativeState: !!served.nativeState, strict: true });
      const validation = await timed(timings, 'validate', log, async () => {
        const tuned = await evaluateRoots({ root, program, driver: served.driver, weights: [tunedBinding], examples: probe,
          arm: 'hierarchy', release: manifest.digest, partition: 'development' });
        const base = await evaluateRoots({ root, program, driver: served.driver, weights: [], examples: probe,
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
