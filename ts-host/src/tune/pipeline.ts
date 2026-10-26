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
import { copyFileSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import type { ProgramDescriptor } from '../adaptation/types.js';
import { sealManifest, type WeightArtifact, type WeightManifest, type FamilyVersion, type FamilyMembership, type SoftState } from '../adaptation/weights/manifest.js';
import { WeightBinding, type ResolvedWeightStack } from '../adaptation/weights/binding.js';
import { activateRelease, readActive, releaseFile, storeRelease, tuneDirectory, tuneJobLock } from '../adaptation/weights/release.js';
import { createNatlangRuntime, type ModelDriver } from '../runtime/runtime.js';
import { loadNatlang } from '../runtime/node-files.js';
import { collect, readRootExamples, scaffoldManifest, splitExamples, type CollectedTurn, type SplitExample } from './collect.js';
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

async function runRoots(root: string, program: ProgramDescriptor, driver: ModelDriver, weights: readonly WeightBinding[], examples: readonly SplitExample[]) {
  const stacks: ResolvedWeightStack[] = [];
  let ok = 0, oracle = 0;
  for (const example of examples) {
    const runtime = createNatlangRuntime({ program, model: driver, weights, trace: trace => {
      const recorded = (trace.adaptation as { weights?: ResolvedWeightStack } | undefined)?.weights; if (recorded) stacks.push(recorded); } });
    try {
      const fn = loadNatlang(resolve(root, example.function), root);
      const params = (fn as unknown as Record<symbol, { definition: { params: { name: string }[] } }>)[Symbol.for('natlang.callable')]!.definition.params;
      const input = example.input;
      const args = input && typeof input === 'object' && !Array.isArray(input) ? params.map(p => (input as Record<string, unknown>)[p.name]) : Array.isArray(input) ? input : [input];
      const value = await runtime.run(() => fn(...args));
      ok++;
      if ('expected' in example && JSON.stringify(value) === JSON.stringify(example.expected)) oracle++;
    } catch { /* counted as not ok */ } finally { runtime.close(); }
  }
  return { roots: examples.length, ok, oracle, stacks };
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
    // 1. Discovery: the lexical scope tree over the program's bound definitions.
    const scaffold = await timed(timings, 'discover', log, () => scaffoldManifest(program, student.base));
    // 2. Data: split before expansion, then collect or reuse whole-program teacher trajectories.
    const all = readRootExamples(root, options.examplesPath);
    const examples = splitExamples(all);
    report.data.examples = all.length; report.data.protected = all.filter(example => example.protected).length;
    for (const example of examples) report.data.components[example.partition] = (report.data.components[example.partition] ?? 0) + 1;
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
      writeFileSync(turnsPath, collected.turns.map(turn => JSON.stringify(turn)).join('\n') + '\n');
      writeFileSync(join(dataDir, 'receipt.json'), JSON.stringify({ schema: 'natlang.tune-data-receipt/v1', key: dataKey, roots: collected.roots,
        turns: collected.turns.length, teacher: options.teacherId, program: program.buildHash }, null, 2));
      return collected.turns;
    }, reusedData);
    report.data.turns = turns.length;
    const trainRows = turns.filter(turn => turn.partition === 'train' && turn.binding && turn.evidence.enclosingSuccess && !turn.reply.truncated);
    report.data.trainTurns = trainRows.length;
    if (!trainRows.length) throw new Error('the teacher produced no verified training trajectories; check the examples and teacher profile');
    // 3. Training: one joint run over all bindings against one input snapshot.
    const trainKey = sha(['natlang.tune-train/v1', dataKey, scaffold.digest, student]).slice(0, 24);
    const jobDir = join(tuneDirectory(root), 'jobs', trainKey), out = join(jobDir, 'out');
    const reusedTraining = existsSync(join(out, 'result.json'));
    const result = await timed(timings, 'train', log, async () => {
      if (!reusedTraining) {
        mkdirSync(out, { recursive: true });
        const rowsPath = join(jobDir, 'train.jsonl');
        writeFileSync(rowsPath, trainRows.map(turn => JSON.stringify(turn)).join('\n') + '\n');
        const plan = join(jobDir, 'plan.json');
        writeFileSync(plan, JSON.stringify({ schema: 'natlang.tune-plan/v1', manifest: scaffold, base: student.base, rows: rowsPath,
          student: { name: student.name, moduleMap: student.moduleMap, checkpoint: student.checkpoint, engine: student.engine ?? {}, ...(student.hfConfigDir ? { hfConfigDir: student.hfConfigDir } : {}),
            ...(student.baseGgufSha256 ? { baseGgufSha256: student.baseGgufSha256 } : {}) },
          exports: options.settings.backend === 'managed-local' ? ['gguf'] : [],
          // Soft instruction/skill states only where the backend can import native state (plan §3.3).
          softStates: options.settings.backend === 'native', recipe: { seed: 0 } }, null, 2));
        await runTrainer(student, plan, out, log);
      }
      return JSON.parse(readFileSync(join(out, 'result.json'), 'utf8')) as TrainerResult;
    }, reusedTraining);
    // 4. Release: content-addressed manifest plus its export files.
    const manifest = await timed(timings, 'export', log, () => {
      const sealed = sealManifest({ ...scaffold, artifacts: result.artifacts, families: result.families ?? [], memberships: result.memberships ?? [], softStates: result.softStates ?? [],
        validatedReductions: result.validatedReductions ?? [], receipts: { data: dataKey, training: result.receipt } } as Omit<WeightManifest, 'digest'>);
      storeRelease(root, sealed);
      const directory = dirname(releaseFile(root, sealed.digest));
      for (const artifact of sealed.artifacts) if (artifact.exports?.gguf) {
        const target = join(directory, artifact.exports.gguf.file);
        mkdirSync(dirname(target), { recursive: true });
        if (!existsSync(target)) copyFileSync(join(out, artifact.exports.gguf.file), target);
      }
      return sealed;
    });
    report.release = manifest.digest;
    // 5. Load on the backend and validate through the normal engine, strict: nothing may silently fall back.
    const directory = dirname(releaseFile(root, manifest.digest));
    const served = await timed(timings, 'load', log, () => options.serve({ manifest, directory }));
    try {
      const development = examples.filter(example => example.partition === 'development');
      const probe = development.length ? development : examples.filter(example => example.partition === 'train').slice(0, 1);
      const tunedBinding = new WeightBinding(manifest, { program, available: served.loaded, nativeState: !!served.nativeState, strict: true });
      const validation = await timed(timings, 'validate', log, async () => {
        const tuned = await runRoots(root, program, served.driver, [tunedBinding], probe);
        const base = await runRoots(root, program, served.driver, [], probe);
        return { tuned, base };
      });
      report.validation = { tuned: { roots: validation.tuned.roots, ok: validation.tuned.ok, oracle: validation.tuned.oracle },
        base: { roots: validation.base.roots, ok: validation.base.ok, oracle: validation.base.oracle } };
      const exercised = validation.tuned.stacks.find(stack => stack.effective.length && stack.status === 'ready');
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
