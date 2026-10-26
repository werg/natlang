/**
 * `natlang tune` (plans/HIERARCHICAL_LORA_TUNING.md §8):
 *
 *   natlang tune <program>                 discover, collect/reuse data, train, export, load, validate, activate
 *   natlang tune <program> --incremental   rebuild what changed (reuses valid data and training receipts)
 *   natlang tune <program> --watch         rebuild in the background on edits; the run/call path keeps serving
 *   natlang tune <program> --report        the latest report, status and the active release
 *   natlang tune <program> --rollback      reactivate the previous coherent release
 *   natlang tune <program> --evaluate <development|confirmation|protected> [--with-teacher]
 *                                          paired arms (untuned, active release, teacher reference) for reports
 *   natlang tune <program> --placement     measured removals/refits on development roots; activates the cheapest
 *                                          candidate within the declared success envelope
 *   natlang tune <program> --sweep <spec>  resumable rank/penalty/data-size sweep journal (sweep.ts)
 *   natlang tune --families                cross-program family discovery and refits
 */
import { existsSync, readdirSync, readFileSync, watch } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildProject } from '../compiler/node-project.js';
import { formatDiagnostics } from '../compiler/project.js';
import { createResolvedModelSession, executorIdentityForChoice, loadModelConfiguration } from '../model/index.js';
import { activateRelease, currentManifest, readActive, readRelease, rollbackRelease, tuneDirectory, tuneStatus } from '../adaptation/weights/release.js';
import type { ModelDriver } from '../runtime/runtime.js';
import type { DecisionRequest, ModelTurnRequest, ModelTurnOptions } from '../contracts.js';
import { loadTuneSettings } from '../tune/settings.js';
import { loadVllmStacks, uploadRelease } from './weights.js';
import { evaluateRelease, prepareData, saveArm, tuneProgram, type TuneOptions, type TuneReport } from '../tune/pipeline.js';
import { evaluateRoots } from '../tune/evaluate.js';
import { protectedExamples, readRootExamples } from '../tune/collect.js';
import { runPlacement } from '../tune/placement.js';
import { runSweep, type SweepSpec } from '../tune/sweep.js';
import { formatOverview, loadOverview } from '../tune/report.js';

const RUNTIME_MODULE = { url: new URL('../index.js', import.meta.url).href, path: fileURLToPath(new URL('../index.js', import.meta.url)),
  types: fileURLToPath(new URL('../index.d.ts', import.meta.url)), specifiers: ['@natlang/node'] };

type Session = ReturnType<typeof createResolvedModelSession>;
const driverOf = (session: Session): ModelDriver => Object.assign(
  (request: ModelTurnRequest, signal?: AbortSignal, options?: ModelTurnOptions) => session.turn(request, signal, options),
  { decide: (request: DecisionRequest, signal?: AbortSignal) => session.decide(request, signal), contextWindow: () => session.contextWindow(),
    ...(session.neuralese ? { neuralese: true as const } : {}) });

function formatReport(report: TuneReport): string {
  const seconds = (ms: number) => `${(ms / 1000).toFixed(1)}s`;
  const lines = [report.activated ? `ready: ${report.program} is tuned for ${report.student} on ${report.backend} (release ${report.release?.slice(0, 12)})`
    : `not activated: ${report.reason ?? 'unknown failure'}${report.release ? ` (release ${report.release.slice(0, 12)} was built)` : ''}`];
  lines.push(`data: ${report.data.examples} root examples (${report.data.protected} protected, excluded), ${report.data.turns} teacher turns, ${report.data.trainTurns} training turns`);
  if (report.validation) lines.push(`validation (development roots): tuned ${report.validation.tuned.oracle}/${report.validation.tuned.roots} correct, ` +
    `${report.validation.tuned.ok} completed; base ${report.validation.base.oracle}/${report.validation.base.roots} correct, ${report.validation.base.ok} completed`);
  lines.push(`time to ${report.activated ? 'first tuned call' : 'stop'}: ${seconds(report.totalMs)} (` + report.timings.map(item => `${item.stage} ${seconds(item.ms)}${item.reused ? ' reused' : ''}`).join(', ') + ')');
  return lines.join('\n');
}

export async function tuneCommand(argv: readonly string[], io: { out: (text: string) => void; err: (text: string) => void } =
  { out: text => process.stdout.write(text + '\n'), err: text => process.stderr.write(text + '\n') }): Promise<number> {
  const valuedNames = ['--examples', '--student', '--teacher', '--evaluate', '--sweep', '--partition'];
  const flags = new Set(argv.filter(arg => arg.startsWith('--') && !valuedNames.includes(arg)));
  const value = (name: string) => { const index = argv.indexOf(name); return index >= 0 ? argv[index + 1] : undefined; };
  const valued = new Set(valuedNames.flatMap(name => { const i = argv.indexOf(name); return i >= 0 ? [i, i + 1] : []; }));
  const words = argv.filter((arg, index) => !arg.startsWith('--') && !valued.has(index));
  const known = new Set(['--incremental', '--watch', '--report', '--rollback', '--families', '--json', '--placement', '--with-teacher']);
  for (const flag of flags) if (!known.has(flag)) throw new Error(`natlang tune: unknown option ${flag}`);
  if (flags.has('--families'))
    throw new Error('natlang tune --families: cross-program family training is not implemented yet (plans/HIERARCHICAL_LORA_TUNING.md §5.5); program tunes run with no family level');
  const root = resolve(words[0] ?? '.');
  if (!existsSync(root)) throw new Error(`natlang tune: ${root} does not exist`);
  if (flags.has('--rollback')) { const record = rollbackRelease(root); io.out(`active release is now ${record.release.slice(0, 12)}`); return 0; }
  if (flags.has('--report')) {
    const reports = join(tuneDirectory(root), 'reports');
    const latest = existsSync(reports) ? readdirSync(reports).sort().at(-1) : undefined;
    const status = tuneStatus(root), overview = loadOverview(root, value('--partition') ?? 'development');
    if (flags.has('--json')) io.out(JSON.stringify({ status, active: readActive(root) ?? null,
      report: latest ? JSON.parse(readFileSync(join(reports, latest), 'utf8')) : null, overview }, null, 2));
    else io.out([`status: ${status.status}${status.reason ? ` (${status.reason})` : ''}`,
      latest ? formatReport(JSON.parse(readFileSync(join(reports, latest), 'utf8')) as TuneReport) : 'no tune report yet',
      formatOverview(overview)].join('\n'));
    return 0;
  }
  const withOptions = async <T>(fn: (options: TuneOptions) => Promise<T>): Promise<T> => {
    const { settings, student } = loadTuneSettings(process.env, { student: value('--student'), teacher: value('--teacher') });
    const inventory = buildProject({ project: root, emit: false, write: false, runtimeModule: RUNTIME_MODULE });
    if (!inventory.ok) throw new Error(formatDiagnostics(inventory.diagnostics));
    const program = inventory.manifest.adaptation!;
    const teacherConfig = loadModelConfiguration(settings.teacher);
    const teacherSession = createResolvedModelSession(teacherConfig.choice, process.env, process.stderr);
    try {
      return await fn({ root, program, settings, student, teacher: driverOf(teacherSession),
        teacherId: executorIdentityForChoice(teacherConfig.choice).id, examplesPath: value('--examples'), log: io.err,
        serve: async ({ manifest, directory }) => {
          if (settings.backend === 'native') {
            const serving = loadModelConfiguration(undefined);
            if (serving.choice.kind !== 'external' || !serving.choice.neuraleseServer)
              throw new Error('natlang tune: the native backend needs a default profile with neuraleseServer: true');
            const session = createResolvedModelSession(serving.choice, process.env, process.stderr);
            await session.prepare();
            const loaded = await uploadRelease(session.nativeArtifacts!()!, directory);
            return { driver: driverOf(session), loaded, nativeState: true, close: () => session.close() };
          }
          if (settings.backend === 'vllm') {
            const serving = loadModelConfiguration(undefined);
            if (serving.choice.kind !== 'external' || !serving.choice.vllm)
              throw new Error('natlang tune: the vllm backend needs a default profile with vllm: true');
            const session = createResolvedModelSession(serving.choice, process.env, process.stderr);
            await session.prepare();
            const loaded = await loadVllmStacks(serving.choice.endpoint, directory, serving.choice.headers ?? {});
            session.vllmStacks?.(loaded.names);
            return { driver: driverOf(session), loaded: loaded.available, close: () => session.close() };
          }
          const adapters = manifest.artifacts.filter(item => item.exports?.gguf).map(item => ({ id: item.artifact, path: join(directory, item.exports!.gguf!.file) }));
          const serving = loadModelConfiguration(undefined);
          if (serving.choice.kind !== 'managed-local') throw new Error('natlang tune: the default profile must use the managed local model to serve tuned weights');
          const session = createResolvedModelSession(serving.choice, process.env, process.stderr, { loraAdapters: () => adapters });
          await session.prepare();
          return { driver: driverOf(session), loaded: new Set(adapters.map(item => item.id)), close: () => session.close() };
        } });
    } finally { await teacherSession.close(); }
  };
  const once = () => withOptions(options => tuneProgram(options));
  const partition = value('--evaluate');
  if (partition !== undefined) {
    if (!['development', 'confirmation', 'protected'].includes(partition)) throw new Error('--evaluate takes development, confirmation or protected');
    await withOptions(async options => {
      const data = await prepareData(options, [], io.err);
      const examples = partition === 'protected' ? protectedExamples(readRootExamples(root, options.examplesPath))
        : data.examples.filter(example => example.partition === partition);
      if (!examples.length) throw new Error(`no ${partition} roots to evaluate`);
      const current = currentManifest(root);
      const arms = [await evaluateRelease(options, data, null, examples, 'untuned', partition)];
      if (current) arms.push(await evaluateRelease(options, data, current, examples, String(current.receipts.variant ?? '').startsWith('placement') ? 'placed' : 'hierarchy', partition));
      // The teacher reference runs as an isolated evaluator: its answers are never training data.
      if (flags.has('--with-teacher')) arms.push(await evaluateRoots({ root, program: options.program, driver: options.teacher, weights: [], examples,
        arm: 'teacher', release: null, partition }));
      for (const arm of arms) saveArm(root, arm);
    });
    io.out(formatOverview(loadOverview(root, partition)));
    return 0;
  }
  if (flags.has('--placement')) {
    const result = await withOptions(async options => {
      const reference = currentManifest(root);
      if (!reference) throw new Error('placement starts from an active release; run natlang tune first');
      const data = await prepareData(options, [], io.err);
      const placed = await runPlacement(options, data, reference, { log: io.err });
      if (placed.selected.release !== reference.digest && placed.selected.exercised) {
        const loaded = readRelease(root, placed.selected.release).artifacts.map(item => item.artifact);
        activateRelease(root, placed.selected.release, { backend: { kind: options.settings.backend, loaded, identity: options.student.base },
          readiness: { definition: data.examples.find(example => example.partition === 'development')!.function, ok: true,
            effectiveIdentity: placed.selected.exercised, ms: placed.selected.meanMs } }, readActive(root)?.release ?? null);
      }
      return placed;
    });
    io.out(flags.has('--json') ? JSON.stringify(result, null, 2) : formatOverview(loadOverview(root, 'development')));
    return 0;
  }
  const sweepPath = value('--sweep');
  if (sweepPath !== undefined) {
    const spec = JSON.parse(readFileSync(resolve(sweepPath), 'utf8')) as SweepSpec;
    const result = await withOptions(async options => runSweep(options, await prepareData(options, [], io.err), spec, io.err));
    io.out(flags.has('--json') ? JSON.stringify(result, null, 2) : `sweep ${spec.name}: ${result.cells.length} cells; selected ` +
      (result.selected ? `${result.selected.arm} ${result.selected.recipe} size ${result.selected.size} (development ${(result.selected.development!.success * 100).toFixed(1)}%)` : 'none') +
      `\njournal: ${result.journal}`);
    return 0;
  }
  if (!flags.has('--watch')) {
    const report = await once();
    io.out(flags.has('--json') ? JSON.stringify(report, null, 2) : formatReport(report));
    return report.activated ? 0 : 1;
  }
  // Watch: coalesce edits, one build at a time; activation compares the pointer, so a stale build cannot win.
  let running = false, again = false, timer: ReturnType<typeof setTimeout> | undefined;
  const build = async () => {
    if (running) { again = true; return; }
    running = true;
    try { io.out(formatReport(await once())); } catch (error) { io.err(`natlang tune: ${(error as Error).message}`); }
    finally { running = false; if (again) { again = false; void build(); } }
  };
  watch(root, { recursive: true }, (_event, file) => {
    if (!file || /(^|[\\/])(\.natlang|node_modules|\.git|dist)([\\/]|$)/.test(String(file))) return;
    clearTimeout(timer); timer = setTimeout(() => void build(), 1500);
  });
  await build();
  await new Promise(() => {});
  return 0;
}
