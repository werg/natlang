/**
 * `natlang tune` (plans/HIERARCHICAL_LORA_TUNING.md §8):
 *
 *   natlang tune <program>                 discover, collect/reuse data, train, export, load, validate, activate
 *   natlang tune <program> --incremental   rebuild what changed: a leaf candidate against pinned parents, the shared
 *                                          refit of dirty ancestors, or a periodic full refit (incremental.ts)
 *   natlang tune <program> --full          a full joint refit of every node (resets the incremental counter)
 *   natlang tune <program> --watch         rebuild in the background on edits; the run/call path keeps serving
 *   natlang tune <program> --report        the latest report, status and the active release
 *   natlang tune <program> --rollback      reactivate the previous coherent release
 *   natlang tune <program> --evaluate <development|confirmation|protected> [--with-teacher]
 *                                          paired arms (untuned, active release, teacher reference) for reports
 *   natlang tune <program> --placement     measured removals/refits on development roots; activates the cheapest
 *                                          candidate within the declared success envelope
 *   natlang tune <program> --sweep <spec>  resumable rank/penalty/data-size sweep journal (sweep.ts)
 *   natlang tune <program> --distill       export the active release's verified training outputs for base SFT (§5.7)
 *   natlang tune <program> --rounds        SFT then RL rounds (tune.rounds), best development release activates (§5.1a)
 *   natlang tune <program> --calibrate     measure the configured teachers on development roots and route one (§4.2)
 *   natlang tune <program> --export-cases <dir> [--include-confirmation]
 *                                          case list of roots needing data, for the reviewed generation pool (pool.ts)
 *   natlang tune <program> --import-cases <results>
 *                                          the pool's collector results → per-root tune data under the actor's identity
 *   natlang tune --families <programs...>  cross-program family discovery, fits and member refits (families.ts)
 */
import { existsSync, readdirSync, readFileSync, watch } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildProject } from '../compiler/node-project.js';
import { formatDiagnostics } from '../compiler/project.js';
import { createResolvedModelSession, executorIdentityForChoice, loadModelConfiguration } from '../model/index.js';
import { activateRelease, currentManifest, currentRelease, readActive, readRelease, rollbackRelease, tuneDirectory, tuneStatus } from '../adaptation/weights/release.js';
import type { ModelDriver } from '../runtime/runtime.js';
import type { DecisionRequest, ModelTurnRequest, ModelTurnOptions } from '../contracts.js';
import { loadTuneSettings } from '../tune/settings.js';
import { loadVllmStacks, uploadRelease } from './weights.js';
import { invalidation } from '../tune/keys.js';
import { scaffoldManifest } from '../tune/collect.js';
import { programSites } from '../tune/sites.js';
import { incrementalSelector, runIncrementalBuilds } from '../tune/incremental.js';
import { exportCases, importCases } from '../tune/pool.js';
import { evaluateRelease, prepareData, runStudentModule, saveArm, tuneProgram, type TuneOptions, type TuneReport } from '../tune/pipeline.js';
import { evaluateRoots } from '../tune/evaluate.js';
import { protectedExamples, readRootExamples } from '../tune/collect.js';
import { runPlacement } from '../tune/placement.js';
import { runSweep, type SweepSpec } from '../tune/sweep.js';
import { formatOverview, loadOverview } from '../tune/report.js';
import { roundsSelector } from '../tune/schedule.js';
import { exportDistillation } from '../tune/distill.js';
import { familyRegistry, familyStore, registerTunedProgram, runFamilies, type FamiliesReport, type FamilyProgram } from '../tune/families.js';
import { calibrateTeachers, routedTeacherProfile, routeTeacher, saveTeacherChoice } from '../tune/teachers.js';
import { currentSplit } from '../tune/pipeline.js';
import { createNatlangRuntime } from '../runtime/runtime.js';
import { loadNatlang } from '../runtime/node-files.js';
import { requireNativeReaderIdentity } from '../model/index.js';

const ROUTE_TEACHER = fileURLToPath(new URL('../../../applications/tuner/routeTeacher.nl', import.meta.url));
const PROPOSE_FAMILIES = fileURLToPath(new URL('../../../applications/tuner/proposeFamilies.nl', import.meta.url));
const identityOf = (profile: string) => executorIdentityForChoice(loadModelConfiguration(profile).choice).id;

const RUNTIME_MODULE = { url: new URL('../index.js', import.meta.url).href, path: fileURLToPath(new URL('../index.js', import.meta.url)),
  types: fileURLToPath(new URL('../index.d.ts', import.meta.url)), specifiers: ['@natlang/node'] };

type Session = ReturnType<typeof createResolvedModelSession>;
const driverOf = (session: Session): ModelDriver => Object.assign(
  (request: ModelTurnRequest, signal?: AbortSignal, options?: ModelTurnOptions) => session.turn(request, signal, options),
  { decide: (request: DecisionRequest, signal?: AbortSignal) => session.decide(request, signal), contextWindow: () => session.contextWindow(),
    ...(session.neuralese ? { neuralese: true as const, neuraleseRuntime: () => session.neuraleseRuntime!() } : {}) });

function formatReport(report: TuneReport): string {
  const seconds = (ms: number) => `${(ms / 1000).toFixed(1)}s`;
  const lines = [report.activated ? `ready: ${report.program} is tuned for ${report.student} on ${report.backend} (release ${report.release?.slice(0, 12)}, ${report.build ?? 'full'} build)`
    : `not activated: ${report.reason ?? 'unknown failure'}${report.release ? ` (release ${report.release.slice(0, 12)} was built)` : ''}`];
  lines.push(`data: ${report.data.examples} root examples (${report.data.protected} protected, excluded), ${report.data.turns} teacher turns, ${report.data.trainTurns} training turns`);
  if (report.validation) lines.push(`validation (development roots): tuned ${report.validation.tuned.oracle}/${report.validation.tuned.roots} correct, ` +
    `${report.validation.tuned.ok} completed; base ${report.validation.base.oracle}/${report.validation.base.roots} correct, ${report.validation.base.ok} completed`);
  lines.push(`time to ${report.activated ? 'first tuned call' : 'stop'}: ${seconds(report.totalMs)} (` + report.timings.map(item => `${item.stage} ${seconds(item.ms)}${item.reused ? ' reused' : ''}`).join(', ') + ')');
  return lines.join('\n');
}

export async function tuneCommand(argv: readonly string[], io: { out: (text: string) => void; err: (text: string) => void } =
  { out: text => process.stdout.write(text + '\n'), err: text => process.stderr.write(text + '\n') }): Promise<number> {
  const valuedNames = ['--examples', '--student', '--teacher', '--evaluate', '--sweep', '--partition', '--export-cases', '--import-cases', '--import-campaign', '--held'];
  const flags = new Set(argv.filter(arg => arg.startsWith('--') && !valuedNames.includes(arg)));
  const value = (name: string) => { const index = argv.indexOf(name); return index >= 0 ? argv[index + 1] : undefined; };
  const valued = new Set(valuedNames.flatMap(name => { const i = argv.indexOf(name); return i >= 0 ? [i, i + 1] : []; }));
  const words = argv.filter((arg, index) => !arg.startsWith('--') && !valued.has(index));
  const known = new Set(['--incremental', '--watch', '--report', '--rollback', '--families', '--json', '--placement', '--with-teacher', '--calibrate', '--include-confirmation', '--rounds', '--distill', '--full']);
  for (const flag of flags) if (!known.has(flag)) throw new Error(`natlang tune: unknown option ${flag}`);
  const root = resolve(words[0] ?? '.');
  if (!existsSync(root)) throw new Error(`natlang tune: ${root} does not exist`);
  if (flags.has('--rollback')) { const record = rollbackRelease(root); io.out(`active release is now ${record.release.slice(0, 12)}`); return 0; }
  if (flags.has('--report')) {
    const reports = join(tuneDirectory(root), 'reports');
    const latest = existsSync(reports) ? readdirSync(reports).sort().at(-1) : undefined;
    const status = tuneStatus(root), overview = loadOverview(root, value('--partition') ?? 'development');
    // What built the active release (§8): full, candidate (with any stale pinned parents) or shared refit.
    const activeReceipts = currentRelease(root)?.manifest.receipts as { build?: string; pinnedFrom?: string; stalePins?: Record<string, unknown> } | undefined;
    const build = activeReceipts ? { kind: activeReceipts.build ?? 'full', ...(activeReceipts.pinnedFrom ? { pinnedFrom: activeReceipts.pinnedFrom } : {}),
      stalePins: Object.keys(activeReceipts.stalePins ?? {}) } : null;
    if (flags.has('--json')) io.out(JSON.stringify({ status, active: readActive(root) ?? null, build,
      report: latest ? JSON.parse(readFileSync(join(reports, latest), 'utf8')) : null, overview }, null, 2));
    else io.out([`status: ${status.status}${status.reason ? ` (${status.reason})` : ''}`,
      ...(build ? [`active build: ${build.kind}${build.stalePins.length ? `; pinned stale shared parents ${build.stalePins.join(', ')} ` +
        '(run natlang tune --incremental to complete the shared refit)' : ''}`] : []),
      latest ? formatReport(JSON.parse(readFileSync(join(reports, latest), 'utf8')) as TuneReport) : 'no tune report yet',
      formatOverview(overview)].join('\n'));
    return 0;
  }
  const withOptions = async <T>(fn: (options: TuneOptions) => Promise<T>, programRoot = root): Promise<T> => {
    const { settings, student } = loadTuneSettings(process.env, { student: value('--student'), teacher: value('--teacher') });
    const inventory = buildProject({ project: programRoot, emit: false, write: false, runtimeModule: RUNTIME_MODULE });
    if (!inventory.ok) throw new Error(formatDiagnostics(inventory.diagnostics));
    const program = inventory.manifest.adaptation!;
    // Several configured teachers: collection uses the calibrated, routed one; an explicit --teacher overrides.
    const teacherProfile = settings.teachers?.length && !value('--teacher')
      ? routedTeacherProfile(programRoot, settings.teachers.map(profile => ({ profile, id: identityOf(profile) }))) : settings.teacher;
    const teacherConfig = loadModelConfiguration(teacherProfile);
    const teacherSession = createResolvedModelSession(teacherConfig.choice, process.env, process.stderr);
    try {
      const assertStudentBase = async () => {
        if (settings.backend !== 'native') return;
        const serving = loadModelConfiguration(undefined);
        if (serving.choice.kind !== 'external' || !serving.choice.neuraleseServer)
          throw new Error('natlang tune: native backend requires a default neuraleseServer profile');
        await requireNativeReaderIdentity(serving.choice.endpoint, student.base, {
          apiKey: process.env[serving.choice.apiKeyEnv], headers: serving.choice.headers });
      };
      return await fn({ root: programRoot, program, settings, student, teacher: driverOf(teacherSession),
        teacherId: executorIdentityForChoice(teacherConfig.choice).id, examplesPath: value('--examples'), log: io.err,
        assertStudentBase,
        serve: async ({ manifest, directory, capture }) => {
          if (settings.backend === 'native') {
            await assertStudentBase();
            const serving = loadModelConfiguration(undefined);
            if (serving.choice.kind !== 'external' || !serving.choice.neuraleseServer)
              throw new Error('natlang tune: the native backend needs a default profile with neuraleseServer: true');
            const session = createResolvedModelSession(serving.choice, process.env, process.stderr, { rlCapture: !!capture });
            await session.prepare();
            const loaded = await uploadRelease(session.nativeArtifacts!()!, directory,
              [...manifest.artifacts.map(item => item.artifact), ...manifest.families.map(item => item.artifact),
                ...manifest.softStates.map(item => item.state)], manifest.receipts.inferenceStateMaterializations);
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
          const adapters = [...manifest.artifacts, ...manifest.families].filter(item => item.exports?.gguf).map(item => ({ id: item.artifact, path: join(directory, item.exports!.gguf!.file) }));
          const serving = loadModelConfiguration(undefined);
          if (serving.choice.kind !== 'managed-local') throw new Error('natlang tune: the default profile must use the managed local model to serve tuned weights');
          // Soft states as exact native state spans on the llama.cpp fork (`--native-state`); refused by other servers.
          const states = manifest.softStates.filter(item => item.exports?.gguf).map(item => ({ id: item.state, path: join(directory, item.exports!.gguf!.file) }));
          const session = createResolvedModelSession(serving.choice, process.env, process.stderr,
            { loraAdapters: () => adapters, nativeStates: () => states });
          await session.prepare();
          return { driver: driverOf(session), loaded: new Set([...adapters, ...states].map(item => item.id)), nativeState: states.length > 0, close: () => session.close() };
        } });
    } finally { await teacherSession.close(); }
  };
  const once = () => withOptions(options => tuneProgram(options, flags.has('--rounds') ? roundsSelector(options, options.settings.rounds) : undefined));
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
      if (flags.has('--with-teacher')) arms.push(await evaluateRoots({ root, program: options.program, driver: options.teacher, weights: [], examples, services: options.services, execution: options.settings.execution,
        arm: 'teacher', release: null, partition }));
      for (const arm of arms) saveArm(root, arm);
    });
    io.out(formatOverview(loadOverview(root, partition)));
    return 0;
  }
  // Every named program stays open (sessions, data) while families are discovered, fit and members refit; held
  // programs are opened too but only measured for transfer.
  const familiesFor = async (programs: readonly string[], heldPrograms: readonly string[]): Promise<FamiliesReport> => {
    if (!programs.length) throw new Error('natlang tune --families: name the tuned programs, or tune programs first (none are registered for this base)');
    // Open members, then held programs, keeping every session open until the run ends.
    let report: FamiliesReport | undefined;
    const openAll = async (rest: readonly string[], opened: FamilyProgram[], heldRest: readonly string[], heldOpened: FamilyProgram[]): Promise<void> => {
      if (rest.length || heldRest.length) return withOptions(async options => {
        const active = currentRelease(options.root);
        if (!active?.active) throw new Error(`natlang tune --families: ${options.root} has no active release; run natlang tune there first`);
        const program = { options, data: await prepareData(options, [], io.err), release: active.manifest, releaseDir: active.directory };
        return rest.length ? openAll(rest.slice(1), [...opened, program], heldRest, heldOpened)
          : openAll(rest, opened, heldRest.slice(1), [...heldOpened, program]);
      }, rest[0] ?? heldRest[0]);
      const settings = opened[0]!.options.settings;
      const nl = async (discovery: string) => {
        const runtime = createNatlangRuntime({ model: opened[0]!.options.teacher });
        try { return await runtime.run(() => (loadNatlang(PROPOSE_FAMILIES) as (...args: unknown[]) => Promise<unknown>)(discovery)); }
        finally { runtime.close(); }
      };
      report = await runFamilies(opened, { ...settings.families, nl, log: io.err, held: heldOpened });
    };
    await openAll(programs, [], heldPrograms, []);
    return report!;
  };
  const familyStoreOf = () => familyStore(loadTuneSettings(process.env, { student: value('--student'), teacher: value('--teacher') }).student.base);
  if (flags.has('--families')) {
    // Without program arguments: every program tuned on this base (the family registry).
    const programs = words.length ? words.map(word => resolve(word)) : Object.keys(familyRegistry(familyStoreOf()).programs);
    const heldPrograms = (value('--held') ?? '').split(',').filter(Boolean).map(item => resolve(item));
    const report = await familiesFor(programs.filter(item => !heldPrograms.includes(item)), heldPrograms);
    io.out(flags.has('--json') ? JSON.stringify(report, null, 2) : [
      `families: ${report.published.map(item => `${item.family}@${item.version} (${item.members.length} members)`).join(', ') || 'none published'}` +
        (report.absent.length ? `; absent: ${report.absent.join(', ')}` : ''),
      ...report.programs.map(item => `${item.program}: ${item.activated ? 'activated' : 'kept previous'} ${item.release?.slice(0, 12) ?? ''}` +
        (item.development ? ` (development ${item.development.previous.toFixed(2)} -> ${item.development.refit.toFixed(2)}` +
          `${item.negativeTransfer ? ', negative transfer' : ''})` : '') + (item.reason ? `: ${item.reason}` : '')),
      ...report.transfer.map(item => `transfer ${item.family}@${item.version} -> ${item.kind} ${item.program}: ` + (item.development
        ? `development ${item.development.current.toFixed(2)} -> ${item.development.withFamily.toFixed(2)} (delta ${item.development.delta.toFixed(3)}` +
          ` [${item.development.interval.map(value => value.toFixed(3)).join(', ')}]${item.development.underpowered ? ', underpowered' : ''})` : item.reason ?? ''))].join('\n'));
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
  const exportDir = value('--export-cases'), importPath = value('--import-cases');
  if (value('--import-campaign') && importPath === undefined) throw new Error('--import-campaign is only valid with --import-cases');
  if (exportDir !== undefined || importPath !== undefined) {
    const { settings, student } = loadTuneSettings(process.env, { student: value('--student'), teacher: value('--teacher') });
    const inventory = buildProject({ project: root, emit: false, write: false, runtimeModule: RUNTIME_MODULE });
    if (!inventory.ok) throw new Error(formatDiagnostics(inventory.diagnostics));
    const program = inventory.manifest.adaptation!;
    const examples = currentSplit(root, value('--examples'));
    const context = { root, program, examples, splitReceipt: examples.receipt,
      scaffold: scaffoldManifest(program, student.base, { sites: programSites(root, program).sites }) };
    if (exportDir !== undefined) {
      const result = exportCases(context, resolve(exportDir), { includeConfirmation: flags.has('--include-confirmation'),
        teacherId: identityOf(settings.teacher) });
      io.out(flags.has('--json') ? JSON.stringify(result, null, 2) : `${result.cases} case(s) written; ${result.skipped.length} root(s) skipped; ` +
        `manifest ${result.manifest}\nnext: root review and dispatch through the reviewed pool (see plans/TUNING_OPERATIONS_HANDOFF_20261026.md)`);
      return 0;
    }
    const python = process.env.NATLANG_PYTHON ?? 'python3';
    const verifier = fileURLToPath(new URL('../../../scripts/neuralese_data/verify_tune_pool_import.py', import.meta.url));
    const snapshotDir = join(tuneDirectory(root), 'pool-import-custody-snapshots');
    const verifyArgs = [verifier, '--input', resolve(importPath!), '--snapshot-dir', snapshotDir,
      ...(value('--import-campaign') ? ['--campaign', resolve(value('--import-campaign')!)] : [])];
    const verified = spawnSync(python, verifyArgs, { cwd: process.cwd(), encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 });
    if (verified.error || verified.status !== 0) throw new Error(`reviewed tune campaign custody verification failed: ${verified.error?.message ?? verified.stderr.trim()}`);
    let custody;
    try { custody = JSON.parse(verified.stdout); }
    catch { throw new Error('reviewed tune campaign custody verifier returned invalid JSON'); }
    const summary = importCases(context, resolve(importPath!), custody);
    io.out(flags.has('--json') ? JSON.stringify(summary, null, 2) : `imported ${summary.imported}/${summary.rows} tune case row(s) ` +
      `(${summary.failures} failed roots kept as failures, ${summary.unboundTurns} turn(s) without a binding); rejected ${summary.rejected.length}` +
      summary.rejected.slice(0, 5).map(item => `\n  ${item.row}: ${item.reason}`).join(''));
    return summary.rejected.length && !summary.imported ? 1 : 0;
  }
  if (flags.has('--calibrate')) {
    const { settings } = loadTuneSettings(process.env, { student: value('--student'), teacher: value('--teacher') });
    const inventory = buildProject({ project: root, emit: false, write: false, runtimeModule: RUNTIME_MODULE });
    if (!inventory.ok) throw new Error(formatDiagnostics(inventory.diagnostics));
    const profiles = value('--teacher') ? [value('--teacher')!] : settings.teachers?.length ? settings.teachers : [settings.teacher];
    const sessions = profiles.map(profile => ({ profile, session: createResolvedModelSession(loadModelConfiguration(profile).choice, process.env, process.stderr) }));
    try {
      const examples = currentSplit(root, value('--examples')).filter(example => example.partition === 'development' && !example.protected);
      const calibrations = await calibrateTeachers({ root, program: inventory.manifest.adaptation!, examples, log: io.err, execution: settings.execution,
        candidates: sessions.map(({ profile, session }) => ({ profile, id: identityOf(profile), driver: driverOf(session) })) });
      // The natural-language policy runs on the default teacher; it only chooses among measured eligible teachers.
      const policySession = sessions.find(item => item.profile === settings.teacher)?.session ?? sessions[0]!.session;
      const nl = async (candidates: string, minSuccess: number) => {
        const runtime = createNatlangRuntime({ model: driverOf(policySession) });
        try { return await runtime.run(() => (loadNatlang(ROUTE_TEACHER) as (...args: unknown[]) => Promise<unknown>)(candidates, minSuccess)); }
        finally { runtime.close(); }
      };
      const choice = await routeTeacher(calibrations, settings.teacherRouting, nl);
      saveTeacherChoice(root, choice);
      io.out(flags.has('--json') ? JSON.stringify(choice, null, 2) : [
        ...calibrations.map(item => `${item.profile}: ${item.success}/${item.roots} correct, ${item.failures} failed, ` +
          `${Math.round(item.tokensPerRoot)} tokens and ${Math.round(item.msPerRoot)} ms per root`),
        choice.teacher ? `teacher: ${choice.profile} (${choice.policy}: ${choice.reason})` : `no teacher chosen: ${choice.reason}`].join('\n'));
      return choice.teacher ? 0 : 1;
    } finally { for (const { session } of sessions) await session.close(); }
  }
  if (flags.has('--distill')) {
    const result = await withOptions(async options => {
      const current = currentManifest(root);
      if (!current) throw new Error('natlang tune --distill exports the active release; run natlang tune first');
      const exported = await exportDistillation(options, await prepareData(options, [], io.err), current, io.err);
      await runStudentModule(options.student, 'natlang_neuralese.tune.distill', ['--export', exported.directory,
        '--checkpoint', options.student.checkpoint, '--device', options.student.engine?.device ?? 'cpu']);
      return exported;
    });
    io.out(flags.has('--json') ? JSON.stringify(result, null, 2) : `distillation export: ${result.manifest.turns.rows} training rows ` +
      `(${result.manifest.turns.bySource.teacher} teacher, ${result.manifest.turns.bySource.stack} tuned stack; ${result.manifest.roots.correct}/` +
      `${result.manifest.roots.rerun} training roots correct) in ${result.directory}; not admitted: register it in training/neuralese_corpora.json to train on it`);
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
  // Incremental builds against the active release (plan §8): a leaf candidate, a shared refit, or a periodic full
  // refit. Undefined when there is no active release (the ordinary full tune runs); null when nothing changed.
  const incrementalBuild = async (): Promise<TuneReport[] | null | undefined> => {
    const active = currentRelease(root);
    if (!active?.active) return undefined;
    const { settings } = loadTuneSettings(process.env, { student: value('--student'), teacher: value('--teacher') });
    const inventory = buildProject({ project: root, emit: false, write: false, runtimeModule: RUNTIME_MODULE });
    if (!inventory.ok) throw new Error(formatDiagnostics(inventory.diagnostics));
    const program = inventory.manifest.adaptation!;
    const current = scaffoldManifest(program, active.manifest.base, { sites: programSites(root, program).sites });
    const change = invalidation(active.manifest, current);
    io.err(`natlang tune: changed ${change.changed.length}, added ${change.added.length}, removed ${change.removed.length}, moved ${change.moved.length}, renamed ${change.rekeyed.length} binding(s); ` +
      `dirty nodes ${change.dirty.length}, dependent residuals ${change.dependents.length}; unchanged roots reuse their data`);
    return runIncrementalBuilds(root, current, program.buildHash, { full: flags.has('--full'), settings: settings.incremental,
      recipe: { seed: 0, ...settings.recipe }, log: io.err }, step => withOptions(options => tuneProgram(options, incrementalSelector(options, step))));
  };
  if ((flags.has('--incremental') || flags.has('--full')) && !flags.has('--watch')) {
    const reports = await incrementalBuild();
    if (reports === null) { io.out('fresh: the active release matches the program; nothing to rebuild'); return 0; }
    if (reports) {
      io.out(flags.has('--json') ? JSON.stringify(reports, null, 2) : reports.map(formatReport).join('\n'));
      return reports.every(report => report.activated) ? 0 : 1;
    }
  }
  if (!flags.has('--watch')) {
    const report = await once();
    io.out(flags.has('--json') ? JSON.stringify(report, null, 2) : formatReport(report));
    if (report.activated && report.release) {
      // Families run after a batch of newly tuned programs (tune.families.every), over every program of this base.
      const store = familyStoreOf(), every = loadTuneSettings(process.env, { student: value('--student') }).settings.families?.every;
      if (registerTunedProgram(store, root, report.release, every)) {
        const programs = Object.keys(familyRegistry(store).programs);
        io.err(`natlang tune: ${every} newly tuned program release(s) since the last family run; running families over ${programs.length} programs`);
        if (programs.length >= 2) await familiesFor(programs, []);
      }
    }
    return report.activated ? 0 : 1;
  }
  // Watch: coalesce edits, one build at a time; activation compares the pointer, so a stale build cannot win.
  let running = false, again = false, timer: ReturnType<typeof setTimeout> | undefined;
  const build = async () => {
    if (running) { again = true; return; }
    running = true;
    try {
      // Edits rebuild incrementally against the active release; the first build (no release yet) is a full tune.
      const reports = await incrementalBuild();
      if (reports === null) return;
      for (const report of reports ?? [await once()]) io.out(formatReport(report));
    } catch (error) { io.err(`natlang tune: ${(error as Error).message}`); }
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
