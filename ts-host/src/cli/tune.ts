/**
 * `natlang tune` (plans/HIERARCHICAL_LORA_TUNING.md §8):
 *
 *   natlang tune <program>                 discover, collect/reuse data, train, export, load, validate, activate
 *   natlang tune <program> --incremental   rebuild what changed (reuses valid data and training receipts)
 *   natlang tune <program> --watch         rebuild in the background on edits; the run/call path keeps serving
 *   natlang tune <program> --report        the latest report, status and the active release
 *   natlang tune <program> --rollback      reactivate the previous coherent release
 *   natlang tune --families                cross-program family discovery and refits
 */
import { existsSync, readdirSync, readFileSync, watch } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildProject } from '../compiler/node-project.js';
import { formatDiagnostics } from '../compiler/project.js';
import { createResolvedModelSession, executorIdentityForChoice, loadModelConfiguration } from '../model/index.js';
import { readActive, rollbackRelease, tuneDirectory, tuneStatus } from '../adaptation/weights/release.js';
import type { ModelDriver } from '../runtime/runtime.js';
import type { DecisionRequest, ModelTurnRequest, ModelTurnOptions } from '../contracts.js';
import { loadTuneSettings } from '../tune/settings.js';
import { tuneProgram, type TuneReport } from '../tune/pipeline.js';

const RUNTIME_MODULE = { url: new URL('../index.js', import.meta.url).href, path: fileURLToPath(new URL('../index.js', import.meta.url)),
  types: fileURLToPath(new URL('../index.d.ts', import.meta.url)), specifiers: ['@natlang/node'] };

type Session = ReturnType<typeof createResolvedModelSession>;
const driverOf = (session: Session): ModelDriver => Object.assign(
  (request: ModelTurnRequest, signal?: AbortSignal, options?: ModelTurnOptions) => session.turn(request, signal, options),
  { decide: (request: DecisionRequest, signal?: AbortSignal) => session.decide(request, signal), contextWindow: () => session.contextWindow() });

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
  const flags = new Set(argv.filter(arg => arg.startsWith('--') && !['--examples', '--student', '--teacher'].includes(arg)));
  const value = (name: string) => { const index = argv.indexOf(name); return index >= 0 ? argv[index + 1] : undefined; };
  const valued = new Set(['--examples', '--student', '--teacher'].flatMap(name => { const i = argv.indexOf(name); return i >= 0 ? [i, i + 1] : []; }));
  const words = argv.filter((arg, index) => !arg.startsWith('--') && !valued.has(index));
  const known = new Set(['--incremental', '--watch', '--report', '--rollback', '--families', '--json']);
  for (const flag of flags) if (!known.has(flag)) throw new Error(`natlang tune: unknown option ${flag}`);
  if (flags.has('--families'))
    throw new Error('natlang tune --families: cross-program family training is not implemented yet (plans/HIERARCHICAL_LORA_TUNING.md §5.5); program tunes run with no family level');
  const root = resolve(words[0] ?? '.');
  if (!existsSync(root)) throw new Error(`natlang tune: ${root} does not exist`);
  if (flags.has('--rollback')) { const record = rollbackRelease(root); io.out(`active release is now ${record.release.slice(0, 12)}`); return 0; }
  if (flags.has('--report')) {
    const reports = join(tuneDirectory(root), 'reports');
    const latest = existsSync(reports) ? readdirSync(reports).sort().at(-1) : undefined;
    const status = tuneStatus(root);
    if (flags.has('--json')) io.out(JSON.stringify({ status, active: readActive(root) ?? null,
      report: latest ? JSON.parse(readFileSync(join(reports, latest), 'utf8')) : null }, null, 2));
    else io.out([`status: ${status.status}${status.reason ? ` (${status.reason})` : ''}`,
      latest ? formatReport(JSON.parse(readFileSync(join(reports, latest), 'utf8')) as TuneReport) : 'no tune report yet'].join('\n'));
    return 0;
  }
  const once = async (): Promise<TuneReport> => {
    const { settings, student } = loadTuneSettings(process.env, { student: value('--student'), teacher: value('--teacher') });
    const inventory = buildProject({ project: root, emit: false, write: false, runtimeModule: RUNTIME_MODULE });
    if (!inventory.ok) throw new Error(formatDiagnostics(inventory.diagnostics));
    const program = inventory.manifest.adaptation!;
    const teacherConfig = loadModelConfiguration(settings.teacher);
    const teacherSession = createResolvedModelSession(teacherConfig.choice, process.env, process.stderr);
    try {
      return await tuneProgram({ root, program, settings, student, teacher: driverOf(teacherSession),
        teacherId: executorIdentityForChoice(teacherConfig.choice).id, examplesPath: value('--examples'), log: io.err,
        serve: async ({ manifest, directory }) => {
          if (settings.backend !== 'managed-local') throw new Error(`natlang tune: serving backend ${settings.backend} is not wired into tune validation yet`);
          const adapters = manifest.artifacts.filter(item => item.exports?.gguf).map(item => ({ id: item.artifact, path: join(directory, item.exports!.gguf!.file) }));
          const serving = loadModelConfiguration(undefined);
          if (serving.choice.kind !== 'managed-local') throw new Error('natlang tune: the default profile must use the managed local model to serve tuned weights');
          const session = createResolvedModelSession(serving.choice, process.env, process.stderr, { loraAdapters: () => adapters });
          await session.prepare();
          return { driver: driverOf(session), loaded: new Set(adapters.map(item => item.id)), close: () => session.close() };
        } });
    } finally { await teacherSession.close(); }
  };
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
