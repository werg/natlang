/** Canonical program-IR execution through the existing evaluation fixture frontend. */
import { mkdtempSync, mkdirSync, writeFileSync, readdirSync, existsSync, symlinkSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, relative, resolve } from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { buildProject } from '../compiler/node-project.js';
import { formatDiagnostics } from '../compiler/project.js';
import { evaluationProjectFiles } from '../evaluation/project-files.js';
import { loadEvaluationSuite } from '../evaluation/suite.js';
import { evaluate } from '../evaluation/runner.js';
import { fingerprint } from '../adaptation/identity.js';
import type { InvocationTrace, ModelDriver } from '../runtime/runtime.js';
import type { ProgramRecord, EvaluationFixtureSpec } from './program.js';
import type { ExecuteOptions, ProgramRun } from './collector.js';

export const EVALUATION_FIXTURE_SCHEMA = 'natlang.evaluation-fixture/1';
const same = (a: unknown, b: unknown) => fingerprint(a) === fingerprint(b);
function pathName(value: unknown, label: string, dot = false): asserts value is string {
  if (dot && value === '.') return;
  if (typeof value !== 'string' || !value || value.includes('\\') || value.includes('\0') ||
      value.startsWith('/') || value.split('/').some(part => !part || part === '.' || part === '..'))
    throw new TypeError(`${label} must be a normalized relative path`);
}

/** Host code, services and oracle assets never become model-owned program files. */
export function validateEvaluationFixture(record: ProgramRecord): void {
  const spec = record.semantics.evaluation_fixture;
  if (record.kind !== 'evaluation_fixture' || !spec || spec.schema !== EVALUATION_FIXTURE_SCHEMA)
    throw new TypeError('evaluation_fixture requires the supported program kind and fixture schema');
  pathName(spec.suite_module, 'fixture suite_module');
  pathName(spec.program_root, 'fixture program_root', true);
  pathName(spec.program_entry, 'fixture program_entry');
  if (!spec.host_files || Array.isArray(spec.host_files) || typeof spec.host_files !== 'object')
    throw new TypeError('fixture host_files must contain complete portable source and assets');
  for (const [path, source] of Object.entries(spec.host_files)) {
    pathName(path, 'fixture host file');
    if (typeof source !== 'string' || path.endsWith('.nl') ||
        path.split('/').some(part => ['node_modules', '.git', '.natlang'].includes(part) || part.startsWith('vendor')) ||
        /(?:^|\/)cases(?:\.[^/]*)?$/.test(path))
      throw new TypeError('fixture host files must omit model NL files, unselected cases and vendored/build outputs');
  }
  if (typeof spec.host_files[spec.suite_module] !== 'string') throw new TypeError('fixture suite_module is absent');
  for (const [path, source] of Object.entries(record.semantics.files)) {
    pathName(path, 'model source file');
    if (typeof source !== 'string' || (!/\.(?:nl|ts|mts|tsx)$/.test(path) &&
        !['package.json', 'natlang.json', 'tsconfig.json'].includes(path)) ||
        path.split('/').includes('__tests__') || /\.(?:test|spec)\./.test(path) ||
        /(?:^|\/)(?:suite|oracle|verifier|cases)(?:\.[^/]*)?$/.test(path))
      throw new TypeError('model files contain a fixture, test, oracle or unsupported source asset');
  }
  if (typeof record.semantics.files[spec.program_entry] !== 'string') throw new TypeError('fixture program_entry is absent from model source');
  if (!spec.case || !spec.case.id || !spec.case.group || !['train', 'validation', 'test'].includes(spec.case.split) ||
      (record.split !== undefined && record.split !== spec.case.split) || !same(spec.case.expected ?? null, record.semantics.expected))
    throw new TypeError('fixture selected case, expected value and split must agree with the task');
  if (!spec.entry || !['natlang-root', 'authored-module'].includes(spec.entry.kind) ||
      !['entry', 'semantic-anchor'].includes(spec.root_role)) throw new TypeError('fixture entry and root_role are required');
  pathName(spec.entry.module, 'fixture entry module');
  if (typeof record.semantics.files[spec.entry.module] !== 'string' ||
      typeof spec.entry.export !== 'string' || !/^[A-Za-z_$][A-Za-z0-9_$]*$/.test(spec.entry.export))
    throw new TypeError('fixture entry must name a real exported model source function');
  if (!Array.isArray(spec.entry.export_path) || !spec.entry.export_path.length || spec.entry.export_path.some(name => typeof name !== 'string' || !/^[A-Za-z_$][A-Za-z0-9_$]*$/.test(name)))
    throw new TypeError('fixture entry export_path must name its actual loaded program binding');
  if (!Array.isArray(spec.entry.input_arguments) || !Array.isArray(spec.service_inputs) ||
      [...spec.entry.input_arguments, ...spec.service_inputs].some(name => typeof name !== 'string' || !/^[A-Za-z_$][A-Za-z0-9_$]*$/.test(name)) ||
      new Set(spec.entry.input_arguments).size !== spec.entry.input_arguments.length ||
      new Set(spec.service_inputs).size !== spec.service_inputs.length ||
      spec.service_inputs.some(name => !spec.entry.input_arguments.includes(name) || Object.hasOwn(record.semantics.inputs, name)) ||
      !same([...spec.entry.input_arguments].sort(), [...Object.keys(record.semantics.inputs), ...spec.service_inputs].sort()))
    throw new TypeError('fixture entry arguments must bind actual JSON inputs or explicitly constructed service inputs');
  if (spec.entry.kind === 'natlang-root' &&
      (spec.root_role !== 'entry' || spec.entry.module !== record.semantics.root || spec.entry.export !== 'default'))
    throw new TypeError('a natlang-root fixture entry must execute its real NL root');
  if (spec.entry.kind === 'authored-module' && !/\.(?:ts|mts|tsx)$/.test(spec.entry.module))
    throw new TypeError('authored-module fixture entry must name TypeScript source');
  if (record.semantics.application_controller || record.semantics.folder_files || record.semantics.effects ||
      record.semantics.trusted_exact_verifier || record.semantics.failure_seed)
    throw new TypeError('evaluation_fixture owns its host world and scorer; it cannot silently ignore another execution contract');
}

function put(root: string, path: string, source: string): void {
  const destination = join(root, path);
  mkdirSync(dirname(destination), { recursive: true }); writeFileSync(destination, source);
}

/** Bind the runtime's APIs/dependencies, without installing packages or copying a runtime into a task. */
function dependencies(root: string): void {
  const modules = join(root, 'node_modules'); mkdirSync(modules, { recursive: true });
  const require = createRequire(import.meta.url);
  for (const name of readdirSync(dirname(dirname(require.resolve('typescript/package.json'))))) {
    if (name.startsWith('.') || name === '@workspace' || name === '@natlang') continue;
    const source = join(dirname(dirname(require.resolve('typescript/package.json'))), name);
    if (!existsSync(join(modules, name))) symlinkSync(source, join(modules, name));
  }
  for (const name of ['@workspace/natlang', '@natlang/node']) {
    const directory = join(modules, name); mkdirSync(directory, { recursive: true });
    put(directory, 'package.json', JSON.stringify({ name, type: 'module', exports: {
      '.': './index.js', './evaluation': './evaluation.js', './model': './evaluation.js' } }));
    for (const [entry, target] of [['index', '../index.js'], ['evaluation', '../entry/evaluation.js']]) {
      const url = new URL(target!, import.meta.url);
      put(directory, `${entry}.js`, `export * from ${JSON.stringify(url.href)};\n`);
      put(directory, `${entry}.d.ts`, `export * from ${JSON.stringify(fileURLToPath(url))};\n`);
    }
  }
}

function sourceOf(trace: InvocationTrace): string | undefined {
  const value = trace.events.find(event => event.kind === 'manifest')?.definition_source;
  return typeof value === 'string' ? value.replace(/\\/g, '/') : undefined;
}

/** Fixture code is a trusted host world; all semantic execution still comes from this IR's model source. */
export async function executeEvaluationFixture(record: ProgramRecord, driver: ModelDriver, options: ExecuteOptions): Promise<ProgramRun> {
  validateEvaluationFixture(record);
  const spec = record.semantics.evaluation_fixture!;
  if (options.calls !== undefined && options.calls !== false) throw new Error('evaluation fixtures cannot transfer a live call store across a worker boundary');
  if (options.trainingGuidance || options.programGuidance || options.neuralese || options.neuraleseService)
    throw new Error('evaluation fixture frontend does not yet carry native state/guidance; use a compatible declared fixture task');
  const scratch = mkdtempSync(join(tmpdir(), 'natlang-evaluation-fixture-'));
  try {
    put(scratch, 'package.json', '{"type":"module"}\n'); dependencies(scratch);
    const host = join(scratch, 'host'), model = join(scratch, 'program');
    for (const [path, source] of Object.entries(spec.host_files)) put(host, path, source);
    for (const [path, source] of Object.entries(record.semantics.files)) put(model, path, source);
    const hostOut = join(host, '.natlang', 'fixture');
    const built = buildProject({ project: host, files: evaluationProjectFiles, outDir: hostOut,
      write: true, writeDeclarations: false, runtimeModule: { specifiers: ['@natlang/node'],
        url: new URL('../index.js', import.meta.url).href, types: fileURLToPath(new URL('../index.d.ts', import.meta.url)) } });
    if (!built.ok) throw new Error(formatDiagnostics(built.diagnostics));
    const suiteModule = pathToFileURL(join(hostOut, spec.suite_module.replace(/\.m?ts$/, '.js'))).href;
    const modelId = (driver as { model?: string }).model ?? 'undeclared:teacher-driver';
    const identity = { id: modelId, configuration: { model: modelId, promptRendering: options.promptRendering ?? 'full' } };
    // Mechanical host wrapper only: reuse the authored fixture/score functions, with the IR's one selected case and source.
    const wrapper = join(scratch, 'suite.mjs');
    const settings = { ...(options.maxTurns === undefined ? {} : { maxTurns: options.maxTurns }),
      ...(options.temperature === undefined ? {} : { temperature: options.temperature }),
      ...(options.contextTokens === undefined ? {} : { contextTokens: options.contextTokens }),
      ...(options.promptRendering === undefined ? {} : { promptRendering: options.promptRendering }) };
    put(scratch, 'suite.mjs', `import authored from ${JSON.stringify(suiteModule)};\n` +
      `export default { ...authored, cases: ${JSON.stringify([spec.case])}, program: { ...authored.program, root: ${JSON.stringify(model)}, entry: ${JSON.stringify(spec.program_entry)} },\n` +
      `executorIdentity: ${JSON.stringify(identity)}, ${options.systemPrompt === undefined ? "" : `systemPrompt: ${JSON.stringify(options.systemPrompt)},`}\n` +
      `policy: { ...authored.policy, ${options.systemPrompt === undefined ? "" : `systemPromptHash: ${JSON.stringify(fingerprint(options.systemPrompt, 'natlang.system-prompt/v1'))},`} settings: { ...authored.policy?.settings, ...${JSON.stringify(settings)} } } };\n`);
    const prepared = await loadEvaluationSuite(wrapper);
    // Hosted teachers demonstrate finite readouts through real generation, without a scored-decision shortcut.
    const generationDriver: ModelDriver = Object.assign((...args: Parameters<ModelDriver>) => driver(...args), { model: modelId });
    const batch = await evaluate({ ...prepared, programRoot: resolve(host, spec.program_root) }, {
      caseIds: [spec.case.id], ...(options.calls === false ? { calls: false } : {}), driver: generationDriver,
      fixtureEntry: { ...spec.entry, inputs: record.semantics.inputs, service_inputs: spec.service_inputs }, seed: options.rootSeed, signal: options.signal, runId: options.runId,
      timeoutMs: 2 ** 31 - 1 });
    const result = batch.results[0]!;
    const traces = [...result.traces];
    const sourcePath = (source: string) => source.startsWith(model + '/') ? relative(model, source) : source.replace(/^\/project\//, '');
    const sourceBinding = traces.every(trace => {
      const source = sourceOf(trace);
      return source ? Object.hasOwn(record.semantics.files, sourcePath(source)) :
        typeof trace.adaptation?.component === 'string' && prepared.program.components.some(component => component.key === trace.adaptation!.component);
    });
    const primary = traces.find(trace => sourceOf(trace) && sourcePath(sourceOf(trace)!) === record.semantics.root) ?? traces.find(trace => !trace.parentCallId) ?? traces[0];
    const events = traces.flatMap(trace => trace.events);
    const returned = result.outcome.kind === 'returned';
    const status = primary?.outcome ?? (returned ? 'done' : result.outcome.error?.outcome ?? 'failed');
    const gates = Object.values(result.gates).every(value => value === true);
    const checks = { expected_status: returned && status === 'done', answer: result.quality === 1,
      fixture_gates: gates, model_source_binding: sourceBinding, fixture_entry_binding: (result.fixtureEntry?.calls ?? 0) > 0,
      fixture_root_binding: spec.root_role === 'semantic-anchor' || traces.some(trace => sourceOf(trace) && sourcePath(sourceOf(trace)!) === record.semantics.root) };
    const accepted = Object.values(checks).every(Boolean);
    return { trace: primary?.events ?? [], invocationTraces: traces.filter(trace => trace !== primary), outcome: {
      status, detail: primary?.detail ?? result.outcome.error?.message ?? '', value: result.outcome.value,
      accepted, checks, rejection_reasons: Object.entries(checks).filter(([, ok]) => !ok).map(([name]) => name),
      oracle: { accepted: returned && result.quality === 1 && gates, level: 'constraints', score: result.quality,
        verdict: 'evaluation-fixture/1' },
      evaluation: { schema: 'natlang.evaluation-fixture-result/1', suite_id: prepared.suite.id,
        suite_hash: fingerprint({ fixture: spec, model: record.semantics.files, root: record.semantics.root,
          inputs: record.semantics.inputs, identity, policy: prepared.suite.policy, systemPrompt: prepared.suite.systemPrompt ?? null },
          'natlang.evaluation-fixture-suite/v1'), case_id: spec.case.id, group: spec.case.group, split: spec.case.split,
        quality: result.quality, outcome: result.outcome, fixture_entry: result.fixtureEntry, gates: result.gates, metrics: result.metrics, feedback: result.feedback,
        observation: result.observation, observations: result.observations, entry: spec.entry, root_role: spec.root_role },
      invocation_ledger: traces.map(trace => ({ invocation_id: trace.callId, parent_invocation_id: trace.parentCallId,
        definition_id: trace.definitionId, name: trace.name, definition_source: sourceOf(trace) ?? null,
        completion_status: trace.outcome, completion_source: 'runtime_invocation_trace' })),
      execution_graph: events.filter(event => typeof event.node === 'string' && Array.isArray(event.inputs)),
      action_ledger: events.filter(event => event.kind === 'action'),
      scope_failures: events.filter(event => event.kind === 'scope_failure'), host_events: events.filter(event => event.kind === 'host'),
    } };
  } finally { rmSync(scratch, { recursive: true, force: true }); }
}
