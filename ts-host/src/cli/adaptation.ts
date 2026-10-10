import { mkdirSync, writeFileSync, readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { buildProject } from '../compiler/node-project.js';
import { formatDiagnostics } from '../compiler/project.js';
import { bindAdaptation } from '../adaptation/compatibility.js';
import { canonical, fingerprint } from '../adaptation/identity.js';
import { loadEvaluationSuite } from '../evaluation/suite.js';
import { evaluate, preflightEvaluation } from '../evaluation/runner.js';
import { loadAdaptation, exportAdaptationPatch, revalidateAdaptation } from '../optimization/export-patch.js';
import { optimize, resumeOptimization } from '../optimization/optimize.js';
import { loadModelConfiguration, createResolvedModelSession, executorIdentityForChoice, localModelPrerequisites } from '../model/index.js';
import type { ModelDriver } from '../runtime/runtime.js';
export async function adaptationCommand(argv: string[]): Promise<number> {
  const [command, ...args] = argv; const words: string[] = [], flags = new Map<string, string | true>();
  const boolean = new Set(['--json', '--baseline', '--dry-run', '--preflight']);
  const valued = new Set(['--project', '--out', '--suite', '--artifact', '--split', '--strategy', '--seed', '--mapping']);
  try {
  for (let at = 0; at < args.length; at++) {
    const value = args[at]!;
    if (!value.startsWith('--')) words.push(value);
    else if (flags.has(value)) throw new Error('duplicate option: ' + value);
    else if (boolean.has(value)) flags.set(value, true);
    else if (!valued.has(value)) throw new Error('unknown option: ' + value);
    else { const next = args[++at]; if (next === undefined || next.startsWith('--')) throw new Error('missing value: ' + value); flags.set(value, next); }
  }
  const option = (name: string) => typeof flags.get(name) === 'string' ? flags.get(name) as string : undefined;
  const output = (value: unknown) => process.stdout.write(JSON.stringify(value, null, 2) + '\n');
  const action = command === 'adapt' ? words[0] : command;
  const allowed: Record<string, readonly string[]> = {
    inspect: ['--json', '--project'], 'export-source': ['--json', '--project', '--out'],
    revalidate: ['--json', '--suite', '--mapping', '--out', '--dry-run', '--preflight'],
    eval: ['--json', '--baseline', '--artifact', '--split', '--dry-run', '--preflight'],
    optimize: ['--json', '--strategy', '--out', '--seed', '--suite', '--dry-run', '--preflight'],
  };
  if (!action || !allowed[action] || !['adapt', 'eval', 'optimize'].includes(command!)) throw new Error('unknown adaptation command');
  for (const flag of flags.keys()) if (!allowed[action]!.includes(flag)) throw new Error('option is unsupported for ' + action + ': ' + flag);
  if (command === 'adapt' && action !== 'inspect' && words.length !== 2 || command === 'eval' && words.length !== 1 ||
    command === 'optimize' && words.length !== (words[0] === 'resume' ? 2 : 1) || command === 'adapt' && action === 'inspect' && words.length > 2)
    throw new Error('supply the expected suite, artifact, or run path for ' + action);
    if (command === 'adapt' && words[0] === 'inspect') {
      const path = resolve(words[1] ?? '.');
      if (path.endsWith('.json')) {
        const artifact = loadAdaptation(path); const project = option('--project');
        if (!project) { output(artifact); return 0; }
        const result = buildProject({ project, programId: artifact.program.id, emit: false, write: false });
        if (!result.ok) throw new Error(formatDiagnostics(result.diagnostics));
        const binding = bindAdaptation(artifact, result.manifest.adaptation!, artifact.executor); output({ compatible: true, artifact: binding.artifact }); return 0;
      }
      const result = buildProject({ project: path, emit: false, write: false });
      if (!result.ok) throw new Error(formatDiagnostics(result.diagnostics));
      output({ ...result.manifest.adaptation, exclusions: [
        { origin: 'runtime-generated', reason: 'eval nl, delegate and generated children have invocation lineage but no persistent authored source contract' },
        { origin: 'unregistered', reason: 'defineNatlang callables without compiler-authored metadata cannot be selected persistently' },
      ] }); return 0;
    }
    if (command === 'adapt' && words[0] === 'export-source') {
      const artifact = loadAdaptation(words[1]!); const project = resolve(option('--project') ?? '.');
      const result = buildProject({ project, programId: artifact.program.id, emit: false, write: false });
      if (!result.ok) throw new Error(formatDiagnostics(result.diagnostics));
      const exported = exportAdaptationPatch(artifact, result.manifest.adaptation!, artifact.executor, project);
      const out = resolve(option('--out') ?? 'adaptation.patch'); mkdirSync(dirname(out), { recursive: true }); writeFileSync(out, exported.patch);
      writeFileSync(out + '.manifest.json', canonical({ schema: 'natlang.adaptation-source-export/v1', files: exported.manifest }));
      if (exported.guidance !== undefined) { writeFileSync(out + '.guidance.txt', exported.guidance); writeFileSync(out + '.guidance-integration.txt', 'Pass this authored guidance through buildProject({ guidance }) and rebuild program metadata before revalidation.'); }
      output({ patch: out, manifest: out + '.manifest.json', guidance: exported.guidance !== undefined ? out + '.guidance.txt' : null }); return 0;
    }
    const resume = command === 'optimize' && words[0] === 'resume';
    const suitePath = command === 'adapt' ? option('--suite') : resume ? option('--suite') : words[0];
    if (!suitePath) throw new Error('supply an evaluation suite module; resume requires --suite');
    const prepared = await loadEvaluationSuite(suitePath);
    const executorConfig = loadModelConfiguration(prepared.suite.models?.executor);
    const executorIdentity = executorIdentityForChoice(executorConfig.choice);
    if (fingerprint(executorIdentity) !== fingerprint(prepared.suite.executorIdentity)) throw new Error('suite executorIdentity does not match resolved profile; inspect executorIdentityForChoice');
    if (flags.has('--dry-run') || flags.has('--preflight')) {
      const models = [executorConfig, ...(prepared.suite.models?.reflection ? [loadModelConfiguration(prepared.suite.models.reflection)] : []),
        ...(prepared.suite.models?.judge ? [loadModelConfiguration(prepared.suite.models.judge)] : [])];
      if (models.some(model => model.choice.kind === 'managed-local') && !localModelPrerequisites().available) throw new Error('managed model is unavailable; run natlang --setup');
      const fixtures = await preflightEvaluation(prepared);
      output({ ok: true, suite: prepared.suite.id, suiteHash: prepared.suiteHash, fixtures,
        components: prepared.components, cases: prepared.cases.map(testCase => ({ id: testCase.id, split: testCase.split })), executorIdentity }); return 0;
    }
    const executor = createResolvedModelSession(executorConfig.choice, process.env, process.stderr);
    let reflection: ReturnType<typeof createResolvedModelSession> | undefined;
    let judge: ReturnType<typeof createResolvedModelSession> | undefined;
    const judgeConfig = prepared.suite.models?.judge ? loadModelConfiguration(prepared.suite.models.judge) : undefined;
    if (judgeConfig) judge = createResolvedModelSession(judgeConfig.choice, process.env, process.stderr);
    const abort = new AbortController(); const interrupt = () => abort.abort(new Error('optimization interrupted'));
    process.once('SIGINT', interrupt);
    try {
      const driver: ModelDriver = (request, signal, options) => executor.turn(request, signal, options);
      if (command === 'eval') {
        const artifactPath = option('--artifact'); if (artifactPath && flags.has('--baseline')) throw new Error('--baseline and --artifact are mutually exclusive');
        const artifact = artifactPath ? loadAdaptation(artifactPath) : null;
        const candidate = artifact ? bindAdaptation(artifact, prepared.program, executorIdentity).candidate : undefined;
        const split = option('--split') ?? 'validation'; if (!['train', 'validation', 'test'].includes(split)) throw new Error('invalid evaluation split');
        const batch = await evaluate(prepared, { split: split as 'validation', candidate, driver, judge: judge ? (request, signal, options) => judge!.turn(request, signal, options) : undefined, signal: abort.signal }); output(batch); return batch.gatesPassed ? 0 : 3;
      }
      if (command === 'adapt' && words[0] === 'revalidate') {
        const old = loadAdaptation(words[1]!); const mapping = option('--mapping');
        const artifact = await revalidateAdaptation(old, prepared, driver, mapping ? JSON.parse(readFileSync(mapping, 'utf8')) : {},
          { signal: abort.signal, judge: judge ? (request, signal, options) => judge!.turn(request, signal, options) : undefined });
        const out = resolve(option('--out') ?? 'revalidated.json'); mkdirSync(dirname(out), { recursive: true }); writeFileSync(out, canonical(artifact)); output({ artifact: out, digest: artifact.digest }); return 0;
      }
      if (command !== 'optimize') throw new Error('unknown adaptation command');
      const reflectionConfig = loadModelConfiguration(prepared.suite.models?.reflection);
      reflection = createResolvedModelSession(reflectionConfig.choice, process.env, process.stderr);
      const strategy = option('--strategy') ?? 'gepa'; if (strategy !== 'gepa' && strategy !== 'reflection') throw new Error('unknown optimization strategy');
      const options = { executor: driver, reflection: (request: Parameters<ModelDriver>[0], signal?: AbortSignal, options?: Parameters<ModelDriver>[2]) => reflection!.turn(request, signal, options), strategy,
        reflectionIdentity: executorIdentityForChoice(reflectionConfig.choice), judgeIdentity: judgeConfig ? executorIdentityForChoice(judgeConfig.choice) : undefined,
        judge: judge ? (request: Parameters<ModelDriver>[0], signal?: AbortSignal, options?: Parameters<ModelDriver>[2]) => judge!.turn(request, signal, options) : undefined,
        out: option('--out'), seed: Number(option('--seed') ?? 0), signal: abort.signal,
        progress: (event: Record<string, unknown>) => process.stderr.write(JSON.stringify(event) + '\n') } as const;
      const result = resume ? await resumeOptimization(prepared, words[1]!, options) : await optimize(prepared, options);
      output({ directory: result.directory, artifact: result.artifact.digest, report: result.report }); return abort.signal.aborted ? 130 : 0;
    } finally { process.removeListener('SIGINT', interrupt); await judge?.close(); await reflection?.close(); await executor.close(); }
  } catch (error) {
    process.stderr.write((error instanceof Error ? error.message : String(error)) + '\n');
    return /interrupted|cancelled/.test(String(error)) ? 130 : /regression|revalidation failed/.test(String(error)) ? 3 : /infrastructure|worker/.test(String(error)) ? 4 : 2;
  }
}
