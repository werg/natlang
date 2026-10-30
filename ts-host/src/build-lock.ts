/**
 * `natlang build [--warm]`: the compiled program's lock (model, runtime, tuning, functions) and its ahead-of-time
 * tracing run (package/warm.ts). `natlang run` reads the lock: the program's model is used unless the command selects
 * one, and the warm prefixes are restored when they were prefilled for exactly the model file and llama.cpp build.
 */
import { existsSync, readdirSync, readFileSync, rmSync, statSync } from 'node:fs';
import { basename, join, relative, resolve } from 'node:path';
import { currentRelease } from './adaptation/weights/release.js';
import { natlangRuntime, type ConfiguredNatlangRuntime } from './configured-runtime.js';
import { readNatlangFrontmatter } from './runtime/loader.js';
import { fileTraceSink, loadNatlang } from './runtime/node-files.js';
import type { LiveSink } from './runtime/runtime.js';
import { parsePackageManifest, type NatlangPackageManifest } from './package/manifest.js';
import { BUILD_LOCK_SCHEMA, readBuildLock, writeBuildLock, type BuildLock, type WarmCaseResult } from './package/warm.js';
import type { LoadedModelConfiguration } from './model/config.js';
import { discoverLlamaRuntime, type LlamaRuntimeDiscovery, type LlamaServerInspection } from './model/llama-runtime.js';
import { runtimeLabel } from './model/local-server.js';

const IGNORED = new Set(['.git', '.natlang', 'node_modules', 'dist', '.venv']);

/** The package manifest at `root`, when there is a valid one. */
export function readProgramManifest(root: string): NatlangPackageManifest | undefined {
  const path = join(root, 'natlang.json');
  if (!existsSync(path)) return undefined;
  return parsePackageManifest(JSON.parse(readFileSync(path, 'utf8')));
}

/** Every named natural-language function of a project: its source, parameters, result type and description. */
export function programFunctions(root: string): BuildLock['functions'] {
  const found: BuildLock['functions'] = [];
  const visit = (directory: string) => {
    for (const name of readdirSync(directory).sort()) {
      const path = join(directory, name), status = statSync(path);
      if (status.isDirectory()) { if (!IGNORED.has(name)) visit(path); continue; }
      if (!name.endsWith('.nl')) continue;
      const match = /^---\r?\n([\s\S]*?)\r?\n---/.exec(readFileSync(path, 'utf8'));
      let meta: Record<string, unknown> = {};
      try { meta = match ? readNatlangFrontmatter(match[1]!) : {}; } catch { /* natlang check reports malformed frontmatter */ }
      const args = (meta.args && typeof meta.args === 'object' ? meta.args : {}) as Record<string, string>;
      found.push({ name: basename(name, '.nl'), source: relative(root, path).split('\\').join('/'),
        params: Object.entries(args).map(([key, type]) => `${key}: ${type}`), returns: String(meta.returns ?? 'string'),
        ...(typeof meta.description === 'string' ? { description: meta.description } : {}) });
    }
  };
  visit(root);
  return found;
}

function modelSection(selection: LoadedModelConfiguration, natlang: ConfiguredNatlangRuntime | undefined): BuildLock['model'] {
  const choice = selection.choice, status = natlang?.modelStatus();
  const release = choice.kind === 'managed-local' ? choice.release : undefined;
  const runtime = choice.kind === 'managed-local' ? discoverLlamaRuntime().selected : null;
  return { profile: selection.name, kind: choice.kind, model: choice.model, executor: natlang?.executorIdentity.id ?? `${choice.kind}:${choice.model}`,
    release: release ? { id: release.id, sha256: release.sha256, file: release.file, quant: release.quant, source: release.source.repository,
      revision: release.source.revision, template: release.template } : null,
    request: 'request' in choice && choice.request ? choice.request : null,
    runtime: runtime?.version && runtime.build ? { version: runtime.version, build: runtime.build, label: runtimeLabel(runtime) } :
      status?.runtimeBuild ? { version: '', build: status.runtimeBuild, label: status.runtime ?? `b${status.runtimeBuild}` } : null };
}

/** The lock `natlang build` writes beside the compiled program; an existing warm section is kept while it still applies. */
export function lockFor(root: string, selection: LoadedModelConfiguration, version: string, natlang?: ConfiguredNatlangRuntime,
  warm?: BuildLock['warm']): BuildLock {
  const manifest = (() => { try { return readProgramManifest(root); } catch { return undefined; } })();
  const release = currentRelease(root);
  const model = modelSection(selection, natlang);
  const previous = readBuildLock(join(root, '.natlang', 'build'))?.warm ?? null;
  const kept = previous && model.release?.sha256 === previous.modelSha256 && model.runtime?.label === previous.runtime ? previous : null;
  return { schema: BUILD_LOCK_SCHEMA, program: { name: manifest?.name ?? null, version: manifest?.version ?? null, root },
    builtAt: new Date().toISOString(), natlang: version, model,
    tuning: release ? { release: release.manifest.digest, base: release.manifest.base.checkpoint_sha256 } : null,
    functions: programFunctions(root), warm: warm ?? kept };
}

/**
 * The ahead-of-time tracing run: each `warmup` call of natlang.json runs on the program's model with traces recorded
 * under .natlang/build/warm/traces; on the managed llama.cpp server the prompt openings those calls send are prefilled
 * and saved (package/warm.ts), so `natlang run` starts warm. The lock records the calls and the openings.
 */
export async function warmProgram(root: string, options: { profile?: string; provider?: string; model?: string; version: string;
  live?: LiveSink; log: NodeJS.WritableStream; ensureRuntime: (discovery: LlamaRuntimeDiscovery) => Promise<LlamaServerInspection | null> }):
  Promise<{ lock: BuildLock; path: string }> {
  const manifest = readProgramManifest(root);
  const cases = manifest?.warmup ?? [];
  const buildDirectory = join(root, '.natlang', 'build'), traces = join(buildDirectory, 'warm', 'traces');
  rmSync(traces, { recursive: true, force: true });
  if (!cases.length) options.log.write('natlang: natlang.json declares no "warmup" calls; nothing to trace or prefill ahead of time\n');
  const natlang = await natlangRuntime({ profile: options.profile, provider: options.provider, model: options.model,
    ...(manifest?.model ? { programModel: manifest.model } : {}), programRoot: root, ensureRuntime: options.ensureRuntime, log: options.log,
    runtime: { trace: fileTraceSink(traces), programRoot: root, ...(options.live ? { onLive: options.live } : {}) } });
  const started = Date.now(), results: WarmCaseResult[] = [];
  try {
    for (const [index, item] of cases.entries()) {
      const fn = loadNatlang(resolve(root, item.function), root) as unknown as ((...args: unknown[]) => Promise<unknown>) &
        Record<symbol, { definition: { params: { name: string }[] } }>;
      const params = fn[Symbol.for('natlang.callable')]!.definition.params;
      const roots: { callId: string; outcome: string }[] = [];
      let calls = 0, tokensIn = 0, tokensOut = 0;
      const at = Date.now();
      let outcome = 'done', detail: string | undefined;
      try {
        await natlang.runtime.run(() => fn(...params.map(parameter => item.inputs[parameter.name])), { onLive: event => {
          if (event.type === 'call_end') { calls++; if (!event.parentCallId) roots.push({ callId: event.callId, outcome: event.outcome }); }
          if (event.type === 'trace' && event.event.kind === 'model_request' && event.event.phase === 'end') {
            tokensIn += Number(event.event.prompt_tokens ?? 0); tokensOut += Number(event.event.completion_tokens ?? 0);
          }
        } });
      } catch (error) { outcome = roots[0]?.outcome && roots[0].outcome !== 'done' ? roots[0].outcome : 'failed'; detail = (error as Error).message.split('\n')[0]!.slice(0, 400); }
      results.push({ name: item.name ?? `${basename(item.function, '.nl')}#${index + 1}`, function: item.function, outcome,
        ...(detail ? { detail } : {}), wall_ms: Date.now() - at, calls, tokens_in: tokensIn, tokens_out: tokensOut, call_id: roots[0]?.callId ?? null });
      options.log.write(`natlang: warm-up ${results.at(-1)!.name}: ${outcome}${detail ? ` (${detail})` : ''} in ${((Date.now() - at) / 1000).toFixed(1)} s\n`);
    }
    const status = natlang.modelStatus();
    const warm: BuildLock['warm'] = cases.length ? { at: new Date().toISOString(), modelSha256: status.release?.sha256 ?? null,
      runtimeBuild: status.runtimeBuild ?? null, runtime: status.runtime ?? null, prefixes: status.warm ?? [], cases: results, wall_ms: Date.now() - started } : null;
    if (cases.length && status.source !== 'managed-local')
      options.log.write(`natlang: traces recorded; prompt openings are prefilled only on the managed local model (this profile is ${status.source})\n`);
    const lock = lockFor(root, natlang.selection, options.version, natlang, warm ?? undefined);
    return { lock, path: writeBuildLock(buildDirectory, lock) };
  } finally { await natlang.close(); }
}
