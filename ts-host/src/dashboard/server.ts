/**
 * `natlang dashboard`: a local web UI for running natlang programs and supervising them. It serves ts-host/dashboard/
 * and a JSON API over the same machine state the CLI uses: model profiles (~/.config/natlang/config.json), stock
 * models, provider credentials, the call record store (traces), compiled programs and their build locks, and jobs
 * (CLI commands it starts and watches, jobs.ts). It binds to loopback and every API request carries the session token
 * printed at startup (a cookie set by the startup URL and a header), so other web pages cannot drive it.
 */
import { randomBytes, timingSafeEqual } from 'node:crypto';
import { existsSync, mkdirSync, readdirSync, readFileSync, realpathSync, renameSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { cpus, freemem, totalmem } from 'node:os';
import { basename, dirname, extname, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { CallStore, machineStoreRoot } from '../calls/store.js';
import { expandedRecord } from '../cli/calls.js';
import { programFunctions, readProgramManifest } from '../cli/build-lock.js';
import { createProgram, readExamples, saveExample, type SavedExample } from '../cli/authoring.js';
import { currentRelease } from '../adaptation/weights/release.js';
import { loadModelConfiguration, resolveModelChoice, type ModelProfile } from '../model/config.js';
import { DEFAULT_LOCAL_MODEL, STOCK_MODELS, stockModel, type LocalModelRelease } from '../model-catalog.js';
import { createResolvedModelSession, ensureStockModel, managedModelSha256, profileEnvironment, stockModelDownload, stockModelPath, localModelPrerequisites, supportsNativeStates, warmTag,
  type ManagedModelSession } from '../model/local-server.js';
import { publishSharedServer, readSharedServer, withdrawSharedServer } from '../model/shared-server.js';
import { discoverLlamaRuntime, LLAMA_RUNTIME_RELEASE } from '../model/llama-runtime.js';
import { LLAMA_FORK_SOURCE } from '../llama-fork-source.js';
import { discoverLocalApplications } from '../package/discover.js';
import { defaultNatlangCacheDirectory, defaultNatlangConfigDirectory, defaultNatlangStateDirectory } from '../package/store.js';
import { readBuildLock } from '../package/warm.js';
import { JobSupervisor, type JobSpec } from './jobs.js';

export type DashboardOptions = { host?: string; port?: number; root?: string; version: string; environment?: NodeJS.ProcessEnv;
  log?: NodeJS.WritableStream };
type Handler = (request: IncomingMessage, params: Record<string, string>, query: URLSearchParams, body: unknown) => Promise<unknown> | unknown;
class HttpError extends Error { constructor(readonly status: number, message: string) { super(message); } }

const STATIC = fileURLToPath(new URL('../../dashboard/', import.meta.url));
const TYPES: Record<string, string> = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8', '.svg': 'image/svg+xml', '.json': 'application/json' };

type ConfigFile = { defaultProfile?: string; profiles?: Record<string, ModelProfile>; nativeSources?: Record<string, string> } & Record<string, unknown>;

export async function startDashboard(options: DashboardOptions): Promise<{ url: string; close(): Promise<void> }> {
  const environment = options.environment ?? process.env, log = options.log ?? process.stderr;
  const root = realpathSync(resolve(options.root ?? '.'));
  const token = randomBytes(18).toString('base64url');
  const jobs = new JobSupervisor(join(defaultNatlangStateDirectory(environment), 'dashboard', 'jobs'), environment);
  const configPath = join(defaultNatlangConfigDirectory(environment), 'config.json');
  const readConfig = (): ConfigFile => existsSync(configPath) ? JSON.parse(readFileSync(configPath, 'utf8')) as ConfigFile : {};
  const writeConfig = (config: ConfigFile) => {
    mkdirSync(dirname(configPath), { recursive: true });
    const temporary = `${configPath}.${process.pid}.tmp`;
    writeFileSync(temporary, JSON.stringify(config, null, 1) + '\n'); renameSync(temporary, configPath);
  };
  const downloads = new Map<string, { received: number; total: number; error?: string; done?: boolean }>();
  const store = (): CallStore => {
    const path = machineStoreRoot(environment);
    if (!path) throw new HttpError(409, 'the call store is off on this machine (NATLANG_CALL_STORE=off)');
    return CallStore.open(path);
  };
  const withStore = <T>(fn: (store: CallStore) => T): T => { const opened = store(); try { return fn(opened); } finally { opened.close(); } };
  const inside = (path: string) => {
    const absolute = resolve(root, path);
    if (absolute !== root && !absolute.startsWith(root + sep)) throw new HttpError(400, `${path} is outside the dashboard's root ${root}`);
    let parent = absolute;
    while (!existsSync(parent)) parent = dirname(parent);
    const actual = realpathSync(parent);
    if (actual !== root && !actual.startsWith(root + sep)) throw new HttpError(400, 'The path points outside the dashboard root.');
    return absolute;
  };
  const stock = (release: LocalModelRelease) => {
    const path = [resolve('models', release.file), stockModelPath(release, environment)].find(existsSync);
    // This dashboard's own download, else one another process (a setup job, a terminal run) is making.
    const own = downloads.get(release.id), other = path || (own && !own.done && !own.error) ? null : stockModelDownload(release, environment);
    return { ...release, installed: !!path, path: path ?? null, download: other ? { received: other.received, total: other.total } : own ?? null,
      default: release.id === DEFAULT_LOCAL_MODEL.id };
  };
  const pi = async () => import('../model/pi-provider.js');
  const selection = (body: Record<string, unknown>) => [
    ...(typeof body.profile === 'string' && body.profile ? ['--profile', body.profile] : []),
    ...(typeof body.provider === 'string' && body.provider ? ['--provider', body.provider] : []),
    ...(typeof body.model === 'string' && body.model ? ['--model', body.model] : [])];
  const string = (value: unknown, label: string): string => {
    if (typeof value !== 'string' || !value.trim()) throw new HttpError(400, `${label} is required`);
    return value;
  };

  const routes: [string, RegExp, Handler][] = [];
  const route = (method: string, pattern: string, handler: Handler) =>
    routes.push([method, new RegExp('^' + pattern.replace(/:([a-z]+)/g, '(?<$1>[^/]+)') + '$'), handler]);

  route('GET', '/api/status', () => {
    const runtime = discoverLlamaRuntime(environment), config = readConfig();
    let calls: Record<string, unknown> | null = null;
    try { calls = withStore(s => ({ root: s.root, calls: (s.db.prepare('SELECT COUNT(*) AS n FROM calls').get() as { n: number }).n,
      bytes: s.diskBytes(), settings: s.settings() })); } catch { /* store off */ }
    let model: Record<string, unknown>;
    const name = config.defaultProfile ?? 'default';
    try {
      const choice = resolveModelChoice(config.profiles?.[name] ?? {});
      model = { profile: name, kind: choice.kind, model: choice.model,
        ...(choice.kind === 'managed-local' && choice.release ? { release: stock(choice.release) } : {}),
        ...(choice.kind === 'external' ? { endpoint: choice.endpoint } : {}), ...(choice.kind === 'pi-provider' ? { provider: choice.provider } : {}) };
    } catch (error) { model = { profile: name, error: (error as Error).message }; }
    return { version: options.version, node: process.version, root, configPath, model, calls,
      runtime: { selected: runtime.selected ? { ...runtime.selected, fork: supportsNativeStates(runtime.selected.path) } : null, recommended: { version: LLAMA_RUNTIME_RELEASE.version, build: LLAMA_RUNTIME_RELEASE.build },
        fork: { repository: LLAMA_FORK_SOURCE.repository, commit: LLAMA_FORK_SOURCE.commit },
        candidates: runtime.candidates.map(item => ({ path: item.path, source: item.source, compatible: item.compatible, reason: item.reason })) },
      machine: { cpus: cpus().length, cpu: cpus()[0]?.model ?? null, memory: totalmem(), free: freemem(), platform: `${process.platform}-${process.arch}` },
      jobs: { running: jobs.running(), total: jobs.list().length } };
  });

  // Warm prompt openings the managed server saved (package/warm.ts).
  const warmDirectory = join(defaultNatlangCacheDirectory(environment), 'warm-prefixes');
  route('GET', '/api/warm', () => existsSync(warmDirectory) ? readdirSync(warmDirectory).filter(name => name.endsWith('.slot')).map(name => {
    const stat = statSync(join(warmDirectory, name));
    let meta: { tokens?: number; chars?: number } = {};
    try { meta = JSON.parse(readFileSync(join(warmDirectory, `${name}.json`), 'utf8')) as typeof meta; } catch { /* no sidecar */ }
    const [, model, build, layout] = /^[0-9a-f]{16}-([0-9a-f]{12})-b(\w+?)(?:-p(\d+x\d+))?\.slot$/.exec(name) ?? [];
    return { file: name, bytes: stat.size, usedAt: stat.mtime.toISOString(), tokens: meta.tokens ?? null, model: model ?? null, build: build ?? null,
      layout: layout ?? null };
  }).sort((a, b) => b.usedAt.localeCompare(a.usedAt)) : []);
  route('DELETE', '/api/warm', () => { rmSync(warmDirectory, { recursive: true, force: true }); return { ok: true }; });

  // The model the dashboard keeps loaded for every natlang process on this machine (model/shared-server.ts).
  let kept: { session: ManagedModelSession; profile: string; log: string[] } | undefined;
  const keptStatus = () => {
    const published = readSharedServer(environment);
    return kept ? { running: true, profile: kept.profile, ...(published?.pid === process.pid ? { server: published } : {}), log: kept.log.slice(-20) } :
      { running: false, ...(published ? { other: published } : {}) };
  };
  const stopKept = async () => {
    if (!kept) return;
    const { session } = kept; kept = undefined;
    withdrawSharedServer(process.pid, environment);
    await session.close();
  };
  route('GET', '/api/model-server', keptStatus);
  route('POST', '/api/model-server', async (_request, _params, _query, body) => {
    await stopKept();
    const requested = (body as { profile?: string } | undefined)?.profile || undefined;
    const selection = loadModelConfiguration(requested, {}, environment);
    if (selection.choice.kind !== 'managed-local') throw new HttpError(400, `profile ${selection.name} is not a local model; only the managed llama.cpp server can be kept loaded`);
    const log: string[] = [];
    const session = createResolvedModelSession(selection.choice, environment, { write: (text: string) => { log.push(...String(text).trim().split('\n')); return true; } } as unknown as NodeJS.WritableStream);
    kept = { session, profile: selection.name, log };
    try {
      await session.prepare();
      const status = session.status(), plan = session.slotPlan();
      if (!status.endpoint || !plan) throw new Error('the model server did not start');
      const sha = await managedModelSha256(profileEnvironment(selection.choice, environment), selection.choice.release);
      publishSharedServer({ pid: process.pid, endpoint: status.endpoint, modelSha256: sha, runtimeBuild: status.runtimeBuild ?? null,
        runtime: status.runtime ?? null, slots: plan.slots, contextPerSlot: plan.contextPerSlot, model: status.model, profile: selection.name, startedAt: new Date().toISOString(),
        warm: environment.NATLANG_WARM_PREFIXES !== 'off' }, environment);
      // Start every slot from the openings earlier runs saved for this model file, build and slot layout, most recent first.
      const tag = `-${warmTag(sha, status.runtime ?? 'bx', plan)}.slot`;
      const saved = existsSync(warmDirectory) ? readdirSync(warmDirectory).filter(name => name.endsWith(tag))
        .sort((a, b) => statSync(join(warmDirectory, b)).mtimeMs - statSync(join(warmDirectory, a)).mtimeMs) : [];
      let restored = 0;
      for (let slot = 0; slot < plan.slots && saved.length; slot++) {
        const response = await fetch(`${status.endpoint}/slots/${slot}?action=restore`, { method: 'POST', headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ filename: saved[slot % saved.length] }) }).catch(() => undefined);
        if (response?.ok) restored++;
      }
      log.push(`kept loaded at ${status.endpoint} with ${plan.slots} slots; ${restored} slot${restored === 1 ? '' : 's'} start from a saved prompt opening`);
      return keptStatus();
    } catch (error) { await stopKept(); throw new HttpError(500, `${(error as Error).message}\n${log.slice(-6).join('\n')}`); }
  });
  route('DELETE', '/api/model-server', async () => { await stopKept(); return keptStatus(); });

  // Model configuration.
  route('GET', '/api/config', async () => {
    const config = readConfig(), credentials = await (await pi()).piCredentials(environment).list();
    const profiles = Object.fromEntries(Object.entries(config.profiles ?? {}).map(([name, profile]) => {
      try { const choice = resolveModelChoice(profile); return [name, { profile, kind: choice.kind, model: choice.model }]; }
      catch (error) { return [name, { profile, error: (error as Error).message }]; }
    }));
    return { path: configPath, defaultProfile: config.defaultProfile ?? null, profiles,
      credentials: credentials.map(item => ({ id: item.providerId, type: item.type })), stock: STOCK_MODELS.map(stock),
      builtinDefault: { kind: 'managed-local', model: DEFAULT_LOCAL_MODEL.id } };
  });
  route('PUT', '/api/profiles/:name', (_request, params, _query, body) => {
    const name = decodeURIComponent(params.name!);
    if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(name)) throw new HttpError(400, 'profile names are letters, digits, . _ and -');
    const { profile, makeDefault } = (body ?? {}) as { profile?: ModelProfile; makeDefault?: boolean };
    if (!profile || typeof profile !== 'object') throw new HttpError(400, 'profile is required');
    try { resolveModelChoice(profile); } catch (error) { throw new HttpError(400, (error as Error).message); }
    const config = readConfig();
    config.profiles = { ...config.profiles, [name]: profile };
    if (makeDefault) config.defaultProfile = name;
    writeConfig(config); return { ok: true };
  });
  route('DELETE', '/api/profiles/:name', (_request, params) => {
    const name = decodeURIComponent(params.name!), config = readConfig();
    if (!config.profiles?.[name]) throw new HttpError(404, `no profile ${name}`);
    delete config.profiles[name];
    if (config.defaultProfile === name) delete config.defaultProfile;
    writeConfig(config); return { ok: true };
  });
  route('POST', '/api/default', (_request, _params, _query, body) => {
    const name = (body as { profile?: string | null })?.profile, config = readConfig();
    if (name && !config.profiles?.[name]) throw new HttpError(404, `no profile ${name}`);
    if (name) config.defaultProfile = name; else delete config.defaultProfile;
    writeConfig(config); return { ok: true };
  });
  route('GET', '/api/stock', () => STOCK_MODELS.map(stock));
  route('POST', '/api/stock/:id/download', (_request, params) => {
    const release = (() => { try { return stockModel(decodeURIComponent(params.id!)); } catch (error) { throw new HttpError(404, (error as Error).message); } })();
    const current = downloads.get(release.id);
    if (current && !current.done && !current.error) return { started: false, download: current };
    const other = stockModelDownload(release, environment);
    if (other) return { started: false, download: { received: other.received, total: other.total } };
    const state = { received: 0, total: release.bytes } as { received: number; total: number; error?: string; done?: boolean };
    downloads.set(release.id, state);
    void ensureStockModel(release, environment, { write: () => true } as unknown as NodeJS.WritableStream, (received, total) => {
      state.received = received; state.total = total;
    }).then(() => { state.done = true; state.received = state.total; }, error => { state.error = (error as Error).message; });
    return { started: true, download: state };
  });
  route('GET', '/api/providers', async () => {
    const module = await pi(), models = module.piModels(environment), stored = new Set((await module.piCredentials(environment).list()).map(item => item.providerId));
    return models.getProviders().map(item => ({ id: item.id, name: item.name, credential: stored.has(item.id) }));
  });
  route('GET', '/api/providers/:id/models', async (_request, params) => {
    const models = (await pi()).piModels(environment), id = decodeURIComponent(params.id!);
    if (!models.getProvider(id)) throw new HttpError(404, `unknown provider ${id}`);
    return models.getModels(id).map(item => ({ id: item.id, name: item.name, reasoning: (item as { reasoning?: boolean }).reasoning ?? null,
      contextWindow: (item as { contextWindow?: number }).contextWindow ?? null }));
  });
  route('PUT', '/api/credentials/:id', async (_request, params, _query, body) => {
    const id = decodeURIComponent(params.id!), key = string((body as { key?: unknown })?.key, 'key');
    if (!/^(?:env:[A-Z_][A-Z0-9_]*|[a-z0-9][a-z0-9._-]*)$/.test(id)) throw new HttpError(400, 'a credential is a provider ID or env:VARIABLE');
    await (await pi()).piCredentials(environment).modify(id, async () => ({ type: 'api_key', key: key.trim() }));
    return { ok: true };
  });
  route('DELETE', '/api/credentials/:id', async (_request, params) => {
    await (await pi()).piCredentials(environment).delete(decodeURIComponent(params.id!)); return { ok: true };
  });

  // Programs.
  route('POST', '/api/programs', (_request, _params, _query, body) => {
    const input = (body ?? {}) as Record<string, unknown>, name = string(input.name, 'directory name');
    if (!/^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$/.test(name)) throw new HttpError(400, 'Use a directory name made of letters, numbers, dots, dashes or underscores.');
    for (const key of ['instruction', 'input', 'returns', 'profile']) if (input[key] !== undefined && typeof input[key] !== 'string') throw new HttpError(400, `${key} must be text`);
    const path = inside(name);
    if (existsSync(path) && (!statSync(path).isDirectory() || readdirSync(path).length)) throw new HttpError(409, 'That directory already contains files. Choose a new name.');
    const directory = createProgram(path, { instruction: input.instruction as string | undefined, input: input.input as string | undefined, returns: input.returns as string | undefined, profile: input.profile as string | undefined });
    return { directory: relative(root, directory) };
  });
  route('GET', '/api/programs/model', (_request, _params, query) => {
    const directory = inside(string(query.get('path'), 'path')), manifest = readProgramManifest(directory);
    const { choice } = loadModelConfiguration(query.get('profile') || undefined, {}, environment, manifest?.model);
    const local = choice.kind === 'managed-local' ? localModelPrerequisites(profileEnvironment(choice, environment), choice.release) : null;
    return { model: choice.model, kind: choice.kind, detail: choice.kind === 'managed-local' ?
      `${local?.modelPath ? 'model downloaded' : `first run downloads ${Math.ceil((choice.release?.bytes ?? 0) / 1_000_000)} MB`}; ${local?.executable ? 'runtime installed' : 'runtime setup required'}. First runs on a CPU may take several minutes.` :
      'Uses the selected endpoint or provider. Use Models → Test to verify an actual request.' };
  });
  route('GET', '/api/programs', () => discoverLocalApplications(root, { libraries: true }).map(item => {
    const directory = dirname(item.path), lock = readBuildLock(join(directory, '.natlang', 'build'));
    return { ...item, directory: relative(root, directory) || '.', functions: programFunctions(directory).length,
      warm: lock?.warm ? { at: lock.warm.at, prefixes: lock.warm.prefixes.length, cases: lock.warm.cases.length,
        done: lock.warm.cases.filter(item => item.outcome === 'done').length } : null, builtAt: lock?.builtAt ?? null };
  }));
  route('GET', '/api/programs/detail', (_request, _params, query) => {
    const directory = inside(string(query.get('path'), 'path'));
    if (!existsSync(join(directory, 'natlang.json'))) throw new HttpError(404, `${query.get('path')} has no natlang.json`);
    let manifest: unknown, problem: string | null = null;
    try { manifest = readProgramManifest(directory); } catch (error) { problem = (error as Error).message; manifest = JSON.parse(readFileSync(join(directory, 'natlang.json'), 'utf8')); }
    const release = currentRelease(directory);
    const readme = ['README.md', 'DECOMPOSITION.md'].map(name => join(directory, name)).find(existsSync);
    const recent = (() => { try { return withStore(s => s.calls({ program: directory, limit: 20, audits: false })); } catch { return []; } })();
    const lastRuns = jobs.list().filter(job => job.kind === 'run' && job.status === 'done' && job.meta?.path === (relative(root, directory) || '.'));
    const lastCalls = jobs.list().filter(job => job.kind === 'call' && job.status === 'done' && job.meta?.program === (relative(root, directory) || '.'));
    let examples = 0, examplesProblem: string | null = null;
    try { examples = readExamples(directory).length; } catch (error) { examplesProblem = (error as Error).message; }
    return { directory: relative(root, directory) || '.', manifest, problem, functions: programFunctions(directory),
      lastRuns: Object.fromEntries(lastRuns.slice().reverse().map(job => [String(job.meta?.target ?? ''), job.meta])), examples, examplesProblem,
      lastCalls: Object.fromEntries(lastCalls.slice().reverse().map(job => [String(job.meta?.path), job.meta])),
      lock: readBuildLock(join(directory, '.natlang', 'build')) ?? null,
      tuning: release ? { release: release.manifest.digest, base: release.manifest.base.checkpoint_sha256 } : null,
      readme: readme ? { name: basename(readme), text: readFileSync(readme, 'utf8').slice(0, 20000) } : null, recent };
  });
  route('GET', '/api/programs/source', (_request, _params, query) => {
    const path = inside(string(query.get('path'), 'path'));
    if (!existsSync(path) || !statSync(path).isFile() || statSync(path).size > 512_000) throw new HttpError(404, 'no such source file');
    return { path: relative(root, path), text: readFileSync(path, 'utf8') };
  });
  route('PUT', '/api/programs/source', (_request, _params, _query, body) => {
    const input = (body ?? {}) as Record<string, unknown>, path = inside(string(input.path, 'path'));
    if (!/\.(nl|ts|json)$/.test(path) || !existsSync(path) || statSync(path).size > 512_000) throw new HttpError(400, 'Choose an existing .nl, .ts or .json source file.');
    if (typeof input.text !== 'string' || input.text.length > 512_000) throw new HttpError(400, 'Source text must be at most 512 KB.');
    if (typeof input.previous !== 'string' || readFileSync(path, 'utf8') !== input.previous) throw new HttpError(409, 'This file changed on disk. Reload its source before saving.');
    const temporary = `${path}.${randomBytes(6).toString('hex')}.tmp`;
    writeFileSync(temporary, input.text); renameSync(temporary, path);
    return { path: relative(root, path) };
  });
  route('POST', '/api/programs/example', (_request, _params, _query, body) => {
    const input = (body ?? {}) as Record<string, unknown>, directory = inside(string(input.path, 'program path'));
    if (!readProgramManifest(directory)) throw new HttpError(400, 'Choose a program with natlang.json first.');
    const example = input.example as SavedExample;
    if (typeof example?.function !== 'string') throw new HttpError(400, 'Choose a function.');
    inside(join(directory, example.function));
    if (!existsSync(join(directory, example.function))) throw new HttpError(404, 'The function does not exist.');
    saveExample(directory, example);
    return { examples: readExamples(directory).length };
  });

  // Jobs: the CLI commands the dashboard starts.
  route('GET', '/api/jobs', () => jobs.list().slice(0, 200));
  route('POST', '/api/jobs', (_request, _params, _query, body) => {
    const input = (body ?? {}) as Record<string, unknown>, kind = string(input.kind, 'kind');
    let spec: JobSpec;
    if (kind === 'run') {
      const path = inside(string(input.path, 'path'));
      const args = Array.isArray(input.args) ? input.args.map(String) : [];
      spec = { kind, title: `run ${relative(root, path) || '.'}${input.target ? `#${input.target}` : ''}${args.length ? ' ' + args.join(' ') : ''}`, cwd: root,
        argv: ['run', path, '--plain', ...(typeof input.target === 'string' && input.target ? ['--target', input.target] : []), ...selection(input),
          '--yes', ...(args.length ? ['--', ...args] : [])], interactive: input.interactive !== false,
        ...(typeof input.stdin === 'string' ? { stdin: input.stdin } : {}), meta: { path: relative(root, path) || '.', target: input.target ?? '', args, profile: input.profile } };
    } else if (kind === 'call') {
      const path = inside(string(input.function, 'function'));
      if (!path.endsWith('.nl')) throw new HttpError(400, 'function must be a .nl file');
      const inputs = input.inputs && typeof input.inputs === 'object' ? input.inputs : {};
      spec = { kind, title: `call ${relative(root, path)}`, cwd: dirname(path), argv: ['call', path, ...selection(input), '--yes', '--json'],
        stdin: JSON.stringify(inputs), meta: { path: relative(root, path), inputs, profile: input.profile, program: typeof input.program === 'string' ? relative(root, inside(input.program)) || '.' : undefined } };
    } else if (kind === 'ask') {
      const instruction = string(input.instruction, 'instruction');
      spec = { kind, title: `ask ${instruction.slice(0, 60)}`, cwd: typeof input.cwd === 'string' ? inside(input.cwd) : root,
        argv: ['ask', instruction, ...(typeof input.returns === 'string' && input.returns ? ['--returns', input.returns] : []), ...selection(input), '--yes'],
        stdin: typeof input.input === 'string' ? input.input : '', meta: { instruction, input: input.input ?? '', returns: input.returns ?? null, profile: input.profile } };
    } else if (kind === 'test') {
      const path = inside(string(input.path, 'path'));
      spec = { kind, title: `test ${relative(root, path) || '.'}`, cwd: root, argv: ['test', path, ...selection(input), '--yes'], meta: { path: relative(root, path) || '.' } };
    } else if (kind === 'build') {
      const path = inside(string(input.path, 'path'));
      spec = { kind, title: `build ${relative(root, path) || '.'}${input.warm ? ' --warm' : ''}`, cwd: root,
        argv: ['build', path, ...(input.warm ? ['--warm'] : []), ...selection(input), '--yes'], meta: { path: relative(root, path) } };
    } else if (kind === 'setup') {
      spec = { kind, title: 'setup', cwd: root, argv: ['setup', '--yes', ...selection(input)] };
    } else if (kind === 'runtime-build') {
      spec = { kind, title: 'build natlang llama.cpp fork', cwd: root, argv: ['runtime', 'build', ...(input.cuda === true ? ['--cuda'] : input.cuda === false ? ['--no-cuda'] : [])] };
    } else if (kind === 'tune') {
      const path = inside(string(input.path, 'path'));
      spec = { kind, title: `tune ${relative(root, path) || '.'}`, cwd: root, argv: ['tune', path], meta: { path: relative(root, path) } };
    } else if (kind === 'specialize') {
      spec = { kind, title: 'specialize', cwd: root, argv: ['specialize'] };
    } else throw new HttpError(400, `unknown job kind ${kind}`);
    return jobs.start(spec);
  });
  route('GET', '/api/jobs/:id', (_request, params, query) => {
    const job = jobs.get(params.id!);
    if (!job) throw new HttpError(404, `no job ${params.id}`);
    const offset = (name: string) => Number(query.get(name) ?? 0) || 0;
    return { job, stdout: jobs.read(job.id, 'stdout.log', offset('stdout')), stderr: jobs.read(job.id, 'stderr.log', offset('stderr')),
      live: jobs.read(job.id, 'live.jsonl', offset('live')) };
  });
  route('POST', '/api/jobs/:id/input', (_request, params, _query, body) => { jobs.input(params.id!, string((body as { text?: unknown })?.text, 'text')); return { ok: true }; });
  route('POST', '/api/jobs/:id/stop', (_request, params) => jobs.stop(params.id!));
  route('DELETE', '/api/jobs/:id', (_request, params) => { jobs.remove(params.id!); return { ok: true }; });

  // Traces: the machine's call record store.
  route('GET', '/api/traces', (_request, _params, query) => withStore(s => s.calls({
    definition: query.get('definition') || undefined, outcome: query.get('outcome') || undefined, executor: query.get('executor') || undefined,
    program: query.get('program') ? inside(query.get('program')!) : undefined, since: query.get('since') || undefined,
    after: query.get('after') || undefined, limit: Math.min(500, Number(query.get('limit') ?? 100) || 100), audits: query.get('all') === '1' })));
  route('GET', '/api/traces/hot', (_request, _params, query) => withStore(s => s.hot({ by: (query.get('by') as 'calls' | 'tokens' | 'wall_ms' | null) ?? 'calls', limit: 30 })));
  route('GET', '/api/traces/:id', (_request, params) => withStore(s => {
    const id = decodeURIComponent(params.id!), record = s.call(id);
    if (!record) {
      const running = s.db.prepare("SELECT call_id, definition_name, outcome, started_at FROM calls WHERE call_id = ?").get(id);
      if (!running) throw new HttpError(404, `no call ${id}`);
      return { running, children: s.children(id), events: s.events(id) ?? [] };
    }
    return { record: expandedRecord(s, record), annotations: s.annotations(id), children: s.children(id), cases: s.callCases(id) };
  }));
  route('GET', '/api/traces/:id/events', (_request, params) => withStore(s => s.events(decodeURIComponent(params.id!)) ?? []));
  route('POST', '/api/traces/:id/feedback', async (_request, params, _query, body) => {
    const { verdict, expected } = (body ?? {}) as { verdict?: string; expected?: unknown };
    if (!['good', 'bad', 'fix'].includes(String(verdict))) throw new HttpError(400, 'verdict is good, bad or fix');
    const { recordFeedback } = await import('../calls/feedback.js');
    return { message: withStore(s => recordFeedback(s, { verdict: verdict as 'good' | 'bad' | 'fix', call: decodeURIComponent(params.id!), dryRun: false,
      ...(expected === undefined || expected === '' ? {} : { expected: typeof expected === 'string' ? expected : JSON.stringify(expected) }) })) };
  });
  route('POST', '/api/traces/:id/pin', (_request, params, _query, body) => withStore(s => {
    s.pin(decodeURIComponent(params.id!), (body as { pinned?: boolean })?.pinned !== false); return { ok: true };
  }));
  route('GET', '/api/compilations', () => withStore(s => s.compilations({ limit: 200 })));
  route('PUT', '/api/store/settings', (_request, _params, _query, body) => withStore(s => s.writeSettings((body ?? {}) as never)));

  const send = (response: ServerResponse, status: number, value: unknown) => {
    const text = JSON.stringify(value);
    response.writeHead(status, { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store' });
    response.end(text);
  };
  const authorized = (request: IncomingMessage) => {
    const given = String(request.headers['x-natlang-token'] ?? ''), cookie = /(?:^|;\s*)natlang_dashboard=([^;]+)/.exec(request.headers.cookie ?? '')?.[1] ?? '';
    const equal = (a: string) => a.length === token.length && timingSafeEqual(Buffer.from(a), Buffer.from(token));
    return equal(given) && equal(cookie);
  };
  let port = options.port ?? 8780;
  const host = options.host ?? '127.0.0.1';
  const server = createServer(async (request, response) => {
    try {
      const url = new URL(request.url ?? '/', 'http://localhost');
      // Only this machine's names: a page on another origin that rebinds a hostname to 127.0.0.1 is refused.
      const hostHeader = String(request.headers.host ?? '').replace(/:\d+$/, '');
      if (!['127.0.0.1', 'localhost', '[::1]', host].includes(hostHeader)) { send(response, 403, { error: 'unexpected Host header' }); return; }
      if (url.pathname.startsWith('/api/')) {
        if (!authorized(request)) { send(response, 401, { error: 'open the dashboard from the URL natlang dashboard printed' }); return; }
        const match = routes.map(([method, pattern, handler]) => ({ method, handler, found: pattern.exec(url.pathname) }))
          .find(item => item.found && item.method === request.method);
        if (!match) { send(response, 404, { error: `no route ${request.method} ${url.pathname}` }); return; }
        let body: unknown;
        if (request.method !== 'GET') {
          const chunks: Buffer[] = [];
          for await (const chunk of request) chunks.push(chunk as Buffer);
          const text = Buffer.concat(chunks).toString('utf8');
          body = text ? JSON.parse(text) : undefined;
        }
        send(response, 200, await match.handler(request, match.found!.groups ?? {}, url.searchParams, body));
        return;
      }
      if (url.searchParams.get('token') === token) {
        response.writeHead(302, { location: url.pathname, 'set-cookie': `natlang_dashboard=${token}; HttpOnly; SameSite=Strict; Path=/` });
        response.end(); return;
      }
      const file = url.pathname === '/' ? 'index.html' : url.pathname.slice(1);
      const path = resolve(STATIC, file);
      if (!path.startsWith(STATIC) || !existsSync(path) || !statSync(path).isFile()) { response.writeHead(404); response.end('not found'); return; }
      let text = readFileSync(path);
      if (file === 'index.html') text = Buffer.from(text.toString('utf8').replace('__NATLANG_TOKEN__', authorizedCookie(request) ? token : ''));
      response.writeHead(200, { 'content-type': TYPES[extname(path)] ?? 'application/octet-stream', 'cache-control': 'no-store' });
      response.end(text);
    } catch (error) {
      const status = error instanceof HttpError ? error.status : error instanceof SyntaxError ? 400 : 500;
      send(response, status, { error: (error as Error).message });
    }
  });
  const authorizedCookie = (request: IncomingMessage) => (/(?:^|;\s*)natlang_dashboard=([^;]+)/.exec(request.headers.cookie ?? '')?.[1] ?? '') === token;
  await new Promise<void>((resolveListen, reject) => {
    const attempt = () => {
      server.once('error', (error: NodeJS.ErrnoException) => {
        if (error.code === 'EADDRINUSE' && options.port === undefined && port < 8800) { port++; attempt(); } else reject(error);
      });
      server.listen(port, host, () => resolveListen());
    };
    attempt();
  });
  const address = server.address();
  if (address && typeof address === 'object') port = address.port;
  const url = `http://${host === '0.0.0.0' ? '127.0.0.1' : host}:${port}/?token=${token}`;
  log.write(`natlang dashboard on ${url}\n`);
  return { url, close: async () => { await jobs.close(); await stopKept(); await new Promise<void>(resolveClose => server.close(() => resolveClose())); } };
}
