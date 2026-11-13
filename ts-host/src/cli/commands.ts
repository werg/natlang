/** The implementations of the natlang commands main.ts dispatches to (help and argument parsing are in main.ts and args.ts). */
import { cpSync, existsSync, mkdirSync, readFileSync, readdirSync, renameSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { spawnSync } from 'node:child_process';
import { basename, dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createInterface } from 'node:readline/promises';
import { isDeepStrictEqual } from 'node:util';
import { answerSource, createProgram, readExamples, sampleInput } from '../package/authoring.js';
import { createPackageArchive, NatlangPackageStore, readPackageArchive,
  writePackageArchive, defaultNatlangStateDirectory } from '../package/index.js';
import { compareVersions, defaultNatlangConfigDirectory, satisfiesVersion } from '../package/store.js';
import { defaultNatlangCacheDirectory } from '../package/directories.js';
import { defaultNatlangRuntimeDirectory } from '../llama/node.js';
import { modelStatus, programForStatus } from './status.js';
import { commandProgress, parseProgressMode } from './progress.js';
import { lockFor, readProgramManifest, warmProgram } from '../build-lock.js';
import { writeBuildLock, type BuildLock } from '../package/warm.js';
import { defaultNativePython, ensureNativeServer, ownsNativeServer, pinNativeCheckpoint, readCheckpointPointer, removeNativePin, exportNativeGguf, findNativeCheckpoint,
  stopNativeServer, type NativeModelSettings } from '../model/native-managed.js';
import { parsePackageManifest, type NatlangTarget } from '../package/manifest.js';
import { discoverLocalApplications, isApplicationManifest } from '../package/discover.js';
import { parseRefinementSettings, type RefinementSettings } from '../native/refinement-settings.js';
import type { TargetContext, TargetExecutable, TargetMain } from '../package/target.js';
import { argumentSignature, argumentUsage, readTargetArguments } from '../runtime/target-arguments.js';
import { LLAMA_FORK_SOURCE } from '../llama-fork-source.js';
import { createResolvedModelSession, loadModelConfiguration, describeLlamaRuntime, discoverLlamaRuntime,
  installManagedLlamaRuntime, buildForkLlamaRuntime, LLAMA_RUNTIME_RELEASE, localModelPrerequisites, localSlotPlan, ensureStockModel,
  type LlamaRuntimeDiscovery, type LlamaServerInspection, type LoadedModelConfiguration,
  type ResolvedModelChoice } from '../model/index.js';
import type { AuthPrompt } from '@earendil-works/pi-ai';
import { openFolder, saveFolder } from '../native/node-files.js';
import { executeNatlangAsk, executeNatlangCall } from '../native/natlang-command.js';
import { inferValueType } from '../native/runtime.js';
import { formatDiagnostics } from '../compiler/project.js';
import type { NatlangDiagnostic } from '../compiler/inline.js';
import { closestName } from '../compiler/suggest.js';
import { buildProject } from '../compiler/node-project.js';
import { type LiveSink, type ModelDriver, type NatlangRuntime } from '../runtime/runtime.js';
import { createNatlangRuntime, loadVirtualNatlang } from '../runtime/node.js';
import type { ModelTurnRequest } from '../contracts.js';
import { fileTraceSink, applicationContextRecords, loadCallables, loadNatlang } from '../runtime/node-files.js';
import { callableMeta } from '../runtime/callable.js';
import { analyzeFusion, prepareFusion } from '../fusion/prepare.js';
import { formatFusionReport } from '../fusion/report.js';
import { parseFusionSettings, type FusionSettings } from '../fusion/settings.js';
import { parseAdaptation } from '../adaptation/schema.js';
import { currentRelease } from '../adaptation/weights/release.js';
import { invokeDefinition } from '../runtime/kernel.js';
import { natlangRuntime, runtimeModel, selectModel, type ConfiguredNatlangRuntime } from '../configured-runtime.js';
import { resolveFrame } from '../runtime/runtime.js';
import '../runtime/node.js';
import { acceptOptions, CliError, EXIT, NATLANG_CLI_VERSION, noTrailingArguments, numericOption, option, output, usageError,
  type Parsed } from './args.js';

/** The runtime module this CLI runs; compiled apps are bound to it so they share one runtime instance. */
const RUNTIME_MODULE = { url: new URL('../index.js', import.meta.url).href, path: fileURLToPath(new URL('../index.js', import.meta.url)),
  types: fileURLToPath(new URL('../index.d.ts', import.meta.url)), specifiers: ['@natlang/node'] };

async function confirm(question: string): Promise<boolean> {
  if (!process.stdin.isTTY || !process.stderr.isTTY) return false;
  const terminal = createInterface({ input: process.stdin, output: process.stderr });
  try { return !/^n(?:o)?$/i.test((await terminal.question(`${question} [Y/n] `)).trim()); }
  finally { terminal.close(); }
}

const megabytes = (bytes: number) => bytes >= 1e9 ? `${(bytes / 1e9).toFixed(1)} GB` : `${Math.ceil(bytes / 1e6)} MB`;
type ManagedChoice = Extract<ResolvedModelChoice, { kind: 'managed-local' }>;
const HOSTED_INSTEAD = 'or use a hosted model with --provider ID (natlang models lists them) or NATLANG_SERVER=URL';

/**
 * Install what a managed local model is missing, the llama.cpp runtime and the stock model, after one question that
 * states their combined size; --yes answers it. Without a terminal and --yes nobody can answer, so this says what to run
 * (exit 3). Returns false when the question is declined. `runtimeOnly` installs the managed llama.cpp build alone.
 */
async function ensureLocalModel(choice: ManagedChoice | undefined, assumeYes: boolean, runtimeOnly = false): Promise<boolean> {
  const discovery = discoverLlamaRuntime();
  const needRuntime = !discovery.selected || (runtimeOnly && discovery.selected.source !== 'managed');
  const release = choice && !runtimeOnly && !process.env.NATLANG_MODEL_PATH && !choice.local?.modelPath ? choice.release : undefined;
  const needModel = !!release && !localModelPrerequisites(process.env, release).modelPath;
  if (!needRuntime && !needModel) return true;
  const artifact = discovery.artifact;
  if (needRuntime) {
    if (discovery.candidates.some(candidate => candidate.source === 'explicit'))
      throw new CliError(`NATLANG_LLAMA_SERVER is incompatible: ${describeLlamaRuntime(discovery)}`, EXIT.unavailable);
    if (!artifact) throw new CliError(`natlang has no managed llama.cpp build for ${process.platform}-${process.arch}; ` +
      'set NATLANG_LLAMA_SERVER to a compatible build', EXIT.unavailable);
  }
  const parts = [...(needRuntime ? [{ what: `llama.cpp ${LLAMA_RUNTIME_RELEASE.version} ${artifact!.backend} runtime`, bytes: artifact!.bytes }] : []),
    ...(needModel ? [{ what: `the ${release!.label} model`, bytes: release!.bytes }] : [])];
  const total = parts.reduce((sum, part) => sum + part.bytes, 0);
  const listing = parts.map(part => `${part.what} (${megabytes(part.bytes)})`).join(' and ');
  const found = needRuntime && discovery.candidates.length ? `Found incompatible ${describeLlamaRuntime(discovery)}. ` : '';
  if (!assumeYes && !(process.stdin.isTTY && process.stderr.isTTY))
    throw new CliError(`${found}the local model needs ${listing}${parts.length > 1 ? `, ${megabytes(total)} in all,` : ''} downloaded once, ` +
      'and without a terminal natlang cannot ask first',
      EXIT.unavailable, `Run natlang setup --yes, pass --yes to this command, ${HOSTED_INSTEAD}.`);
  if (!assumeYes && !await confirm(`${found}Download ${listing}${parts.length > 1 ? `, ${megabytes(total)} in all` : ''}?`)) return false;
  if (needRuntime) await installManagedLlamaRuntime({ error: process.stderr });
  if (needModel) await ensureStockModel(release!);
  return true;
}
/** The managed session's own runtime check (a server started by a command that did not preflight its model). */
const sessionRuntime = (assumeYes: boolean) => async (_discovery: LlamaRuntimeDiscovery): Promise<LlamaServerInspection | null> =>
  await ensureLocalModel(undefined, assumeYes) ? discoverLlamaRuntime().selected : null;

/** The environment variable that holds a hosted provider's API key (pi-ai's own table), when it has one. */
async function providerKeyName(provider: string): Promise<string | undefined> {
  try {
    const { findEnvKeys } = await import(new URL('./env-api-keys.js', import.meta.resolve('@earendil-works/pi-ai')).href) as
      { findEnvKeys(provider: string, env: Record<string, string>): string[] | undefined };
    const names = findEnvKeys(provider, new Proxy({}, { get: () => 'set' })) ?? [];
    return names.find(name => name.endsWith('_API_KEY')) ?? names[0];
  } catch { return undefined; }
}
/** A hosted provider without credentials, as the variable to set and, when the provider offers it, subscription sign-in. */
async function missingCredentials(provider: string, apiKeyEnv?: string): Promise<CliError> {
  const { piModels } = await import('../model/pi-provider.js');
  const key = apiKeyEnv ?? await providerKeyName(provider), oauth = !!piModels().getProvider(provider)?.auth.oauth;
  const ways = [...(key ? [`set ${key}`] : []), ...(oauth ? [`sign in with natlang auth login ${provider}`] : [])];
  return new CliError(`provider ${provider} has no credentials`, EXIT.unavailable,
    ways.length ? `${ways.join(', or ')}.` : 'Set its API key environment variable (natlang models --json shows the provider).');
}

/** Whether an OpenAI-compatible endpoint answers: its model catalog, without running inference. */
async function probeEndpoint(choice: Extract<ResolvedModelChoice, { kind: 'external' }>): Promise<{ reachable: boolean; status?: number }> {
  const headers: Record<string, string> = { ...choice.headers };
  const key = process.env[choice.apiKeyEnv];
  if (key) headers.authorization ??= `Bearer ${key}`;
  const base = choice.endpoint.replace(/\/$/, ''), url = base + (base.endsWith('/v1') ? '/models' : '/v1/models');
  try {
    const response = await fetch(url, { headers, signal: AbortSignal.timeout(5000) });
    await response.body?.cancel();
    return { reachable: response.ok, status: response.status };
  } catch { return { reachable: false }; }
}

/**
 * Before any call: the selected model can be used (`reach: false` checks only what must be asked before downloading). A managed model's files are installed (after asking), a hosted
 * provider has credentials, an endpoint answers. Each failure is reported once, with the next action, as exit 3.
 */
async function preflightModel(natlang: ConfiguredNatlangRuntime, assumeYes: boolean, reach = true): Promise<void> {
  const { choice } = natlang.selection;
  if (choice.kind === 'managed-local') {
    if (!await ensureLocalModel(choice, assumeYes)) throw new CliError('the local model is not installed; nothing was downloaded', EXIT.unavailable,
      `Run natlang setup when ready, ${HOSTED_INSTEAD}.`);
  } else if (!reach) return;
  else if (choice.kind === 'pi-provider') {
    try { await natlang.prepare(); }
    catch (error) {
      if (/no credentials/.test((error as Error).message)) throw await missingCredentials(choice.provider, choice.apiKeyEnv);
      throw new CliError((error as Error).message, EXIT.unavailable);
    }
  } else if (!choice.managedNative) {
    const probe = await probeEndpoint(choice);
    if (probe.status === 401 || probe.status === 403) throw new CliError(`the model endpoint ${choice.endpoint} refused the request (HTTP ${probe.status})`,
      EXIT.unavailable, `Set ${choice.apiKeyEnv} to its API key.`);
    if (!probe.reachable && probe.status === undefined) throw new CliError(`the model endpoint ${choice.endpoint} is not reachable`, EXIT.unavailable,
      `Start the model server, or correct ${process.env.NATLANG_SERVER ? 'NATLANG_SERVER' : `the endpoint of profile ${natlang.selection.name}`}; natlang doctor checks it.`);
  }
}

function runtimeReport(discovery = discoverLlamaRuntime()): Record<string, unknown> {
  return { ok: Boolean(discovery.selected), required: LLAMA_RUNTIME_RELEASE.compatible,
    recommended: { version: LLAMA_RUNTIME_RELEASE.version, build: LLAMA_RUNTIME_RELEASE.build,
      artifact: discovery.artifact?.key ?? null, bytes: discovery.artifact?.bytes ?? null },
    selected: discovery.selected, candidates: discovery.candidates, runtimeRoot: discovery.runtimeRoot };
}
function outputRuntime(discovery: LlamaRuntimeDiscovery, json: boolean): void {
  if (json) { output(runtimeReport(discovery), true); return; }
  if (discovery.selected) {
    output(`llama.cpp runtime ready\nsource: ${discovery.selected.source}\npath: ${discovery.selected.path}\nversion: ${discovery.selected.version} ${discovery.selected.source === 'fork' ?
      `(natlang fork ${LLAMA_FORK_SOURCE.repository} at ${LLAMA_FORK_SOURCE.commit.slice(0, 12)})` : `(build ${discovery.selected.build})`}`, false);
    return;
  }
  output(`llama.cpp runtime unavailable\n${describeLlamaRuntime(discovery)}`, false);
}

/**
 * Prefill, once per model and llama.cpp build, the prompt opening that a new program's `answer.nl` sends, and with it the
 * shared base every function with the same system prompt builds on (llama/warm.ts): the first `natlang run` then starts
 * warm, and a new function's first call prefills about 300 tokens instead of the whole opening. The opening is
 * rendered by the runtime itself (a driver that captures the first request); no model call is made.
 */
async function prewarmStarter(choice: Extract<LoadedModelConfiguration['choice'], { kind: 'managed-local' }>): Promise<void> {
  if (process.env.NATLANG_WARM_PREFIXES === 'off') return;
  let opening: ModelTurnRequest | undefined;
  const capture = Object.assign(async (request: ModelTurnRequest) => { opening = request; throw new Error('opening captured'); },
    { model: choice.model }) as unknown as ModelDriver;
  const { model } = await runtimeModel(choice, capture);
  const answer = loadVirtualNatlang({ 'answer.nl': answerSource() }, 'answer.nl');
  await createNatlangRuntime({ model: model!, seed: { mode: 'backend' } }).run(() => answer(sampleInput)).catch(() => undefined);
  if (!opening) return;
  process.stderr.write('natlang: preparing the prompt cache so first runs start warm (once per model; minutes on a CPU)\n');
  const session = createResolvedModelSession(choice, process.env, process.stderr);
  try { await session.warm?.(opening); }
  catch (error) { process.stderr.write(`natlang: prompt cache not prepared: ${(error as Error).message.split('\n')[0]}; the first run prepares it\n`); }
  finally { await session.close(); }
}

const selectionOptions = (parsed: Parsed) => ({ profile: option(parsed, '--profile'), provider: option(parsed, '--provider'),
  model: option(parsed, '--model') });
function modelSelection(parsed: Parsed): LoadedModelConfiguration { return selectModel(selectionOptions(parsed)); }
/** `--escalate PROFILE`, or `none` to turn a profile's escalation off. */
const escalateOption = (parsed: Parsed) => { const value = option(parsed, '--escalate'); return value === 'none' ? false as const : value; };

const readConfigFile = (): NatlangConfigFile => {
  const path = join(defaultNatlangConfigDirectory(), 'config.json');
  return (existsSync(path) ? JSON.parse(readFileSync(path, 'utf8')) : {}) as NatlangConfigFile;
};

/** `natlang setup --program DIR`: pin the checkpoint a tuned program's active release was tuned on, ahead of its first run. */
async function setupProgram(parsed: Parsed, root: string): Promise<number> {
  const release = currentRelease(resolve(root));
  if (!release) throw new Error(`${root} has no active tune release (.natlang/tune/active.json)`);
  const sha256 = release.manifest.base.checkpoint_sha256, { nativeSources } = loadModelConfiguration(option(parsed, '--profile'));
  const found = findNativeCheckpoint(sha256, { programRoot: resolve(root), sources: nativeSources });
  if (!found) throw new Error(`checkpoint ${sha256} that ${root} was tuned on is not available here; natlang setup --native CHECKPOINT pins it`);
  const pin = await pinNativeCheckpoint(found.pinned ? { path: found.pinned, sha256, ...(found.step !== undefined ? { step: found.step } : {}) } : found.pointer!, defaultNativePython());
  const report = { ok: true, program: resolve(root), release: release.manifest.digest, checkpoint: pin.checkpoint, sha256, reused: pin.reused,
    actorView: pin.actorView ?? null };
  output(parsed.options.has('--json') ? report : `${basename(resolve(root))} runs its tuned release ${release.manifest.digest.slice(0, 12)} on ` +
    `${sha256.slice(0, 12)} (pinned at ${pin.checkpoint})`, parsed.options.has('--json'));
  return 0;
}

type NatlangConfigFile = { defaultProfile?: string; profiles?: Record<string, import('../model/config.js').ModelProfile>;
  nativeSources?: Record<string, string> } & Record<string, unknown>;

/** `natlang setup --native SOURCE`: pin the checkpoint and write (or update) its profile. */
async function setupNative(parsed: Parsed, source: string): Promise<number> {
  const configPath = join(defaultNatlangConfigDirectory(), 'config.json');
  const config = readConfigFile();
  const named = config.nativeSources?.[source];
  const name = option(parsed, '--profile') ?? (named ? source : 'native');
  const pointer = readCheckpointPointer(named ?? source);
  const existing = config.profiles?.[name] ?? {};
  if (!existing.native && (existing.endpoint || existing.provider || existing.local))
    throw new Error(`profile ${name} is not a native profile; choose another --profile`);
  const previous = existing.native;
  const pin = await pinNativeCheckpoint(pointer, { ...defaultNativePython(), ...(previous?.python ? { python: previous.python, pythonPath: previous.pythonPath } : {}) });
  const port = option(parsed, '--port'), device = option(parsed, '--device');
  const native: NativeModelSettings = { ...defaultNativePython(), ...previous, checkpoint: pin.checkpoint, sha256: pin.sha256, sourceSha256: pin.sourceSha256,
    source: named ? source : resolve(source), ...(pointer.step !== undefined ? { step: pointer.step } : { step: undefined }), actorView: pin.actorView,
    ...(device ? { device } : {}), ...(port ? { port: Number(port) } : {}) };
  // --escalate PROFILE (or none): both profiles hand off live invocations to this fallback.
  const escalate = escalateOption(parsed);
  if (typeof escalate === 'string' && !config.profiles?.[escalate]) throw new Error(`--escalate names no profile ${escalate}`);
  const withEscalation = <T extends { escalate?: string }>(profile: T): T => {
    if (escalate === undefined) return profile;
    const { escalate: _previous, ...rest } = profile;
    return (escalate === false ? rest : { ...rest, escalate }) as T;
  };
  config.profiles = { ...config.profiles, [name]: withEscalation({ ...existing, native }) };
  // --gguf: the same weights as a GGUF for the managed llama.cpp backend (faster; ordinary tokens only).
  // A refresh keeps an existing GGUF sibling current (its export goes with the superseded pin).
  const ggufName = `${name}-gguf`;
  const outtype = option(parsed, '--gguf') ?? /^native-gguf-[0-9a-f]{12}-(\w+)$/.exec(config.profiles[ggufName]?.local?.modelId ?? '')?.[1];
  const gguf = outtype ? await exportNativeGguf(native, outtype) : undefined;
  if (gguf) {
    const previousGguf = config.profiles[ggufName] ?? {};
    if (previousGguf.endpoint || previousGguf.provider || previousGguf.native)
      throw new Error(`profile ${ggufName} is not a GGUF profile; rename it first`);
    config.profiles[ggufName] = withEscalation({ ...previousGguf,
      local: { contextTokens: 16384, ...previousGguf.local, modelPath: gguf.modelPath, template: gguf.template,
        modelId: `native-gguf-${pin.sha256.slice(0, 12)}-${outtype}` },
      // The native server decodes greedily unless asked otherwise; llama-server would sample at 0.8.
      request: { temperature: 0, ...previousGguf.request, chat_template_kwargs: gguf.templateKwargs, logit_bias: gguf.logitBias } });
  }
  if (parsed.options.has('--default')) config.defaultProfile = gguf ? ggufName : name;
  mkdirSync(dirname(configPath), { recursive: true });
  const temporary = `${configPath}.${process.pid}.tmp`;
  writeFileSync(temporary, JSON.stringify(config, null, 1) + '\n'); renameSync(temporary, configPath);
  // The superseded pin goes once no profile names it; pins are only ever copies.
  let removed: string | undefined;
  if (previous && previous.checkpoint !== pin.checkpoint && !Object.values(config.profiles).some(profile => profile.native?.checkpoint === previous.checkpoint)
    && removeNativePin(previous.checkpoint)) removed = previous.checkpoint;
  const report = { ok: true, profile: name, default: config.defaultProfile === name, checkpoint: pin.checkpoint, sha256: pin.sha256,
    step: pointer.step ?? null, sourceSha256: pin.sourceSha256, bytes: pin.bytes, reused: pin.reused, actorView: pin.actorView ?? null,
    ...(gguf ? { ggufProfile: ggufName, gguf: gguf.modelPath, ggufSha256: gguf.sha256 } : {}), ...(removed ? { removedPin: removed } : {}) };
  output(parsed.options.has('--json') ? report : `native profile ${name} ready${report.default ? ' (default)' : ''}\n` +
    `checkpoint: ${pin.sha256.slice(0, 12)}${pointer.step !== undefined ? ` (step ${pointer.step})` : ''}\n` +
    `use: natlang run APP --profile ${name}   (natlang serve --profile ${name} keeps the model loaded)` +
    (gguf ? `\nGGUF: natlang run APP --profile ${ggufName}   (llama.cpp; faster, no Neuralese production)` : ''), parsed.options.has('--json'));
  return 0;
}

/** `natlang serve`: start the profile's native server and hold it until interrupted. */
async function serveCommand(parsed: Parsed): Promise<number> {
  const { name, choice } = modelSelection(parsed);
  if (choice.kind !== 'external' || !choice.managedNative)
    throw new Error(`profile ${name} has no native model to serve; run natlang setup --native SOURCE`);
  await ensureNativeServer(choice.managedNative);
  if (!ownsNativeServer(choice.endpoint)) { output(`already serving ${choice.managedNative.sha256.slice(0, 12)} at ${choice.endpoint}`, false); return 0; }
  output(`serving profile ${name} at ${choice.endpoint}; other natlang commands use it until you press Ctrl-C`, false);
  // Signal listeners do not keep the process alive; the health check does, and ends the command if the server dies.
  const failed = await new Promise<boolean>(resolveStop => {
    const health = setInterval(() => void fetch(choice.endpoint + '/health', { signal: AbortSignal.timeout(10_000) })
      .then(response => { if (!response.ok) throw new Error(); }).catch(() => { clearInterval(health); resolveStop(true); }), 30_000);
    const stop = () => { clearInterval(health); resolveStop(false); };
    process.once('SIGINT', stop); process.once('SIGTERM', stop);
  });
  if (failed) process.stderr.write(`natlang: the native server at ${choice.endpoint} stopped answering\n`);
  await stopNativeServer(choice.endpoint);
  return failed ? 1 : 0;
}

function modelSession(choice: ResolvedModelChoice, assumeYes = false, loraAdapters?: () => readonly { id: string; path: string }[],
  nativeStates?: () => readonly { id: string; path: string }[]) {
  return createResolvedModelSession(choice, process.env, process.stderr,
  { ensureRuntime: sessionRuntime(assumeYes), ...(loraAdapters ? { loraAdapters } : {}), ...(nativeStates ? { nativeStates } : {}) });
}

function selectTarget(manifest: { name: string; targets?: Record<string, unknown> }, requested?: string): string {
  const targets = Object.keys(manifest.targets ?? {});
  if (!targets.length) throw usageError(`${manifest.name} has no runnable targets. Use natlang call FUNCTION.nl, or add a main.ts target to natlang.json.`);
  const selected = requested ?? (targets.length === 1 ? targets[0] : undefined);
  if (!selected) throw usageError(`${manifest.name} has multiple targets; use --target ${targets.join('|')}`);
  if (!manifest.targets?.[selected]) throw usageError(`unknown target ${selected} in ${manifest.name}; choose ${targets.join(', ')}`);
  return selected;
}


function checkEngines(engines: { node?: string; natlang?: string } | undefined): void {
  if (engines?.node && !satisfiesVersion(process.versions.node, engines.node))
    throw new Error(`package needs Node ${engines.node}; this process is ${process.versions.node}`);
  if (engines?.natlang && !satisfiesVersion(NATLANG_CLI_VERSION, engines.natlang))
    throw new Error(`package needs natlang ${engines.natlang}; this CLI is ${NATLANG_CLI_VERSION}`);
}
function commandAvailable(command: string): boolean {
  const probe = spawnSync(process.platform === 'win32' ? 'where.exe' : 'which', [command], { stdio: 'ignore' });
  return probe.status === 0;
}


/** Which llama.cpp build serves managed models, and what natlang's fork would add when it is not the one. */
function llamaRuntimeNote(): string {
  const selected = discoverLlamaRuntime().selected;
  if (!selected) return 'not installed: natlang setup';
  if (selected.source === 'fork') return `natlang fork ${LLAMA_FORK_SOURCE.commit.slice(0, 12)}, built from source`;
  if (selected.source === 'managed') return `natlang fork release b${selected.build}`;
  return `${selected.source} b${selected.build}`;
}
async function doctorReport(parsed: Parsed, store: NatlangPackageStore): Promise<{ report: Record<string, unknown>; okay: boolean }> {
  const selected = modelSelection(parsed), choice = selected.choice;
  const local = choice.kind === 'managed-local' ? localModelPrerequisites(process.env, choice.release) : null;
  const slotPlan = choice.kind === 'managed-local' ? localSlotPlan(choice.local, process.env, local?.modelPath ?? null, choice.release) : null;
  let piReady = false;
  if (choice.kind === 'pi-provider') {
    let session: ReturnType<typeof createResolvedModelSession> | undefined;
    try {
      session = createResolvedModelSession(choice);
      await session.prepare();
      piReady = true;
    } catch { /* doctor reports a provider profile that cannot prepare as unavailable */ }
    finally { await session?.close(); }
  }
  let connection: { checked: boolean; reachable: boolean | null; detail: string; next?: string } = {
    checked: false, reachable: null, detail: 'Not contacted; local model files and runtime are checked without starting a server.' };
  if (choice.kind === 'managed-local' && (!local?.available || !local.modelPath)) connection = {
    checked: false, reachable: null, detail: 'Local model setup is incomplete.', next: 'Run natlang setup to install the runtime and download the model.' };
  if (choice.kind === 'pi-provider') connection = { checked: false, reachable: null,
    detail: piReady ? 'Provider credentials are configured; inference has not been tested.' : 'Provider credentials are unavailable.',
    ...(!piReady ? { next: (await missingCredentials(choice.provider, choice.apiKeyEnv)).next! } : {}) };
  if (choice.kind === 'external') {
    const probe = await probeEndpoint(choice);
    connection = probe.reachable ? { checked: true, reachable: true, detail: 'Model catalog reachable; inference has not been tested.' } :
      probe.status !== undefined ? { checked: true, reachable: false, detail: `Model catalog returned HTTP ${probe.status}.`,
        next: 'Check the endpoint URL and credentials, then test the profile in Models.' } :
      { checked: true, reachable: false, detail: 'Could not reach the model endpoint.', next: 'Start the server or correct the endpoint URL, then run natlang doctor again.' };
  }
  const modelOkay = choice.kind === 'pi-provider' ? piReady : choice.kind === 'external' ? connection.reachable === true : Boolean(local?.available && local.modelPath);
  const okay = Boolean(modelOkay);
  // Native models, tuning and the program (cli/status.ts): pin, newer checkpoints, server, release base, pending examples.
  const status = await modelStatus(selected, readConfigFile().profiles ?? {}, programForStatus(option(parsed, '--program')));
  return { okay, report: { ok: okay, node: process.version, packageStore: store.root,
    cacheDirectory: defaultNatlangCacheDirectory(), runtimeDirectory: defaultNatlangRuntimeDirectory(),
    installedPackages: store.list().length, config: selected.configPath, profile: selected.name, configurationValid: true, connection,
    inferenceTested: false,
    modelSource: choice.kind, endpoint: choice.kind === 'external' ? choice.endpoint : null,
    model: choice.model, provider: choice.kind === 'pi-provider' ? choice.provider : null,
    modelPath: local?.modelPath ?? null, modelServer: local?.executable ?? null,
    ...(choice.kind === 'managed-local' ? { llamaRuntime: llamaRuntimeNote() } : {}),
    modelDownloadAvailable: local?.downloadable ?? null,
    apiKey: choice.kind === 'external' ? Boolean(process.env[choice.apiKeyEnv]) : null,
    providerAuthConfigured: choice.kind === 'pi-provider' ? piReady : null,
    // Batched execution (plans/BATCHED_EXECUTION.md): the slots a managed server starts with and the scheduler's limits.
    serverSlots: slotPlan?.slots ?? null, kvBudgetMiB: slotPlan?.budgetBytes == null ? null : Math.round(slotPlan.budgetBytes / 2 ** 20),
    kvBudgetSource: slotPlan?.budgetSource ?? null, slotPlan: slotPlan ?? null,
    batching: choice.kind === 'pi-provider' ? null : { mode: choice.batching?.mode ?? 'server-continuous',
      maxConcurrent: choice.batching?.maxConcurrent ?? choice.concurrency ?? slotPlan?.slots ?? null,
      coalesceMs: choice.batching?.coalesceMs ?? null, scoreEndpoint: choice.batching?.scoreEndpoint ?? null }, ...status } };
}

/** natlang doctor without --json: the model, how it runs, whether it answers, where files live, and the next action. */
function doctorText(report: Record<string, unknown>): string {
  const connection = report.connection as { detail: string; next?: string };
  const where = report.modelSource === 'managed-local' ? `local (${report.modelPath ?? 'not downloaded yet'}); llama.cpp: ${report.llamaRuntime}` :
    report.modelSource === 'pi-provider' ? `hosted provider ${report.provider}` : `endpoint ${report.endpoint}`;
  const native = report.nativeModel as Record<string, unknown> | undefined, tuning = report.tuning as Record<string, unknown> | undefined;
  return [`model: ${report.model} (profile ${report.profile})`, `runs on: ${where}`, `connection: ${connection.detail}`,
    `files: cache ${report.cacheDirectory}, runtimes ${report.runtimeDirectory}`,
    ...(native ? [`native: checkpoint ${native.checkpoint}${native.step != null ? ` (step ${native.step})` : ''}${native.newer ? `; newer ${native.newer}` : ''}`,
      `tuning: ${tuning && !('unavailable' in tuning) ? `student ${tuning.student} on ${tuning.base}` : String(tuning?.unavailable ?? 'not configured')}`] : []),
    `next: ${report.ok ? 'ready; natlang doctor --json shows the full report' : connection.next ?? 'natlang doctor --json shows what is missing'}`].join('\n');
}

function applicationSpecifier(store: NatlangPackageStore, query: string, requestedTarget?: string): string {
  const packages = store.list().filter(item => {
    const short = item.name.includes('/') ? item.name.slice(item.name.lastIndexOf('/') + 1) : item.name;
    return item.name === query || short === query;
  });
  const names = [...new Set(packages.map(item => item.name))];
  if (!packages.length) throw usageError(`${query} is neither a directory nor an installed application; natlang apps lists the applications here`);
  if (names.length > 1) throw usageError(`application name is ambiguous: ${names.join(', ')}`);
  packages.sort((left, right) => compareVersions(right.version, left.version));
  const selected = packages[0]!, manifest = store.manifest(`${selected.name}@${selected.version}`);
  const targets = Object.keys(manifest.targets ?? {});
  const target = requestedTarget ?? (targets.length === 1 ? targets[0] : undefined);
  if (!target) throw usageError(`${selected.name} has multiple targets; use --target ${targets.join('|')}`);
  if (!manifest.targets?.[target]) throw usageError(`unknown target ${target} in ${selected.name}; choose ${targets.join(', ')}`);
  return `${selected.name}@${selected.version}#${target}`;
}


/** Nearest directory at or above `start` holding a project marker. */
function projectRoot(start: string): string {
  let dir = statSync(start).isDirectory() ? start : dirname(start);
  while (true) {
    if (['tsconfig.json', 'natlang.json', 'package.json'].some(name => existsSync(join(dir, name)))) return dir;
    const parent = dirname(dir);
    if (parent === dir) return statSync(start).isDirectory() ? start : dirname(start);
    dir = parent;
  }
}

/** Compile a project for this process's runtime; returns the emitted path of `entry`. */
function compileFor(root: string, entry: string, outDir: string, installed: boolean, programId?: string): string {
  const result = buildProject({ project: root, outDir, programId, runtimeModule: RUNTIME_MODULE, writeDeclarations: !installed });
  if (!result.ok) throw new Error(`natlang build failed:\n${formatDiagnostics(result.diagnostics)}`);
  // A `.nl` entry runs through the loader (as natlang call and natlang test do); the build above has checked it.
  if (/\.(?:m?js|nl)$/.test(entry)) return entry;
  const emitted = join(outDir, relative(root, entry)).replace(/\.m?ts$/, '.js');
  if (!existsSync(emitted)) throw new Error(`the build did not emit ${relative(root, entry)}`);
  return emitted;
}

type Launch = { root: string; target: NatlangTarget; targetName: string; installed?: boolean;
  package?: TargetContext['package']; dependencies: TargetContext['dependencies']; stateKey: string };

/** The recording, specialization and refinement settings of the package at `root` (natlang.json). */
function readManifestSettings(root: string): { recording?: { exclude?: string[] }; specialization?: 'off' | 'shadow' | 'on';
  refinements?: RefinementSettings } {
  let manifest;
  try { manifest = parsePackageManifest(JSON.parse(readFileSync(join(root, 'natlang.json'), 'utf8'))); } catch { return {}; }
  return { ...(manifest.recording ? { recording: manifest.recording } : {}), ...(manifest.specialization ? { specialization: manifest.specialization } : {}),
    // A malformed refinements block is an error, not a silently unchecked program.
    ...(manifest.refinements ? { refinements: parseRefinementSettings(manifest.refinements) } : {}) };
}

async function launch(parsed: Parsed, spec: Launch): Promise<number> {
  const stateDirectory = resolve(option(parsed, '--state') ?? join(defaultNatlangStateDirectory(),
    encodeURIComponent(spec.stateKey), spec.targetName));
  const traceDirectory = resolve(option(parsed, '--traces') ?? join(stateDirectory, 'traces'));
  const workspace = resolve(option(parsed, '--workspace') ?? '.');
  const entry = join(spec.root, ...spec.target.entry.split('/'));
  const selection = option(parsed, '--adaptation') ?? spec.target.adaptation;
  if (option(parsed, '--adaptation') && parsed.options.has('--no-adaptation')) throw usageError('--adaptation and --no-adaptation are mutually exclusive');
  const artifact = !parsed.options.has('--no-adaptation') && selection ? parseAdaptation(readFileSync(
    option(parsed, '--adaptation') ? existingPath(selection, '--adaptation') : resolve(spec.root, selection), 'utf8')) : null;
  // A `.nl` entry is the target's function itself: its parameters are the target's arguments unless natlang.json declares them.
  const entryFunction = spec.target.entry.endsWith('.nl') ? loadNatlang(entry, spec.root) : undefined;
  const entryParams = entryFunction ? callableMeta(entryFunction)!.definition.params : undefined;
  const targetArguments = spec.target.arguments ?? entryParams?.map(param => ({ name: param.name, type: param.type, required: !param.optional }));
  // Declared arguments are checked before building or starting a model, so a wrong one fails at once.
  const declaredArguments = readTargetArguments(targetArguments, parsed.rest, spec.root, workspace);
  if (declaredArguments.problems.length) throw usageError(`${declaredArguments.problems.map(problem => problem.message).join('\n')}\n` +
    `usage: natlang run ${relative(process.cwd(), spec.root) || '.'} --target ${spec.targetName} -- ${argumentUsage(targetArguments!)}`);
  const buildDirectory = spec.installed ? join(stateDirectory, 'build', spec.package?.digest ?? 'local') : join(spec.root, '.natlang', 'build');
  process.stderr.write(`natlang: checking and building ${relative(process.cwd(), spec.root) || '.'}\n`);
  const compiled = compileFor(spec.root, entry, buildDirectory, spec.installed === true, artifact?.program.id);
  const inventory = buildProject({ project: spec.root, programId: artifact?.program.id, emit: false, write: false, runtimeModule: RUNTIME_MODULE });
  if (!inventory.ok) throw new Error(formatDiagnostics(inventory.diagnostics));
  const manifest = spec.package ? readManifestSettings(spec.root) : {};
  // Crisp checkers of refined predicates come from the program's `refinements.ts`, which its compiled `.nl` modules load.
  const refinements = manifest.refinements;
  const progress = progressFor(parsed);
  // The compiled program's own model (natlang.json "model"); the command's selection wins.
  const declared = spec.package ? (() => { try { return readProgramManifest(spec.root); } catch { return undefined; } })() : undefined;
  const natlang = await natlangRuntime({ ...selectionOptions(parsed), ...(declared?.model ? { programModel: declared.model } : {}),
    programRoot: spec.root, program: inventory.manifest.adaptation!,
    weights: !parsed.options.has('--no-weights'), escalate: escalateOption(parsed), ...(artifact ? { adaptation: artifact } : {}),
    ensureRuntime: sessionRuntime(parsed.options.has('--yes')),
    runtime: { trace: fileTraceSink(traceDirectory), programRoot: spec.root, ...manifest, ...(refinements ? { refinements } : {}),
      onLive: progress.live } });
  const { runtime, driver, executorIdentity, adaptation } = natlang, program = natlang.program!, choice = natlang.selection.choice;
  process.stderr.write(`natlang: model ${choice.model} via ${choice.kind} (profile ${natlang.selection.name})\n`);
  try {
    // A TypeScript entry may not call the model at all: only a managed model's download is settled before it starts.
    await preflightModel(natlang, parsed.options.has('--yes'), !!entryFunction);
    // Fused hand-offs (plans/FUSED_PIPELINES.md): off unless natlang.json says otherwise.
    const fusionSetting = spec.package ? fusionSettings(parsed, spec.root) : undefined;
    if (fusionSetting && fusionSetting.mode !== 'off') {
      const fusion = await prepareFusion(runtime, spec.root, fusionSetting);
      if (fusion) (runtime.options as { fusion?: typeof fusion }).fusion = fusion;
    }
    if (entryFunction && entryParams) {
      const value = await runtime.run(() => entryFunction(...entryParams.map(param => declaredArguments.values[param.name])));
      process.stdout.write(`${pipelineValue(value)}\n`);
      return 0;
    }
    const module = await import(pathToFileURL(compiled).href) as Record<string, unknown>;
    const name = spec.target.export ?? 'main', main = module[name];
    if (typeof main !== 'function') throw new Error(`${spec.target.entry} does not export ${name}()`);
    const context: TargetContext = { package: spec.package, dependencies: spec.dependencies, targetName: spec.targetName,
      target: spec.target, workspace, stateDirectory, traceDirectory, args: parsed.rest, arguments: declaredArguments.values,
      io: { input: process.stdin, output: process.stdout, error: process.stderr,
        color: !parsed.options.has('--plain') && !parsed.options.has('--no-color') && Boolean(process.stdout.isTTY) },
      model: driver, runtime, program, adaptation, executorIdentity,
      ...(choice.kind === 'external' ? { modelEndpoint: { endpoint: choice.endpoint, model: choice.model, apiKeyEnv: choice.apiKeyEnv } } : {}) };
    const value = await (main as TargetMain)(context);
    if (value && typeof value === 'object' && typeof (value as TargetExecutable).run === 'function') {
      const executable = value as TargetExecutable;
      try { return Number(await executable.run() ?? 0); } finally { await executable.close?.(); }
    }
    return Number(value ?? 0);
  } finally { progress.close(); reportCalls(progress.roots, runtime); await natlang.close(); }
}

async function runTarget(parsed: Parsed, value = '.'): Promise<number> {
  acceptOptions(parsed, ['--progress', '--target', '--export', '--profile', '--provider', '--model', '--escalate', '--workspace', '--state', '--traces', '--store',
    '--plain', '--no-color', '--yes', '--adaptation', '--no-adaptation', '--no-weights']);
  const path = resolve(value);
  if (existsSync(path) && statSync(path).isFile() && /\.(?:m?ts|m?js)$/.test(path)) {
    const root = projectRoot(path);
    return launch(parsed, { root, targetName: 'main', dependencies: {}, stateKey: basename(root),
      target: { entry: relative(root, path).split('\\').join('/'), ...(option(parsed, '--export') ? { export: option(parsed, '--export')! } : {}) } });
  }
  if (existsSync(path)) {
    const manifestPath = statSync(path).isDirectory() ? join(path, 'natlang.json') : path;
    if (!isApplicationManifest(manifestPath)) throw usageError(`${value} has no valid natlang.json manifest. Use natlang init DIRECTORY to create a program, or natlang run main.ts for a TypeScript entry.`);
    const root = dirname(manifestPath);
    const archive = createPackageArchive(JSON.parse(readFileSync(manifestPath, 'utf8')), root);
    const store = new NatlangPackageStore(option(parsed, '--store'));
    const targetName = selectTarget(archive.manifest, option(parsed, '--target'));
    return launch(parsed, { root, targetName, target: archive.manifest.targets![targetName]!,
      package: { name: archive.manifest.name, version: archive.manifest.version, digest: archive.digest, root },
      dependencies: store.resolveDependencies(archive.manifest.dependencies),
      stateKey: `${archive.manifest.name}/${archive.manifest.version}-local.${archive.digest.slice(0, 12)}` });
  }
  const store = new NatlangPackageStore(option(parsed, '--store'));
  const specifier = value.includes('#') ? value : applicationSpecifier(store, value, option(parsed, '--target'));
  const marker = specifier.lastIndexOf('#');
  const packageSpecifier = specifier.slice(0, marker), targetName = specifier.slice(marker + 1);
  const installed = store.resolve(packageSpecifier), manifest = store.manifest(packageSpecifier);
  checkEngines(manifest.engines);
  const target = manifest.targets?.[targetName];
  if (!target) throw usageError(`unknown target ${targetName} in ${manifest.name}`);
  return launch(parsed, { root: installed.root, targetName, target, installed: true,
    package: { name: installed.name, version: installed.version, digest: installed.digest, root: installed.root },
    dependencies: installed.dependencies, stateKey: `${installed.name}/${installed.version}` });
}

/** The fusion settings of the package at `root` (natlang.json), with `--fusion-planner` over them. */
function fusionSettings(parsed: Parsed, root: string): FusionSettings {
  let raw: unknown;
  try { raw = (JSON.parse(readFileSync(join(root, 'natlang.json'), 'utf8')) as { fusion?: unknown }).fusion; } catch { raw = undefined; }
  const settings = parseFusionSettings(raw), planner = option(parsed, '--fusion-planner');
  if (planner && planner !== 'crisp' && planner !== 'nl' && planner !== 'shadow') throw usageError('--fusion-planner must be crisp, nl or shadow');
  return planner ? { ...settings, planner: planner as FusionSettings["planner"] } : settings;
}

/** The fusion report for `natlang check`; analysis problems never fail a check. */
async function fusionForCheck(parsed: Parsed, project: string) {
  try {
    const root = resolve(project), settings = fusionSettings(parsed, root);
    if (settings.planner === 'crisp') return (await analyzeFusion(root, settings)).report;
    return await withModelRuntime(parsed, async runtime => (await analyzeFusion(root, settings, runtime)).report, undefined, { calls: false });
  } catch (error) {
    if (error instanceof TypeError) throw error;
    return undefined;
  }
}

/** `natlang build`: write the program's lock; with --warm, run its warm-up calls and prefill their prompt prefixes first. */
async function buildLock(parsed: Parsed, root: string): Promise<{ lock: BuildLock; path: string }> {
  if (parsed.options.has('--warm')) {
    const declared = (() => { try { return readProgramManifest(root); } catch { return undefined; } })();
    const { choice } = selectModel({ ...selectionOptions(parsed), ...(declared?.model ? { programModel: declared.model } : {}) });
    if (choice.kind === 'managed-local' && !await ensureLocalModel(choice, parsed.options.has('--yes')))
      throw new CliError('the local model is not installed; nothing was downloaded', EXIT.unavailable, `run natlang setup when ready, ${HOSTED_INSTEAD}`);
    const progress = progressFor(parsed);
    try {
      return await warmProgram(root, { ...selectionOptions(parsed), version: NATLANG_CLI_VERSION, log: process.stderr,
        live: progress.live, ensureRuntime: sessionRuntime(parsed.options.has('--yes')) });
    } finally { progress.close(); }
  }
  const manifest = (() => { try { return readProgramManifest(root); } catch { return undefined; } })();
  const lock = lockFor(root, selectModel({ ...selectionOptions(parsed), ...(manifest?.model ? { programModel: manifest.model } : {}) }), NATLANG_CLI_VERSION);
  return { lock, path: writeBuildLock(join(root, '.natlang', 'build'), lock) };
}

function describeLock(lock: BuildLock, path: string): string {
  const warm = lock.warm;
  return [`lock: ${relative(process.cwd(), path) || path}`,
    `model: ${lock.model.release ? `${lock.model.release.id} (${lock.model.release.source}@${lock.model.release.revision.slice(0, 8)})` : lock.model.model}` +
      ` via ${lock.model.kind}${lock.model.runtime ? `, llama.cpp build ${lock.model.runtime.build}` : ''} (profile ${lock.model.profile})`,
    `tuning: ${lock.tuning ? `release ${lock.tuning.release.slice(0, 12)} on ${lock.tuning.base.slice(0, 12)}` : 'none'}`,
    `functions: ${lock.functions.map(item => item.name).join(', ') || '(none)'}`,
    warm ? `warm: ${warm.cases.filter(item => item.outcome === 'done').length}/${warm.cases.length} warm-up calls done, ` +
      `${warm.prefixes.length} prompt prefix${warm.prefixes.length === 1 ? '' : 'es'} prefilled (${warm.prefixes.map(item => item.tokens).join(', ') || 0} tokens)` :
      'warm: none (natlang build --warm runs the natlang.json warmup calls and prefills their prompts)'].join('\n');
}

/**
 * The progress display of a command that runs natural-language calls (`--progress`, NATLANG_PROGRESS), with the IDs of
 * its root calls, which `reportCalls` names at the end so `natlang traces show` finds them.
 */
function progressFor(parsed: Parsed, options: { roots?: boolean } = {}) {
  const progress = commandProgress(process.stderr, parseProgressMode(option(parsed, '--progress') ?? process.env.NATLANG_PROGRESS), process.env, options);
  const roots: string[] = [];
  const live: LiveSink = event => { if (event.type === 'call_start' && event.parentCallId === null) roots.push(event.callId); progress.live?.(event); };
  return { live, roots, close: progress.close };
}
function reportCalls(roots: readonly string[], runtime: NatlangRuntime): void {
  if (!roots.length || !runtime.callStore()) return;
  process.stderr.write(roots.length <= 3 ? roots.map(id => `natlang: call ${id} (natlang traces show ${id})\n`).join('') :
    `natlang: ${roots.length} calls (natlang traces list)\n`);
}

async function withModelRuntime<T>(parsed: Parsed, fn: (runtime: NatlangRuntime) => Promise<T>, sourceRoot = projectRoot(process.cwd()),
  options: { calls?: boolean } = {}): Promise<T> {
  const trace = option(parsed, '--trace'), weights = !parsed.options.has('--no-weights');
  if (option(parsed, '--adaptation') && parsed.options.has('--no-adaptation')) throw usageError('--adaptation and --no-adaptation are mutually exclusive');
  const artifact = option(parsed, '--adaptation') ? parseAdaptation(readFileSync(existingPath(option(parsed, '--adaptation')!, '--adaptation'), 'utf8')) : null;
  // A tuned project is identified from the working directory, without flags.
  const release = weights ? currentRelease(process.cwd()) : undefined;
  const inventory = artifact || release ? buildProject({ project: process.cwd(), programId: artifact?.program.id ?? release?.manifest.program.id,
    emit: false, write: false, runtimeModule: RUNTIME_MODULE }) : null;
  if (inventory && !inventory.ok) throw new Error(formatDiagnostics(inventory.diagnostics));
  const progress = progressFor(parsed);
  const manifest = readProgramManifest(sourceRoot);
  const natlang = await natlangRuntime({ ...selectionOptions(parsed), ...(manifest?.model ? { programModel: manifest.model } : {}), ...(inventory ? { programRoot: process.cwd(), program: inventory.manifest.adaptation } : {}),
    weights, escalate: escalateOption(parsed), ...(artifact ? { adaptation: artifact } : {}),
    ensureRuntime: sessionRuntime(parsed.options.has('--yes')),
    runtime: { ...(trace ? { trace: fileTraceSink(resolve(trace)) } : {}), onLive: progress.live,
      // Calls of a program's functions are recorded under that program (natlang traces list inside it shows them).
      ...(existsSync(join(sourceRoot, 'natlang.json')) ? { programRoot: sourceRoot } : {}) } });
  try {
    await preflightModel(natlang, parsed.options.has('--yes'));
    await natlang.prepare();
    return await fn(natlang.runtime);
  } finally {
    progress.close();
    if (options.calls !== false) reportCalls(progress.roots, natlang.runtime);
    await natlang.close();
  }
}

const tunedProject = (parsed: Parsed) => !parsed.options.has('--no-weights') && !!currentRelease(process.cwd());
/** The nearest directory at or above `directory` holding natlang.json, if any. */
function enclosingProgram(directory: string): string | undefined {
  for (let at = directory; ; at = dirname(at)) {
    if (existsSync(join(at, 'natlang.json'))) return at;
    if (dirname(at) === at) return undefined;
  }
}

/** `path`, which must exist (a usage error names it otherwise). */
function existingPath(path: string, what?: string): string {
  const resolved = resolve(path);
  if (!existsSync(resolved)) throw usageError(`${what ? `${what} ` : ''}${path} does not exist`);
  return resolved;
}

function commandCallable(name: string, adaptationRoot?: string): ((...args: unknown[]) => Promise<unknown>) &
  { [key: symbol]: { definition: { params: { name: string; type: string; optional?: boolean }[]; subtype: string } } } {
  let value: unknown;
  // A program's function is loaded relative to its program, so its recorded source is the program's path for it.
  if (name.endsWith('.nl')) value = loadNatlang(existingPath(name), adaptationRoot ?? enclosingProgram(dirname(resolve(name))));
  else {
    const context = applicationContextRecords(process.cwd());
    if (!context) throw usageError(`${name} is not a .nl file, and there is no natlang.d context here to find it in`);
    value = name.split('.').reduce<unknown>((part, key) =>
      part && typeof part === 'object' || typeof part === 'function' ?
        (part as Record<string, unknown>)[key] : undefined, loadCallables(context.dir));
  }
  if (typeof value !== 'function' || !(Symbol.for('natlang.callable') in value))
    throw usageError(`${name} is not a natural-language function`);
  return value as ReturnType<typeof commandCallable>;
}

async function pipedStdin(waitForEnd = false): Promise<string> {
  if (process.stdin.isTTY) return '';
  return new Promise<string>(resolve => {
    const chunks: string[] = [];
    let timer: ReturnType<typeof setTimeout> | undefined;
    const finish = () => {
      if (timer) clearTimeout(timer);
      process.stdin.off('data', data); process.stdin.off('end', end);
      process.stdin.pause(); resolve(chunks.join(''));
    };
    const data = (chunk: Buffer | string) => { chunks.push(chunk.toString()); if (timer) { clearTimeout(timer); timer = undefined; } };
    const end = () => finish();
    process.stdin.on('data', data); process.stdin.once('end', end);
    if (!waitForEnd) timer = setTimeout(finish, 150);
    process.stdin.resume();
  });
}

function commandInputArgs(input: unknown, params: { name: string }[]): unknown[] {
  if (input && typeof input === 'object' && !Array.isArray(input))
    return params.map(parameter => (input as Record<string, unknown>)[parameter.name]);
  if (params.length === 1) return [input];
  throw usageError('call input must be a JSON object keyed by parameter name');
}

/** A result as a command prints it: text as text, everything else (and everything under --json) as JSON. */
const pipelineValue = (value: unknown, json = false) =>
  json || value === null || typeof value === 'object' ? JSON.stringify(value) : String(value);
const callRoot = (name: string) => existsSync(resolve(name)) ? projectRoot(resolve(name)) : projectRoot(process.cwd());

async function callCommand(parsed: Parsed, name: string, words: string[] = []): Promise<number> {
  acceptOptions(parsed, ['--progress', '--inputs', '--trace', '--profile', '--provider', '--model', '--json', '--yes', '--lines', '--jsonl', '--adaptation', '--no-adaptation', '--no-weights', '--escalate']);
  if (words.length || parsed.rest.length) return callWithArguments(parsed, name, [...words, ...parsed.rest]);
  if (parsed.options.has('--lines') && parsed.options.has('--jsonl')) throw usageError('--lines and --jsonl are mutually exclusive');
  if (option(parsed, '--inputs') && (parsed.options.has('--lines') || parsed.options.has('--jsonl')))
    throw usageError('--inputs cannot be combined with --lines or --jsonl');
  const fn = commandCallable(name, option(parsed, '--adaptation') || tunedProject(parsed) ? process.cwd() : undefined), definition = fn[Symbol.for('natlang.callable')]!.definition;
  const params = definition.params;
  if (definition.subtype === 'directory-reducer') throw usageError(`${name} is a directory reducer; use natlang apply`);
  if (parsed.options.has('--lines') && params.length !== 1) throw usageError('--lines requires a function with one parameter');
  const inputsPath = option(parsed, '--inputs');
  const stdin = inputsPath ? readFileSync(existingPath(inputsPath, '--inputs'), 'utf8') : await pipedStdin(
    parsed.options.has('--lines') || parsed.options.has('--jsonl'));
  // No input at all is a usage error before any model work, unless the function needs none.
  const declared = params.map(param => ({ name: param.name, type: param.type, required: !param.optional }));
  if (!stdin.trim() && declared.some(param => param.required))
    throw usageError(`${name} needs its input: natlang call ${name} ${argumentUsage(declared)}, or pipe it on stdin`);
  const mode = parsed.options.has('--lines') ? 'line' : parsed.options.has('--jsonl') ? 'jsonl' : 'whole';
  const text = params.length === 1 && params[0]!.type === 'string' && !inputsPath;
  // The result prints before the runtime closes, so the call's ID line follows it.
  await withModelRuntime(parsed, async runtime => process.stdout.write(await runtime.run(() => executeNatlangCall(name, stdin, mode,
    (_name, input) => fn(...commandInputArgs(input, params)), { text, json: parsed.options.has('--json') }))), callRoot(name));
  return 0;
}

/** `natlang call FN ARG...`: the words fill the function's parameters in order, read as `natlang run` reads target
 * arguments (text as written, other types as JSON), so `natlang call answer.nl "some text"` needs no JSON file. */
async function callWithArguments(parsed: Parsed, name: string, raw: string[]): Promise<number> {
  for (const flag of ['--inputs', '--lines', '--jsonl'])
    if (parsed.options.has(flag)) throw usageError(`${flag} cannot be combined with arguments; pass the input either way, not both`);
  const fn = commandCallable(name, option(parsed, '--adaptation') || tunedProject(parsed) ? process.cwd() : undefined);
  const definition = callableMeta(fn)!.definition;
  if ((definition as { subtype?: string }).subtype === 'directory-reducer') throw usageError(`${name} is a directory reducer; use natlang apply`);
  const root = callRoot(name);
  const declared = definition.params.map(param => ({ name: param.name, type: param.type, required: !param.optional }));
  const { values, problems } = readTargetArguments(declared, raw, root, process.cwd());
  if (problems.length) throw usageError(`${problems.map(problem => problem.message).join('\n')}\nusage: natlang call ${name} ${argumentUsage(declared)}`);
  await withModelRuntime(parsed, async runtime => process.stdout.write(`${pipelineValue(
    await runtime.run(() => fn(...definition.params.map(param => values[param.name]))), parsed.options.has('--json'))}\n`), root);
  return 0;
}

/** The coding-agent skills: `skills/` in the installed package, else at the root of this checkout (beside ts-host). */
const SKILLS_ROOT = ['../../skills/', '../../../skills/'].map(path => fileURLToPath(new URL(path, import.meta.url)))
  .find(path => existsSync(join(path, 'natlang-authoring', 'SKILL.md'))) ?? fileURLToPath(new URL('../../../skills/', import.meta.url));
const SKILL_NAMES = ['natlang-authoring', 'natlang-integration'];

/** Whether two folders hold the same files with the same bytes. */
function sameTree(left: string, right: string): boolean {
  const files = (root: string, at = ''): string[] => readdirSync(join(root, at), { withFileTypes: true }).flatMap(entry =>
    entry.isDirectory() ? files(root, join(at, entry.name)) : [join(at, entry.name)]).sort();
  const names = files(left);
  return isDeepStrictEqual(names, files(right)) && names.every(name => readFileSync(join(left, name)).equals(readFileSync(join(right, name))));
}

/** `natlang skills`: where the skills are; `install` copies them into Claude Code's and/or Codex's skill folders. */
function skillsCommand(parsed: Parsed, action: string): number {
  const json = parsed.options.has('--json');
  const missing = SKILL_NAMES.filter(name => !existsSync(join(SKILLS_ROOT, name, 'SKILL.md')));
  if (missing.length) throw new Error(`the natlang skills are not in this installation (${SKILLS_ROOT}); reinstall natlang or use a checkout of the natlang repository`);
  const home = homedir(), homes = { claude: join(home, '.claude', 'skills'), codex: join(home, '.codex', 'skills') };
  if (action === 'list') {
    output(json ? { root: SKILLS_ROOT, skills: SKILL_NAMES } : `${SKILL_NAMES.map(name => `${name}  ${join(SKILLS_ROOT, name)}`).join('\n')}\n\n` +
      'natlang skills install copies them into ~/.claude/skills and/or ~/.codex/skills (--claude, --codex, --dir DIR).', json);
    return 0;
  }
  const explicit = [...(parsed.options.has('--claude') ? [homes.claude] : []), ...(parsed.options.has('--codex') ? [homes.codex] : []),
    ...(option(parsed, '--dir') ? [resolve(option(parsed, '--dir')!)] : [])];
  // Without a choice, every agent that is set up on this machine (its ~/.claude or ~/.codex exists).
  const destinations = explicit.length ? explicit : Object.values(homes).filter(path => existsSync(dirname(path)));
  if (!destinations.length) throw new Error('no ~/.claude or ~/.codex found; choose a destination with --claude, --codex or --dir DIR');
  const installed: string[] = [], replaced: string[] = [];
  for (const destination of destinations) for (const name of SKILL_NAMES) {
    const target = join(destination, name);
    // A skill folder is replaced whole, so files removed upstream do not linger; one that differs from this version is named.
    if (existsSync(target) && !sameTree(join(SKILLS_ROOT, name), target)) replaced.push(target);
    rmSync(target, { recursive: true, force: true });
    cpSync(join(SKILLS_ROOT, name), target, { recursive: true });
    installed.push(target);
  }
  output(json ? { installed, replaced } : `installed\n${installed.map(path => `  ${path}`).join('\n')}` +
    (replaced.length ? `\nreplaced earlier copies that differed (local edits or another version):\n${replaced.map(path => `  ${path}`).join('\n')}` : '') +
    '\nRestart or refresh the agent to load them.', json);
  return 0;
}

async function applyCommand(parsed: Parsed, name: string, directory: string): Promise<number> {
  acceptOptions(parsed, ['--progress', '--trace', '--profile', '--provider', '--model', '--json', '--yes', '--adaptation', '--no-adaptation', '--no-weights', '--escalate']); noTrailingArguments(parsed);
  const fn = commandCallable(name, option(parsed, '--adaptation') || tunedProject(parsed) ? process.cwd() : undefined), definition = fn[Symbol.for('natlang.callable')]!.definition;
  if (definition.subtype !== 'directory-reducer') throw usageError(`${name} is not a directory reducer`);
  if (definition.params.length) throw usageError(`${name} needs parameters; natlang apply accepts a folder-only reducer`);
  const path = resolve(directory), folder = openFolder(path, 'write');
  const value = await withModelRuntime(parsed, runtime => runtime.run(() => folder.apply(fn)), existsSync(resolve(name)) ? projectRoot(resolve(name)) : projectRoot(process.cwd()));
  saveFolder(path, folder);
  process.stdout.write(`${pipelineValue(value)}\n`);
  return 0;
}

async function askCommand(parsed: Parsed, instruction: string): Promise<number> {
  if (parsed.options.has('--adaptation') || parsed.options.has('--no-adaptation')) throw usageError('ad hoc ask functions are not registered persistent adaptation components');
  acceptOptions(parsed, ['--progress', '--trace', '--profile', '--provider', '--model', '--yes', '--lines', '--jsonl', '--filter', '--escalate',
    '--returns', '--files', '--jobs']); noTrailingArguments(parsed);
  const context = applicationContextRecords(process.cwd());
  const folder = openFolder(process.cwd());
  const stdin = option(parsed, '--files') ? '' : await pipedStdin(parsed.options.has('--lines') || parsed.options.has('--jsonl'));
  const jobs = numericOption(parsed, '--jobs', 1);
  const result = await withModelRuntime(parsed, runtime => runtime.run(() => executeNatlangAsk(folder,
    instruction.trim(), stdin, {
      lines: parsed.options.has('--lines'), jsonl: parsed.options.has('--jsonl'),
      filter: parsed.options.has('--filter'), files: option(parsed, '--files'),
      returns: option(parsed, '--returns'), jobs,
    // model-call-kind: runtime-authored-call (natlang ask) -- training/model-call-kinds.json
    }, (instructions, returns, input) => invokeDefinition(resolveFrame(), {
      id: `natlang:ask:${instructions}`, name: 'ask', body: `${instructions}\n`,
      params: [{ name: inferValueType(input) === 'Folder' ? 'project' : 'input', type: inferValueType(input) }], returns,
      types: {}, codebase: context?.records ?? {}, subtype: 'function',
    }, [input]))));
  process.stdout.write(result);
  return 0;
}

function inspectCommand(parsed: Parsed, value: string): Record<string, unknown> {
  acceptOptions(parsed, ['--target', '--store', '--json']); noTrailingArguments(parsed);
  const path = resolve(value);
  if (existsSync(path) && path.endsWith('.nl')) {
    const fn = loadNatlang(path) as unknown as Record<symbol, { definition: { name: string; params: unknown[]; returns: string; codebase: object } }>;
    const definition = fn[Symbol.for('natlang.callable')]!.definition;
    return { kind: 'function', path, function: definition.name, params: definition.params, returns: definition.returns,
      children: Object.keys(definition.codebase) };
  }
  if (existsSync(path)) {
    const manifestPath = statSync(path).isDirectory() ? join(path, 'natlang.json') : path;
    if (!isApplicationManifest(manifestPath)) throw usageError(`${value} has no natlang.package/v2 manifest`);
    const archive = createPackageArchive(JSON.parse(readFileSync(manifestPath, 'utf8')), dirname(manifestPath));
    const targetName = selectTarget(archive.manifest, option(parsed, '--target'));
    return { kind: 'application', path: manifestPath, name: archive.manifest.name, version: archive.manifest.version,
      ...inspectTarget(archive.manifest, targetName, manifestPath) };
  }
  const store = new NatlangPackageStore(option(parsed, '--store'));
  const specifier = value.includes('#') ? value : applicationSpecifier(store, value, option(parsed, '--target'));
  const marker = specifier.lastIndexOf('#');
  const packageSpecifier = specifier.slice(0, marker), targetName = specifier.slice(marker + 1);
  const manifest = store.manifest(packageSpecifier), installed = store.resolve(packageSpecifier);
  return { kind: 'installed-application', root: installed.root, ...inspectTarget(manifest, targetName, packageSpecifier) };
}

function inspectTarget(manifest: ReturnType<NatlangPackageStore['manifest']>, targetName: string, label: string): Record<string, unknown> {
  const target = manifest.targets?.[targetName];
  if (!target) throw usageError(`unknown target ${targetName} in ${label}`);
  let engineError: string | null = null;
  try { checkEngines(manifest.engines); } catch (error) { engineError = error instanceof Error ? error.message : String(error); }
  return { package: label, target: targetName, entry: target.entry, export: target.export ?? 'main',
    // The declared arguments as a signature, and the command line that passes them.
    arguments: target.arguments ? argumentSignature(target.arguments) : '(undeclared: words reach context.args as text)',
    usage: `natlang run ${label.endsWith('natlang.json') ? relative(process.cwd(), dirname(label)) || '.' : label} --target ${targetName}${target.arguments?.length ? ` -- ${argumentUsage(target.arguments)}` : ' [-- ARGS]'}`,
    authority: target.authority ?? [],
    commands: Object.fromEntries((target.commands ?? []).map(command => [command, commandAvailable(command)])),
    engines: manifest.engines ?? {}, engineError };
}

async function authCommand(parsed: Parsed, words: string[]): Promise<number> {
  acceptOptions(parsed, ['--json']); noTrailingArguments(parsed);
  const [action, provider] = words;
  const { piModels } = await import('../model/pi-provider.js');
  const models = piModels();
  if (action === 'status' && !provider) {
    const rows = await Promise.all(models.getProviders().map(async item => {
      const auth = await models.checkAuth(item.id);
      return auth ? { provider: item.id, type: auth.type, source: auth.source ?? null } : null;
    }));
    const configured = rows.filter((row): row is NonNullable<typeof row> => row !== null);
    output(parsed.options.has('--json') ? configured : configured.length ?
      configured.map(row => `${row.provider}: ${row.source ?? row.type}`).join('\n') : 'No hosted model provider credentials found.',
    parsed.options.has('--json'));
    return 0;
  }
  if (provider && !models.getProvider(provider)) throw unknownProvider(provider, models.getProviders().map(item => item.id));
  if (!provider) throw usageError('usage: natlang auth login|logout PROVIDER, or natlang auth status');
  if (action === 'logout') { await models.logout(provider); output(`Logged out of ${provider}.`, false); return 0; }
  if (action !== 'login') throw usageError('usage: natlang auth login|logout PROVIDER, or natlang auth status');
  if (!process.stdin.isTTY || !process.stderr.isTTY) throw usageError(`provider sign-in needs an interactive terminal; run natlang auth login ${provider} in one, or use natlang dashboard (Models)`);
  if (!models.getProvider(provider)?.auth.oauth) {
    const key = await providerKeyName(provider);
    throw usageError(`${provider} has no subscription sign-in; set ${key ?? 'its API key environment variable'} instead`);
  }
  const terminal = createInterface({ input: process.stdin, output: process.stderr });
  try {
    await models.login(provider, 'oauth', {
      async prompt(prompt: AuthPrompt) {
        if (prompt.type === 'select') {
          for (const choice of prompt.options) process.stderr.write(`  ${choice.id}: ${choice.label}${choice.description ? ` — ${choice.description}` : ''}\n`);
        }
        return terminal.question(`${prompt.message}: `, { signal: prompt.signal });
      },
      notify(event) {
        if (event.type === 'auth_url') process.stderr.write(`${event.instructions ?? 'Open this URL'}: ${event.url}\n`);
        else if (event.type === 'device_code') process.stderr.write(`Open ${event.verificationUri} and enter ${event.userCode}\n`);
        else if (event.type === 'info' || event.type === 'progress') process.stderr.write(`${event.message}\n`);
      },
    });
  } finally { terminal.close(); }
  output(`Logged in to ${provider}.`, false);
  return 0;
}

/**
 * `--provider ID` without `--model` runs the provider's recommended model (the newest Luna or Haiku it serves,
 * model/provider-recommendations.ts) unless the selected profile already names a model of that provider.
 */
async function defaultProviderModel(parsed: Parsed): Promise<void> {
  const provider = option(parsed, '--provider');
  if (!provider) return;
  const { piModels } = await import('../model/pi-provider.js');
  const models = piModels();
  if (!models.getProvider(provider)) throw unknownProvider(provider, models.getProviders().map(item => item.id));
  if (parsed.options.has('--model') || process.env.NATLANG_MODEL) return;
  const config = readConfigFile();
  const profile = config.profiles?.[option(parsed, '--profile') ?? process.env.NATLANG_PROFILE ?? config.defaultProfile ?? 'default'];
  if (profile?.provider === provider && profile.model) return;
  const { recommendedProviderModel } = await import('../model/provider-recommendations.js');
  const ids = models.getModels(provider).map(item => item.id);
  const model = recommendedProviderModel(provider, ids).model ?? ids[0];
  if (model) parsed.options.set('--model', model);
}
const unknownProvider = (provider: string, known: string[]) => {
  const near = closestName(provider, known);
  return usageError(`unknown provider ${provider}${near ? ` (did you mean ${near}?)` : ''}; natlang models lists them`);
};

async function modelsCommand(parsed: Parsed, provider?: string): Promise<number> {
  acceptOptions(parsed, ['--json', '--refresh']); noTrailingArguments(parsed);
  const { piModels } = await import('../model/pi-provider.js');
  const models = piModels();
  if (provider && !models.getProvider(provider)) throw unknownProvider(provider, models.getProviders().map(item => item.id));
  if (parsed.options.has('--refresh')) {
    const refreshed = await models.refresh({ force: true, ...(provider ? { providers: [provider] } : {}) });
    if (refreshed.errors.size) throw new Error([...refreshed.errors].map(([id, error]) =>
      `${id}: ${error.message}`).join('\n'));
  }
  if (!provider) {
    const rows = models.getProviders().map(item => ({ provider: item.id, name: item.name }));
    output(parsed.options.has('--json') ? rows : rows.map(row => `${row.provider}\t${row.name}`).join('\n'),
      parsed.options.has('--json'));
    return 0;
  }
  const rows = models.getModels(provider);
  const { recommendedProviderModel } = await import('../model/provider-recommendations.js');
  const recommended = recommendedProviderModel(provider, rows.map(row => row.id)).model;
  output(parsed.options.has('--json') ? rows.map(row => ({ ...row, recommended: row.id === recommended })) :
    rows.map(row => `${row.id}\t${row.name}${row.id === recommended ? '\t(default for --provider)' : ''}`).join('\n'),
    parsed.options.has('--json'));
  return 0;
}

/** A diagnostic as check --json reports it: where, the code, the message, and a further detail or hint when there is one. */
const jsonDiagnostic = (item: NatlangDiagnostic) => ({ file: item.file, line: item.line, column: item.column, code: item.code, message: item.message,
  ...(item.detail ? { detail: item.detail } : {}), ...(item.hint ? { hint: item.hint } : {}), ...(item.severity === 'warning' ? { severity: item.severity } : {}) });

/** Run one natlang command (main.ts parses and dispatches). */
export async function runCommand(command: string, words: string[], parsed: Parsed): Promise<number> {
  await defaultProviderModel(parsed);
  const json = parsed.options.has('--json');
  if (command === 'init') {
    acceptOptions(parsed, []); noTrailingArguments(parsed);
    if (words.length > 1) throw usageError('usage: natlang init [DIRECTORY]');
    const target = resolve(words[0] ?? 'my-program');
    const enclosing = [process.cwd(), dirname(target)].find(directory => existsSync(join(directory, 'natlang.json')));
    if (enclosing) process.stderr.write(`natlang: note: ${relative(process.cwd(), enclosing) || '.'} is already a program (natlang.json); ` +
      'the new program is created inside it as a separate program\n');
    const root = createProgram(words[0] ?? 'my-program');
    output(`Created ${root}\n\nNext:\n  cd ${JSON.stringify(root)}\n  natlang run                      # answer.nl on the sample input\n  natlang call answer.nl "Your text"\n  natlang dev                      # rerun after each edit\n\nEdit answer.nl to change behavior and examples.json to save expectations; natlang test checks them.\nnatlang dashboard --open shows results, traces and model settings. AGENTS.md guides coding agents.`, false);
    return 0;
  }
  if (command === 'dev') {
    acceptOptions(parsed, ['--progress', '--target', '--export', '--profile', '--provider', '--model', '--escalate', '--workspace', '--state', '--traces', '--store', '--plain', '--no-color', '--yes', '--adaptation', '--no-adaptation', '--no-weights']);
    if (words.length > 1) throw usageError('usage: natlang dev [SOURCE] [OPTIONS] [-- ARGS]');
    const args = [words[0] ?? '.', ...[...parsed.options].flatMap(([key, value]) => value === true ? [key] : [key, value]), ...(parsed.rest.length ? ['--', ...parsed.rest] : [])];
    return (await import('./dev.js')).develop(words[0] ?? '.', args);
  }
  if (command === 'test') {
    acceptOptions(parsed, ['--profile', '--provider', '--model', '--yes', '--progress', '--json']); noTrailingArguments(parsed);
    if (words.length > 1) throw usageError('usage: natlang test [DIRECTORY]');
    const root = existingPath(words[0] ?? '.'), examples = readExamples(root);
    if (!examples.length) throw new Error('No saved examples. Add { function, inputs, expected } cases to examples.json, or save a test in the dashboard function form.');
    const checked = buildProject({ project: root, emit: false, runtimeModule: RUNTIME_MODULE });
    if (!checked.ok) {
      if (json) { output({ ok: false, diagnostics: checked.diagnostics }, true); return EXIT.failed; }
      throw new Error(formatDiagnostics(checked.diagnostics));
    }
    // Each example prints its own outcome, so the display leaves root-call failures to it.
    const manifest = readProgramManifest(root), progress = progressFor(parsed, { roots: false });
    const natlang = await natlangRuntime({ ...selectionOptions(parsed), programRoot: root, programModel: manifest?.model,
      ensureRuntime: sessionRuntime(parsed.options.has('--yes')),
      runtime: { onLive: progress.live } });
    let failed = 0;
    try {
      // A model that cannot be used is one failure, before any example runs (and nothing is recorded as a failed call).
      try { await preflightModel(natlang, parsed.options.has('--yes')); }
      catch (error) {
        if (!json || !(error instanceof CliError) || error.exitCode !== EXIT.unavailable) throw error;
        output({ ok: false, error: { kind: 'model-unavailable', message: error.message, next: error.next ?? null } }, true);
        return EXIT.unavailable;
      }
      // Examples run together (the runtime's scheduler fills the model's slots); each result prints when it finishes,
      // so a run that is stopped keeps what it found.
      const started = Date.now();
      let finished = 0;
      type Outcome = { name: string; function: string; ok: boolean; call_id: string | null; seconds: number; expected: unknown; received?: unknown; error?: string };
      // --json prints one report at the end for agents and scripts; text prints each result as it finishes.
      const report = (result: Outcome) => { finished++; if (!result.ok) failed++;
        if (!json) output(`[${finished}/${examples.length}] ${result.ok ? 'PASS' : 'FAIL'} ${result.name}` + (result.error ? `: ${result.error}` :
          ` (${result.seconds.toFixed(1)} s)` + (result.ok ? '' : `\n  expected: ${JSON.stringify(result.expected)}\n  received: ${JSON.stringify(result.received)}`)) +
          (result.ok || !result.call_id || !natlang.runtime.callStore() ? '' : `\n  natlang traces show ${result.call_id}`), false);
        return result; };
      const results = await Promise.all(examples.map(async (example, index) => report(await (async (): Promise<Outcome> => {
        const name = example.name ?? `${example.function} #${index + 1}`, began = Date.now();
        let callId: string | null = null;
        const base = () => ({ name, function: example.function, call_id: callId, expected: example.expected });
        try {
          const fn = loadNatlang(resolve(root, example.function), root);
          const params = callableMeta(fn)!.definition.params;
          const received = await natlang.runtime.run(() => fn(...params.map(param => example.inputs[param.name])), { onLive: event => {
            if (event.type === 'call_start' && event.parentCallId === null) callId ??= event.callId; } });
          return { ...base(), ok: isDeepStrictEqual(received, example.expected), seconds: (Date.now() - began) / 1000, received };
        } catch (error) { return { ...base(), ok: false, seconds: (Date.now() - began) / 1000, error: (error as Error).message }; }
      })())));
      if (json) output({ ok: failed === 0, passed: examples.length - failed, failed, seconds: (Date.now() - started) / 1000, results }, true);
      else output(`${examples.length - failed}/${examples.length} passed in ${((Date.now() - started) / 1000).toFixed(1)} s.`, false);
    } finally { progress.close(); await natlang.close(); }
    return failed ? EXIT.failed : 0;
  }
  if (command === 'check' || command === 'build') {
    acceptOptions(parsed, command === 'build' ? ['--out', '--target', '--json', '--manifest', '--warm', '--profile', '--provider', '--model', '--yes', '--progress'] :
      ['--json', '--manifest', '--fusion', '--fusion-planner', '--profile', '--provider', '--model', '--yes']);
    if (words.length > 1) throw usageError(`usage: natlang ${command} [PROJECT]`);
    if (words[0]) existingPath(words[0]);
    noTrailingArguments(parsed);
    const target = option(parsed, '--target');
    if (target && target !== 'node' && target !== 'browser') throw usageError('--target must be node or browser');
    const result = buildProject({ project: words[0] ?? '.', emit: command === 'build', outDir: option(parsed, '--out'),
      runtimeTypes: { specifiers: RUNTIME_MODULE.specifiers, types: RUNTIME_MODULE.types },
      ...(target ? { target: target as 'node' | 'browser' } : {}) });
    // Fused pipelines (plans/FUSED_PIPELINES.md): the candidate hand-offs and the selected planner's plan.
    const fusion = command === 'check' && result.ok ? await fusionForCheck(parsed, words[0] ?? '.') : undefined;
    // The compiled program's lock: its model, runtime, tuning and functions, and with --warm its ahead-of-time run.
    const lock = command === 'build' && result.ok && target !== 'browser' ? await buildLock(parsed, projectRoot(resolve(words[0] ?? '.'))) : undefined;
    // check --json is { ok, diagnostics, functions }; build adds where it wrote and the lock; --manifest adds the build manifest.
    if (json) output({ ok: result.ok, diagnostics: result.diagnostics.map(jsonDiagnostic),
      functions: [...result.manifest.named.map(({ source, signature }) => ({ source, signature })),
        ...result.manifest.inline.map(({ source, line, signature }) => ({ source: `${source}:${line}`, signature }))],
      ...(command === 'build' ? { outDir: result.outDir } : {}), ...(parsed.options.has('--manifest') ? { manifest: result.manifest } : {}),
      ...(result.refinedSlots ? { refinedSlots: result.refinedSlots } : {}), ...(fusion ? { fusion } : {}), ...(lock ? { lock } : {}) }, true);
    else output(result.ok ? `${command === 'build' ? `built ${Object.keys(result.outputs).length} files into ${result.outDir}` : 'ok'}` +
      (lock ? `\n${describeLock(lock.lock, lock.path)}` : '') +
      (result.refinedSlots?.length ? `\nrefined slots (checked at run time):\n${result.refinedSlots.map(slot =>
        `  ${slot.function} ${slot.slot}: ${JSON.stringify(slot.predicate)}`).join('\n')}` : '') +
      (fusion && fusion.counts.edges ? `\n${formatFusionReport(fusion, parsed.options.has('--fusion'))}` : '') +
      (result.diagnostics.length ? `\n${formatDiagnostics(result.diagnostics)}` : '') :
      formatDiagnostics(result.diagnostics), false);
    return result.ok ? 0 : 1;
  }
  if (command === 'run') return runTarget(parsed, words[0]);
  if (command === 'call') { if (!words.length) throw usageError('usage: natlang call NAME|FILE.nl [ARGUMENT...]'); return callCommand(parsed, words[0]!, words.slice(1)); }
  if (command === 'apply') { if (words.length !== 2) throw usageError('usage: natlang apply REDUCER DIR'); return applyCommand(parsed, words[0]!, words[1]!); }
  if (command === 'ask') { if (!words.length) throw usageError('usage: natlang ask INSTRUCTION...'); return askCommand(parsed, words.join(' ')); }
  if (command === 'auth') return authCommand(parsed, words);
  if (command === 'models') { if (words.length > 1) throw usageError('usage: natlang models [PROVIDER]'); return modelsCommand(parsed, words[0]); }
  if (command === 'serve') { acceptOptions(parsed, ['--profile']); noTrailingArguments(parsed); return serveCommand(parsed); }
  if (command === 'setup' && option(parsed, '--program')) {
    acceptOptions(parsed, ['--program', '--profile', '--json']); noTrailingArguments(parsed);
    return setupProgram(parsed, option(parsed, '--program')!);
  }
  if (command === 'setup' && option(parsed, '--native')) {
    acceptOptions(parsed, ['--native', '--profile', '--default', '--device', '--port', '--gguf', '--escalate', '--json']); noTrailingArguments(parsed);
    return setupNative(parsed, option(parsed, '--native')!);
  }
  if (command === 'setup' || command === 'runtime') {
    const action = command === 'setup' ? 'ensure' : words[0] ?? 'status';
    acceptOptions(parsed, command === 'setup' ? ['--yes', '--json', '--profile', '--provider', '--model'] : action === 'install' ? ['--yes', '--json'] :
      action === 'build' ? ['--cuda', '--no-cuda', '--jobs', '--json'] : ['--json']);
    noTrailingArguments(parsed);
    if (action === 'build') {
      await buildForkLlamaRuntime({ cuda: parsed.options.has('--cuda') ? true : parsed.options.has('--no-cuda') ? false : undefined,
        jobs: parsed.options.has('--jobs') ? numericOption(parsed, '--jobs', 1) : undefined });
      outputRuntime(discoverLlamaRuntime(), json); return 0;
    }
    if (command === 'setup') {
      const { choice } = modelSelection(parsed);
      if (choice.kind !== 'managed-local') {
        if (choice.kind === 'pi-provider') {
          const session = modelSession(choice, parsed.options.has('--yes'));
          try { await session.prepare(); } finally { await session.close(); }
        }
        const report = { ok: true, modelSource: choice.kind,
          endpoint: choice.kind === 'external' ? choice.endpoint : null,
          provider: choice.kind === 'pi-provider' ? choice.provider : null,
          model: choice.model, runtimeRequired: false };
        output(json ? report : `model profile ready\n${report.provider ? `provider: ${report.provider}` : `endpoint: ${report.endpoint}`}\nmodel: ${report.model}`, json);
        return 0;
      }
    }
    if (action === 'status') { const discovery = discoverLlamaRuntime(); outputRuntime(discovery, json); return discovery.selected ? 0 : 1; }
    if (action === 'ensure' || action === 'install') {
      // setup also fetches and verifies the profile's stock model, so the first run does not wait on a download.
      const { choice } = action === 'ensure' ? modelSelection(parsed) : { choice: undefined };
      const managed = choice?.kind === 'managed-local' ? choice : undefined;
      if (!await ensureLocalModel(managed, parsed.options.has('--yes'), action === 'install')) {
        output(json ? { ...runtimeReport(), declined: true } : 'Setup cancelled; nothing was downloaded. Run natlang setup when ready, ' +
          'or configure a hosted model with natlang dashboard (Models).', json);
        return EXIT.unavailable;
      }
      const release = managed && !process.env.NATLANG_MODEL_PATH && !managed.local?.modelPath ? managed.release : undefined;
      const modelPath = release ? await ensureStockModel(release) : undefined;
      if (managed) await prewarmStarter(managed);
      if (json) output({ ...runtimeReport(), ...(release ? { model: release.id, modelPath } : {}) }, true);
      else { outputRuntime(discoverLlamaRuntime(), false); if (release) output(`model: ${release.label} (${modelPath})`, false); }
      return 0;
    }
    throw usageError('usage: natlang runtime status|install|build');
  }
  if (command === 'skills') {
    acceptOptions(parsed, ['--claude', '--codex', '--dir', '--json']); noTrailingArguments(parsed);
    const action = words[0] ?? 'list';
    if (words.length > 1 || (action !== 'list' && action !== 'install')) throw usageError('usage: natlang skills [list|install] [--claude] [--codex] [--dir DIR]');
    return skillsCommand(parsed, action);
  }
  if (command === 'apps') {
    acceptOptions(parsed, ['--json']); noTrailingArguments(parsed);
    const root = words[0] ?? '.', rows = discoverLocalApplications(root).map(item => ({ ...item,
      path: relative(process.cwd(), item.path) || basename(item.path) }));
    output(json ? rows : rows.length ? rows.map(item =>
      `${item.path}\n  ${item.name}@${item.version}  targets: ${item.targets.join(', ')}${item.description ? `\n  ${item.description}` : ''}`).join('\n') :
      `No natlang applications found under ${resolve(root)}. Applications contain a natlang.json manifest.`, json);
    return 0;
  }
  if (command === 'packages') {
    acceptOptions(parsed, ['--store', '--json']); noTrailingArguments(parsed);
    const store = new NatlangPackageStore(option(parsed, '--store')), rows = store.list();
    output(json ? rows : rows.length ? rows.map(row => `${row.name}@${row.version} ${row.digest}`).join('\n') :
      `No distribution packages are installed in ${store.root}.`, json);
    return 0;
  }
  if (command === 'inspect') {
    if (words.length !== 1) throw usageError('usage: natlang inspect SOURCE [--json]');
    output(inspectCommand(parsed, words[0]!), json); return 0;
  }
  if (command === 'package') {
    const [action, argument] = words;
    if (action === 'pack' && argument) {
      acceptOptions(parsed, ['--out', '--json']); noTrailingArguments(parsed);
      const path = resolve(argument), manifestPath = statSync(path).isDirectory() ? join(path, 'natlang.json') : path;
      const archive = createPackageArchive(JSON.parse(readFileSync(manifestPath, 'utf8')), dirname(manifestPath));
      const out = resolve(option(parsed, '--out') ?? `${archive.manifest.name.replace('/', '-')}-${archive.manifest.version}.nlpkg`);
      writePackageArchive(out, archive);
      output({ archive: out, digest: archive.digest, name: archive.manifest.name, version: archive.manifest.version }, json); return 0;
    }
    if (action === 'verify' && argument) {
      acceptOptions(parsed, ['--json']); noTrailingArguments(parsed);
      const archive = readPackageArchive(resolve(argument));
      output({ valid: true, digest: archive.digest, name: archive.manifest.name, version: archive.manifest.version, files: archive.files.length }, json);
      return 0;
    }
    if (action === 'install' && argument) {
      acceptOptions(parsed, ['--store', '--json']); noTrailingArguments(parsed);
      const installed = new NatlangPackageStore(option(parsed, '--store')).installMany(words.slice(1).map(path => resolve(path)));
      output(installed.length === 1 ? installed[0] : installed, json); return 0;
    }
    throw usageError('usage: natlang package pack|verify|install ...');
  }
  if (command === 'dashboard') {
    acceptOptions(parsed, ['--port', '--host', '--root', '--open']); noTrailingArguments(parsed);
    const { startDashboard } = await import('../dashboard/server.js');
    const port = numericOption(parsed, '--port', 1);
    const dashboard = await startDashboard({ version: NATLANG_CLI_VERSION, root: option(parsed, '--root') ?? '.',
      ...(port ? { port } : {}), ...(option(parsed, '--host') ? { host: option(parsed, '--host')! } : {}) });
    if (parsed.options.has('--open')) {
      const opener = process.platform === 'darwin' ? 'open' : process.platform === 'win32' ? 'explorer.exe' : 'xdg-open';
      try { spawnSync(opener, [dashboard.url], { stdio: 'ignore', timeout: 5000 }); } catch { /* the URL is printed */ }
    }
    await new Promise<void>(resolveStop => { process.once('SIGINT', () => resolveStop()); process.once('SIGTERM', () => resolveStop()); });
    await dashboard.close();
    return 0;
  }
  if (command === 'doctor') {
    acceptOptions(parsed, ['--store', '--profile', '--provider', '--model', '--program', '--json']); noTrailingArguments(parsed);
    const { report, okay } = await doctorReport(parsed, new NatlangPackageStore(option(parsed, '--store')));
    output(json ? report : doctorText(report), json); return okay ? 0 : 1;
  }
  throw usageError(`unknown command ${command}; run natlang help`);
  throw usageError(`unknown command ${command}; run natlang help`);
}
