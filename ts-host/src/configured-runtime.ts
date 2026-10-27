/**
 * A natlang runtime from the machine's model configuration (docs/NATIVE_MODELS.md), the one implementation behind
 * `natlang run|call|ask` and TypeScript hosts:
 *
 *   const natlang = await natlangRuntime({ programRoot: import.meta.dirname });
 *   const lane = await natlang.runtime.run(() => route(ticket));
 *   await natlang.close();
 *
 * It selects the profile (config.json; `profile`, NATLANG_PROFILE or the default), runs a tuned program on the exact
 * checkpoint its active release was tuned on and serves that release, starts and stops natlang-owned model servers,
 * wires a profile's `escalate` teacher through the tiered engine, and reports each of these on `log` in one line.
 */
import { existsSync, readFileSync } from 'node:fs';
import { basename, join, resolve } from 'node:path';
import type { ProgramDescriptor } from './adaptation/types.js';
import { currentRelease } from './adaptation/weights/release.js';
import { bindAdaptation } from './adaptation/compatibility.js';
import type { AdaptationArtifact } from './adaptation/types.js';
import { TierEngine } from './calls/tiers.js';
import type { DecisionRequest, ModelTurnOptions, ModelTurnRequest } from './contracts.js';
import { buildProject } from './compiler/node-project.js';
import { executorIdentityForChoice, loadModelConfiguration, resolveModelChoice, type LoadedModelConfiguration,
  type ModelProfile, type ModelSelectionOverrides, type ResolvedModelChoice } from './model/config.js';
import type { LlamaRuntimeDiscovery, LlamaServerInspection } from './model/llama-runtime.js';
import { createResolvedModelSession, type ManagedModelSession } from './model/local-server.js';
import { defaultNativePython, ensureNativeServer, findNativeCheckpoint, newerNativeCheckpoint, pinNativeCheckpoint,
  type NativeModelSettings } from './model/native-managed.js';
import { defaultNatlangConfigDirectory } from './package/store.js';
import { createNatlangRuntime, type InvocationTrace, type ModelDriver, type NatlangRuntime, type NatlangRuntimeOptions,
  type TraceSink } from './runtime/runtime.js';
import { tunedServing, type TunedServing } from './tuned-serving.js';

export type NatlangRuntimeConfiguration = {
  /** Model selection: a profile name, or a Pi provider / model override (as `--profile`, `--provider`, `--model`). */
  profile?: string; provider?: string; model?: string;
  /** A tuned program's root: its active release is served, on the checkpoint it was tuned on. */
  programRoot?: string;
  /** The program's descriptor, when the caller has built it; otherwise it is built when a release or adaptation needs it. */
  program?: ProgramDescriptor;
  /** false runs the selected model untuned (`--no-weights`). */
  weights?: boolean;
  /** A profile that answers calls the model fails, or false; default: the profile's `escalate`. */
  escalate?: string | false;
  /** An adaptation artifact to bind to the program and executor. */
  adaptation?: AdaptationArtifact;
  /** Installs a managed llama.cpp runtime when none is compatible (the CLI asks first); default: none is installed. */
  ensureRuntime?: (discovery: LlamaRuntimeDiscovery) => Promise<LlamaServerInspection | null>;
  environment?: NodeJS.ProcessEnv;
  /** One-line notices (tuned binding, newer checkpoints, escalations, server starts). Default: stderr. */
  log?: NodeJS.WritableStream;
  /** Further runtime options (services, trace, programRoot, refinements, ...). */
  runtime?: Omit<NatlangRuntimeOptions, 'model' | 'models' | 'tiers' | 'weights' | 'neuralese' | 'seed' | 'executorIdentity'
    | 'program' | 'adaptation'>;
};

export type ConfiguredNatlangRuntime = {
  runtime: NatlangRuntime;
  /** The model driver of the selected profile (for hosts that also call the model directly). */
  driver: ModelDriver;
  selection: LoadedModelConfiguration;
  executorIdentity: ReturnType<typeof executorIdentityForChoice>;
  program?: ProgramDescriptor;
  adaptation: ReturnType<typeof bindAdaptation> | null;
  tuned: TunedServing;
  /** Start the model server now rather than on the first call. */
  prepare(): Promise<void>;
  /** Close the runtime and every model session and server it started. */
  close(): Promise<void>;
};

type ConfigFile = { profiles?: Record<string, ModelProfile> } & Record<string, unknown>;
const readConfigFile = (environment: NodeJS.ProcessEnv): ConfigFile => {
  const path = join(defaultNatlangConfigDirectory(environment), 'config.json');
  return (existsSync(path) ? JSON.parse(readFileSync(path, 'utf8')) : {}) as ConfigFile;
};

/** The configured profile, with a notice when its native checkpoint source has a newer checkpoint. */
export function selectModel(options: Pick<NatlangRuntimeConfiguration, 'profile' | 'provider' | 'model' | 'environment' | 'log'> = {}):
  LoadedModelConfiguration {
  const environment = options.environment ?? process.env, log = options.log ?? process.stderr;
  const overrides: ModelSelectionOverrides = { provider: options.provider, model: options.model };
  const loaded = loadModelConfiguration(options.profile, overrides, environment);
  const native = loaded.choice.kind === 'external' ? loaded.choice.managedNative : undefined;
  const newer = native && newerNativeCheckpoint(native, loaded.nativeSources);
  if (newer) log.write(`natlang: ${native!.source} has a newer checkpoint${newer.step !== undefined ? ` (step ${newer.step}` +
    `${native!.step !== undefined ? `; profile ${loaded.name} pins ${native!.step}` : ''})` : ''}; run natlang setup --native ${native!.source}` +
    `${loaded.name === native!.source ? '' : ` --profile ${loaded.name}`} to use it\n`);
  return loaded;
}

/**
 * A tuned program runs on the exact checkpoint its active release was tuned on (the release's `base`): the adapters and
 * soft states are only valid there. A native profile on another checkpoint, or the default profile, is rebound to that
 * base, pinned and served natively with the launch settings of the selected (or first configured) native profile. An
 * explicitly selected non-native backend is kept and runs untuned with the usual notice.
 */
export async function bindTunedModel(selected: LoadedModelConfiguration, programRoot: string | undefined,
  options: { explicit?: boolean; environment?: NodeJS.ProcessEnv; log?: NodeJS.WritableStream } = {}): Promise<LoadedModelConfiguration> {
  const environment = options.environment ?? process.env, log = options.log ?? process.stderr;
  const release = programRoot ? currentRelease(programRoot) : undefined;
  if (!release) return selected;
  const sha256 = release.manifest.base.checkpoint_sha256, choice = selected.choice;
  const native = choice.kind === 'external' ? choice.managedNative : undefined;
  if (native?.sha256 === sha256) return selected;
  const explicit = options.explicit || environment.NATLANG_SERVER || environment.NATLANG_PROVIDER || environment.NATLANG_PROFILE;
  // An unmanaged native server checks its own identity in weight binding; an explicit other backend is the user's call.
  if (!native && (explicit || (choice.kind === 'external' && choice.neuraleseServer))) return selected;
  const program = basename(programRoot!);
  const found = findNativeCheckpoint(sha256, { programRoot, sources: selected.nativeSources }, environment);
  if (!found) {
    log.write(`natlang: ${program} was tuned on checkpoint ${sha256.slice(0, 12)}, which is not available here; ` +
      `natlang setup --native CHECKPOINT pins it. Running ${selected.name} untuned.\n`);
    return selected;
  }
  const launch = native ?? Object.values(readConfigFile(environment).profiles ?? {}).find(profile => profile.native)?.native;
  const { checkpoint: _c, sha256: _s, sourceSha256: _o, source: _r, step: _t, port: _p, actorView: _v, ...settings } = launch ?? {} as NativeModelSettings;
  const interpreter = { ...defaultNativePython(environment), ...(settings.python ? { python: settings.python, pythonPath: settings.pythonPath } : {}) };
  const pin = await pinNativeCheckpoint(found.pinned ? { path: found.pinned, sha256, ...(found.step !== undefined ? { step: found.step } : {}) } : found.pointer!,
    interpreter, environment, log);
  const step = pin.step ?? found.step;
  const profile: ModelProfile = { ...(native ? selected.profile : {}), native: { ...defaultNativePython(environment), ...settings,
    checkpoint: pin.checkpoint, sha256, sourceSha256: sha256, ...(step !== undefined ? { step } : {}), ...(pin.actorView ? { actorView: pin.actorView } : {}) },
    ...(selected.profile.escalate ? { escalate: selected.profile.escalate } : {}) };
  log.write(`natlang: ${program} runs its tuned release ${release.manifest.digest.slice(0, 12)} on the checkpoint it was tuned on ` +
    `(${sha256.slice(0, 12)}${step !== undefined ? `, step ${step}` : ''}) instead of ${selected.name}; --no-weights runs ${selected.name} untuned\n`);
  return { ...selected, name: `${program}-tuned`, profile, choice: resolveModelChoice(profile) };
}

/** A model session's turns, decisions and context window as a runtime model driver. */
export function sessionDriver(session: ManagedModelSession): ModelDriver {
  return Object.assign((request: ModelTurnRequest, signal?: AbortSignal, options?: ModelTurnOptions) => session.turn(request, signal, options),
    { decide: (request: DecisionRequest, signal?: AbortSignal) => session.decide(request, signal), contextWindow: () => session.contextWindow(),
      ...(session.neuralese ? { neuralese: true as const, neuraleseRuntime: () => session.neuraleseRuntime!() } : {}) });
}

/** The runtime's `model` (and seed, native-state reader) for a resolved choice and its driver. */
export async function runtimeModel(choice: ResolvedModelChoice, driver: ModelDriver): Promise<Pick<NatlangRuntimeOptions, 'model' | 'seed' | 'neuralese'>> {
  const { seed, ...settings } = choice.runtime ?? {};
  const neuralese = await driver.neuraleseRuntime?.();
  return { ...(neuralese ? { neuralese } : {}), model: { ...settings, promptRendering: choice.promptRendering ?? 'full', driver }, ...(seed ? { seed } : {}) };
}

/**
 * Escalation (a profile's `escalate`): every function runs on the selected model as the student tier, and a call it
 * fails (an error, or a budget or turn limit) runs again on the teacher profile (calls/tiers.ts tiers 1 and 0). Each
 * escalation is announced in one line and recorded with the call.
 */
async function escalation(name: string | undefined, selected: LoadedModelConfiguration, student: NatlangRuntimeOptions['model'],
  configuration: NatlangRuntimeConfiguration, log: NodeJS.WritableStream) {
  if (!name || !student) return undefined;
  if (name === selected.name) throw new Error(`profile ${name} cannot escalate to itself`);
  const environment = configuration.environment ?? process.env;
  const teacher = loadModelConfiguration(name, {}, environment);
  const session = createResolvedModelSession(teacher.choice, environment, log,
    configuration.ensureRuntime ? { ensureRuntime: configuration.ensureRuntime } : {});
  const { model } = await runtimeModel(teacher.choice, sessionDriver(session));
  const notice: TraceSink = (trace: InvocationTrace) => {
    const deopt = trace.events.find(event => event.kind === 'tier_deopt' && event.tier === 'tier1' && event.reason_kind !== 'guard');
    if (!deopt) return;
    const served = trace.events.some(event => event.kind === 'tier_served' && event.tier === 'tier0');
    log.write(`natlang: ${String(deopt.function)} failed on ${selected.name} (${String(deopt.reason ?? deopt.reason_kind).split('\n')[0]!.slice(0, 160)}); ` +
      `${served ? `answered by ${name}` : `${name} failed too`}. Record the right answer with natlang traces good|fix\n`);
  };
  return { options: { models: { student, teacher: model! },
    tiers: new TierEngine({ functions: { '*': { tier1: { model: 'student' }, tier0: { model: 'teacher' } } } }) },
    notice, close: () => session.close() };
}

const traceSinks = (...sinks: (TraceSink | undefined)[]): Pick<NatlangRuntimeOptions, 'trace'> => {
  const present = sinks.filter((sink): sink is TraceSink => !!sink);
  return present.length === 1 ? { trace: present[0]! } : present.length ? { trace: trace => { for (const sink of present) sink(trace); } } : {};
};

/** A natlang runtime configured as `natlang run` configures it; see the module comment. */
export async function natlangRuntime(configuration: NatlangRuntimeConfiguration = {}): Promise<ConfiguredNatlangRuntime> {
  const environment = configuration.environment ?? process.env, log = configuration.log ?? process.stderr;
  const root = configuration.programRoot ? resolve(configuration.programRoot) : undefined;
  const weights = configuration.weights !== false;
  const explicit = !!(configuration.profile || configuration.provider);
  const selection = await bindTunedModel(selectModel({ ...configuration, environment, log }), weights ? root : undefined,
    { explicit, environment, log });
  const { choice } = selection;
  const executorIdentity = executorIdentityForChoice(choice);
  const release = weights && root ? currentRelease(root) : undefined;
  let program = configuration.program;
  if (!program && root && (release || configuration.adaptation)) {
    const inventory = buildProject({ project: root, programId: configuration.adaptation?.program.id ?? release?.manifest.program.id,
      emit: false, write: false });
    if (!inventory.ok) throw new Error(inventory.diagnostics.map(item => item.message).join('\n'));
    program = inventory.manifest.adaptation;
  }
  // Tuned artifacts are uploaded to a native server before the first turn, so it must be serving by then.
  if (release && choice.kind === 'external' && choice.managedNative) await ensureNativeServer(choice.managedNative, environment, log);
  const tuned = release ? await tunedServing(root!, program, choice, environment) : {};
  if (tuned.notice) log.write(tuned.notice + '\n');
  const session = createResolvedModelSession(choice, environment, log, { ...(configuration.ensureRuntime ? { ensureRuntime: configuration.ensureRuntime } : {}),
    ...(tuned.loraAdapters ? { loraAdapters: tuned.loraAdapters } : {}), ...(tuned.nativeStates ? { nativeStates: tuned.nativeStates } : {}) });
  const closers: (() => Promise<void>)[] = [() => session.close()];
  try {
    if (tuned.vllmStacks) session.vllmStacks?.(tuned.vllmStacks);
    const driver = sessionDriver(session);
    const student = await runtimeModel(choice, driver);
    const escalateTo = configuration.escalate === false ? undefined : configuration.escalate ?? selection.profile.escalate;
    const escalating = await escalation(escalateTo === 'none' ? undefined : escalateTo, selection, student.model, configuration, log);
    if (escalating) closers.push(escalating.close);
    const adaptation = configuration.adaptation && program ? bindAdaptation(configuration.adaptation, program, executorIdentity) : null;
    const runtime = createNatlangRuntime({ ...configuration.runtime, ...student, ...(program ? { program } : {}), adaptation, executorIdentity,
      ...(root ? { programRoot: configuration.runtime?.programRoot ?? root } : {}),
      ...traceSinks(configuration.runtime?.trace, escalating?.notice), ...escalating?.options,
      ...(tuned.weights ? { weights: tuned.weights } : {}) });
    let closed = false;
    return { runtime, driver, selection, executorIdentity, ...(program ? { program } : {}), adaptation, tuned,
      prepare: async () => { await session.prepare(); },
      close: async () => {
        if (closed) return;
        closed = true;
        runtime.close();
        for (const close of closers) await close();
      } };
  } catch (error) {
    for (const close of closers) await close().catch(() => undefined);
    throw error;
  }
}
