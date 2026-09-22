#!/usr/bin/env node
import { existsSync, lstatSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { basename, dirname, join, relative, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { createInterface } from 'node:readline/promises';
import { createPackageArchive, NatlangPackageStore, readPackageArchive,
  writePackageArchive, defaultNatlangConfigDirectory, defaultNatlangStateDirectory } from '../package/index.js';
import { compareVersions, satisfiesVersion } from '../package/store.js';
import type { PackageTargetContext, PackageTargetFactory } from '../package/target.js';
import { createManagedModelSession, DEFAULT_LOCAL_MODEL, describeLlamaRuntime, discoverLlamaRuntime,
  installManagedLlamaRuntime, LLAMA_RUNTIME_RELEASE, localModelPrerequisites,
  type LlamaRuntimeDiscovery, type LlamaServerInspection } from '../model/index.js';
import { NativeNatlangHost } from '../native/host.js';
import { loadAnonymousInstruction, loadFunctionFile } from '../native/source.js';
import { FILE_TREE_LEAF_TYPE, NodeFileTree } from '../native/node-files.js';
import { formatType } from '../native/types.js';
import { TypeScriptEnvironment } from '../environment.js';
import { TerminalNatlangApplication } from '../terminal/application.js';
import { TerminalSessionStore } from '../terminal/session.js';
import { TerminalEventQueue } from '../terminal/events.js';
import { runTerminalShell } from '../terminal/shell.js';
import { renderTerminalView } from '../terminal/view.js';

export const NATLANG_CLI_VERSION = '0.1.0';

type Parsed = { words: string[]; options: Map<string, string | true>; rest: string[] };
function parseArgs(args: string[]): Parsed {
  const words: string[] = [], options = new Map<string, string | true>(), rest: string[] = [];
  const boolean = new Set(['--json', '--plain', '--no-color', '--help', '-h', '--version', '--yes']);
  let separated = false;
  for (let index = 0; index < args.length; index++) {
    const value = args[index]!;
    if (separated) { rest.push(value); continue; }
    if (value === '--') { separated = true; continue; }
    if (boolean.has(value)) {
      if (options.has(value)) throw new Error(`duplicate option ${value}`);
      options.set(value, true); continue;
    }
    if (value.startsWith('--')) {
      const equal = value.indexOf('=');
      if (equal >= 0) {
        const name = value.slice(0, equal);
        if (boolean.has(name)) throw new Error(`${name} does not take a value`);
        if (options.has(name)) throw new Error(`duplicate option ${name}`);
        options.set(name, value.slice(equal + 1));
      }
      else {
        if (options.has(value)) throw new Error(`duplicate option ${value}`);
        const next = args[++index];
        if (next === undefined) throw new Error(`${value} needs a value`);
        options.set(value, next);
      }
    } else words.push(value);
  }
  return { words, options, rest };
}
const option = (parsed: Parsed, name: string): string | undefined => {
  const value = parsed.options.get(name); return typeof value === 'string' ? value : undefined;
};
function numericOption(parsed: Parsed, name: string, minimum: number): number | undefined {
  const raw = option(parsed, name);
  if (raw === undefined) return;
  const value = Number(raw);
  if (!Number.isSafeInteger(value) || value < minimum) throw new Error(`${name} must be an integer >= ${minimum}`);
  return value;
}
function acceptOptions(parsed: Parsed, allowed: readonly string[]): void {
  const accepted = new Set([...allowed, '--help', '-h', '--version']);
  const unknown = [...parsed.options.keys()].filter(name => !accepted.has(name));
  if (unknown.length) throw new Error(`option ${unknown[0]} is not valid here`);
}
function noTrailingArguments(parsed: Parsed): void {
  if (parsed.rest.length) throw new Error('arguments after -- are only valid for applications');
}

function help(): string { return `natlang ${NATLANG_CLI_VERSION}

Usage:
  natlang SOURCE [OPTIONS]       Run a program file, application directory,
                                 manifest, or installed application.
  natlang --apps [DIRECTORY]     Find runnable source applications.
  natlang --inspect SOURCE       Explain how a source resolves without running it.
  natlang --packages             List installed distribution packages.
  natlang --package COMMAND      Pack, verify, or install a distribution archive.
  natlang --setup                Prepare the local model runtime.
  natlang --runtime COMMAND      Inspect or install the model runtime.
  natlang --doctor               Check this natlang installation.

Examples:
  natlang examples/triage/main.nl --inputs inputs.json
  natlang codebases/semantic_terminal
  natlang summarize the available codebase functions
  natlang --apps codebases
  natlang --inspect codebases/semantic_terminal

Run natlang --help COMMAND for focused usage and options.

Semantic execution lazily starts and owns its configured local model server.
Run natlang --setup once to inspect or prepare it.`; }

function topicHelp(topic: string): string {
  if (topic === 'source') return `Run source or an installed application:
  natlang SOURCE [OPTIONS] [-- APPLICATION_ARGS...]

SOURCE may be a .nl, .ts, .json, or .yaml program, an application directory
containing natlang.json, a manifest file, NAME, or NAME@VERSION#TARGET.
Two or more words, or one quoted multiword argument, which do not form an
existing path become an anonymous Lambda<{ files: Dict<ProjectFile> }, Text>
instruction over the current directory's top-level functions and lazy file tree.

Program options:
  --inputs FILE       JSON object containing function inputs.
  --trace FILE        Write the execution trace as JSONL.
  --seed NUMBER       Root seed for deterministic model turns.
  --timeout MS        Explicit execution timeout.
  --profile NAME      Select a model profile.
  --json              Print the complete run result.

Application options:
  --target NAME       Select one target when the manifest has several.
  --root DIRECTORY    Override source-root inference for a local manifest.
  --workspace DIR     Set the application's working directory.
  --state DIRECTORY   Override durable application state.
  --traces DIRECTORY  Override application trace storage.
  --profile NAME      Select a model profile.
  --plain             Disable interactive terminal formatting.
  --no-color          Disable terminal color.
  --yes               Permit a required managed runtime download.`;
  if (topic === 'apps') return `Discover source applications:
  natlang --apps [DIRECTORY] [--json]

This recursively finds natlang.json manifests. It does not read the installed
package store and does not require applications to be packaged first.`;
  if (topic === 'inspect') return `Resolve source without running it:
  natlang --inspect SOURCE [--target NAME] [--root DIR] [--store DIR] [--json]

Programs report their function signature and implementation. Applications
report their selected target, requested host authority, commands, and engines.`;
  if (topic === 'packages') return `List installed distribution packages:
  natlang --packages [--store DIR] [--json]

This reads the content addressed package store. Use natlang --apps to find local
source applications.`;
  if (topic === 'package') return `Distribution packages are optional deployment artifacts:
  natlang --package pack MANIFEST_OR_DIRECTORY [--root DIR] [--out FILE]
  natlang --package verify ARCHIVE [--json]
  natlang --package install ARCHIVE... [--store DIR] [--json]
  natlang --packages [--store DIR] [--json]

Source programs and applications do not need to be packaged before running.`;
  if (topic === 'runtime') return `Model runtime commands:
  natlang --setup [--yes] [--json]
  natlang --runtime status [--json]
  natlang --runtime install [--yes] [--json]

Setup reuses a compatible explicit, managed, or PATH llama-server. If none is
compatible, interactive use asks before installing the verified managed build.
Profiles live in ~/.config/natlang/config.json. Environment overrides are
NATLANG_SERVER, NATLANG_MODEL, NATLANG_MODEL_PATH, NATLANG_LLAMA_SERVER,
NATLANG_RUNTIME_HOME, and NATLANG_API_KEY.`;
  if (topic === 'doctor') return `Check the natlang installation and model configuration:
  natlang --doctor [--profile NAME] [--store DIR] [--json]

Use natlang --inspect SOURCE for source and application checks.`;
  throw new Error(`unknown help topic ${topic}; choose SOURCE, apps, inspect, packages, package, runtime, or doctor`);
}

function output(value: unknown, json: boolean): void {
  process.stdout.write(json ? JSON.stringify(value, null, 2) + '\n' : typeof value === 'string' ? value + '\n' :
    Object.entries(value as Record<string, unknown>).map(([key, item]) => `${key}: ${String(item)}`).join('\n') + '\n');
}

type Profile = { endpoint?: string; model?: string; apiKeyEnv?: string; headers?: Record<string, string>;
  request?: Record<string, unknown> };
function loadProfile(name?: string): { name: string; profile: Profile; configPath: string } {
  const configPath = join(defaultNatlangConfigDirectory(), 'config.json');
  let config: { defaultProfile?: string; profiles?: Record<string, Profile> } = {};
  if (existsSync(configPath)) config = JSON.parse(readFileSync(configPath, 'utf8')) as typeof config;
  const selected = name ?? process.env.NATLANG_PROFILE ?? config.defaultProfile ?? 'default';
  const configured = config.profiles?.[selected] ?? {};
  return { name: selected, configPath, profile: { ...configured,
    endpoint: process.env.NATLANG_SERVER ?? configured.endpoint,
    model: process.env.NATLANG_MODEL ?? configured.model } };
}
async function confirm(question: string): Promise<boolean> {
  if (!process.stdin.isTTY || !process.stderr.isTTY) return false;
  const terminal = createInterface({ input: process.stdin, output: process.stderr });
  try { return !/^n(?:o)?$/i.test((await terminal.question(`${question} [Y/n] `)).trim()); }
  finally { terminal.close(); }
}

async function ensureRuntime(discovery: LlamaRuntimeDiscovery, assumeYes: boolean,
  forceInstall = false): Promise<LlamaServerInspection | null> {
  if (discovery.selected && (!forceInstall || discovery.selected.source === 'managed')) return discovery.selected;
  const explicit = discovery.candidates.find(candidate => candidate.source === 'explicit');
  if (explicit) throw new Error(`NATLANG_LLAMA_SERVER is incompatible: ${describeLlamaRuntime(discovery)}`);
  if (!discovery.artifact) throw new Error(`natlang has no managed llama.cpp build for ${process.platform}-${process.arch}; set NATLANG_LLAMA_SERVER to a compatible build`);
  const found = discovery.candidates.length ? `Found incompatible ${describeLlamaRuntime(discovery)}. ` : '';
  const artifact = discovery.artifact;
  const allowed = assumeYes || await confirm(`${found}Download and install the verified llama.cpp ${LLAMA_RUNTIME_RELEASE.version} ${artifact.backend} runtime (${Math.ceil(artifact.bytes / 1_000_000)} MB)?`);
  if (!allowed) return null;
  return installManagedLlamaRuntime({ error: process.stderr });
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
    output(`llama.cpp runtime ready\nsource: ${discovery.selected.source}\npath: ${discovery.selected.path}\nversion: ${discovery.selected.version} (build ${discovery.selected.build})`, false);
    return;
  }
  output(`llama.cpp runtime unavailable\n${describeLlamaRuntime(discovery)}`, false);
}

function modelSession(profileName?: string, assumeYes = false) {
  const { profile } = loadProfile(profileName);
  return createManagedModelSession({ ...profile,
    request: profile.request, headers: profile.headers }, process.env, process.stderr,
  { ensureRuntime: discovery => ensureRuntime(discovery, assumeYes) });
}

type RunnablePackage = { name: string; version: string; digest: string; root: string;
  dependencies: Record<string, { name: string; version: string; digest: string; root: string }> };

function selectTarget(manifest: { name: string; targets?: Record<string, unknown> }, requested?: string): string {
  const targets = Object.keys(manifest.targets ?? {});
  const selected = requested ?? (targets.length === 1 ? targets[0] : undefined);
  if (!selected) throw new Error(`${manifest.name} has multiple targets; use --target ${targets.join('|')}`);
  if (!manifest.targets?.[selected]) throw new Error(`unknown target ${selected} in ${manifest.name}; choose ${targets.join(', ')}`);
  return selected;
}

async function executeTarget(parsed: Parsed, installed: RunnablePackage,
  manifest: ReturnType<NatlangPackageStore['manifest']>, targetName: string, local = false): Promise<number> {
  checkEngines(manifest.engines);
  const target = manifest.targets?.[targetName];
  if (!target) throw new Error(`unknown target ${targetName} in ${manifest.name}`);
  const stateVersion = local ? `${installed.version}-local.${installed.digest.slice(0, 12)}` : installed.version;
  const stateDirectory = resolve(option(parsed, '--state') ?? join(defaultNatlangStateDirectory(),
    encodeURIComponent(installed.name), stateVersion, targetName));
  const traceDirectory = resolve(option(parsed, '--traces') ?? join(stateDirectory, 'traces'));
  const workspace = resolve(option(parsed, '--workspace') ?? '.');
  const entry = join(installed.root, ...target.entry.split('/'));
  const model = modelSession(option(parsed, '--profile'), parsed.options.has('--yes'));
  try {
    const module = await import(pathToFileURL(entry).href) as Record<string, unknown>;
    const factoryName = target.export ?? 'createTarget', factory = module[factoryName];
    if (typeof factory !== 'function') throw new Error(`target entry does not export ${factoryName}()`);
    const context: PackageTargetContext = { package: installed, dependencies: installed.dependencies,
      targetName, target, workspace,
      stateDirectory, traceDirectory, args: parsed.rest, io: { input: process.stdin, output: process.stdout,
        error: process.stderr, color: !parsed.options.has('--plain') && !parsed.options.has('--no-color') && Boolean(process.stdout.isTTY) },
      modelTurn: request => model.turn(request), runtime: Object.freeze({ NativeNatlangHost,
        TypeScriptEnvironment, TerminalNatlangApplication, TerminalSessionStore, TerminalEventQueue,
        NodeFileTree, runTerminalShell, renderTerminalView }) };
    const executable = await (factory as PackageTargetFactory)(context);
    if (!executable || typeof executable.run !== 'function') throw new Error('target factory must return an executable with run()');
    try { return Number(await executable.run() ?? 0); }
    finally { await executable.close?.(); }
  } finally { await model.close(); }
}

async function runTarget(parsed: Parsed, specifier: string): Promise<number> {
  const marker = specifier.lastIndexOf('#');
  if (marker < 1 || marker === specifier.length - 1) throw new Error('run specifier must be NAME@VERSION#TARGET');
  const packageSpecifier = specifier.slice(0, marker), targetName = specifier.slice(marker + 1);
  const store = new NatlangPackageStore(option(parsed, '--store'));
  const installed = store.resolve(packageSpecifier), manifest = store.manifest(packageSpecifier);
  return executeTarget(parsed, installed, manifest, targetName);
}

function localManifestPath(value: string): string {
  const path = resolve(value);
  if (!existsSync(path)) throw new Error(`application path does not exist: ${value}`);
  const status = statSync(path);
  if (status.isFile()) return path;
  if (!status.isDirectory()) throw new Error(`application path is neither a file nor directory: ${value}`);
  const candidate = join(path, 'natlang.json');
  if (!isApplicationManifest(candidate))
    throw new Error(`application directory does not contain a valid natlang.json: ${value}`);
  return candidate;
}

type LocalSource = { kind: 'application'; path: string } | { kind: 'program'; path: string };
function isApplicationManifest(path: string): boolean {
  if (!existsSync(path) || !lstatSync(path).isFile()) return false;
  try {
    const value = JSON.parse(readFileSync(path, 'utf8')) as Record<string, unknown>;
    return value.schema === 'natlang.package/v1' && Boolean(value.targets && typeof value.targets === 'object');
  } catch { return false; }
}
function resolveLocalSource(value: string): LocalSource | null {
  const path = resolve(value);
  if (!existsSync(path)) return null;
  const status = statSync(path);
  if (status.isFile()) return isApplicationManifest(path) ? { kind: 'application', path } : { kind: 'program', path };
  if (!status.isDirectory()) throw new Error(`source is neither a file nor directory: ${value}`);
  const manifest = join(path, 'natlang.json');
  if (isApplicationManifest(manifest)) return { kind: 'application', path: manifest };
  for (const name of ['main.nl', 'main.ts', 'index.nl', 'index.ts']) {
    const entry = join(path, name); if (existsSync(entry)) return { kind: 'program', path: entry };
  }
  const functions = readdirSync(path).filter(name => name !== 'types.ts' && ['.nl', '.ts'].includes(name.slice(name.lastIndexOf('.'))));
  if (functions.length === 1) return { kind: 'program', path: join(path, functions[0]!) };
  const detail = functions.length ? ` Found function files: ${functions.sort().join(', ')}.` : '';
  throw new Error(`directory is not directly runnable: ${value}. Add natlang.json for an application or main.nl for a program.${detail}`);
}

function discoverLocalApplications(value = '.'): Array<{ path: string; name: string; version: string;
  targets: string[]; description: string }> {
  const root = resolve(value);
  if (!existsSync(root)) throw new Error(`application search path does not exist: ${value}`);
  const manifests: string[] = [], ignored = new Set(['.git', '.natlang', 'node_modules', '.venv', 'dist']);
  const visit = (path: string, explicitRoot = false) => {
    const status = explicitRoot ? statSync(path) : lstatSync(path);
    if (status.isSymbolicLink()) return;
    if (status.isFile()) { if ((basename(path) === 'natlang.json' || path.endsWith('.natlang.json')) && isApplicationManifest(path)) manifests.push(path); return; }
    if (!status.isDirectory() || ignored.has(basename(path))) return;
    for (const name of readdirSync(path).sort()) visit(join(path, name));
  };
  visit(root, true);
  return manifests.map(path => {
    const manifest = JSON.parse(readFileSync(path, 'utf8')) as { name: string; version: string;
      description?: string; targets?: Record<string, unknown> };
    return { path, name: manifest.name, version: manifest.version,
      targets: Object.keys(manifest.targets ?? {}), description: manifest.description ?? '' };
  }).sort((left, right) => left.path.localeCompare(right.path));
}

function loadLocalApplication(parsed: Parsed, value: string) {
  const manifestPath = localManifestPath(value);
  const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
  const explicitRoot = option(parsed, '--root');
  const roots: string[] = [];
  if (explicitRoot) roots.push(resolve(explicitRoot));
  else {
    let candidate = dirname(manifestPath);
    while (true) {
      roots.push(candidate);
      const parent = dirname(candidate);
      if (parent === candidate) break;
      candidate = parent;
    }
  }
  let failure: unknown;
  for (const root of roots) {
    try { return { archive: createPackageArchive(manifest, root), root, manifestPath }; }
    catch (error) { failure = error; }
  }
  const detail = failure instanceof Error ? failure.message : String(failure);
  throw new Error(`cannot load application ${manifestPath}: ${detail}${explicitRoot ? '' : '; pass --root if its source root is elsewhere'}`);
}

async function runLocalApplication(parsed: Parsed, value: string): Promise<number> {
  const { archive, root } = loadLocalApplication(parsed, value);
  const store = new NatlangPackageStore(option(parsed, '--store'));
  const installed: RunnablePackage = { name: archive.manifest.name, version: archive.manifest.version,
    digest: archive.digest, root, dependencies: store.resolveDependencies(archive.manifest.dependencies) };
  const target = selectTarget(archive.manifest, option(parsed, '--target'));
  return executeTarget(parsed, installed, archive.manifest, target, true);
}

async function runProgramPath(parsed: Parsed, value: string): Promise<number> {
  const path = resolve(value);
  const inputsPath = option(parsed, '--inputs');
  const inputs = inputsPath ? JSON.parse(readFileSync(resolve(inputsPath), 'utf8')) as Record<string, unknown> : undefined;
  if (inputs !== undefined && (!inputs || typeof inputs !== 'object' || Array.isArray(inputs)))
    throw new Error('--inputs must contain a JSON object');
  const timeoutMs = numericOption(parsed, '--timeout', 1), seed = numericOption(parsed, '--seed', 0);
  const host = new NativeNatlangHost();
  const model = modelSession(option(parsed, '--profile'), parsed.options.has('--yes'));
  try {
    const result = await host.run({ source: { kind: 'file', path }, inputs,
      modelTurn: request => model.turn(request),
      tracePath: option(parsed, '--trace') ? resolve(option(parsed, '--trace')!) : undefined,
      timeoutMs,
      options: seed === undefined ? undefined : { seed: { mode: 'derived', root: seed } } });
    if (parsed.options.has('--json')) output(result, true);
    else process.stdout.write(JSON.stringify(result.value, null, 2) + '\n');
    if (result.outcome.kind !== 'done') process.stderr.write(`natlang: ${result.outcome.kind}: ${result.outcome.detail}\n`);
    return result.outcome.kind === 'done' ? 0 : 1;
  } finally { host.close(); await model.close(); }
}

async function runAnonymousInstruction(parsed: Parsed, instruction: string): Promise<number> {
  acceptOptions(parsed, ['--profile', '--trace', '--json', '--timeout', '--seed', '--yes']);
  noTrailingArguments(parsed);
  const timeoutMs = numericOption(parsed, '--timeout', 1), seed = numericOption(parsed, '--seed', 0);
  const host = new NativeNatlangHost();
  const model = modelSession(option(parsed, '--profile'), parsed.options.has('--yes'));
  try {
    const program = loadAnonymousInstruction(process.cwd(), instruction);
    const body = program.$lambda as Record<string, unknown>;
    body.type = 'Lambda<{ files: Dict<ProjectFile> }, Text>';
    body.types = { ...body.types as Record<string, string> ?? {}, ProjectFile: FILE_TREE_LEAF_TYPE };
    const result = await host.run({ source: { kind: 'program', program },
      inputs: { files: new NodeFileTree(process.cwd()) },
      modelTurn: request => model.turn(request),
      tracePath: option(parsed, '--trace') ? resolve(option(parsed, '--trace')!) : undefined,
      timeoutMs,
      options: seed === undefined ? undefined : { seed: { mode: 'derived', root: seed } } });
    if (parsed.options.has('--json')) output(result, true);
    else if (typeof result.value === 'string') process.stdout.write(result.value + (result.value.endsWith('\n') ? '' : '\n'));
    else process.stdout.write(JSON.stringify(result.value, null, 2) + '\n');
    if (result.outcome.kind !== 'done') process.stderr.write(`natlang: ${result.outcome.kind}: ${result.outcome.detail}\n`);
    return result.outcome.kind === 'done' ? 0 : 1;
  } finally { host.close(); await model.close(); }
}

const programOptions = ['--inputs', '--profile', '--trace', '--json', '--timeout', '--seed', '--yes'];
const applicationOptions = ['--root', '--target', '--profile', '--workspace', '--state', '--traces',
  '--store', '--plain', '--no-color', '--yes'];

async function runSource(parsed: Parsed, value: string): Promise<number> {
  const local = resolveLocalSource(value);
  if (local?.kind === 'application') {
    acceptOptions(parsed, applicationOptions);
    return runLocalApplication(parsed, local.path);
  }
  if (local?.kind === 'program') {
    acceptOptions(parsed, programOptions); noTrailingArguments(parsed);
    return runProgramPath(parsed, local.path);
  }
  if (value.startsWith('.') || value.startsWith('/') || (!value.startsWith('@') && value.includes('/')) ||
      /\.(?:nl|ts|json|ya?ml)$/.test(value)) throw new Error(`source path does not exist: ${value}`);
  acceptOptions(parsed, applicationOptions.filter(name => name !== '--root'));
  return runTarget(parsed, value.includes('#') ? value : applicationSpecifier(
    new NatlangPackageStore(option(parsed, '--store')), value, option(parsed, '--target')));
}

function inspectSource(parsed: Parsed, value: string): Record<string, unknown> {
  noTrailingArguments(parsed);
  const local = resolveLocalSource(value);
  if (local?.kind === 'program') {
    acceptOptions(parsed, ['--json']);
    const source = loadFunctionFile(local.path);
    return { kind: 'program', path: local.path, function: source.functionName,
      signature: formatType(source.type), implementation: source.kind, engine: source.engine,
      effects: source.effects, functions: Object.keys(source.codebase) };
  }
  if (local?.kind === 'application') {
    acceptOptions(parsed, ['--root', '--target', '--json']);
    const { archive, root, manifestPath } = loadLocalApplication(parsed, local.path);
    const targetName = selectTarget(archive.manifest, option(parsed, '--target'));
    return { kind: 'application', path: manifestPath, root, name: archive.manifest.name,
      version: archive.manifest.version, ...inspectTarget(archive.manifest, targetName, manifestPath) };
  }
  acceptOptions(parsed, ['--target', '--store', '--json']);
  const store = new NatlangPackageStore(option(parsed, '--store'));
  const specifier = value.includes('#') ? value : applicationSpecifier(store, value, option(parsed, '--target'));
  const marker = specifier.lastIndexOf('#');
  if (marker < 1) throw new Error('installed source must resolve to NAME@VERSION#TARGET');
  const packageSpecifier = specifier.slice(0, marker), targetName = specifier.slice(marker + 1);
  const manifest = store.manifest(packageSpecifier), installed = store.resolve(packageSpecifier);
  return { kind: 'installed-application', root: installed.root,
    ...inspectTarget(manifest, targetName, packageSpecifier) };
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

function inspectTarget(manifest: ReturnType<NatlangPackageStore['manifest']>, targetName: string,
  packageLabel: string): Record<string, unknown> {
  const target = manifest.targets?.[targetName];
  if (!target) throw new Error(`unknown target ${targetName} in ${packageLabel}`);
  let engineError: string | null = null;
  try { checkEngines(manifest.engines); }
  catch (error) { engineError = error instanceof Error ? error.message : String(error); }
  return { package: packageLabel, target: targetName, authority: target.authority ?? [],
    commands: Object.fromEntries((target.commands ?? []).map(command => [command, commandAvailable(command)])),
    engines: manifest.engines ?? {}, engineError };
}

function doctorReport(parsed: Parsed, store: NatlangPackageStore): { report: Record<string, unknown>; okay: boolean } {
  const selected = loadProfile(option(parsed, '--profile'));
  const hasExternal = Boolean(selected.profile.endpoint);
  const partialExternal = Boolean(selected.profile.model) && !hasExternal;
  const local = localModelPrerequisites();
  const modelOkay = hasExternal || (!partialExternal && local.available);
  const okay = Boolean(modelOkay);
  return { okay, report: { ok: okay, node: process.version, packageStore: store.root,
    installedPackages: store.list().length, config: selected.configPath, profile: selected.name,
    modelSource: hasExternal ? 'external' : partialExternal ? 'invalid-partial-profile' : 'managed-local',
    endpoint: selected.profile.endpoint ?? null,
    model: selected.profile.model ?? DEFAULT_LOCAL_MODEL.id,
    modelPath: hasExternal ? null : local.modelPath, modelServer: hasExternal ? null : local.executable,
    modelDownloadAvailable: hasExternal ? null : local.downloadable,
    apiKey: Boolean(process.env[selected.profile.apiKeyEnv ?? 'NATLANG_API_KEY']) } };
}

function applicationSpecifier(store: NatlangPackageStore, query: string, requestedTarget?: string): string {
  const packages = store.list().filter(item => {
    const short = item.name.includes('/') ? item.name.slice(item.name.lastIndexOf('/') + 1) : item.name;
    return item.name === query || short === query;
  });
  const names = [...new Set(packages.map(item => item.name))];
  if (!packages.length) throw new Error(`no installed application matches ${query}`);
  if (names.length > 1) throw new Error(`application name is ambiguous: ${names.join(', ')}`);
  packages.sort((left, right) => compareVersions(right.version, left.version));
  const selected = packages[0]!, manifest = store.manifest(`${selected.name}@${selected.version}`);
  const targets = Object.keys(manifest.targets ?? {});
  const target = requestedTarget ?? (targets.length === 1 ? targets[0] : undefined);
  if (!target) throw new Error(`${selected.name} has multiple targets; use --target ${targets.join('|')}`);
  if (!manifest.targets?.[target]) throw new Error(`unknown target ${target} in ${selected.name}; choose ${targets.join(', ')}`);
  return `${selected.name}@${selected.version}#${target}`;
}

export async function main(argv = process.argv.slice(2)): Promise<number> {
  const administrativeCommands = new Map([
    ['--apps', 'apps'], ['--inspect', 'inspect'], ['--packages', 'packages'],
    ['--package', 'package'], ['--setup', 'setup'], ['--runtime', 'runtime'], ['--doctor', 'doctor'],
  ]);
  const administrativeCommand = administrativeCommands.get(argv[0] ?? '');
  const parsed = parseArgs(administrativeCommand ? [administrativeCommand, ...argv.slice(1)] : argv);
  const json = parsed.options.has('--json');
  if (parsed.options.has('--version')) { output(NATLANG_CLI_VERSION, false); return 0; }
  if (parsed.options.has('--help') || parsed.options.has('-h')) {
    const topic = parsed.words[0]?.toLowerCase();
    output(topic && ['apps', 'inspect', 'packages', 'package', 'runtime', 'setup', 'doctor'].includes(topic) ?
      topicHelp(topic === 'setup' ? 'runtime' : topic) :
      topic ? topicHelp('source') : help(), false); return 0;
  }
  if (!parsed.words.length) { acceptOptions(parsed, []); output(help(), false); return 0; }
  if (administrativeCommand && (parsed.words[0] === 'setup' || parsed.words[0] === 'runtime')) {
    const action = parsed.words[0] === 'setup' ? 'ensure' : parsed.words[1] ?? 'status';
    if (parsed.words[0] === 'setup') {
      acceptOptions(parsed, ['--yes', '--json', '--profile']);
      if (parsed.words.length !== 1) throw new Error('usage: natlang --setup [--yes] [--json]');
    } else {
      acceptOptions(parsed, action === 'install' ? ['--yes', '--json'] : ['--json']);
      if (parsed.words.length > 2) throw new Error('usage: natlang --runtime status|install');
    }
    noTrailingArguments(parsed);
    if (parsed.words[0] === 'setup') {
      const selected = loadProfile(option(parsed, '--profile'));
      if (selected.profile.endpoint) {
        const report = { ok: true, modelSource: 'external', endpoint: selected.profile.endpoint,
          model: selected.profile.model ?? DEFAULT_LOCAL_MODEL.id, runtimeRequired: false };
        output(json ? report : `external model profile ready\nendpoint: ${report.endpoint}\nmodel: ${report.model}`, json);
        return 0;
      }
    }
    if (action === 'status') {
      const discovery = discoverLlamaRuntime(); outputRuntime(discovery, json);
      return discovery.selected ? 0 : 1;
    }
    if (action === 'ensure' || action === 'install') {
      const discovery = discoverLlamaRuntime();
      const selected = await ensureRuntime(discovery, parsed.options.has('--yes'), action === 'install');
      if (!selected && action === 'ensure') { outputRuntime(discovery, json); return 0; }
      if (!selected) throw new Error('llama.cpp installation was declined; run natlang --setup when ready');
      outputRuntime(discoverLlamaRuntime(), json); return 0;
    }
    throw new Error('unknown runtime command; run natlang --help');
  }
  if (administrativeCommand === 'apps') {
    acceptOptions(parsed, ['--json']); noTrailingArguments(parsed);
    if (parsed.words.length > 2) throw new Error('usage: natlang --apps [DIRECTORY] [--json]');
    const root = parsed.words[1] ?? '.', rows = discoverLocalApplications(root).map(item => ({ ...item,
      path: relative(process.cwd(), item.path) || basename(item.path) }));
    output(json ? rows : rows.length ? rows.map(item =>
      `${item.path}\n  ${item.name}@${item.version}  targets: ${item.targets.join(', ')}${item.description ? `\n  ${item.description}` : ''}`).join('\n') :
      `No natlang applications found under ${resolve(root)}. Applications contain a natlang.json manifest.`, json);
    return 0;
  }
  if (administrativeCommand === 'packages') {
    acceptOptions(parsed, ['--store', '--json']); noTrailingArguments(parsed);
    if (parsed.words.length !== 1) throw new Error('usage: natlang --packages [--store DIR] [--json]');
    const store = new NatlangPackageStore(option(parsed, '--store')), rows = store.list();
    output(json ? rows : rows.length ? rows.map(row => `${row.name}@${row.version} ${row.digest}`).join('\n') :
      `No distribution packages are installed in ${store.root}. Source applications can run without installation.`, json);
    return 0;
  }
  if (administrativeCommand === 'inspect') {
    if (parsed.words.length !== 2) throw new Error('usage: natlang --inspect SOURCE [--json]');
    output(inspectSource(parsed, parsed.words[1]!), json); return 0;
  }
  if (administrativeCommand === 'package') {
    const action = parsed.words[1], argument = parsed.words[2];
    if (action === 'pack' && !argument) throw new Error('usage: natlang --package pack MANIFEST_OR_DIRECTORY [--root DIR] [--out FILE]');
    if (action === 'verify' && !argument) throw new Error('usage: natlang --package verify ARCHIVE [--json]');
    if (action === 'install' && !argument) throw new Error('usage: natlang --package install ARCHIVE... [--store DIR] [--json]');
    if (action === 'pack' && argument) {
      acceptOptions(parsed, ['--root', '--out', '--json']); noTrailingArguments(parsed);
      if (parsed.words.length !== 3) throw new Error('usage: natlang --package pack MANIFEST_OR_DIRECTORY [--root DIR] [--out FILE]');
      const { archive } = loadLocalApplication(parsed, argument);
      const out = resolve(option(parsed, '--out') ?? `${archive.manifest.name.replace('/', '-')}-${archive.manifest.version}.nlpkg`);
      writePackageArchive(out, archive); output({ archive: out, digest: archive.digest,
        name: archive.manifest.name, version: archive.manifest.version }, json); return 0;
    }
    if (action === 'verify' && argument) {
      acceptOptions(parsed, ['--json']); noTrailingArguments(parsed);
      if (parsed.words.length !== 3) throw new Error('usage: natlang --package verify ARCHIVE [--json]');
      const archive = readPackageArchive(resolve(argument)); output({ valid: true, digest: archive.digest,
        name: archive.manifest.name, version: archive.manifest.version, files: archive.files.length }, json); return 0;
    }
    const store = new NatlangPackageStore(option(parsed, '--store'));
    if (action === 'install' && argument) {
      acceptOptions(parsed, ['--store', '--json']); noTrailingArguments(parsed);
      const installed = store.installMany(parsed.words.slice(2).map(path => resolve(path)));
      output(installed.length === 1 ? installed[0] : installed, json); return 0;
    }
    if (action === 'list') throw new Error('use `natlang --packages`');
    if (action === 'inspect') throw new Error('use `natlang --inspect SOURCE`');
    throw new Error('unknown package command; choose pack, verify, or install');
  }
  if (administrativeCommand === 'doctor') {
    acceptOptions(parsed, ['--store', '--profile', '--json']); noTrailingArguments(parsed);
    if (parsed.words.length !== 1) throw new Error('use `natlang --inspect SOURCE` to inspect a program or application');
    const store = new NatlangPackageStore(option(parsed, '--store'));
    const { report, okay } = doctorReport(parsed, store);
    output(report, json); return okay ? 0 : 1;
  }
  if (parsed.words.length > 1 || /\s/.test(parsed.words[0]!)) {
    const value = parsed.words.join(' ');
    if (existsSync(resolve(value))) return runSource(parsed, value);
    return runAnonymousInstruction(parsed, value);
  }
  return runSource(parsed, parsed.words[0]!);
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href)
  main().then(code => { process.exitCode = code; }, error => { process.stderr.write(`natlang: ${error instanceof Error ? error.message : String(error)}\n`); process.exitCode = 1; });
