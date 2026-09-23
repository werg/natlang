#!/usr/bin/env node
import { existsSync, lstatSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { basename, dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createInterface } from 'node:readline/promises';
import { createPackageArchive, NatlangPackageStore, readPackageArchive,
  writePackageArchive, defaultNatlangConfigDirectory, defaultNatlangStateDirectory } from '../package/index.js';
import { compareVersions, satisfiesVersion } from '../package/store.js';
import type { NatlangTarget } from '../package/manifest.js';
import type { TargetContext, TargetExecutable, TargetMain } from '../package/target.js';
import { createManagedModelSession, DEFAULT_LOCAL_MODEL, describeLlamaRuntime, discoverLlamaRuntime,
  installManagedLlamaRuntime, LLAMA_RUNTIME_RELEASE, localModelPrerequisites,
  type LlamaRuntimeDiscovery, type LlamaServerInspection } from '../model/index.js';
import { openFolder } from '../native/node-files.js';
import { formatDiagnostics } from '../compiler/project.js';
import { buildProject } from '../compiler/node-project.js';
import { createNatlangRuntime, type ModelDriver, type NatlangRuntime } from '../runtime/runtime.js';
import { fileTraceSink, applicationContextRecords, loadNatlang } from '../runtime/node-files.js';
import { invokeDefinition } from '../runtime/kernel.js';
import { resolveFrame } from '../runtime/runtime.js';
import '../runtime/node.js';

export const NATLANG_CLI_VERSION = '0.2.0';
/** The runtime module this CLI runs; compiled apps are bound to it so they share one runtime instance. */
const RUNTIME_MODULE = { url: new URL('../index.js', import.meta.url).href, path: fileURLToPath(new URL('../index.js', import.meta.url)),
  types: fileURLToPath(new URL('../index.d.ts', import.meta.url)), specifiers: ['@natlang/node'] };

function help(): string { return `natlang ${NATLANG_CLI_VERSION}

Usage:
  natlang run [SOURCE] [OPTIONS] [-- ARGS]   Build and run an application or TS entry module.
  natlang check [PROJECT]                    Type-check and natlang-check a project.
  natlang build [PROJECT] [--out DIR]        Compile a project (lowering nl, embedding .nl functions).
  natlang call FILE.nl [--inputs FILE]       Call one named natural-language function.
  natlang ask INSTRUCTION...                 Answer an instruction over the local natlang.d/ and files.
  natlang apps [DIRECTORY]                   Find runnable applications (natlang.json).
  natlang inspect SOURCE                     Explain how a source resolves without running it.
  natlang packages                           List installed distribution packages.
  natlang package pack|verify|install ...    Pack, verify, or install a distribution archive.
  natlang setup                              Prepare the local model runtime.
  natlang runtime status|install             Inspect or install the model runtime.
  natlang doctor                             Check this natlang installation.

Run natlang help COMMAND for focused usage and options.`; }

function topicHelp(topic: string): string {
  if (topic === 'run') return `Build and run an application:
  natlang run [SOURCE] [OPTIONS] [-- APPLICATION_ARGS...]

SOURCE is an application directory or natlang.json manifest (default: .), a
TypeScript entry module, NAME, or NAME@VERSION#TARGET of an installed package.
The project is compiled with natlang build into .natlang/build and the entry's
exported function (default main) receives the target context: args, io,
workspace, state and trace directories, the configured model, and a runtime.

Options:
  --target NAME       Select one target when the manifest has several.
  --export NAME       Entry function for a TypeScript entry module (default main).
  --workspace DIR     Set the application's working directory.
  --state DIRECTORY   Override durable application state.
  --traces DIRECTORY  Override application trace storage.
  --profile NAME      Select a model profile.
  --store DIR         Package store for installed applications.
  --plain             Disable interactive terminal formatting.
  --no-color          Disable terminal color.
  --yes               Permit a required managed runtime download.`;
  if (topic === 'check' || topic === 'build') return `Check or compile a project:
  natlang check [PROJECT] [--json]
  natlang build [PROJECT] [--out DIR] [--target node|browser] [--json]

PROJECT is a directory or tsconfig.json (default: .). Both commands write a
typed foo.d.nl.ts beside every foo.nl so editors and tsc see typed imports.`;
  if (topic === 'call') return `Call one named natural-language function:
  natlang call FILE.nl [--inputs FILE] [--trace DIR] [--profile NAME] [--json]

--inputs is a JSON object keyed by parameter name.`;
  if (topic === 'ask') return `Answer an instruction:
  natlang ask INSTRUCTION... [--trace DIR] [--profile NAME]

The instruction runs as an inline natural-language function over the nearest
natlang.d/ callable folder and a read-only view of the current directory's files.`;
  if (topic === 'apps') return `Discover applications:
  natlang apps [DIRECTORY] [--json]`;
  if (topic === 'inspect') return `Resolve a source without running it:
  natlang inspect SOURCE [--target NAME] [--store DIR] [--json]`;
  if (topic === 'packages' || topic === 'package') return `Distribution packages are optional deployment artifacts:
  natlang package pack MANIFEST_OR_DIRECTORY [--out FILE]
  natlang package verify ARCHIVE [--json]
  natlang package install ARCHIVE... [--store DIR] [--json]
  natlang packages [--store DIR] [--json]`;
  if (topic === 'runtime' || topic === 'setup') return `Model runtime commands:
  natlang setup [--yes] [--json]
  natlang runtime status [--json]
  natlang runtime install [--yes] [--json]

Setup reuses a compatible explicit, managed, or PATH llama-server. If none is
compatible, interactive use asks before installing the verified managed build.
Profiles live in ~/.config/natlang/config.json. Environment overrides are
NATLANG_SERVER, NATLANG_MODEL, NATLANG_MODEL_PATH, NATLANG_LLAMA_SERVER,
NATLANG_RUNTIME_HOME, and NATLANG_API_KEY.`;
  if (topic === 'doctor') return `Check the natlang installation and model configuration:
  natlang doctor [--profile NAME] [--store DIR] [--json]`;
  throw new Error(`unknown help topic ${topic}`);
}
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
  if (parsed.rest.length) throw new Error('arguments after -- are only valid for natlang run');
}


function output(value: unknown, json: boolean): void {
  process.stdout.write(json ? JSON.stringify(value, null, 2) + '\n' : typeof value === 'string' ? value + '\n' :
    Object.entries(value as Record<string, unknown>).map(([key, item]) => `${key}: ${text(item)}`).join('\n') + '\n');
}
/** One line of plain output: scalars as text, lists joined, records as `key=value` pairs. */
function text(item: unknown): string {
  if (Array.isArray(item)) return item.map(text).join(', ') || '(none)';
  if (item && typeof item === 'object') {
    const entries = Object.entries(item);
    return entries.length ? entries.map(([key, child]) => `${key}=${typeof child === 'object' && child ? JSON.stringify(child) : String(child)}`).join(', ') : '(none)';
  }
  return String(item);
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


function selectTarget(manifest: { name: string; targets?: Record<string, unknown> }, requested?: string): string {
  const targets = Object.keys(manifest.targets ?? {});
  const selected = requested ?? (targets.length === 1 ? targets[0] : undefined);
  if (!selected) throw new Error(`${manifest.name} has multiple targets; use --target ${targets.join('|')}`);
  if (!manifest.targets?.[selected]) throw new Error(`unknown target ${selected} in ${manifest.name}; choose ${targets.join(', ')}`);
  return selected;
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


function isApplicationManifest(path: string): boolean {
  if (!existsSync(path) || !lstatSync(path).isFile()) return false;
  try {
    const value = JSON.parse(readFileSync(path, 'utf8')) as Record<string, unknown>;
    return value.schema === 'natlang.package/v2' && Boolean(value.targets && typeof value.targets === 'object');
  } catch { return false; }
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
function compileFor(root: string, entry: string, outDir: string, installed: boolean): string {
  const result = buildProject({ project: root, outDir, runtimeModule: RUNTIME_MODULE, writeDeclarations: !installed });
  if (!result.ok) throw new Error(`natlang build failed:\n${formatDiagnostics(result.diagnostics)}`);
  if (/\.(?:m?js)$/.test(entry)) return entry;
  const emitted = join(outDir, relative(root, entry)).replace(/\.m?ts$/, '.js');
  if (!existsSync(emitted)) throw new Error(`the build did not emit ${relative(root, entry)}`);
  return emitted;
}

type Launch = { root: string; target: NatlangTarget; targetName: string; installed?: boolean;
  package?: TargetContext['package']; dependencies: TargetContext['dependencies']; stateKey: string };

async function launch(parsed: Parsed, spec: Launch): Promise<number> {
  const stateDirectory = resolve(option(parsed, '--state') ?? join(defaultNatlangStateDirectory(),
    encodeURIComponent(spec.stateKey), spec.targetName));
  const traceDirectory = resolve(option(parsed, '--traces') ?? join(stateDirectory, 'traces'));
  const workspace = resolve(option(parsed, '--workspace') ?? '.');
  const entry = join(spec.root, ...spec.target.entry.split('/'));
  const compiled = compileFor(spec.root, entry, spec.installed ? join(stateDirectory, 'build', spec.package?.digest ?? 'local') :
    join(spec.root, '.natlang', 'build'), spec.installed === true);
  const model = modelSession(option(parsed, '--profile'), parsed.options.has('--yes'));
  const driver: ModelDriver = request => model.turn(request);
  const runtime = createNatlangRuntime({ model: driver, trace: fileTraceSink(traceDirectory) });
  try {
    const module = await import(pathToFileURL(compiled).href) as Record<string, unknown>;
    const name = spec.target.export ?? 'main', main = module[name];
    if (typeof main !== 'function') throw new Error(`${spec.target.entry} does not export ${name}()`);
    const context: TargetContext = { package: spec.package, dependencies: spec.dependencies, targetName: spec.targetName,
      target: spec.target, workspace, stateDirectory, traceDirectory, args: parsed.rest,
      io: { input: process.stdin, output: process.stdout, error: process.stderr,
        color: !parsed.options.has('--plain') && !parsed.options.has('--no-color') && Boolean(process.stdout.isTTY) },
      model: driver, runtime };
    const value = await (main as TargetMain)(context);
    if (value && typeof value === 'object' && typeof (value as TargetExecutable).run === 'function') {
      const executable = value as TargetExecutable;
      try { return Number(await executable.run() ?? 0); } finally { await executable.close?.(); }
    }
    return Number(value ?? 0);
  } finally { runtime.close(); await model.close(); }
}

async function runCommand(parsed: Parsed, value = '.'): Promise<number> {
  acceptOptions(parsed, ['--target', '--export', '--profile', '--workspace', '--state', '--traces', '--store',
    '--plain', '--no-color', '--yes']);
  const path = resolve(value);
  if (existsSync(path) && statSync(path).isFile() && /\.(?:m?ts|m?js)$/.test(path)) {
    const root = projectRoot(path);
    return launch(parsed, { root, targetName: 'main', dependencies: {}, stateKey: basename(root),
      target: { entry: relative(root, path).split('\\').join('/'), ...(option(parsed, '--export') ? { export: option(parsed, '--export')! } : {}) } });
  }
  if (existsSync(path)) {
    const manifestPath = statSync(path).isDirectory() ? join(path, 'natlang.json') : path;
    if (!isApplicationManifest(manifestPath)) throw new Error(`${value} has no natlang.package/v2 natlang.json manifest`);
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
  if (!target) throw new Error(`unknown target ${targetName} in ${manifest.name}`);
  return launch(parsed, { root: installed.root, targetName, target, installed: true,
    package: { name: installed.name, version: installed.version, digest: installed.digest, root: installed.root },
    dependencies: installed.dependencies, stateKey: `${installed.name}/${installed.version}` });
}

async function withModelRuntime<T>(parsed: Parsed, fn: (runtime: NatlangRuntime) => Promise<T>): Promise<T> {
  const model = modelSession(option(parsed, '--profile'), parsed.options.has('--yes'));
  const trace = option(parsed, '--trace');
  const runtime = createNatlangRuntime({ model: request => model.turn(request), ...(trace ? { trace: fileTraceSink(resolve(trace)) } : {}) });
  try { await model.prepare(); return await fn(runtime); }
  finally { runtime.close(); await model.close(); }
}

async function callCommand(parsed: Parsed, file: string): Promise<number> {
  acceptOptions(parsed, ['--inputs', '--trace', '--profile', '--json', '--yes']); noTrailingArguments(parsed);
  const fn = loadNatlang(resolve(file));
  const inputsPath = option(parsed, '--inputs');
  const inputs = inputsPath ? JSON.parse(readFileSync(resolve(inputsPath), 'utf8')) as Record<string, unknown> : {};
  if (!inputs || typeof inputs !== 'object' || Array.isArray(inputs)) throw new Error('--inputs must contain a JSON object');
  const record = (fn as unknown as { [key: symbol]: { definition: { params: { name: string }[] } } })[Symbol.for('natlang.callable')]!;
  const args = record.definition.params.map(parameter => inputs[parameter.name]);
  const value = await withModelRuntime(parsed, runtime => runtime.run(() => fn(...args)));
  output(value, true);
  return 0;
}

async function askCommand(parsed: Parsed, instruction: string): Promise<number> {
  acceptOptions(parsed, ['--trace', '--profile', '--yes']); noTrailingArguments(parsed);
  const context = applicationContextRecords(process.cwd());
  const value = await withModelRuntime(parsed, runtime => runtime.run(() => invokeDefinition(resolveFrame(), {
    id: 'natlang:ask', name: 'ask', body: `${instruction.trim()}\n`,
    params: [{ name: 'project', type: 'Folder' }], returns: 'string',
    types: {}, codebase: context?.records ?? {}, subtype: 'function' },
    [openFolder(process.cwd()).root()])));
  process.stdout.write(`${String(value).replace(/\n?$/, '\n')}`);
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
    if (!isApplicationManifest(manifestPath)) throw new Error(`${value} has no natlang.package/v2 manifest`);
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
  if (!target) throw new Error(`unknown target ${targetName} in ${label}`);
  let engineError: string | null = null;
  try { checkEngines(manifest.engines); } catch (error) { engineError = error instanceof Error ? error.message : String(error); }
  return { package: label, target: targetName, entry: target.entry, export: target.export ?? 'main',
    authority: target.authority ?? [],
    commands: Object.fromEntries((target.commands ?? []).map(command => [command, commandAvailable(command)])),
    engines: manifest.engines ?? {}, engineError };
}

export async function main(argv = process.argv.slice(2)): Promise<number> {
  const parsed = parseArgs(argv);
  const json = parsed.options.has('--json');
  const [command, ...words] = parsed.words;
  if (parsed.options.has('--version')) { output(NATLANG_CLI_VERSION, false); return 0; }
  if (!command || command === 'help' || parsed.options.has('--help') || parsed.options.has('-h')) {
    const topic = command === 'help' ? words[0] : command;
    output(topic ? topicHelp(topic) : help(), false); return 0;
  }
  if (command === 'check' || command === 'build') {
    acceptOptions(parsed, command === 'build' ? ['--out', '--target', '--json'] : ['--json']); noTrailingArguments(parsed);
    const target = option(parsed, '--target');
    if (target && target !== 'node' && target !== 'browser') throw new Error('--target must be node or browser');
    const result = buildProject({ project: words[0] ?? '.', emit: command === 'build', outDir: option(parsed, '--out'),
      runtimeTypes: { specifiers: RUNTIME_MODULE.specifiers, types: RUNTIME_MODULE.types },
      ...(target ? { target: target as 'node' | 'browser' } : {}) });
    if (json) output({ ok: result.ok, diagnostics: result.diagnostics, outDir: result.outDir, manifest: result.manifest }, true);
    else output(result.ok ? `${command === 'build' ? `built ${Object.keys(result.outputs).length} files into ${result.outDir}` : 'ok'}` :
      formatDiagnostics(result.diagnostics), false);
    return result.ok ? 0 : 1;
  }
  if (command === 'run') return runCommand(parsed, words[0]);
  if (command === 'call') { if (words.length !== 1) throw new Error('usage: natlang call FILE.nl [--inputs FILE]'); return callCommand(parsed, words[0]!); }
  if (command === 'ask') { if (!words.length) throw new Error('usage: natlang ask INSTRUCTION...'); return askCommand(parsed, words.join(' ')); }
  if (command === 'setup' || command === 'runtime') {
    const action = command === 'setup' ? 'ensure' : words[0] ?? 'status';
    acceptOptions(parsed, command === 'setup' ? ['--yes', '--json', '--profile'] : action === 'install' ? ['--yes', '--json'] : ['--json']);
    noTrailingArguments(parsed);
    if (command === 'setup') {
      const selected = loadProfile(option(parsed, '--profile'));
      if (selected.profile.endpoint) {
        const report = { ok: true, modelSource: 'external', endpoint: selected.profile.endpoint,
          model: selected.profile.model ?? DEFAULT_LOCAL_MODEL.id, runtimeRequired: false };
        output(json ? report : `external model profile ready\nendpoint: ${report.endpoint}\nmodel: ${report.model}`, json);
        return 0;
      }
    }
    if (action === 'status') { const discovery = discoverLlamaRuntime(); outputRuntime(discovery, json); return discovery.selected ? 0 : 1; }
    if (action === 'ensure' || action === 'install') {
      const discovery = discoverLlamaRuntime();
      const selected = await ensureRuntime(discovery, parsed.options.has('--yes'), action === 'install');
      if (!selected && action === 'ensure') { outputRuntime(discovery, json); return 0; }
      if (!selected) throw new Error('llama.cpp installation was declined; run natlang setup when ready');
      outputRuntime(discoverLlamaRuntime(), json); return 0;
    }
    throw new Error('usage: natlang runtime status|install');
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
    if (words.length !== 1) throw new Error('usage: natlang inspect SOURCE [--json]');
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
    throw new Error('usage: natlang package pack|verify|install ...');
  }
  if (command === 'doctor') {
    acceptOptions(parsed, ['--store', '--profile', '--json']); noTrailingArguments(parsed);
    const { report, okay } = doctorReport(parsed, new NatlangPackageStore(option(parsed, '--store')));
    output(report, json); return okay ? 0 : 1;
  }
  throw new Error(`unknown command ${command}; run natlang help`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href)
  main().then(code => { process.exitCode = code; }, error => { process.stderr.write(`natlang: ${error instanceof Error ? error.message : String(error)}\n`); process.exitCode = 1; });
