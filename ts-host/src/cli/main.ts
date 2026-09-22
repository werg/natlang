#!/usr/bin/env node
import { existsSync, readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { basename, dirname, join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { createInterface } from 'node:readline/promises';
import { createPackageArchive, NatlangPackageStore, parsePackageArchive, readPackageArchive,
  writePackageArchive, defaultNatlangConfigDirectory, defaultNatlangStateDirectory } from '../package/index.js';
import { compareVersions, satisfiesVersion } from '../package/store.js';
import type { PackageTargetContext, PackageTargetFactory } from '../package/target.js';
import { createManagedModelSession, DEFAULT_LOCAL_MODEL, describeLlamaRuntime, discoverLlamaRuntime,
  installManagedLlamaRuntime, LLAMA_RUNTIME_RELEASE, localModelPrerequisites,
  type LlamaRuntimeDiscovery, type LlamaServerInspection } from '../model/index.js';
import { NativeNatlangHost } from '../native/host.js';
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
    if (boolean.has(value)) { options.set(value, true); continue; }
    if (value.startsWith('--')) {
      const equal = value.indexOf('=');
      if (equal >= 0) options.set(value.slice(0, equal), value.slice(equal + 1));
      else {
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

function help(): string { return `natlang ${NATLANG_CLI_VERSION}

Usage:
  natlang package pack MANIFEST [--root DIR] [--out FILE]
  natlang package verify ARCHIVE [--json]
  natlang package install ARCHIVE [--store DIR] [--json]
  natlang package list [--store DIR] [--json]
  natlang package inspect NAME@VERSION [--store DIR] [--json]
  natlang run PATH [--inputs FILE] [--profile NAME] [--trace FILE] [--json]
  natlang run NAME@VERSION#TARGET [--store DIR] [--profile NAME] [--workspace DIR] [-- ARGS...]
  natlang app list [--store DIR] [--json]
  natlang app run PATH|PACKAGE [--root DIR] [--target NAME] [--profile NAME] [--workspace DIR] [-- ARGS...]
  natlang app doctor PACKAGE [--target NAME] [--profile NAME] [--json]
  natlang setup [--yes] [--json]
  natlang runtime status [--json]
  natlang runtime install [--yes] [--json]
  natlang doctor [--profile NAME] [--json]

Without model configuration, semantic turns lazily start an owned local llama-server
with natlang's release default model. Model settings can override this through
~/.config/natlang/config.json, NATLANG_SERVER, NATLANG_MODEL, NATLANG_MODEL_PATH,
NATLANG_LLAMA_SERVER, NATLANG_RUNTIME_HOME, and NATLANG_API_KEY. NATLANG_HOME
selects the package store.`; }

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
        runTerminalShell, renderTerminalView }) };
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
  const candidate = join(path, 'natlang.json');
  return existsSync(candidate) ? candidate : path;
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
  const host = new NativeNatlangHost();
  const model = modelSession(option(parsed, '--profile'), parsed.options.has('--yes'));
  try {
    const result = await host.run({ source: { kind: 'file', path }, inputs,
      modelTurn: request => model.turn(request),
      tracePath: option(parsed, '--trace') ? resolve(option(parsed, '--trace')!) : undefined,
      timeoutMs: option(parsed, '--timeout') ? Number(option(parsed, '--timeout')) : undefined,
      options: option(parsed, '--seed') ? { seed: { mode: 'derived', root: Number(option(parsed, '--seed')) } } : undefined });
    if (parsed.options.has('--json')) output(result, true);
    else process.stdout.write(JSON.stringify(result.value, null, 2) + '\n');
    if (result.outcome.kind !== 'done') process.stderr.write(`natlang: ${result.outcome.kind}: ${result.outcome.detail}\n`);
    return result.outcome.kind === 'done' ? 0 : 1;
  } finally { host.close(); await model.close(); }
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

function doctorReport(parsed: Parsed, store: NatlangPackageStore,
  targetReport: Record<string, unknown> | null): { report: Record<string, unknown>; okay: boolean } {
  const selected = loadProfile(option(parsed, '--profile'));
  const targetOkay = !targetReport || (!targetReport.engineError &&
    Object.values(targetReport.commands as object).every(Boolean));
  const hasExternal = Boolean(selected.profile.endpoint);
  const partialExternal = Boolean(selected.profile.model) && !hasExternal;
  const local = localModelPrerequisites();
  const modelOkay = hasExternal || (!partialExternal && local.available);
  const okay = Boolean(modelOkay && targetOkay);
  return { okay, report: { ok: okay, node: process.version, packageStore: store.root,
    installedPackages: store.list().length, config: selected.configPath, profile: selected.name,
    modelSource: hasExternal ? 'external' : partialExternal ? 'invalid-partial-profile' : 'managed-local',
    endpoint: selected.profile.endpoint ?? null,
    model: selected.profile.model ?? DEFAULT_LOCAL_MODEL.id,
    modelPath: hasExternal ? null : local.modelPath, modelServer: hasExternal ? null : local.executable,
    modelDownloadAvailable: hasExternal ? null : local.downloadable,
    apiKey: Boolean(process.env[selected.profile.apiKeyEnv ?? 'NATLANG_API_KEY']), target: targetReport } };
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

function applicationRows(store: NatlangPackageStore) {
  return store.list().flatMap(installed => {
    const manifest = store.manifest(`${installed.name}@${installed.version}`);
    return Object.entries(manifest.targets ?? {}).map(([target, definition]) => ({
      name: installed.name.includes('/') ? installed.name.slice(installed.name.lastIndexOf('/') + 1) : installed.name,
      package: installed.name, version: installed.version, target, kind: definition.kind,
      description: definition.description ?? '', specifier: `${installed.name}@${installed.version}#${target}`,
    }));
  }).sort((left, right) => left.name.localeCompare(right.name) ||
    compareVersions(right.version, left.version) || left.target.localeCompare(right.target));
}

export async function main(argv = process.argv.slice(2)): Promise<number> {
  const parsed = parseArgs(argv), json = parsed.options.has('--json');
  if (parsed.options.has('--version')) { output(NATLANG_CLI_VERSION, false); return 0; }
  if (parsed.options.has('--help') || parsed.options.has('-h') || !parsed.words.length) { output(help(), false); return 0; }
  if (parsed.words[0] === 'setup' || parsed.words[0] === 'runtime') {
    const action = parsed.words[0] === 'setup' ? 'ensure' : parsed.words[1] ?? 'status';
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
      if (!selected) throw new Error('llama.cpp installation was declined; run natlang setup when ready');
      outputRuntime(discoverLlamaRuntime(), json); return 0;
    }
    throw new Error('unknown runtime command; run natlang --help');
  }
  if (parsed.words[0] === 'package') {
    const action = parsed.words[1], argument = parsed.words[2];
    if (action === 'pack' && argument) {
      const manifestPath = resolve(argument), root = resolve(option(parsed, '--root') ?? dirname(manifestPath));
      const archive = createPackageArchive(JSON.parse(readFileSync(manifestPath, 'utf8')), root);
      const out = resolve(option(parsed, '--out') ?? `${archive.manifest.name.replace('/', '-')}-${archive.manifest.version}.nlpkg`);
      writePackageArchive(out, archive); output({ archive: out, digest: archive.digest,
        name: archive.manifest.name, version: archive.manifest.version }, json); return 0;
    }
    if (action === 'verify' && argument) {
      const archive = readPackageArchive(resolve(argument)); output({ valid: true, digest: archive.digest,
        name: archive.manifest.name, version: archive.manifest.version, files: archive.files.length }, json); return 0;
    }
    const store = new NatlangPackageStore(option(parsed, '--store'));
    if (action === 'install' && argument) {
      const installed = store.installMany(parsed.words.slice(2).map(path => resolve(path)));
      output(installed.length === 1 ? installed[0] : installed, json); return 0;
    }
    if (action === 'list') { const rows = store.list(); output(json ? rows : rows.map(row => `${row.name}@${row.version} ${row.digest}`).join('\n'), json); return 0; }
    if (action === 'inspect' && argument) { const installed = store.resolve(argument);
      output({ ...installed, manifest: store.manifest(argument) }, json); return 0; }
    throw new Error('unknown package command; run natlang --help');
  }
  if (parsed.words[0] === 'run' && parsed.words[1]) {
    const value = parsed.words[1], path = resolve(value);
    if (existsSync(path)) {
      const manifest = existsSync(join(path, 'natlang.json')) || basename(path) === 'natlang.json' || value.endsWith('.natlang.json');
      return manifest ? runLocalApplication(parsed, value) : runProgramPath(parsed, value);
    }
    return runTarget(parsed, value);
  }
  if (parsed.words[0] === 'app') {
    const action = parsed.words[1], query = parsed.words[2], store = new NatlangPackageStore(option(parsed, '--store'));
    if (action === 'list') {
      const rows = applicationRows(store);
      output(json ? rows : rows.map(row => `${row.name.padEnd(20)} ${row.target.padEnd(12)} ${row.version.padEnd(24)} ${row.description}`).join('\n'), json);
      return 0;
    }
    if ((action === 'run' || action === 'doctor') && query) {
      if (existsSync(resolve(query))) {
        if (action === 'run') return runLocalApplication(parsed, query);
        const store = new NatlangPackageStore(option(parsed, '--store'));
        const { archive, manifestPath } = loadLocalApplication(parsed, query);
        const targetName = selectTarget(archive.manifest, option(parsed, '--target'));
        const { report, okay } = doctorReport(parsed, store,
          inspectTarget(archive.manifest, targetName, manifestPath));
        output(report, json); return okay ? 0 : 1;
      }
      const specifier = applicationSpecifier(store, query, option(parsed, '--target'));
      if (action === 'run') return runTarget(parsed, specifier);
      const forwarded = ['doctor', specifier, ...[...parsed.options.entries()].flatMap(([key, value]) =>
        value === true ? [key] : [key, value])];
      return main(forwarded);
    }
    throw new Error('unknown app command; run natlang --help');
  }
  if (parsed.words[0] === 'doctor') {
    const store = new NatlangPackageStore(option(parsed, '--store'));
    let targetReport: Record<string, unknown> | null = null;
    if (parsed.words[1]) {
      const marker = parsed.words[1].lastIndexOf('#');
      if (marker < 1) throw new Error('doctor target must be NAME@VERSION#TARGET');
      const packageSpecifier = parsed.words[1].slice(0, marker), targetName = parsed.words[1].slice(marker + 1);
      const manifest = store.manifest(packageSpecifier);
      targetReport = inspectTarget(manifest, targetName, packageSpecifier);
    }
    const { report, okay } = doctorReport(parsed, store, targetReport);
    output(report, json); return okay ? 0 : 1;
  }
  throw new Error('unknown command; run natlang --help');
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href)
  main().then(code => { process.exitCode = code; }, error => { process.stderr.write(`natlang: ${error instanceof Error ? error.message : String(error)}\n`); process.exitCode = 1; });
