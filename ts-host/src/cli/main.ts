#!/usr/bin/env node
import { existsSync, readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { dirname, join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { createPackageArchive, NatlangPackageStore, parsePackageArchive, readPackageArchive,
  writePackageArchive, defaultNatlangConfigDirectory, defaultNatlangStateDirectory } from '../package/index.js';
import { satisfiesVersion } from '../package/store.js';
import type { PackageTargetContext, PackageTargetFactory } from '../package/target.js';
import { openAICompatibleModelTurn } from '../model/index.js';
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
  const boolean = new Set(['--json', '--plain', '--no-color', '--help', '-h', '--version']);
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
  natlang run NAME@VERSION#TARGET [--store DIR] [--profile NAME] [--workspace DIR] [-- ARGS...]
  natlang doctor [--profile NAME] [--json]

Model settings come from ~/.config/natlang/config.json, NATLANG_SERVER,
NATLANG_MODEL, and NATLANG_API_KEY. NATLANG_HOME selects the package store.`; }

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
function modelDriver(profileName?: string) {
  const { profile } = loadProfile(profileName);
  if (!profile.endpoint || !profile.model) return undefined;
  const apiKey = process.env[profile.apiKeyEnv ?? 'NATLANG_API_KEY'];
  return openAICompatibleModelTurn({ endpoint: profile.endpoint, model: profile.model, apiKey,
    headers: profile.headers, request: profile.request, toolAliases: { call: 'call_function' } });
}

async function runTarget(parsed: Parsed, specifier: string): Promise<number> {
  const marker = specifier.lastIndexOf('#');
  if (marker < 1 || marker === specifier.length - 1) throw new Error('run specifier must be NAME@VERSION#TARGET');
  const packageSpecifier = specifier.slice(0, marker), targetName = specifier.slice(marker + 1);
  const store = new NatlangPackageStore(option(parsed, '--store'));
  const installed = store.resolve(packageSpecifier), manifest = store.manifest(packageSpecifier);
  checkEngines(manifest.engines);
  const target = manifest.targets?.[targetName];
  if (!target) throw new Error(`unknown target ${targetName} in ${packageSpecifier}`);
  const stateDirectory = resolve(option(parsed, '--state') ?? join(defaultNatlangStateDirectory(),
    encodeURIComponent(installed.name), installed.version, targetName));
  const traceDirectory = resolve(option(parsed, '--traces') ?? join(stateDirectory, 'traces'));
  const workspace = resolve(option(parsed, '--workspace') ?? '.');
  const entry = join(installed.root, ...target.entry.split('/'));
  const module = await import(pathToFileURL(entry).href) as Record<string, unknown>;
  const factoryName = target.export ?? 'createTarget', factory = module[factoryName];
  if (typeof factory !== 'function') throw new Error(`target entry does not export ${factoryName}()`);
  const context: PackageTargetContext = { package: installed, targetName, target, workspace,
    stateDirectory, traceDirectory, args: parsed.rest, io: { input: process.stdin, output: process.stdout,
      error: process.stderr, color: !parsed.options.has('--plain') && !parsed.options.has('--no-color') && Boolean(process.stdout.isTTY) },
    modelTurn: modelDriver(option(parsed, '--profile')), runtime: Object.freeze({ NativeNatlangHost,
      TypeScriptEnvironment, TerminalNatlangApplication, TerminalSessionStore, TerminalEventQueue,
      runTerminalShell, renderTerminalView }) };
  const executable = await (factory as PackageTargetFactory)(context);
  if (!executable || typeof executable.run !== 'function') throw new Error('target factory must return an executable with run()');
  try { return Number(await executable.run() ?? 0); }
  finally { await executable.close?.(); }
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

export async function main(argv = process.argv.slice(2)): Promise<number> {
  const parsed = parseArgs(argv), json = parsed.options.has('--json');
  if (parsed.options.has('--version')) { output(NATLANG_CLI_VERSION, false); return 0; }
  if (parsed.options.has('--help') || parsed.options.has('-h') || !parsed.words.length) { output(help(), false); return 0; }
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
  if (parsed.words[0] === 'run' && parsed.words[1]) return runTarget(parsed, parsed.words[1]);
  if (parsed.words[0] === 'doctor') {
    const selected = loadProfile(option(parsed, '--profile')), store = new NatlangPackageStore(option(parsed, '--store'));
    let targetReport: Record<string, unknown> | null = null;
    if (parsed.words[1]) {
      const marker = parsed.words[1].lastIndexOf('#');
      if (marker < 1) throw new Error('doctor target must be NAME@VERSION#TARGET');
      const packageSpecifier = parsed.words[1].slice(0, marker), targetName = parsed.words[1].slice(marker + 1);
      const manifest = store.manifest(packageSpecifier), target = manifest.targets?.[targetName];
      if (!target) throw new Error(`unknown target ${targetName} in ${packageSpecifier}`);
      let engineError: string | null = null;
      try { checkEngines(manifest.engines); } catch (error) { engineError = error instanceof Error ? error.message : String(error); }
      const commands = Object.fromEntries((target.commands ?? []).map(command => [command, commandAvailable(command)]));
      targetReport = { package: packageSpecifier, target: targetName, authority: target.authority ?? [],
        commands, engines: manifest.engines ?? {}, engineError };
    }
    const targetOkay = !targetReport || (!targetReport.engineError && Object.values(targetReport.commands as object).every(Boolean));
    const report = { ok: Boolean(selected.profile.endpoint && selected.profile.model && targetOkay), node: process.version,
      packageStore: store.root, installedPackages: store.list().length, config: selected.configPath,
      profile: selected.name, endpoint: selected.profile.endpoint ?? null, model: selected.profile.model ?? null,
      apiKey: Boolean(process.env[selected.profile.apiKeyEnv ?? 'NATLANG_API_KEY']), target: targetReport };
    output(report, json); return report.ok ? 0 : 1;
  }
  throw new Error('unknown command; run natlang --help');
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href)
  main().then(code => { process.exitCode = code; }, error => { process.stderr.write(`natlang: ${error instanceof Error ? error.message : String(error)}\n`); process.exitCode = 1; });
