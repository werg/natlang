import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { accessSync, chmodSync, constants, createReadStream, createWriteStream, existsSync, lstatSync,
  mkdirSync, mkdtempSync, readFileSync, readdirSync, renameSync, rmSync, writeFileSync } from 'node:fs';
import { homedir, platform } from 'node:os';
import { basename, delimiter, dirname, join, relative, resolve, sep } from 'node:path';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { LLAMA_RUNTIME_RELEASE } from '../llama-runtime-release.js';

export type LlamaRuntimeArtifact = { key: string; platform: NodeJS.Platform; arch: string;
  backend: 'cpu' | 'metal'; archive: 'tar.gz' | 'zip'; url: string; bytes: number; sha256: string };
export type LlamaRuntimeRelease = { version: string; build: number; commit: string;
  compatible: { minimumVersion: string; maximumVersionExclusive: string; minimumBuild: number };
  artifacts: readonly LlamaRuntimeArtifact[] };
export type LlamaServerInspection = { path: string; source: 'explicit' | 'managed' | 'path';
  compatible: boolean; version: string | null; build: number | null; commit: string | null;
  reason: string | null; output: string };
export type LlamaRuntimeDiscovery = { selected: LlamaServerInspection | null;
  candidates: LlamaServerInspection[]; artifact: LlamaRuntimeArtifact | null; runtimeRoot: string };

export { LLAMA_RUNTIME_RELEASE } from '../llama-runtime-release.js';

function semver(value: string): [number, number, number] | null {
  const match = /^(\d+)\.(\d+)\.(\d+)(?:-[0-9A-Za-z.-]+)?$/.exec(value);
  return match ? [Number(match[1]), Number(match[2]), Number(match[3])] : null;
}
function compare(left: [number, number, number], right: [number, number, number]): number {
  for (let index = 0; index < 3; index++) if (left[index] !== right[index]) return left[index]! - right[index]!;
  return 0;
}
export function isCompatibleLlamaVersion(version: string, build: number,
  release: LlamaRuntimeRelease = LLAMA_RUNTIME_RELEASE): boolean {
  const actual = semver(version), minimum = semver(release.compatible.minimumVersion),
    maximum = semver(release.compatible.maximumVersionExclusive);
  return Boolean(actual && minimum && maximum && compare(actual, minimum) >= 0 && compare(actual, maximum) < 0 &&
    build >= release.compatible.minimumBuild);
}

function commandPath(command: string, environment: NodeJS.ProcessEnv): string | null {
  if (command.includes('/') || command.includes('\\')) return existsSync(resolve(command)) ? resolve(command) : null;
  const extensions = process.platform === 'win32'
    ? (environment.PATHEXT ?? '.EXE;.CMD;.BAT;.COM').split(';') : [''];
  for (const directory of (environment.PATH ?? '').split(delimiter).filter(Boolean)) {
    for (const extension of extensions) {
      const candidate = resolve(directory, command + extension.toLowerCase());
      try { accessSync(candidate, process.platform === 'win32' ? constants.F_OK : constants.X_OK); return candidate; }
      catch { /* keep searching */ }
    }
  }
  return null;
}

export function inspectLlamaServer(path: string, source: LlamaServerInspection['source'],
  release: LlamaRuntimeRelease = LLAMA_RUNTIME_RELEASE, environment: NodeJS.ProcessEnv = process.env): LlamaServerInspection {
  const resolved = resolve(path);
  if (!existsSync(resolved)) return { path: resolved, source, compatible: false, version: null, build: null,
    commit: null, reason: 'executable does not exist', output: '' };
  const result = spawnSync(resolved, ['--version'], { encoding: 'utf8', env: environment, timeout: 10000 });
  const output = `${result.stdout ?? ''}${result.stderr ?? ''}`.trim();
  if (result.error || result.status !== 0) return { path: resolved, source, compatible: false, version: null,
    build: null, commit: null, reason: result.error?.message ?? `version probe exited ${result.status}`, output };
  const match = /version:\s*(\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?)\s*\(build\s+(\d+),\s*commit\s+([0-9a-f]+)\)/i.exec(output);
  if (!match) return { path: resolved, source, compatible: false, version: null, build: null, commit: null,
    reason: 'unrecognized llama-server version output', output };
  const version = match[1]!, build = Number(match[2]), commit = match[3]!;
  const compatible = isCompatibleLlamaVersion(version, build, release);
  return { path: resolved, source, compatible, version, build, commit,
    reason: compatible ? null : `needs >=${release.compatible.minimumVersion} build ${release.compatible.minimumBuild} and <${release.compatible.maximumVersionExclusive}`,
    output };
}

export function defaultNatlangRuntimeDirectory(environment: NodeJS.ProcessEnv = process.env): string {
  if (environment.NATLANG_RUNTIME_HOME) return resolve(environment.NATLANG_RUNTIME_HOME);
  if (platform() === 'win32') return join(environment.LOCALAPPDATA ?? homedir(), 'natlang', 'runtimes');
  if (platform() === 'darwin') return join(homedir(), 'Library', 'Application Support', 'natlang', 'runtimes');
  return join(environment.XDG_DATA_HOME ?? join(homedir(), '.local', 'share'), 'natlang', 'runtimes');
}

function artifactForCurrentPlatform(release: LlamaRuntimeRelease): LlamaRuntimeArtifact | null {
  return release.artifacts.find(item => item.platform === process.platform && item.arch === process.arch) ?? null;
}
function managedRecordPath(root: string, release: LlamaRuntimeRelease, artifact: LlamaRuntimeArtifact): string {
  return join(root, 'llama.cpp', `b${release.build}`, artifact.key, 'runtime.json');
}
function managedExecutable(root: string, release: LlamaRuntimeRelease, artifact: LlamaRuntimeArtifact): string | null {
  const record = managedRecordPath(root, release, artifact);
  if (!existsSync(record)) return null;
  try {
    const value = JSON.parse(readFileSync(record, 'utf8')) as { executable?: string };
    if (!value.executable) return null;
    const base = dirname(record), path = resolve(base, value.executable), child = relative(base, path);
    if (!child || child === '..' || child.startsWith(`..${sep}`) || resolve(child) === child) return null;
    return existsSync(path) ? path : null;
  } catch { return null; }
}

export function discoverLlamaRuntime(environment: NodeJS.ProcessEnv = process.env,
  release: LlamaRuntimeRelease = LLAMA_RUNTIME_RELEASE): LlamaRuntimeDiscovery {
  const runtimeRoot = defaultNatlangRuntimeDirectory(environment), artifact = artifactForCurrentPlatform(release);
  const candidates: LlamaServerInspection[] = [];
  if (environment.NATLANG_LLAMA_SERVER) {
    const explicit = commandPath(environment.NATLANG_LLAMA_SERVER, environment) ?? resolve(environment.NATLANG_LLAMA_SERVER);
    candidates.push(inspectLlamaServer(explicit, 'explicit', release, environment));
    return { selected: candidates[0]!.compatible ? candidates[0]! : null, candidates, artifact, runtimeRoot };
  }
  if (artifact) {
    const managed = managedExecutable(runtimeRoot, release, artifact);
    if (managed) candidates.push(inspectLlamaServer(managed, 'managed', release, environment));
  }
  const ambient = commandPath('llama-server', environment);
  if (ambient && !candidates.some(item => item.path === ambient))
    candidates.push(inspectLlamaServer(ambient, 'path', release, environment));
  return { selected: candidates.find(item => item.compatible) ?? null, candidates, artifact, runtimeRoot };
}

async function hashFile(path: string): Promise<string> {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest('hex');
}
function findExecutable(root: string): string | null {
  const expected = process.platform === 'win32' ? 'llama-server.exe' : 'llama-server';
  for (const name of readdirSync(root)) {
    const path = join(root, name), status = lstatSync(path);
    if (status.isDirectory()) { const nested = findExecutable(path); if (nested) return nested; }
    else if (status.isFile() && name === expected) return path;
  }
  return null;
}
function extractArchive(archive: string, destination: string): void {
  const result = spawnSync('tar', ['-xf', archive, '-C', destination], { encoding: 'utf8', timeout: 120000 });
  if (result.status !== 0) throw new Error(`could not extract the llama.cpp archive with tar: ${(result.stderr || result.error?.message || '').trim()}`);
}

export async function installManagedLlamaRuntime(options: { environment?: NodeJS.ProcessEnv;
  release?: LlamaRuntimeRelease; error?: NodeJS.WritableStream } = {}): Promise<LlamaServerInspection> {
  const environment = options.environment ?? process.env, release = options.release ?? LLAMA_RUNTIME_RELEASE;
  const root = defaultNatlangRuntimeDirectory(environment), artifact = artifactForCurrentPlatform(release);
  if (!artifact) throw new Error(`no managed llama.cpp runtime is published for ${process.platform}-${process.arch}; set NATLANG_LLAMA_SERVER`);
  const existing = managedExecutable(root, release, artifact);
  if (existing) {
    const inspected = inspectLlamaServer(existing, 'managed', release, environment);
    if (inspected.compatible) return inspected;
  }
  mkdirSync(root, { recursive: true });
  const temporary = mkdtempSync(join(root, '.llama-install-'));
  const archive = join(temporary, basename(new URL(artifact.url).pathname)), content = join(temporary, 'content');
  mkdirSync(content);
  options.error?.write(`natlang: downloading llama.cpp ${release.version} ${artifact.key} (${Math.ceil(artifact.bytes / 1_000_000)} MB)\n`);
  try {
    const response = await fetch(artifact.url, { redirect: 'follow' });
    if (!response.ok || !response.body) throw new Error(`llama.cpp download failed: HTTP ${response.status}`);
    await pipeline(Readable.from(response.body as AsyncIterable<Uint8Array>), createWriteStream(archive, { flags: 'wx' }));
    const status = lstatSync(archive);
    if (status.size !== artifact.bytes || await hashFile(archive) !== artifact.sha256)
      throw new Error('downloaded llama.cpp archive failed its size or SHA-256 check');
    extractArchive(archive, content);
    const executable = findExecutable(content);
    if (!executable) throw new Error('downloaded llama.cpp archive does not contain llama-server');
    if (process.platform !== 'win32') chmodSync(executable, 0o755);
    const relative = executable.slice(content.length + 1).split('\\').join('/');
    writeFileSync(join(content, 'runtime.json'), JSON.stringify({ version: release.version, build: release.build,
      commit: release.commit, artifact: artifact.key, sha256: artifact.sha256, executable: relative }, null, 2) + '\n');
    const destination = dirname(managedRecordPath(root, release, artifact));
    mkdirSync(dirname(destination), { recursive: true });
    try { renameSync(content, destination); }
    catch (failure) { if (!existsSync(destination)) throw failure; }
    const installed = managedExecutable(root, release, artifact);
    if (!installed) throw new Error('managed llama.cpp runtime was not installed correctly');
    const inspected = inspectLlamaServer(installed, 'managed', release, environment);
    if (!inspected.compatible) throw new Error(`installed llama-server is incompatible: ${inspected.reason}`);
    return inspected;
  } finally { rmSync(temporary, { recursive: true, force: true }); }
}

export function describeLlamaRuntime(discovery: LlamaRuntimeDiscovery): string {
  if (discovery.selected) return `${discovery.selected.source} ${discovery.selected.path} (${discovery.selected.version}, build ${discovery.selected.build})`;
  const incompatible = discovery.candidates.map(item => `${item.path}: ${item.reason}`).join('; ');
  return incompatible || `no llama-server found for ${process.platform}-${process.arch}`;
}
