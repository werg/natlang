import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { accessSync, chmodSync, constants, cpSync, createReadStream, createWriteStream, existsSync, lstatSync,
  mkdirSync, mkdtempSync, readFileSync, readdirSync, renameSync, rmSync, writeFileSync } from 'node:fs';
import { cpus, homedir, platform } from 'node:os';
import { basename, delimiter, dirname, join, relative, resolve, sep } from 'node:path';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { LLAMA_RUNTIME_RELEASE } from '../llama-runtime-release.js';
import { LLAMA_FORK_SOURCE } from '../llama-fork-source.js';

export type LlamaRuntimeArtifact = { key: string; platform: NodeJS.Platform; arch: string;
  backend: 'cpu' | 'metal'; archive: 'tar.gz' | 'zip'; url: string; bytes: number; sha256: string };
export type LlamaRuntimeRelease = { version: string; build: number; commit: string;
  compatible: { minimumVersion: string; maximumVersionExclusive: string; minimumBuild: number };
  artifacts: readonly LlamaRuntimeArtifact[] };
export type LlamaServerInspection = { path: string; source: 'explicit' | 'managed' | 'fork' | 'path';
  compatible: boolean; version: string | null; build: number | null; commit: string | null;
  reason: string | null; output: string };
export type LlamaRuntimeDiscovery = { selected: LlamaServerInspection | null;
  candidates: LlamaServerInspection[]; artifact: LlamaRuntimeArtifact | null; runtimeRoot: string };

export { LLAMA_RUNTIME_RELEASE } from '../llama-runtime-release.js';
export { LLAMA_FORK_SOURCE } from '../llama-fork-source.js';

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
  // A source build of the pinned fork commit is that commit, whatever build number its checkout counted.
  const compatible = source === 'fork' ? commit.length >= 7 && LLAMA_FORK_SOURCE.commit.startsWith(commit) :
    isCompatibleLlamaVersion(version, build, release);
  if (source === 'fork') return { path: resolved, source, compatible, version, build, commit,
    reason: compatible ? null : `a build of fork commit ${commit}, not the pinned ${LLAMA_FORK_SOURCE.commit.slice(0, 9)}; run natlang runtime build`, output };
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

const forkDirectory = (root: string) => join(root, 'llama.cpp', `fork-${LLAMA_FORK_SOURCE.commit.slice(0, 12)}-${process.platform}-${process.arch}`);
function forkExecutable(root: string): string | null {
  const executable = join(forkDirectory(root), 'bin', process.platform === 'win32' ? 'llama-server.exe' : 'llama-server');
  return existsSync(join(forkDirectory(root), 'runtime.json')) && existsSync(executable) ? executable : null;
}

/**
 * `natlang runtime build`: fetch the pinned commit of natlang's llama.cpp fork and compile llama-server for this machine
 * (native CPU flags; CUDA when nvcc is found or `cuda` is set; Metal on macOS) into the runtime directory, where
 * discovery prefers it. Needs git, CMake and a C++ compiler.
 */
export async function buildForkLlamaRuntime(options: { environment?: NodeJS.ProcessEnv; error?: NodeJS.WritableStream;
  cuda?: boolean; jobs?: number } = {}): Promise<LlamaServerInspection> {
  const environment = options.environment ?? process.env, log = options.error ?? process.stderr;
  const root = defaultNatlangRuntimeDirectory(environment), destination = forkDirectory(root);
  for (const tool of ['git', 'cmake']) if (!commandPath(tool, environment)) throw new Error(`natlang runtime build needs ${tool} on PATH`);
  const cuda = options.cuda ?? Boolean(commandPath('nvcc', environment));
  const run = (command: string, args: string[], cwd: string) => {
    log.write(`natlang: ${command} ${args.join(' ')}\n`);
    const result = spawnSync(command, args, { cwd, env: environment, stdio: ['ignore', 'inherit', 'inherit'] });
    if (result.status !== 0) throw new Error(`${command} ${args[0]} failed (${result.error?.message ?? `exit ${result.status}`})`);
  };
  mkdirSync(root, { recursive: true });
  const work = mkdtempSync(join(root, '.fork-build-'));
  try {
    const source = join(work, 'llama.cpp');
    mkdirSync(source);
    run('git', ['init', '-q'], source);
    run('git', ['fetch', '-q', '--depth', '1', `https://github.com/${LLAMA_FORK_SOURCE.repository}`, LLAMA_FORK_SOURCE.commit], source);
    run('git', ['checkout', '-q', 'FETCH_HEAD'], source);
    const rpath = process.platform === 'darwin' ? '@loader_path' : '$ORIGIN';
    run('cmake', ['-S', '.', '-B', 'build', '-DCMAKE_BUILD_TYPE=Release', '-DLLAMA_CURL=OFF', '-DLLAMA_BUILD_TESTS=OFF',
      // natlang does not serve llama.cpp's web UI, which would otherwise be fetched from an unpinned "latest" bucket.
      '-DLLAMA_BUILD_EXAMPLES=OFF', '-DLLAMA_USE_PREBUILT_UI=OFF', '-DLLAMA_BUILD_UI=OFF', `-DCMAKE_INSTALL_RPATH=${rpath}`, '-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON',
      ...(cuda ? ['-DGGML_CUDA=ON', '-DCMAKE_CUDA_ARCHITECTURES=native'] : [])], source);
    run('cmake', ['--build', 'build', '--config', 'Release', '--target', 'llama-server', '-j', String(options.jobs ?? Math.max(1, cpus().length))], source);
    const built = [join(source, 'build', 'bin'), join(source, 'build', 'bin', 'Release')].find(path => existsSync(join(path,
      process.platform === 'win32' ? 'llama-server.exe' : 'llama-server')));
    if (!built) throw new Error('the fork build produced no llama-server');
    const staged = join(work, 'runtime');
    cpSync(built, join(staged, 'bin'), { recursive: true });
    writeFileSync(join(staged, 'runtime.json'), JSON.stringify({ fork: true, repository: LLAMA_FORK_SOURCE.repository,
      commit: LLAMA_FORK_SOURCE.commit, backend: cuda ? 'cuda' : process.platform === 'darwin' ? 'metal' : 'cpu',
      builtAt: new Date().toISOString() }, null, 2) + '\n');
    rmSync(destination, { recursive: true, force: true });
    mkdirSync(dirname(destination), { recursive: true });
    renameSync(staged, destination);
    const inspected = inspectLlamaServer(join(destination, 'bin', process.platform === 'win32' ? 'llama-server.exe' : 'llama-server'), 'fork',
      LLAMA_RUNTIME_RELEASE, environment);
    if (!inspected.compatible) throw new Error(`the built llama-server is not the pinned fork commit: ${inspected.reason ?? inspected.output}`);
    return inspected;
  } finally { rmSync(work, { recursive: true, force: true }); }
}

export function discoverLlamaRuntime(environment: NodeJS.ProcessEnv = process.env,
  release: LlamaRuntimeRelease = LLAMA_RUNTIME_RELEASE): LlamaRuntimeDiscovery {
  const runtimeRoot = defaultNatlangRuntimeDirectory(environment), artifact = artifactForCurrentPlatform(release);
  const candidates: LlamaServerInspection[] = [];
  if (environment.NATLANG_LLAMA_SERVER) {
    const explicit = commandPath(environment.NATLANG_LLAMA_SERVER, environment) ?? resolve(environment.NATLANG_LLAMA_SERVER);
    const inspected = inspectLlamaServer(explicit, 'explicit', release, environment);
    candidates.push(inspected);
    // An explicitly chosen build (a CUDA build, the Neuralese fork) runs even when it predates the pinned release; the
    // reason stays as a warning (LFM3, the default model, needs build 11938 or later).
    const selected = inspected.compatible ? inspected : inspected.version ? { ...inspected, compatible: true,
      reason: `older than natlang's pinned llama.cpp ${release.version} (${inspected.reason}); used because it was chosen explicitly` } : null;
    return { selected, candidates, artifact, runtimeRoot };
  }
  // natlang's fork, built here by `natlang runtime build`, comes before the downloaded upstream release.
  const fork = forkExecutable(runtimeRoot);
  if (fork) candidates.push(inspectLlamaServer(fork, 'fork', release, environment));
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
