#!/usr/bin/env node
/** Run the official OpenCode CLI server + CLI-backed Natlang adapter in bwrap. */
import { access, mkdir, readFile, realpath, stat, writeFile } from 'node:fs/promises';
import { constants as fsConstants } from 'node:fs';
import { spawn } from 'node:child_process';
import { dirname, isAbsolute, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = dirname(fileURLToPath(import.meta.url));
const repository = resolve(scriptDir, '..');
const workspaceRuns = resolve(repository, 'runs');
const bootstrap = resolve(scriptDir, 'opencode-cli-loopback-bootstrap.mjs');
const credentialFile = '/home/werg/.config/natlang/opencode.env';
let launchPhase = 'argument-validation';

async function readConfiguredApiKey() {
  const metadata = await stat(credentialFile).catch(() => null);
  if (!metadata?.isFile() || (metadata.mode & 0o777) !== 0o600) throw new Error('configured credential is missing or not mode 0600');
  const content = await readFile(credentialFile, 'utf8');
  let key;
  for (const line of content.split(/\r?\n/)) {
    const value = line.trim();
    if (!value || value.startsWith('#')) continue;
    const match = /^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$/.exec(value);
    if (!match) throw new Error('configured credential file has invalid env syntax');
    if (match[1] !== 'OPENCODE_API_KEY') continue;
    if (key !== undefined) throw new Error('configured credential file has duplicate API key entries');
    const raw = match[2];
    key = raw.length >= 2 && ((raw.startsWith('"') && raw.endsWith('"')) || (raw.startsWith("'") && raw.endsWith("'")))
      ? raw.slice(1, -1) : raw;
  }
  if (!key) throw new Error('configured credential file lacks OPENCODE_API_KEY');
  return key;
}

function parse(argv) {
  const values = {};
  for (let i = 0; i < argv.length; i++) {
    const key = argv[i];
    if (!['--sdk-module', '--client-bin', '--out', '--model', '--variant', '--max-request-ms', '--tool-surface'].includes(key) ||
        i + 1 >= argv.length || Object.hasOwn(values, key)) throw new Error(`invalid or duplicate option ${key}`);
    values[key] = argv[++i];
  }
  for (const key of ['--sdk-module', '--client-bin', '--out']) if (!values[key]) throw new Error(`${key} is required`);
  values['--model'] ??= 'ling-3.1-flash-free';
  values['--max-request-ms'] ??= '180000';
  if (values['--tool-surface'] !== undefined && !['standard', 'natlang-only'].includes(values['--tool-surface']))
    throw new Error('--tool-surface must be standard or natlang-only');
  if (!/^[a-z0-9][a-z0-9._-]*-free$/i.test(values['--model'])) throw new Error('--model must name an explicit free OpenCode model');
  if (values['--variant'] !== undefined && (values['--model'] !== 'step-5-preview-free' ||
      !['low', 'medium', 'high'].includes(values['--variant'])))
    throw new Error('--variant is supported only for step-5-preview-free with low, medium, or high');
  if (!/^\d+$/.test(values['--max-request-ms']) || Number(values['--max-request-ms']) < 1) throw new Error('--max-request-ms must be positive');
  return values;
}

function isWithin(parent, child) {
  const rel = relative(parent, child);
  return rel === '' || (!rel.startsWith(`..${process.platform === 'win32' ? '\\' : '/'}`) && rel !== '..' && !isAbsolute(rel));
}
async function findNodeModulesRoot(path) {
  let current = resolve(path);
  for (;;) { if (current.endsWith('/node_modules')) return current; const parent = dirname(current); if (parent === current) break; current = parent; }
  throw new Error('SDK and official CLI must be installed beneath node_modules');
}
function ancestors(path) { const rows = []; let current = resolve(path); while (current !== dirname(current)) { rows.unshift(current); current = dirname(current); } return rows; }

export function buildBubblewrapArgs({ sdkModule, clientBin, output, nodeModules, forwardedOptions = {} }) {
  const dirs = new Set(['/home', '/home/werg', '/home/werg/natlang', '/home/werg/natlang/scripts',
    ...ancestors(nodeModules).filter(path => path !== '/'), ...ancestors(output).filter(path => path !== '/')]);
  const args = ['--die-with-parent', '--new-session', '--unshare-user-try', '--unshare-pid', '--as-pid-1',
    '--unshare-uts', '--unshare-ipc', '--share-net', '--ro-bind', '/usr', '/usr', '--ro-bind', '/lib', '/lib',
    '--ro-bind-try', '/lib64', '/lib64', '--ro-bind', '/bin', '/bin', '--ro-bind', '/etc/resolv.conf', '/etc/resolv.conf',
    '--ro-bind', '/etc/hosts', '/etc/hosts', '--ro-bind', '/etc/nsswitch.conf', '/etc/nsswitch.conf',
    '--ro-bind', '/etc/ssl/certs', '/etc/ssl/certs', '--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp'];
  for (const path of dirs) args.push('--dir', path);
  args.push('--ro-bind', resolve(scriptDir), '/home/werg/natlang/scripts', '--ro-bind', nodeModules, nodeModules,
    '--bind', output, output, '--chdir', output, process.execPath, bootstrap,
    '--sdk-module', sdkModule, '--client-bin', clientBin, '--out', output);
  for (const key of ['--model', '--variant', '--max-request-ms', '--tool-surface'])
    if (key in forwardedOptions) args.push(key, forwardedOptions[key]);
  return args;
}

async function main() {
  const options = parse(process.argv.slice(2));
  const sdkModule = resolve(options['--sdk-module']), clientBin = resolve(options['--client-bin']), output = resolve(options['--out']);
  launchPhase = 'output-path-validation';
  if (!isWithin(workspaceRuns, output) || output === workspaceRuns) throw new Error('--out must be a dedicated directory under runs/');
  launchPhase = 'official-client-validation';
  await access(sdkModule, fsConstants.R_OK); await access(clientBin, fsConstants.R_OK | fsConstants.X_OK);
  if (!(await stat(sdkModule)).isFile() || !(await stat(clientBin)).isFile()) throw new Error('SDK module and official CLI must be files');
  const nodeModules = await findNodeModulesRoot(await realpath(sdkModule));
  if (nodeModules !== await findNodeModulesRoot(await realpath(clientBin))) throw new Error('SDK and CLI must use the same pinned node_modules tree');
  launchPhase = 'output-parent-validation';
  const runsReal = await realpath(workspaceRuns);
  let existingParent = dirname(output);
  for (;;) {
    try { await access(existingParent); break; }
    catch (error) {
      if (error?.code !== 'ENOENT') throw error;
      const parent = dirname(existingParent);
      if (parent === existingParent) throw error;
      existingParent = parent;
    }
  }
  if (!isWithin(runsReal, await realpath(existingParent)))
    throw new Error('--out parent resolves outside runs/');
  launchPhase = 'output-parent-creation';
  await mkdir(dirname(output), { recursive: true, mode: 0o700 });
  if (!isWithin(runsReal, await realpath(dirname(output))))
    throw new Error('--out parent resolves outside runs/');
  launchPhase = 'fresh-output-leaf-creation';
  await mkdir(output, { recursive: false, mode: 0o700 });
  launchPhase = 'credential-validation';
  const apiKey = await readConfiguredApiKey();
  const forwarded = Object.fromEntries(['--model', '--variant', '--max-request-ms', '--tool-surface'].filter(key => key in options).map(key => [key, options[key]]));
  launchPhase = 'bubblewrap-argument-build';
  const args = buildBubblewrapArgs({ sdkModule, clientBin, output, nodeModules, forwardedOptions: forwarded });
  launchPhase = 'bubblewrap-spawn';
  const child = spawn('/usr/bin/bwrap', args, { env: { PATH: '/usr/bin:/bin', HOME: output, TMPDIR: '/tmp',
    LANG: 'C.UTF-8', OPENCODE_API_KEY: apiKey }, stdio: 'inherit' });
  let shutdownRequested = false;
  const forward = signal => {
    if (shutdownRequested || child.exitCode !== null) return;
    shutdownRequested = true;
    void writeFile(resolve(output, 'shutdown.request'), `${signal}\n`, { flag: 'w', mode: 0o600 }).then(async () => {
      const closed = new Promise(resolveClose => child.once('close', resolveClose));
      if (await Promise.race([closed.then(() => true), new Promise(resolveWait => setTimeout(() => resolveWait(false), 20_000))])) return;
      child.kill('SIGTERM');
      if (await Promise.race([closed.then(() => true), new Promise(resolveWait => setTimeout(() => resolveWait(false), 5_000))])) return;
      child.kill('SIGKILL');
    }).catch(() => child.kill('SIGTERM'));
  };
  process.on('SIGINT', forward); process.on('SIGTERM', forward);
  child.once('error', error => { process.stderr.write(`Bubblewrap launch failed (${error.code ?? error.name}).\n`); process.exitCode = 1; });
  child.once('exit', (code, signal) => { process.removeListener('SIGINT', forward); process.removeListener('SIGTERM', forward); process.exitCode = code ?? (signal ? 1 : 0); });
}
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url))
  main().catch(error => {
    const code = typeof error?.code === 'string' ? `/${error.code}` : '';
    process.stderr.write(`CLI bwrap launch refused at ${launchPhase} (${error?.name ?? 'Error'}${code}).\n`);
    process.exitCode = 1;
  });
