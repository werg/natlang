#!/usr/bin/env node
/** Run the OpenCode SDK collector bridge with bubblewrap filesystem isolation. */
import { access, mkdir, readFile, realpath, stat } from 'node:fs/promises';
import { constants as fsConstants } from 'node:fs';
import { spawn } from 'node:child_process';
import { dirname, isAbsolute, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = dirname(fileURLToPath(import.meta.url));
const repository = resolve(scriptDir, '..');
const workspaceRuns = resolve(repository, 'runs');
const bootstrap = resolve(scriptDir, 'opencode-loopback-bootstrap.mjs');
const credentialFile = '/home/werg/.config/natlang/opencode.env';

async function readConfiguredApiKey() {
  const metadata = await stat(credentialFile).catch(() => null);
  if (!metadata?.isFile() || (metadata.mode & 0o777) !== 0o600)
    throw new Error('configured OpenCode credential file is missing or not mode 0600');
  const content = await readFile(credentialFile, 'utf8');
  let key;
  for (const line of content.split(/\r?\n/)) {
    const value = line.trim();
    if (!value || value.startsWith('#')) continue;
    const match = /^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$/.exec(value);
    if (!match) throw new Error('configured OpenCode credential file has invalid env syntax');
    if (match[1] !== 'OPENCODE_API_KEY') continue;
    if (key !== undefined) throw new Error('configured OpenCode credential file has duplicate API key entries');
    const raw = match[2];
    key = raw.length >= 2 && ((raw.startsWith('"') && raw.endsWith('"')) ||
      (raw.startsWith("'") && raw.endsWith("'"))) ? raw.slice(1, -1) : raw;
  }
  if (!key) throw new Error('configured OpenCode credential file does not contain OPENCODE_API_KEY');
  return key;
}

function parse(argv) {
  const values = {};
  for (let index = 0; index < argv.length; index++) {
    const key = argv[index];
    if (!['--sdk-module', '--client-bin', '--out', '--model', '--max-request-ms', '--max-concurrency'].includes(key) ||
        index + 1 >= argv.length || Object.hasOwn(values, key))
      throw new Error(`invalid or duplicate option ${key}`);
    values[key] = argv[++index];
  }
  for (const key of ['--sdk-module', '--client-bin', '--out'])
    if (!values[key]) throw new Error(`${key} is required`);
  return values;
}

function isWithin(parent, child) {
  const rel = relative(parent, child);
  return rel === '' || (!rel.startsWith(`..${process.platform === 'win32' ? '\\' : '/'}`) && rel !== '..' && !isAbsolute(rel));
}

async function findNodeModulesRoot(path) {
  let current = resolve(path);
  for (;;) {
    if (current.endsWith('/node_modules')) return current;
    const parent = dirname(current);
    if (parent === current) break;
    current = parent;
  }
  throw new Error('SDK and client must be installed beneath node_modules');
}

function ancestors(path) {
  const rows = [];
  let current = resolve(path);
  while (current !== dirname(current)) {
    rows.unshift(current);
    current = dirname(current);
  }
  return rows;
}

export function buildBubblewrapArgs({ sdkModule, clientBin, output, nodeModules, forwardedOptions = {} }) {
  const dirs = new Set([
    '/home', '/home/werg', '/home/werg/natlang', '/home/werg/natlang/scripts',
    ...ancestors(nodeModules).filter(path => path !== '/'),
    ...ancestors(output).filter(path => path !== '/'),
  ]);
  const args = [
    '--die-with-parent', '--new-session', '--unshare-user-try', '--unshare-pid', '--as-pid-1',
    '--unshare-uts', '--unshare-ipc', '--share-net',
    '--ro-bind', '/usr', '/usr', '--ro-bind', '/lib', '/lib', '--ro-bind-try', '/lib64', '/lib64',
    '--ro-bind', '/bin', '/bin', '--ro-bind', '/etc/resolv.conf', '/etc/resolv.conf',
    '--ro-bind', '/etc/hosts', '/etc/hosts', '--ro-bind', '/etc/nsswitch.conf', '/etc/nsswitch.conf',
    '--ro-bind', '/etc/ssl/certs', '/etc/ssl/certs', '--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp'
  ];
  for (const path of dirs) args.push('--dir', path);
  args.push('--ro-bind', resolve(scriptDir), '/home/werg/natlang/scripts');
  args.push('--ro-bind', nodeModules, nodeModules);
  args.push('--bind', output, output, '--chdir', output, process.execPath,
    bootstrap, '--sdk-module', sdkModule, '--client-bin', clientBin, '--out', output);
  for (const key of ['--model', '--max-request-ms', '--max-concurrency'])
    if (key in forwardedOptions) args.push(key, forwardedOptions[key]);
  return args;
}

async function main() {
  const options = parse(process.argv.slice(2));
  const sdkModule = resolve(options['--sdk-module']);
  const clientBin = resolve(options['--client-bin']);
  const output = resolve(options['--out']);
  if (!isWithin(workspaceRuns, output) || output === workspaceRuns)
    throw new Error('--out must be a dedicated directory under this checkout runs/');
  await access(sdkModule, fsConstants.R_OK);
  await access(clientBin, fsConstants.R_OK | fsConstants.X_OK);
  if (!(await stat(sdkModule)).isFile() || !(await stat(clientBin)).isFile())
    throw new Error('SDK module and official client must be files');
  const nodeModules = await findNodeModulesRoot(await realpath(sdkModule));
  const clientNodeModules = await findNodeModulesRoot(await realpath(clientBin));
  if (nodeModules !== clientNodeModules)
    throw new Error('SDK and official client must use the same pinned node_modules tree');
  await mkdir(output, { recursive: false, mode: 0o700 });
  const forwardedOptions = Object.fromEntries(['--model', '--max-request-ms', '--max-concurrency']
    .filter(key => key in options).map(key => [key, options[key]]));
  const args = buildBubblewrapArgs({ sdkModule, clientBin, output, nodeModules, forwardedOptions });
  const env = { PATH: '/usr/bin:/bin', HOME: output, TMPDIR: '/tmp', LANG: 'C.UTF-8',
    OPENCODE_API_KEY: await readConfiguredApiKey() };
  const child = spawn('/usr/bin/bwrap', args, { env, stdio: 'inherit' });
  const forward = signal => { if (child.exitCode === null) child.kill(signal); };
  process.on('SIGINT', forward);
  process.on('SIGTERM', forward);
  child.once('error', error => {
    process.stderr.write(`Bubblewrap launch failed (${error.code ?? error.name}).\n`);
    process.exitCode = 1;
  });
  child.once('exit', (code, signal) => {
    process.removeListener('SIGINT', forward);
    process.removeListener('SIGTERM', forward);
    process.exitCode = code ?? (signal ? 1 : 0);
  });
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url))
  main().catch(error => {
    process.stderr.write(`Bubblewrap launch refused (${error?.name ?? 'Error'}).\n`);
    process.exitCode = 1;
  });
