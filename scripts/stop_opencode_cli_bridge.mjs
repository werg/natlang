#!/usr/bin/env node
/** Stop one owned isolated OpenCode bridge and wait for its lifecycle receipt. */
import { readFile, realpath } from 'node:fs/promises';
import { resolve, relative, isAbsolute } from 'node:path';
import { fileURLToPath } from 'node:url';

const repository = resolve(fileURLToPath(new URL('..', import.meta.url)));
const runs = resolve(repository, 'runs');
const sleep = ms => new Promise(resolveSleep => setTimeout(resolveSleep, ms));

function parse(argv) {
  const options = new Map();
  for (let i = 0; i < argv.length; i += 2) {
    const key = argv[i], value = argv[i + 1];
    if (!['--launcher-pid-file', '--bridge-output', '--timeout-ms'].includes(key) || !value || options.has(key))
      throw new Error(`invalid or duplicate option ${key}`);
    options.set(key, value);
  }
  for (const key of ['--launcher-pid-file', '--bridge-output'])
    if (!options.has(key)) throw new Error(`${key} is required`);
  const timeoutMs = Number(options.get('--timeout-ms') ?? 45_000);
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1 || timeoutMs > 120_000)
    throw new Error('--timeout-ms must be an integer from 1 through 120000');
  return { pidFile: resolve(options.get('--launcher-pid-file')),
    output: resolve(options.get('--bridge-output')), timeoutMs };
}

function inside(parent, path) {
  const rel = relative(parent, path);
  return rel === '' || (rel !== '..' && !rel.startsWith(`..${process.platform === 'win32' ? '\\' : '/'}`) && !isAbsolute(rel));
}

async function alive(pid) {
  try { process.kill(pid, 0); return true; }
  catch (error) { if (error?.code === 'ESRCH') return false; if (error?.code === 'EPERM') return true; throw error; }
}

async function lifecycleAt(output) {
  try { return JSON.parse(await readFile(resolve(output, 'lifecycle.json'), 'utf8')); }
  catch (error) { if (error?.code === 'ENOENT') return null; throw error; }
}

async function verifyOwner(pid, output) {
  let bytes;
  try { bytes = await readFile(`/proc/${pid}/cmdline`); }
  catch (error) { if (error?.code === 'ENOENT') return false; throw error; }
  const args = bytes.toString('utf8').split('\0').filter(Boolean);
  const script = resolve(repository, 'scripts/opencode-cli-loopback-bwrap-launch.mjs');
  const outIndex = args.indexOf('--out');
  return args.includes(script) && outIndex >= 0 && resolve(args[outIndex + 1] ?? '') === output;
}

async function stop({ pidFile, output, timeoutMs }) {
  const runsReal = await realpath(runs);
  if (!inside(runsReal, await realpath(output))) throw new Error('bridge output must resolve under runs/');
  const pidText = (await readFile(pidFile, 'utf8')).trim();
  if (!/^\d+$/.test(pidText) || Number(pidText) < 1) throw new Error('launcher PID file is invalid');
  const pid = Number(pidText);
  let lifecycle = await lifecycleAt(output);
  if (lifecycle?.status === 'stopped') {
    if (await alive(pid)) throw new Error('lifecycle is stopped but the owned launcher PID is still alive');
    return { ok: true, pid, status: 'already-stopped', lifecycle_at: lifecycle.at ?? null };
  }
  if (await alive(pid)) {
    if (!await verifyOwner(pid, output)) throw new Error('launcher PID does not belong to the requested bridge output');
    process.kill(pid, 'SIGTERM');
  } else {
    throw new Error(`launcher exited before lifecycle stopped (status=${lifecycle?.status ?? 'missing'})`);
  }
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    lifecycle = await lifecycleAt(output);
    const launcherAlive = await alive(pid);
    if (lifecycle?.status === 'stopped' && !launcherAlive)
      return { ok: true, pid, status: 'stopped', lifecycle_at: lifecycle.at ?? null };
    if (!launcherAlive && lifecycle?.status !== 'stopped')
      throw new Error(`launcher exited before lifecycle stopped (status=${lifecycle?.status ?? 'missing'})`);
    await sleep(100);
  }
  throw new Error(`timed out waiting for stopped lifecycle (status=${lifecycle?.status ?? 'missing'})`);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  stop(parse(process.argv.slice(2)))
    .then(result => process.stdout.write(`${JSON.stringify(result)}\n`))
    .catch(error => { process.stderr.write(`OpenCode bridge stop refused: ${error.message}\n`); process.exitCode = 1; });
}
