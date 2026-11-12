/**
 * The supervisor of a shared managed model server (llama/shared.ts), run detached by the Node host:
 * `node supervisor.js SPEC.json`. It starts llama-server, publishes its record once the server answers, and stops it
 * when its lease directory has held no live lease for the idle time while no slot was processing, when the server
 * exits, or on SIGTERM/SIGINT. A start failure is written to `<key>.failed` for the waiting starter.
 */
import { spawn } from 'node:child_process';
import { existsSync, openSync, readFileSync, rmdirSync, rmSync, writeFileSync } from 'node:fs';
import { liveLeases, readSharedServerAt, sharedPaths, writeSharedServer, type SupervisedServer } from './shared.js';

const specPath = process.argv[2];
if (!specPath) { process.stderr.write('usage: supervisor.js SPEC.json\n'); process.exit(2); }
const spec = JSON.parse(readFileSync(specPath, 'utf8')) as SupervisedServer;
rmSync(specPath, { force: true });
const paths = sharedPaths(spec.directory, spec.key);
const log = openSync(paths.log, 'w');
const child = spawn(spec.command, [...spec.args], { stdio: ['ignore', log, log] });
let ready = false, stopping = false, failedStart = false;

const withdraw = () => { if (readSharedServerAt(paths.record)?.pid === process.pid) rmSync(paths.record, { force: true }); };
const stop = (why: string) => {
  if (stopping) return;
  stopping = true;
  withdraw();
  writeFileSync(log, `natlang supervisor: stopping (${why})\n`);
  // llama-server finishes the batch in progress on its first signal and exits at once on a second (another kind).
  child.kill('SIGTERM'); child.kill('SIGINT');
  setTimeout(() => { child.kill('SIGKILL'); process.exit(0); }, 5000).unref();
};
child.once('exit', code => {
  withdraw();
  // A clean stop leaves nothing behind; a failed start keeps its log for the waiting starter and the user.
  // A newer start of the same server may already hold the lock, a record or a lease: only what is unused goes.
  if (stopping && ready && !failedStart && !existsSync(paths.lock) && !existsSync(paths.record)) {
    try { rmdirSync(paths.leases); } catch { /* holds a newer lease, or is gone */ }
    for (const path of [paths.log, paths.log.replace(/\.log$/, '.supervisor.log')]) rmSync(path, { force: true });
  }
  if (!ready) {
    let output = '';
    try { output = readFileSync(paths.log, 'utf8').slice(-4000); } catch { /* no log */ }
    writeFileSync(paths.failed, `managed model server exited during startup (code ${code})\n${output.trim()}\n`);
    rmSync(paths.lock, { force: true });
  }
  process.exit(0);
});
for (const signal of ['SIGTERM', 'SIGINT', 'SIGHUP'] as const) process.on(signal, () => stop(signal));

async function idle(): Promise<boolean> {
  try {
    const slots = await (await fetch(spec.endpoint + '/slots')).json() as { is_processing?: boolean }[];
    return !Array.isArray(slots) || slots.every(slot => !slot.is_processing);
  } catch { return true; }
}

async function main() {
  const deadline = Date.now() + spec.startTimeoutMs;
  while (!ready) {
    if (stopping) return;
    if (Date.now() > deadline) {
      writeFileSync(paths.failed, `managed model server did not become ready within the startup timeout (see ${paths.log})\n`);
      rmSync(paths.lock, { force: true });
      ready = true; failedStart = true; // the exit handler must not overwrite the failure, and keeps the log
      stop('startup timeout');
      return;
    }
    try { if ((await fetch(spec.endpoint + '/health')).ok) ready = true; } catch { /* still loading */ }
    if (!ready) await new Promise(resolveWait => setTimeout(resolveWait, 250));
  }
  writeSharedServer(spec.directory, { schema: 'natlang.shared-model-server/2', key: spec.key, pid: process.pid, serverPid: child.pid!,
    startedAt: new Date().toISOString(), ...spec.record });
  rmSync(paths.lock, { force: true });
  let unusedSince = Date.now();
  for (;;) {
    await new Promise(resolveWait => setTimeout(resolveWait, 1000));
    if (stopping) return;
    if (liveLeases(spec.directory, spec.key) > 0 || !await idle()) { unusedSince = Date.now(); continue; }
    if (Date.now() - unusedSince >= spec.record.idleSeconds * 1000) { stop(`unused for ${spec.record.idleSeconds} s`); return; }
  }
}
void main();
