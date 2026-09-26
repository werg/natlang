/**
 * One task in a disposable child process, for code that must not be trusted to stop or to stay contained. The code
 * a task runs is bounded where it runs (an eval or vm time limit); the limits here are safety nets for the child
 * itself, and they count from when the child is ready: starting Node and loading what a task needs can alone take
 * longer than a task on a busy machine, and must not be mistaken for a runaway task.
 */
import { fork } from 'node:child_process';

/** Run `message` in a child started from `moduleUrl` with `--worker`; resolves with the `result` it sends back. */
export function runIsolated(moduleUrl, message, { timeout, startup = 120000, name = 'isolated task' }) {
  return new Promise((accept, reject) => {
    const child = fork(moduleUrl, ['--worker'], { stdio: ['ignore', 'ignore', 'pipe', 'ipc'], execArgv: [] });
    let settled = false;
    const finish = (error, result) => {
      if (settled) return; settled = true; clearTimeout(timer); child.kill('SIGKILL');
      error ? reject(error) : accept(result);
    };
    let timer = setTimeout(() => finish(new Error(`${name} worker did not start`)), startup);
    child.on('message', reply => {
      if (!reply.ready) return finish(reply.error ? new Error(reply.error) : null, reply.result);
      clearTimeout(timer); timer = setTimeout(() => finish(new Error(`${name} timeout`)), timeout);
      child.send(message);
    });
    child.on('error', error => finish(error));
    child.on('exit', code => finish(new Error(`${name} worker exited ${code}`)));
  });
}

/** In a `--worker` child: load `preload`, report ready, run one message through `handle`, send back its result. */
export async function serveIsolated(handle, preload = async () => {}) {
  process.once('message', async message => {
    try { process.send({ result: await handle(message) }); } catch (error) { process.send({ error: String(error) }); }
  });
  await preload();
  process.send({ ready: true });
}
