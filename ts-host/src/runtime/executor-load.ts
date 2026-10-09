/**
 * How busy a vLLM executor is, and waiting for it to be idle. Programs that run beside the executor they use (the
 * specializer, the heartbeat) call this so they do not compete with the programs it serves. Exported from `@natlang/node`
 * because callable folders may import only siblings and packages.
 */

/** Requests the vLLM server runs or queues, summed from its Prometheus text; null when the text has no such metric. */
export function vllmLoad(metrics: string): { running: number, waiting: number } | null {
  let running = 0, waiting = 0, found = false;
  for (const line of metrics.split('\n')) {
    const match = /^vllm:num_requests_(running|waiting)(?:\{[^}]*\})? ([\d.]+)$/.exec(line);
    if (!match) continue;
    found = true;
    if (match[1] === 'running') running += Number(match[2]); else waiting += Number(match[2]);
  }
  return found ? { running, waiting } : null;
}

export type ExecutorIdleWait = {
  /** The text of the executor's `/metrics`; null when it did not answer. */
  readMetrics: () => Promise<string | null>;
  /** Wait while running plus waiting requests exceed this. */
  maxBusy: number;
  /** Go ahead after waiting this many seconds. */
  idleWaitSeconds: number;
  log: (line: string) => void;
  now?: () => number;
  sleep?: (ms: number) => Promise<void>;
  /** Stop waiting when this becomes true. */
  stopping?: () => boolean;
  pollMs?: number;
};

/**
 * Wait until the executor is not busy (running plus waiting requests at most `maxBusy`). Gives up after `idleWaitSeconds`
 * and goes ahead. An executor without metrics, or one that does not answer, is never waited for.
 */
export async function waitForExecutorIdle(options: ExecutorIdleWait): Promise<void> {
  const now = options.now ?? Date.now, sleep = options.sleep ?? (ms => new Promise<void>(done => setTimeout(done, ms)));
  const busy = async (): Promise<number | undefined> => {
    try {
      const text = await options.readMetrics();
      const load = text === null ? null : vllmLoad(text);
      return load ? load.running + load.waiting : undefined;
    } catch { return undefined; }
  };
  const started = now();
  let said = false;
  for (let load = await busy(); load !== undefined && load > options.maxBusy && !options.stopping?.(); load = await busy()) {
    if (now() - started > options.idleWaitSeconds * 1000) { options.log(`the executor is still busy (${load} requests); going ahead`); return; }
    if (!said) { options.log(`waiting for the executor to be idle (${load} requests, at most ${options.maxBusy})`); said = true; }
    await sleep(options.pollMs ?? 30_000);
  }
}
