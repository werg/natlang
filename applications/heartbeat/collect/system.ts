// Sources "gpu", "teacher", "git" and "disk": the machine's resources.
import type { Host } from '../host.js';
import { failure, makeReading, type Reading } from './reading.js';

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

export async function collectGpu(host: Host): Promise<{ reading: Reading, utilization: number | null }> {
  const result = await host.exec(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], { timeoutMs: 20_000 });
  if (!result.ok) return { reading: failure(host, 'gpu', 'gpu', result.stderr || result.stdout || 'nvidia-smi is not available'), utilization: null };
  const first = result.stdout.trim().split('\n')[0]?.split(',').map(part => part.trim()) ?? [];
  const utilization = /^\d+(\.\d+)?$/.test(first[0] ?? '') ? Number(first[0]) : null;
  return { reading: makeReading(host, 'gpu', 'gpu', true, `utilization.gpu=${first[0] ?? '?'}% memory.used=${first[1] ?? '?'} MiB\n${result.stdout.trim()}`), utilization };
}

export async function collectTeacher(host: Host, metricsUrl: string, windowFile: string | null): Promise<{ reading: Reading, load: number | null }> {
  const metrics = await host.fetchText(metricsUrl, 5000);
  const window = windowFile ? await host.readText(windowFile) : null;
  const load = metrics === null ? null : vllmLoad(metrics);
  const lines = [load ? `vllm:num_requests_running=${load.running} vllm:num_requests_waiting=${load.waiting}` : metrics === null ? `the teacher at ${metricsUrl} did not answer` : 'the teacher exposes no request metrics',
    window ? `window file ${windowFile}: ${window.text.trim().slice(0, 1500)}` : windowFile ? `window file ${windowFile} does not exist` : ''].filter(Boolean);
  return { reading: makeReading(host, 'teacher', 'teacher', metrics !== null, lines.join('\n')), load: load ? load.running + load.waiting : null };
}

export async function collectGit(host: Host, repo: string): Promise<Reading> {
  const run = (...args: string[]) => host.exec(['git', '-C', repo, ...args], { timeoutMs: 20_000 });
  const [status, log, ahead] = await Promise.all([run('status', '--porcelain=v1', '-b'), run('log', '--oneline', '-5'), run('rev-list', '--count', 'origin/main..HEAD')]);
  if (!status.ok) return failure(host, 'git', 'repo', status.stderr || status.stdout);
  return makeReading(host, 'git', 'repo', true, `${status.stdout.trim()}\n--\n${log.stdout.trim()}\n--\ncommits ahead of origin/main: ${ahead.ok ? ahead.stdout.trim() : 'unknown'}`);
}

export async function collectDisk(host: Host, roots: string[]): Promise<Reading> {
  const [df, memory] = await Promise.all([host.exec(['df', '-h', ...roots], { timeoutMs: 20_000 }), host.readText('/proc/meminfo')]);
  const available = /^MemAvailable:\s+(\d+) kB/m.exec(memory?.text ?? '');
  const lines = [df.ok ? df.stdout.trim() : `df failed: ${df.stderr.trim()}`, available ? `MemAvailable: ${(Number(available[1]) / 2 ** 20).toFixed(1)} GiB` : 'MemAvailable: unknown'];
  return makeReading(host, 'disk', 'disk', df.ok, lines.join('\n'));
}
