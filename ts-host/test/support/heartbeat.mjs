/** A heartbeat Host over fixtures: commands answer from files under test/fixtures/heartbeat, and every effect is recorded. */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const dir = fileURLToPath(new URL('../fixtures/heartbeat/', import.meta.url));
export const fixture = name => readFileSync(`${dir}${name}`, 'utf8');
export const NOW = '2026-10-09T12:17:00.000Z';
export const REPO = '/fx/repo';
export const CONFIG = fileURLToPath(new URL('../../../applications/heartbeat', import.meta.url));

export const WATCH = {
  schema: 'natlang.heartbeat-watch/1',
  runs: [
    { run_id: 'v20-train', machine: 'dgx', owner: 'dgx-agent', unit: 'natlang-v20-train.service', kind: 'training', expected: 'finishes', log: '/fx/v20-train.log',
      progress_marker: 'step \\d+ loss', stall_after_minutes: 45, gate_paths: [], gate_fields: [], next_steps_doc: '/fx/v20-next.md' },
    { run_id: 'gen-teacher', machine: 'dgx', owner: 'dgx-agent', unit: 'natlang-gen-teacher.service', kind: 'generation', expected: 'finishes', log: '/fx/gen-teacher.log',
      progress_marker: 'generated batch', stall_after_minutes: 60, gate_paths: [], gate_fields: [], next_steps_doc: null },
    { run_id: 'foundation-gate', machine: 'dgx', owner: 'dgx-agent', unit: 'natlang-foundation-gate.service', kind: 'eval', expected: 'finishes', log: null,
      progress_marker: 'done', stall_after_minutes: 30, gate_paths: ['/fx/foundation-gate.json'],
      gate_fields: ['token_aligned_reference_passed', 'causal_distillation_passed', 'details.nested.cosine'], next_steps_doc: null },
    { run_id: 'stalled-eval', machine: 'dgx', owner: 'dgx-agent', unit: 'natlang-stalled-eval.service', kind: 'eval', expected: 'finishes', log: '/fx/stalled-eval.log',
      progress_marker: 'eval case \\d+ done', stall_after_minutes: 60, gate_paths: [], gate_fields: [], next_steps_doc: null },
    { run_id: 'other-machine-run', machine: 'pop', owner: 'pop-agent', unit: 'natlang-pop-thing.service', kind: 'eval', expected: 'finishes', log: null,
      progress_marker: 'x', stall_after_minutes: 30, gate_paths: [], gate_fields: [], next_steps_doc: null },
  ],
};

const FILES = () => ({
  '/fx/v20-train.log': fixture('v20-train.log'), '/fx/gen-teacher.log': fixture('gen-teacher.log'), '/fx/stalled-eval.log': fixture('stalled-eval.log'),
  '/fx/foundation-gate.json': fixture('foundation-gate.json'), '/fx/v20-next.md': '1. Relaunch v20 with the smaller micro-batch.\n2. Run the held-out gate.\n',
  [`${CONFIG}/watch.json`]: JSON.stringify(WATCH), [`${CONFIG}/actions.json`]: readFileSync(`${CONFIG}/actions.json`, 'utf8'),
  '/proc/meminfo': 'MemTotal: 125000000 kB\nMemAvailable: 52000000 kB\n',
});

/**
 * options.files adds or overrides files; options.commands maps a command prefix (joined argv) to `{ ok, code, stdout, stderr }` or a function;
 * options.urls maps URLs to text (absent: unreachable); options.now sets the clock.
 */
export function fixtureHost(options = {}) {
  const files = { ...FILES(), ...(options.files ?? {}) };
  const mtimes = options.mtimes ?? {};
  const calls = [], writes = [], appends = [];
  const result = (stdout, extra = {}) => ({ ok: true, code: 0, stdout, stderr: '', ...extra });
  const defaults = {
    'python3 /fx/repo/scripts/memory_ledger.py status': () => result(fixture('ledger-status.json')),
    'python3 /fx/repo/scripts/memory_ledger.py release-cache': () => result('{"released": 0}\n'),
    'journalctl --user -u': () => result('Oct 09 11:06:03 mltick kernel: Out of memory: Killed process 1\n'),
    'systemctl --user list-units': () => result(fixture('unit-list.txt')),
    'nvidia-smi': () => result(fixture('nvidia-smi.txt')),
    'python3 /fx/repo/scripts/coord.py --repo /fx/repo --as dgx-heartbeat inbox --json': () => result(fixture('inbox.json')),
    'python3 /fx/repo/scripts/coord.py --repo /fx/repo --as dgx-heartbeat status': () => result('── pop status, 2026-10-09T11:00:00Z by pop-agent\n\n   running the corpus build\n'),
    'python3 /fx/repo/scripts/coord.py --repo /fx/repo --as dgx-heartbeat log': () => result('[]'),
    'git -C /fx/repo status': () => result('## main...origin/main\n'),
    'git -C /fx/repo log --oneline': () => result('abc1234 a commit\n'),
    'git -C /fx/repo log --since': () => result(''),
    'git -C /fx/repo rev-list': () => result('0\n'),
    'df -h': () => result('Filesystem Size Used Avail Use% Mounted on\n/dev/nvme0n1 3.6T 1.2T 2.2T 36% /home/werg/data\n'),
  };
  const commands = { ...defaults, ...(options.commands ?? {}) };
  const unitKey = 'systemctl --user show ';
  const host = {
    calls, writes, appends, files,
    async exec(argv, execOptions = {}) {
      const line = argv.join(' ');
      calls.push({ argv, line, input: execOptions.input, env: execOptions.env });
      if (line.startsWith(unitKey)) {
        const unit = argv[3];
        const text = files[`unit:${unit}`] ?? (() => { try { return fixture(`unit-${unit}.txt`); } catch { return null; } })();
        return text === null ? { ok: false, code: 4, stdout: '', stderr: `Unit ${unit} could not be found.` } : result(text);
      }
      const key = Object.keys(commands).sort((a, b) => b.length - a.length).find(prefix => line.startsWith(prefix));
      if (!key) return { ok: false, code: 127, stdout: '', stderr: `fixture host: no answer for ${line}` };
      const answer = commands[key];
      return typeof answer === 'function' ? answer(argv, execOptions) : answer;
    },
    async readText(path) { return path in files ? { text: files[path], mtimeMs: mtimes[path] ?? Date.parse(NOW) - 60_000 } : null; },
    async readTail(path, bytes) { return path in files ? { text: files[path].slice(-bytes), mtimeMs: mtimes[path] ?? Date.parse(NOW) - 60_000 } : null; },
    async appendText(path, text) { appends.push({ path, text }); files[path] = (files[path] ?? '') + text; },
    async writeText(path, text) { writes.push({ path, text }); files[path] = text; },
    async fetchText(url) { return options.urls && url in options.urls ? options.urls[url] : null; },
    now: () => new Date(options.now ?? NOW),
    machine: () => 'dgx',
  };
  return host;
}
