#!/usr/bin/env node
/**
 * Run test files each in its own process, a few at a time, so one file that runs out of memory or hangs is that file's
 * result instead of the end of the run. Run it under the memory ledger with `--oom-policy continue`, so the kernel's
 * kill of one file's process does not stop the unit:
 *
 *   python3 scripts/memory_ledger.py run --unit natlang-tests --budget-gb 8 --oom-policy continue --workdir ts-host \
 *     node scripts/test-files.mjs --concurrency 2 --json /tmp/tests.json
 *
 * Options: --concurrency N (2), --timeout SECONDS per file (900), --alone REGEX for files that run by themselves after
 * the rest (default: the Neuralese training and server tests), --json FILE for the summary, and file paths or name
 * substrings to select (default: test/*.test.mjs). Each file gets its own call store under a temporary directory.
 */
import { spawn } from 'node:child_process';
import { mkdtempSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

const argv = process.argv.slice(2);
const option = (name, fallback) => { const at = argv.indexOf(name); if (at < 0) return fallback; const value = argv[at + 1]; argv.splice(at, 2); return value; };
const concurrency = Number(option('--concurrency', 2));
const timeout = Number(option('--timeout', 900)) * 1000;
const alone = new RegExp(option('--alone', 'neuralese-(learning|server)\\.test'));
const jsonOut = option('--json', undefined);
const root = resolve(import.meta.dirname, '..');
const all = readdirSync(join(root, 'test')).filter(name => name.endsWith('.test.mjs')).sort().map(name => join('test', name));
const files = argv.length ? all.filter(file => argv.some(word => file.includes(word))) : all;
const scratch = mkdtempSync(join(tmpdir(), 'natlang-test-files-'));

/** One file in a child process: its counts, failures, and how it ended. */
function runFile(file) {
  return new Promise(done => {
    const started = Date.now();
    const child = spawn(process.execPath, ['--test', '--test-reporter=tap', file], { cwd: root,
      env: { ...process.env, NATLANG_CALL_STORE: join(scratch, file.replace(/\W+/g, '_')) }, stdio: ['ignore', 'pipe', 'pipe'] });
    let output = '';
    child.stdout.on('data', chunk => { output += chunk; });
    child.stderr.on('data', chunk => { output += chunk; });
    let timedOut = false;
    const timer = setTimeout(() => { timedOut = true; child.kill('SIGKILL'); }, timeout);
    child.on('close', (code, signal) => {
      clearTimeout(timer);
      const count = name => Number(new RegExp(`^# ${name} (\\d+)$`, 'm').exec(output)?.[1] ?? 0);
      const failures = [...output.matchAll(/^not ok \d+ - (.+)$/gm)].map(match => match[1]).filter(name => !name.endsWith('.mjs'));
      const ended = timedOut ? 'timeout' : signal === 'SIGKILL' ? 'killed (out of memory?)' : signal ? `signal ${signal}` : code === 0 ? 'ok' : 'failed';
      done({ file, ended, pass: count('pass'), fail: count('fail'), cancelled: count('cancelled'), failures, seconds: Math.round((Date.now() - started) / 1000),
        ...(ended !== 'ok' && !failures.length ? { tail: output.split('\n').slice(-15).join('\n') } : {}) });
    });
  });
}

async function runAll(list, limit) {
  const queue = [...list], results = [];
  await Promise.all(Array.from({ length: Math.min(limit, queue.length) }, async () => {
    for (let file = queue.shift(); file; file = queue.shift()) {
      const result = await runFile(file);
      results.push(result);
      if (result.ended !== 'ok') process.stdout.write(`${result.ended.padEnd(8)} ${file} (${result.fail} failed, ${result.seconds}s)\n`);
    }
  }));
  return results;
}

const results = [...await runAll(files.filter(file => !alone.test(file)), concurrency), ...await runAll(files.filter(file => alone.test(file)), 1)];
rmSync(scratch, { recursive: true, force: true });
const total = key => results.reduce((sum, result) => sum + result[key], 0);
const bad = results.filter(result => result.ended !== 'ok');
process.stdout.write(`\n${results.length} files: ${results.length - bad.length} ok, ${bad.length} not; ${total('pass')} tests passed, ${total('fail')} failed, ${total('cancelled')} cancelled\n`);
for (const result of bad) process.stdout.write(`\n${result.file}: ${result.ended}\n${result.failures.map(name => `  - ${name}`).join('\n')}${result.tail ? `\n${result.tail}` : ''}\n`);
if (jsonOut) writeFileSync(jsonOut, JSON.stringify({ files: results.length, ok: results.length - bad.length, pass: total('pass'), fail: total('fail'), results }, null, 2) + '\n');
process.exit(bad.length ? 1 : 0);
