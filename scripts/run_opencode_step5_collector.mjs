#!/usr/bin/env node
/** Verify pinned model routing, then run the exact collector argv saved in JSON. */
import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import { verifyStep5ModelPair } from './opencode-step5-preflight.mjs';

const hash = value => createHash('sha256').update(value).digest('hex');
async function main(argv) {
  const options = new Map();
  for (let i = 0; i < argv.length; i += 2) {
    const key = argv[i], value = argv[i + 1];
    if (!['--plan', '--bootstrap-config', '--collector-argv-json', '--receipt'].includes(key) || !value || options.has(key))
      throw new Error(`invalid or duplicate option ${key}`);
    options.set(key, value);
  }
  for (const key of ['--plan', '--bootstrap-config', '--collector-argv-json', '--receipt'])
    if (!options.has(key)) throw new Error(`${key} is required`);
  const [planText, bridgeText, argvText] = await Promise.all([
    readFile(options.get('--plan'), 'utf8'), readFile(options.get('--bootstrap-config'), 'utf8'),
    readFile(options.get('--collector-argv-json'), 'utf8')
  ]);
  const plan = JSON.parse(planText), bootstrapConfig = JSON.parse(bridgeText), collectorArgv = JSON.parse(argvText);
  const pairing = verifyStep5ModelPair({ plan, bootstrapConfig, collectorArgv });
  const timeoutMs = Number(plan.bounds?.outer_wall_clock_seconds) * 1000;
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1) throw new Error('plan has no valid outer wall-clock bound');
  const receiptPath = options.get('--receipt');
  const receipt = { schema: 'natlang.opencode_step5_collector_launch/1', plan_sha256: hash(planText),
    bridge_config_sha256: hash(bridgeText), collector_argv_sha256: hash(argvText), ...pairing,
    collector_argv: collectorArgv, parent_pid: process.pid, started_at: new Date().toISOString(), status: 'starting' };
  await writeFile(receiptPath, `${JSON.stringify(receipt, null, 2)}\n`, { flag: 'wx', mode: 0o600 });
  const child = spawn(process.execPath, collectorArgv, { stdio: 'inherit' });
  receipt.collector_pid = child.pid;
  receipt.status = 'running';
  await writeFile(receiptPath, `${JSON.stringify(receipt, null, 2)}\n`, { mode: 0o600 });
  let timedOut = false, escalation;
  const timer = setTimeout(() => {
    timedOut = true; child.kill('SIGINT');
    escalation = setTimeout(() => child.kill('SIGKILL'), 30_000);
  }, timeoutMs);
  const result = await new Promise((resolve, reject) => {
    child.once('error', reject);
    child.once('close', (code, signal) => resolve({ code, signal }));
  }).finally(() => { clearTimeout(timer); clearTimeout(escalation); });
  receipt.status = timedOut ? 'timed_out' : (result.code === 0 ? 'completed' : 'failed');
  receipt.exit_code = result.code; receipt.signal = result.signal;
  receipt.finished_at = new Date().toISOString();
  await writeFile(receiptPath, `${JSON.stringify(receipt, null, 2)}\n`, { mode: 0o600 });
  process.exitCode = timedOut ? 124 : (result.code ?? 1);
}

if (process.argv[1]?.endsWith('/run_opencode_step5_collector.mjs'))
  main(process.argv.slice(2)).catch(error => { process.stderr.write(`Step 5 collector launch refused: ${error.message}\n`); process.exitCode = 1; });
