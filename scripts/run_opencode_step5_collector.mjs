#!/usr/bin/env node
/** Verify pinned model routing, then run the exact collector argv saved in JSON.
 * Parent SIGINT/SIGTERM is forwarded only to this owned collector child. The
 * receipt records interrupted, timed-out, spawn-error, and normal exits distinctly.
 */
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
  const pairing = verifyStep5ModelPair({ plan, bootstrapConfig, collectorArgv, bootstrapConfigText: bridgeText });
  const timeoutMs = Number(plan.bounds?.outer_wall_clock_seconds) * 1000;
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1) throw new Error('plan has no valid outer wall-clock bound');
  const receiptPath = options.get('--receipt');
  const receipt = { schema: 'natlang.opencode_step5_collector_launch/1', plan_sha256: hash(planText),
    bridge_config_sha256: hash(bridgeText), collector_argv_sha256: hash(argvText), ...pairing,
    collector_argv: collectorArgv, parent_pid: process.pid, started_at: new Date().toISOString(), status: 'starting' };
  await writeFile(receiptPath, `${JSON.stringify(receipt, null, 2)}\n`, { flag: 'wx', mode: 0o600 });
  let receiptWrite = Promise.resolve();
  const persistReceipt = () => {
    const serialized = `${JSON.stringify(receipt, null, 2)}\n`;
    receiptWrite = receiptWrite.then(() => writeFile(receiptPath, serialized, { mode: 0o600 }));
    return receiptWrite;
  };
  let child;
  try {
    child = spawn(process.execPath, collectorArgv, { stdio: 'inherit' });
  } catch (error) {
    receipt.status = 'spawn_error';
    receipt.spawn_error = { name: error?.name ?? 'Error', code: error?.code ?? null,
      message: String(error?.message ?? error) };
    receipt.finished_at = new Date().toISOString();
    await persistReceipt();
    process.exitCode = 1;
    return;
  }
  receipt.collector_pid = child.pid ?? null;
  receipt.status = 'running';
  let stopCause = null, escalation;
  const stopChild = (cause, signal) => {
    if (stopCause) return;
    stopCause = cause;
    receipt.status = cause === 'timeout' ? 'stopping_timeout' : 'stopping_signal';
    if (cause === 'signal') receipt.stop_signal = signal;
    receipt.stop_requested_at = new Date().toISOString();
    child.kill(signal);
    void persistReceipt()
      .catch(error => process.stderr.write(`Could not update Step 5 collector receipt: ${error.message}\n`));
    escalation = setTimeout(() => child.kill('SIGKILL'), 30_000);
  };
  const onSigint = () => stopChild('signal', 'SIGINT');
  const onSigterm = () => stopChild('signal', 'SIGTERM');
  process.on('SIGINT', onSigint);
  process.on('SIGTERM', onSigterm);
  const timer = setTimeout(() => stopChild('timeout', 'SIGINT'), timeoutMs);
  const resultPromise = new Promise(resolve => {
    child.once('error', error => resolve({ error }));
    child.once('close', (code, signal) => resolve({ code, signal }));
  });
  const runningReceiptWrite = persistReceipt();
  await runningReceiptWrite;
  const result = await resultPromise.finally(() => {
    clearTimeout(timer); clearTimeout(escalation);
    process.off('SIGINT', onSigint); process.off('SIGTERM', onSigterm);
  });
  if (result.error) {
    receipt.status = 'spawn_error';
    receipt.spawn_error = { name: result.error?.name ?? 'Error', code: result.error?.code ?? null,
      message: String(result.error?.message ?? result.error) };
  } else if (stopCause === 'timeout') {
    receipt.status = 'timed_out';
  } else if (stopCause === 'signal') {
    receipt.status = 'interrupted';
  } else {
    receipt.status = result.code === 0 ? 'completed' : 'failed';
  }
  receipt.exit_code = result.code ?? (result.error ? 1 : null); receipt.signal = result.signal ?? null;
  receipt.finished_at = new Date().toISOString();
  await persistReceipt();
  process.exitCode = stopCause === 'timeout' ? 124
    : (stopCause === 'signal' ? (receipt.stop_signal === 'SIGINT' ? 130 : 143) : (result.code ?? 1));
}

if (process.argv[1]?.endsWith('/run_opencode_step5_collector.mjs'))
  main(process.argv.slice(2)).catch(error => { process.stderr.write(`Step 5 collector launch refused: ${error.message}\n`); process.exitCode = 1; });
