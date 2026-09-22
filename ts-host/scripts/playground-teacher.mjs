#!/usr/bin/env node
/** Run the Node collector and immediately project its native trajectory rows into training turns. */
import { spawn } from 'node:child_process';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

function run(script, args) {
  return new Promise((resolveRun, reject) => {
    const child = spawn(process.execPath, [script, ...args], { cwd: process.cwd(), env: process.env,
      stdio: 'inherit' });
    child.once('error', reject);
    child.once('exit', (code, signal) => signal ? reject(new Error(`${script} stopped by ${signal}`)) :
      resolveRun(code ?? 1));
  });
}

async function main() {
  const [ir, jobs, rawOutput, turnsOutput, ...collectorArgs] = process.argv.slice(2);
  if (!ir || !jobs || !rawOutput || !turnsOutput) throw new Error(
    'usage: playground-teacher.mjs IR JOBS RAW.jsonl TURNS.jsonl --model-id ID --root-seed N [options]');
  const collector = 'ts-host/scripts/teacher-collector.mjs';
  const materializer = 'ts-host/scripts/materialize-native-teacher.mjs';
  const collected = await run(collector, [ir, jobs, rawOutput, ...collectorArgs]);
  if (collected !== 0) process.exitCode = collected;
  else {
    const projected = await run(materializer, [rawOutput, turnsOutput]);
    if (projected !== 0) process.exitCode = projected;
  }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url))
  main().catch(error => { console.error(error instanceof Error ? error.stack : String(error)); process.exitCode = 1; });
