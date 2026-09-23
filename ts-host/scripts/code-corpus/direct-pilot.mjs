#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync, copyFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { extractFunctions } from './extract.mjs';
import { instrumentSource } from './capture.mjs';

const UPSTREAM = 'd3/d3-array';
const REVISION = 'd1c89b8ae796cd232eeeda3bdbd856a92d2edd56';
const FUNCTIONS = {
  ascending: { source: 'src/ascending.js', spec: 'test/ascending-test.js', instruction: 'Compare two values in ascending order and return their relative ordering.', dependencies: [] },
  mean: { source: 'src/mean.js', spec: 'test/mean-test.js', instruction: 'Calculate the arithmetic mean of the supplied values.', dependencies: ['src/number.js'] },
  transpose: { source: 'src/transpose.js', spec: 'test/transpose-test.js', instruction: 'Transpose the rows and columns of a matrix.', dependencies: ['src/min.js', 'src/number.js'] },
};
const here = dirname(fileURLToPath(import.meta.url));
const captureRuntime = join(here, 'capture-runtime.cjs');

export function parseArgs(argv) {
  const value = (name) => { const i = argv.indexOf(name); return i < 0 ? undefined : argv[i + 1]; };
  return { execute: argv.includes('--execute'), output: value('--output'), functionName: value('--function') ?? 'ascending' };
}
const hash = (data) => createHash('sha256').update(data).digest('hex');
function run(command, args, options = {}) {
  const result = spawnSync(command, args, { encoding: 'utf8', timeout: 240_000, maxBuffer: 24 * 1024 * 1024, ...options });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`${command} ${args.join(' ')} failed (${result.status ?? result.signal})\n${result.stderr || result.stdout}`);
  return result;
}
function writeJsonl(path, rows) { writeFileSync(path, `${rows.map(row => JSON.stringify(row)).join('\n')}\n`); }

async function main() {
  const { execute, output: outArg, functionName } = parseArgs(process.argv.slice(2));
  if (!execute || !outArg) throw new Error('Trusted execution is opt-in. Usage: node scripts/code-corpus/direct-pilot.mjs --execute --output NEW_DIR [--function ascending|mean|transpose]');
  if (!Object.hasOwn(FUNCTIONS, functionName)) throw new Error(`Unsupported pilot function ${functionName}; choose ascending, mean or transpose`);
  const { source: sourcePath, spec: specPath, instruction, dependencies } = FUNCTIONS[functionName];
  const output = resolve(outArg);
  if (existsSync(output)) throw new Error(`Refusing to overwrite existing output: ${output}`);
  const work = mkdtempSync(join(tmpdir(), 'natlang-direct-pilot-'));
  try {
    const checkout = join(work, 'checkout');
    run('git', ['clone', '--no-checkout', '--filter=blob:none', `https://github.com/${UPSTREAM}.git`, checkout]);
    run('git', ['-C', checkout, 'checkout', '--detach', REVISION]);
    const originalSource = readFileSync(join(checkout, sourcePath), 'utf8');
    const spec = readFileSync(join(checkout, specPath), 'utf8');
    const pkg = JSON.parse(readFileSync(join(checkout, 'package.json'), 'utf8'));
    const originalPackage = readFileSync(join(checkout, 'package.json'));
    const instrumented = instrumentSource(originalSource, { path: sourcePath, sourceName: UPSTREAM, revision: REVISION });
    if (!instrumented.functions.some(f => f.name === functionName && f.instrumented)) throw new Error(`${functionName} was not instrumented`);
    writeFileSync(join(checkout, sourcePath), instrumented.code);
    const install = run('npm', ['install', '--ignore-scripts', '--no-audit', '--no-fund', '--legacy-peer-deps'], { cwd: checkout });
    writeFileSync(join(checkout, 'node-test-setup.mjs'), `import { it } from 'node:test';\nimport ${JSON.stringify(pathToFileURL(captureRuntime).href)};\nglobalThis.it = it;\n`);
    const captureFile = join(checkout, 'captures.jsonl');
    const testRun = run(process.execPath, ['--test', '--import', './node-test-setup.mjs', specPath], {
      cwd: checkout, env: { ...process.env, CODE_CORPUS_CAPTURE: captureFile },
    });
    const captures = existsSync(captureFile) ? readFileSync(captureFile, 'utf8').split('\n').filter(Boolean).map(JSON.parse) : [];
    const nonportableReasonCounts = Object.fromEntries([...captures.flatMap(capture => capture.reasons.map(reason => reason.reason))]
      .reduce((counts, reason) => counts.set(reason, (counts.get(reason) ?? 0) + 1), new Map()));
    const tasks = extractFunctions(originalSource, { path: sourcePath, sourceName: UPSTREAM, revision: REVISION, license: pkg.license, instruction })
      .filter(task => task.function.name === functionName);
    mkdirSync(dirname(output), { recursive: true });
    mkdirSync(output);
    const retained = join(output, 'upstream');
    mkdirSync(dirname(join(retained, sourcePath)), { recursive: true });
    mkdirSync(dirname(join(retained, specPath)), { recursive: true });
    for (const dependency of dependencies) mkdirSync(dirname(join(retained, dependency)), { recursive: true });
    writeFileSync(join(retained, sourcePath), originalSource);
    writeFileSync(join(retained, specPath), spec);
    for (const dependency of dependencies) copyFileSync(join(checkout, dependency), join(retained, dependency));
    writeFileSync(join(retained, 'package.json'), originalPackage);
    writeFileSync(join(output, `${functionName}.instrumented.js`), instrumented.code);
    copyFileSync(join(checkout, 'package.json'), join(output, 'package.json'));
    copyFileSync(join(checkout, 'package-lock.json'), join(output, 'package-lock.json'));
    copyFileSync(captureRuntime, join(output, 'capture-runtime.cjs'));
    writeJsonl(join(output, 'tasks.jsonl'), tasks);
    writeJsonl(join(output, 'captures.jsonl'), captures);
    writeFileSync(join(output, 'original-test.stdout.txt'), testRun.stdout);
    writeFileSync(join(output, 'original-test.stderr.txt'), testRun.stderr);
    const manifest = {
      pilot: `d3-array/${functionName}`, upstream: `https://github.com/${UPSTREAM}`, revision: REVISION,
      source: { path: sourcePath, sha256: hash(originalSource) }, spec: { path: specPath, sha256: hash(spec) },
      preserved_dependencies: dependencies.map(path => ({ path, sha256: hash(readFileSync(join(checkout, path))) })),
      dependencies: { ...pkg.devDependencies, install_scripts: false },
      execution: { command: `node --test --import ./node-test-setup.mjs ${specPath}`, exit_code: testRun.status,
        captures: captures.length, portable_captures: captures.filter(c => c.portable).length,
        nonportable_captures: captures.filter(c => !c.portable).length,
        nonportable_reason_counts: nonportableReasonCounts,
        return_captures: captures.filter(c => c.outcome === 'return').length },
      hashes: { instrumented: hash(instrumented.code), tasks: hash(readFileSync(join(output, 'tasks.jsonl'))), captures: hash(readFileSync(join(output, 'captures.jsonl'))), package: hash(originalPackage), package_lock: hash(readFileSync(join(output, 'package-lock.json'))) },
      npm_install_note: install.stderr.trim().split('\n').slice(-2).join('\n'),
    };
    let replayError;
    try {
      const replay = run(process.execPath, [join(here, 'replay.mjs'), '--execute', '--input', join(output, 'tasks.jsonl'), '--captures', join(output, 'captures.jsonl'), '--output', join(output, 'native-replay.jsonl'), '--cases', '30', '--workspace', checkout], { cwd: join(here, '../../..') });
      writeFileSync(join(output, 'native-replay.stdout.txt'), replay.stdout);
      const replaySummary = JSON.parse(replay.stdout.trim());
      manifest.native_replay = { ...replaySummary, command: 'replay.mjs --execute --workspace <disposable-upstream-checkout>',
        rows_sha256: hash(readFileSync(join(output, 'native-replay.jsonl'))),
        turns_sha256: hash(readFileSync(join(output, 'native-replay.jsonl.turns.jsonl'))),
        rejected_sha256: hash(readFileSync(join(output, 'native-replay.jsonl.rejected.jsonl'))) };
    } catch (error) {
      replayError = error;
      manifest.native_replay = { command: 'replay.mjs --execute', error: String(error) };
    }
    writeFileSync(join(output, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
    console.log(`${JSON.stringify(manifest, null, 2)}`);
    if (replayError) throw replayError;
  } finally {
    rmSync(work, { recursive: true, force: true });
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch(error => { console.error(error.stack ?? String(error)); process.exitCode = 1; });
}
