#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync, copyFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { extractFunctions } from './extract.mjs';
import { instrumentSource } from './capture.mjs';

const UPSTREAM = 'toss/es-toolkit';
const REVISION = 'ee72fc74b763d8cb48095e1981a5bcef19ba5cee';
const FUNCTIONS = { chunk: 'src/array/chunk', compact: 'src/array/compact' };
const VITEST_VERSION = '2.1.9';
const FAST_CHECK_VERSION = '3.23.2';
const here = dirname(fileURLToPath(import.meta.url));
const captureRuntime = join(here, 'capture-runtime.cjs');

function hash(data) { return createHash('sha256').update(data).digest('hex'); }
function argsOf(argv) {
  const args = new Set(argv);
  const index = argv.indexOf('--output');
  const functionIndex = argv.indexOf('--function');
  return { execute: args.has('--execute'), output: index >= 0 ? argv[index + 1] : undefined,
    functionName: functionIndex >= 0 ? argv[functionIndex + 1] : 'chunk' };
}
async function fetchText(url) {
  const response = await fetch(url, { headers: { 'User-Agent': 'natlang-code-corpus-pilot', Accept: 'application/vnd.github+json' } });
  if (!response.ok) throw new Error(`GET ${url}: HTTP ${response.status}`);
  return await response.text();
}
function writeJsonl(file, rows) { writeFileSync(file, `${rows.map((row) => JSON.stringify(row)).join('\n')}\n`); }
function run(command, args, options) {
  const result = spawnSync(command, args, { encoding: 'utf8', timeout: 180_000, maxBuffer: 16 * 1024 * 1024, ...options });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`${command} ${args.join(' ')} failed (${result.status ?? result.signal})\n${result.stderr || result.stdout}`);
  return result;
}

async function main() {
  const { execute, output: outArg, functionName } = argsOf(process.argv.slice(2));
  if (!execute || !outArg) throw new Error('Trusted execution is opt-in. Usage: node scripts/code-corpus/pilot.mjs --execute --output NEW_DIR [--function chunk|compact]');
  if (!Object.hasOwn(FUNCTIONS, functionName)) throw new Error(`Unsupported pilot function ${functionName}; choose chunk or compact`);
  const output = resolve(outArg);
  if (existsSync(output)) throw new Error(`Refusing to overwrite existing output: ${output}`);
  const work = mkdtempSync(join(tmpdir(), 'natlang-code-corpus-pilot-'));
  try {
    const chunkPath = FUNCTIONS[functionName];
    const api = JSON.parse(await fetchText(`https://api.github.com/repos/${UPSTREAM}/commits/${REVISION}`));
    if (api.sha !== REVISION) throw new Error(`GitHub resolved unexpected revision ${api.sha}`);
    const sourcePath = `${chunkPath}.ts`;
    const specPath = `${chunkPath}.spec.ts`;
    const source = await fetchText(`https://raw.githubusercontent.com/${UPSTREAM}/${REVISION}/${sourcePath}`);
    const spec = await fetchText(`https://raw.githubusercontent.com/${UPSTREAM}/${REVISION}/${specPath}`);
    const packageRoot = join(work, 'checkout');
    const chunkDir = join(packageRoot, 'src/array');
    mkdirSync(chunkDir, { recursive: true });
    writeFileSync(join(chunkDir, `${functionName}.upstream.ts`), source);
    writeFileSync(join(chunkDir, `${functionName}.spec.ts`), spec);
    writeFileSync(join(chunkDir, `${functionName}.ts`), instrumentSource(source, { path: sourcePath, sourceName: UPSTREAM, revision: REVISION }).code);
    writeFileSync(join(packageRoot, 'package.json'), JSON.stringify({ name: 'es-toolkit-corpus-pilot', private: true, type: 'module' }, null, 2));
    writeFileSync(join(packageRoot, 'vitest.config.mjs'), `export default { test: { include: ['src/array/${functionName}.spec.ts'], setupFiles: ['./capture-setup.mjs'], reporters: ['json'], outputFile: './vitest-results.json' } };\n`);
    writeFileSync(join(packageRoot, 'capture-setup.mjs'), `import ${JSON.stringify(`file://${captureRuntime}`)};\n`);

    const install = run('npm', ['install', '--ignore-scripts', '--no-audit', '--no-fund', `vitest@${VITEST_VERSION}`, `fast-check@${FAST_CHECK_VERSION}`], { cwd: packageRoot });
    const captureFile = join(packageRoot, 'captures.jsonl');
    run(process.execPath, [join(packageRoot, 'node_modules/vitest/vitest.mjs'), 'run', '--config', 'vitest.config.mjs'], {
      cwd: packageRoot, env: { ...process.env, CODE_CORPUS_CAPTURE: captureFile },
    });
    const results = JSON.parse(readFileSync(join(packageRoot, 'vitest-results.json'), 'utf8'));
    const resultFiles = results.testResults ?? [];
    const tests = resultFiles.flatMap((f) => f.assertionResults ?? []);
    const captures = existsSync(captureFile) ? readFileSync(captureFile, 'utf8').split('\n').filter(Boolean).map(JSON.parse) : [];
    const tasks = extractFunctions(source, { path: sourcePath, sourceName: UPSTREAM, revision: REVISION, license: 'MIT' });

    mkdirSync(dirname(output), { recursive: true });
    mkdirSync(output);
    const retained = join(output, 'upstream');
    mkdirSync(dirname(join(retained, sourcePath)), { recursive: true });
    writeFileSync(join(retained, sourcePath), source);
    writeFileSync(join(retained, specPath), spec);
    writeFileSync(join(output, `${functionName}.instrumented.mjs`), readFileSync(join(packageRoot, `src/array/${functionName}.ts`)));
    copyFileSync(join(packageRoot, 'package-lock.json'), join(output, 'package-lock.json'));
    copyFileSync(captureRuntime, join(output, 'capture-runtime.cjs'));
    writeJsonl(join(output, 'tasks.jsonl'), tasks);
    writeJsonl(join(output, 'captures.jsonl'), captures);
    writeFileSync(join(output, 'vitest-results.json'), JSON.stringify(results, null, 2));
    const manifest = {
      pilot: `es-toolkit/${functionName}`, upstream: `https://github.com/${UPSTREAM}`, revision: REVISION,
      source: { path: sourcePath, sha256: hash(source) }, spec: { path: specPath, sha256: hash(spec) },
      dependencies: { vitest: VITEST_VERSION, 'fast-check': FAST_CHECK_VERSION, install_scripts: false },
      execution: { command: 'vitest run --config vitest.config.mjs', test_files: resultFiles.length, tests: tests.length,
        passed: tests.filter((t) => t.status === 'passed').length, failed: tests.filter((t) => t.status === 'failed').length,
        captures: captures.length, portable_captures: captures.filter((c) => c.portable).length },
      hashes: { instrumented: hash(readFileSync(join(output, `${functionName}.instrumented.mjs`))), tasks: hash(readFileSync(join(output, 'tasks.jsonl'))), captures: hash(readFileSync(join(output, 'captures.jsonl'))), package_lock: hash(readFileSync(join(output, 'package-lock.json'))) },
      npm_install_note: install.stderr.trim().split('\n').slice(-2).join('\n'),
    };
    writeFileSync(join(output, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
    console.log(JSON.stringify(manifest, null, 2));
  } finally {
    rmSync(work, { recursive: true, force: true });
  }
}

main().catch((error) => { console.error(error.stack ?? String(error)); process.exitCode = 1; });
