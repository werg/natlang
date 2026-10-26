#!/usr/bin/env node
/** Compile isolated native application packages; no model requests. */
import { readdir, readFile, mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { dirname, join, resolve } from 'node:path';
import { checkProject } from '../../dist/compiler/node-project.js';
import { FIXTURE_ROOT, REPO, SOURCE_ROOT, json, readApplication, safeRelative, sourceDigest, sourceFiles } from './lib.mjs';

const args = process.argv.slice(2);
if (args[0] === '--app' && (args.length === 2 || (args.length === 4 && args[2] === '--result'))) {
  const name = safeRelative(args[1]);
  const root = join(REPO, SOURCE_ROOT, name), sources = await sourceFiles(root), before = sourceDigest(sources);
  // Check the same explicit snapshot that the host and training packager load.
  // Retired working-tree files and generated declarations are not part of that program.
  await mkdir(join(REPO, '.natlang'), { recursive: true });
  const frozen = await mkdtemp(join(REPO, '.natlang/design-brief-source-'));
  let result;
  try {
    for (const [name, text] of Object.entries(sources)) {
      const path = join(frozen, name); await mkdir(dirname(path), { recursive: true }); await writeFile(path, text);
    }
    result = checkProject(frozen, { write: false, writeDeclarations: false });
  } finally { await rm(frozen, { recursive: true, force: true }); }
  let application, applicationError;
  try {
    const sourceIndex = JSON.parse(await readFile(join(REPO, FIXTURE_ROOT, 'source-manifest.json'), 'utf8'));
    application = await readApplication(REPO, name, sourceIndex);
  } catch (error) { applicationError = String(error.message); }
  const after = sourceDigest(await sourceFiles(root));
  const report = { application: name, ok: result.ok && !applicationError && before === after, diagnostics: result.diagnostics,
    definitions: result.manifest.named.length, source_sha256: before, source_stable: before === after,
    compiler_input: 'the explicit frozen native source inventory used by the host and packet',
    ...(applicationError ? { application_error: applicationError } : { interfaces: application.interfaces,
      native_functions: application.native_functions, domain_contracts: application.domain_contracts,
      fixture_tasks: application.records.length, callable_scope: application.callable_scope,
      source_coverage: application.source_coverage,
      potential_calls: application.potential_calls }) };
  if (args[2] === '--result') await writeFile(resolve(args[3]), json(report), { flag: 'wx' });
  else console.log(JSON.stringify(report));
  process.exitCode = report.ok ? 0 : 1;
} else {
  if (args.length !== 2 || args[0] !== '--report') throw new Error('usage: node check.mjs --report REPORT.json | --app NNN-slug');
  const names = (await readdir(join(REPO, SOURCE_ROOT), { withFileTypes: true }))
    .filter(entry => entry.isDirectory() && /^\d{3}-/.test(entry.name)).map(entry => entry.name).sort();
  const results = [];
  await mkdir(join(REPO, '.natlang'), { recursive: true });
  const scratch = await mkdtemp(join(REPO, '.natlang/design-brief-compiler-'));
  for (const name of names) {
    const resultPath = join(scratch, name + '.json');
    const child = spawnSync(process.execPath, [process.argv[1], '--app', name, '--result', resultPath], { cwd: REPO, encoding: 'utf8',
      maxBuffer: 16 * 1024 * 1024 });
    let result;
    try {
      result = JSON.parse(await readFile(resultPath, 'utf8'));
      if (result.application !== name || !/^[a-f0-9]{64}$/.test(result.source_sha256))
        throw new Error('compiler result identity mismatch');
    }
    catch { result = { application: name, ok: false, error: child.error?.message ?? child.stderr ?? child.stdout }; }
    if (child.status !== 0) result.ok = false;
    results.push(result);
    console.log(`${name}: ${result.ok ? 'pass' : 'FAIL'}`);
  }
  const report = { schema: 'natlang.design-brief-compiler-check/1', applications: names.length,
    passed: results.filter(item => item.ok).length, all_passed: names.length === 100 && results.every(item => item.ok),
    evidence: 'actual repository checkProject; no live-model execution', results };
  const path = resolve(args[1]);
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, json(report));
  await rm(scratch, { recursive: true, force: true });
  console.log(JSON.stringify({ applications: report.applications, passed: report.passed, report: path }));
  process.exitCode = report.all_passed ? 0 : 1;
}
