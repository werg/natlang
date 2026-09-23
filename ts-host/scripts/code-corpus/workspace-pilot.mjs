#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { access, cp, mkdir, mkdtemp, readFile, rename, rm, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, isAbsolute, join, relative, resolve, sep } from 'node:path';
import { spawnSync } from 'node:child_process';
import ts from 'typescript';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { extractFunctions } from './extract.mjs';
import { instrumentSource } from './capture.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const runtime = join(here, 'capture-runtime.cjs');
const sha256 = value => createHash('sha256').update(value).digest('hex');
const jsonl = rows => `${rows.map(row => JSON.stringify(row)).join('\n')}\n`;

function parseArgs(argv) {
  const get = key => { const i = argv.indexOf(key); return i < 0 ? undefined : argv[i + 1]; };
  return { workspace:get('--workspace'), source:get('--source'), test:get('--test'), names:argv.flatMap((arg,index)=>arg==='--function'?[argv[index+1]]:[]).filter(Boolean), output:get('--output'),
    license:get('--license'), instruction:get('--instruction'), execute:argv.includes('--execute') };
}
function run(command, args, options = {}) {
  const result = spawnSync(command, args, { encoding:'utf8', timeout:240_000, maxBuffer:32*1024*1024, ...options });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`${command} ${args.join(' ')} failed (${result.status ?? result.signal})\n${result.stderr || result.stdout}`);
  return result;
}
function safeRelative(value, label) {
  if (!value || isAbsolute(value)) throw new Error(`${label} must be a workspace-relative path`);
  const normalized = relative('/workspace', resolve('/workspace', value));
  if (!normalized || normalized === '..' || normalized.startsWith(`..${sep}`)) throw new Error(`${label} escapes the workspace`);
  return normalized;
}
async function fileHash(path) {
  try { return sha256(await readFile(path)); } catch (error) { if (error.code === 'ENOENT') return null; throw error; }
}
async function exists(path) { try { await access(path); return true; } catch (error) { if (error.code === 'ENOENT') return false; throw error; } }
async function preserveImportClosure(workspace, outputDir, startPaths) {
  const visited = new Set();
  const extensions = ['', '.ts', '.tsx', '.js', '.jsx', '.mjs', '.cjs', '.json', '/index.ts', '/index.js', '/index.mjs'];
  const visit = async relativePath => {
    if (visited.has(relativePath)) return;
    visited.add(relativePath);
    const source = await readFile(join(workspace,relativePath),'utf8');
    const destination = join(outputDir,relativePath);
    await mkdir(dirname(destination),{recursive:true});
    await writeFile(destination,source);
    const file = ts.createSourceFile(relativePath,source,ts.ScriptTarget.Latest,true,
      /\.tsx?$/.test(relativePath)?ts.ScriptKind.TS:ts.ScriptKind.JS);
    for(const statement of file.statements) {
      const specifier = ts.isImportDeclaration(statement) || ts.isExportDeclaration(statement)
        ? statement.moduleSpecifier?.text : null;
      if(!specifier?.startsWith('.')) continue;
      const base = resolve(dirname(join(workspace,relativePath)),specifier);
      if (base !== workspace && !base.startsWith(`${workspace}${sep}`))
        throw new Error(`Relative import escapes workspace: ${relativePath} imports ${specifier}`);
      let resolved;
      for(const extension of extensions) if(await exists(`${base}${extension}`)) { resolved=`${base}${extension}`; break; }
      if(resolved) await visit(relative(workspace,resolved));
    }
  };
  for(const path of startPaths) await visit(path);
  return [...visited].sort();
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (process.argv.includes('--help') || process.argv.includes('-h')) {
    console.log('Usage: workspace-pilot.mjs --execute --workspace APP_ROOT --source src/file.ts --test test/file.test.js --function NAME --output NEW_DIR [--license LICENSE] [--instruction TEXT]\nRuns one Node test file in a disposable workspace copy; requires APP_ROOT/package.json and an existing APP_ROOT/node_modules for dependencies.');
    return;
  }
  if (!args.execute || !args.workspace || !args.source || !args.test || !args.names.length || !args.output)
    throw new Error('Trusted execution is opt-in. Use --execute with --workspace, --source, --test, one or more --function options and --output.');
  const workspace = resolve(args.workspace), output = resolve(args.output);
  const sourcePath = safeRelative(args.source, '--source'), testPath = safeRelative(args.test, '--test');
  if (output === workspace || output.startsWith(`${workspace}${sep}`) || workspace.startsWith(`${output}${sep}`))
    throw new Error('output must be outside the input workspace');
  if (await exists(output)) throw new Error(`Refusing to overwrite existing output: ${output}`);
  const packageBytes = await readFile(join(workspace, 'package.json'));
  const pkg = JSON.parse(packageBytes);
  const sourceFile = join(workspace, sourcePath), testFile = join(workspace, testPath);
  const originalSource = await readFile(sourceFile, 'utf8');
  await readFile(testFile);
  const nodeModules = join(workspace, 'node_modules');
  if (!(await exists(nodeModules))) throw new Error('workspace/node_modules is required; install dependencies before capture');
  const revisionResult = spawnSync('git', ['-C', workspace, 'rev-parse', 'HEAD'], { encoding:'utf8' });
  const revision = revisionResult.status === 0 ? revisionResult.stdout.trim() : 'working-tree';
  const sourceName = pkg.name ?? 'local-workspace';
  const instrumented = instrumentSource(originalSource, { path:sourcePath, sourceName, revision });
  const instrumentedNames = new Set(instrumented.functions.filter(item=>item.instrumented).map(item=>item.name));
  const missingNames = args.names.filter(name=>!instrumentedNames.has(name));
  if (missingNames.length) throw new Error(`Functions were not instrumented: ${missingNames.join(', ')}`);
  const requestedNames = new Set(args.names);
  const tasks = extractFunctions(originalSource, { path:sourcePath, sourceName, revision, license:args.license ?? pkg.license, instruction:args.instruction })
    .filter(task => requestedNames.has(task.function.name));
  const extractedNames = new Set(tasks.map(task=>task.function.name));
  const unextractableNames = args.names.filter(name=>!extractedNames.has(name));
  if (unextractableNames.length) throw new Error(`Functions were not extractable: ${unextractableNames.join(', ')}; provide --instruction if they have no documentation`);

  const work = await mkdtemp(join(tmpdir(), 'natlang-workspace-pilot-'));
  await mkdir(dirname(output), { recursive:true });
  const artifact = await mkdtemp(join(dirname(output), '.workspace-pilot-building-'));
  try {
    const checkout = join(work, 'workspace');
    await cp(workspace, checkout, { recursive:true, filter: source => {
      const path = relative(workspace, source);
      return path !== '.git' && !path.startsWith(`.git${sep}`) && path !== 'node_modules' && !path.startsWith(`node_modules${sep}`);
    } });
    await symlink(nodeModules, join(checkout, 'node_modules'), 'dir');
    await writeFile(join(checkout, sourcePath), instrumented.code);
    const setupPath = join(checkout, '.code-corpus-test-setup.mjs');
    await writeFile(setupPath, `import { it } from 'node:test';\nimport ${JSON.stringify(pathToFileURL(runtime).href)};\nglobalThis.it = it;\n`);
    const capturePath = join(work, 'captures.jsonl');
    const testRun = run(process.execPath, ['--test', '--test-isolation=none', '--import', './.code-corpus-test-setup.mjs', testPath], {
      cwd:checkout, env:{ ...process.env, CODE_CORPUS_CAPTURE:capturePath },
    });
    const captures = await exists(capturePath) ? (await readFile(capturePath,'utf8')).split('\n').filter(Boolean).map(JSON.parse) : [];
    const selectedTasks = tasks.map(task => ({ ...task, cases:[] }));
    const selectedIds = new Set(selectedTasks.map(task => task.id));
    const selectedCaptures = captures.filter(capture => selectedIds.has(capture.key));
    await writeFile(join(artifact,'tasks.jsonl'), jsonl(selectedTasks));
    await writeFile(join(artifact,'captures.jsonl'), jsonl(selectedCaptures));
    await writeFile(join(artifact,'original-test.stdout.txt'), testRun.stdout);
    await writeFile(join(artifact,'original-test.stderr.txt'), testRun.stderr);
    await writeFile(join(artifact,'source.instrumented.js'), instrumented.code);
    const dependencyFiles = ['package.json','package-lock.json','npm-shrinkwrap.json','yarn.lock','pnpm-lock.yaml'];
    const dependencyManifest = Object.fromEntries(await Promise.all(dependencyFiles.map(async path => [path, await fileHash(join(workspace,path))])));
    const preservedPaths = await preserveImportClosure(workspace,join(artifact,'upstream'),[sourcePath,testPath]);
    for(const path of dependencyFiles) if(dependencyManifest[path]) {
      const destination=join(artifact,'upstream',path); await mkdir(dirname(destination),{recursive:true});
      await writeFile(destination,await readFile(join(workspace,path)));
    }
    const captureRuntimeResult = run(process.execPath, [join(here,'replay.mjs'),'--execute','--input',join(artifact,'tasks.jsonl'),
      '--captures',join(artifact,'captures.jsonl'),'--output',join(artifact,'native-replay.jsonl'),'--workspace',checkout],
    { cwd:resolve(here,'../../..') });
    await writeFile(join(artifact,'native-replay.stdout.txt'), captureRuntimeResult.stdout);
    const replay = JSON.parse(captureRuntimeResult.stdout.trim());
    const manifest = { pilot:`${sourceName}/${args.names.join(',')}`, functions:args.names, workspace:workspace, revision, source:{path:sourcePath,sha256:sha256(originalSource)},
      test:{path:testPath,sha256:await fileHash(testFile)}, package:{name:pkg.name ?? null,version:pkg.version ?? null,license:args.license ?? pkg.license ?? 'unknown',
        manifest_sha256:sha256(packageBytes),lockfile_sha256:dependencyManifest},
      imports_by_function:Object.fromEntries(tasks.map(task=>[task.function.name,task.function.imports])),
      preserved_paths:preservedPaths,
      execution:{command:`node --test --test-isolation=none --import ./.code-corpus-test-setup.mjs ${testPath}`,exit_code:testRun.status,
        captured_calls:captures.length,selected_function_captures:selectedCaptures.length,portable:selectedCaptures.filter(row=>row.portable).length,
        nonportable:selectedCaptures.filter(row=>!row.portable).length},
      native_replay:{...replay,workspace_before:undefined,workspace_after:undefined,
        rows_sha256:await fileHash(join(artifact,'native-replay.jsonl')),turns_sha256:await fileHash(join(artifact,'native-replay.jsonl.turns.jsonl'))},
      dependency_evidence:'package and lockfile hashes recorded; local workspace node_modules linked read-only by convention; source/test ran from a disposable copy'};
    await writeFile(join(artifact,'manifest.json'), `${JSON.stringify(manifest,null,2)}\n`);
    await rename(artifact,output);
    console.log(JSON.stringify(manifest,null,2));
  } finally { await rm(work,{recursive:true,force:true}); await rm(artifact,{recursive:true,force:true}); }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href)
  main().catch(error => { console.error(error.stack ?? String(error)); process.exitCode=1; });
