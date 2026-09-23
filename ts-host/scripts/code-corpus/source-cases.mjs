#!/usr/bin/env node
import ts from 'typescript';
import { fork } from 'node:child_process';
import { createHash, randomUUID } from 'node:crypto';
import { mkdir, readFile, rename, unlink, writeFile, link } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { readJsonl } from './common.mjs';

const digest = value => createHash('sha256').update(JSON.stringify(value)).digest('hex');
const CASE_TIMEOUT_MS = 900;
const safeMethods = new Set(['map','filter','reduce','reduceRight','slice','concat','flat','flatMap','find','findIndex','some','every','includes','indexOf','lastIndexOf','join','at','trim','trimStart','trimEnd','split','toLowerCase','toUpperCase','startsWith','endsWith','charAt','charCodeAt','codePointAt','fromCharCode','substring','substr','replace','replaceAll','padStart','padEnd','repeat','match','test','forEach','toString']);
const safeGlobals = new Set(['Math','Number','String','Boolean','Array','Object','JSON','Set','Map','NaN','Infinity','undefined','parseInt','parseFloat','isNaN','isFinite']);
const blocked = /\b(?:process|globalThis|global|require|module|exports|fetch|XMLHttpRequest|WebSocket|Date|performance|eval|Function|import|constructor|__proto__|prototype)\b/;

function typeShape(type) {
  const match = /^(string|number|boolean)((?:\[\]){0,2})$/.exec(type ?? '');
  if (!match) return null;
  return { primitive: match[1], depth: match[2].length / 2 };
}
function valuesFor(shape, edge) {
  if (shape.depth) {
    const child = { primitive: shape.primitive, depth: shape.depth - 1 };
    if (edge) return [];
    if (child.depth) return [valuesFor(child, false), valuesFor(child, true)];
    return shape.primitive === 'number' ? [0, 2, -1] : shape.primitive === 'boolean' ? [true, false] : ['', 'oak', 'blue'];
  }
  if (shape.primitive === 'number') return edge ? 0 : 3;
  if (shape.primitive === 'boolean') return edge ? false : true;
  return edge ? '' : 'sample text';
}
function safeRecord(record) {
  if (record.verification?.status === 'rejected') return 'source_task_rejected';
  if (record.kind !== 'function' || !['javascript','typescript'].includes(record.language)) return 'not_a_js_ts_function';
  if (record.function?.imports?.length) return 'has_imports';
  if (!Array.isArray(record.function?.parameters) || !record.function.parameters.length) return 'missing_parameters';
  if (record.function.parameters.some(parameter => !/^[A-Za-z_$][\w$]*$/.test(parameter.name) || !typeShape(parameter.type))) return 'parameter_type_not_portable';
  const knownHelpers = new Set((record.function.helpers ?? []).map(source => source.match(/function\s+([\w$]+)/)?.[1]).filter(Boolean));
  const text = `${(record.function.helpers ?? []).join('\n')}\n${record.function.body}`;
  if (blocked.test(text)) return 'unsafe_global_or_dynamic_code';
  const file = ts.createSourceFile('candidate.js', text, ts.ScriptTarget.Latest, true, ts.ScriptKind.JS);
  if (file.parseDiagnostics.length) return 'source_parse_error';
  let rejected = null;
  const visit = node => {
    if (rejected) return;
    if (ts.isNewExpression(node) || ts.isAwaitExpression(node) || ts.isYieldExpression(node) || ts.isDeleteExpression(node)) { rejected = 'effectful_or_async_syntax'; return; }
    if (ts.isBinaryExpression(node) && node.operatorToken.kind >= ts.SyntaxKind.FirstAssignment && node.operatorToken.kind <= ts.SyntaxKind.LastAssignment && (ts.isPropertyAccessExpression(node.left) || ts.isElementAccessExpression(node.left))) { rejected = 'property_mutation'; return; }
    if ((ts.isPrefixUnaryExpression(node) || ts.isPostfixUnaryExpression(node)) && (ts.isPropertyAccessExpression(node.operand) || ts.isElementAccessExpression(node.operand))) { rejected = 'property_mutation'; return; }
    if (ts.isCallExpression(node)) {
      const expression = node.expression;
      if (ts.isPropertyAccessExpression(expression)) {
        if (['sort','reverse','splice','push','pop','shift','unshift','fill','copyWithin'].includes(expression.name.text)) { rejected = 'mutating_method'; return; }
        if (!safeMethods.has(expression.name.text) && !(['max','min','abs','floor','ceil','round','trunc','sign','sqrt','pow','log','exp'].includes(expression.name.text) && ts.isIdentifier(expression.expression) && expression.expression.text === 'Math')) { rejected = `unsupported_call:${expression.name.text}`; return; }
      } else if (ts.isIdentifier(expression)) {
        if (!safeGlobals.has(expression.text) && !knownHelpers.has(expression.text)) { rejected = `unsupported_call:${expression.text}`; return; }
      } else { rejected = 'dynamic_call'; return; }
    }
    ts.forEachChild(node, visit);
  };
  visit(file);
  if (rejected) return rejected;
  if ((record.verification?.reasons ?? []).some(reason => /mutation detected|async function|generator function|rest parameters|callback arguments/.test(reason))) return 'extractor_flagged_effect_or_signature';
  const freeReason = (record.verification?.reasons ?? []).find(reason => reason.startsWith('free runtime bindings:'));
  if (freeReason) {
    const names = freeReason.slice('free runtime bindings:'.length).split(',').map(name => name.trim());
    if (names.some(name => !safeGlobals.has(name) && !knownHelpers.has(name))) return 'free_runtime_binding';
  }
  return null;
}

function portable(value, depth = 0) {
  if (depth > 4) return false;
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return true;
  if (typeof value === 'number') return Number.isFinite(value);
  return Array.isArray(value) && value.every(item => portable(item, depth + 1));
}

export function sourceCases(record) {
  const reason = safeRecord(record);
  if (reason) return { eligible: false, reason, cases: [] };
  const args = [0, 1].map(edge => record.function.parameters.map(parameter => valuesFor(typeShape(parameter.type), Boolean(edge))));
  return { eligible: true, reason: null, cases: args.map(values => ({ args: values })) };
}

export function observeSourceCase(record, args, timeout = CASE_TIMEOUT_MS) {
  return new Promise((resolvePromise, reject) => {
    const child = fork(new URL('./source-cases.mjs', import.meta.url), ['--worker'], { stdio: ['ignore','ignore','pipe','ipc'], execArgv: [] });
    let settled = false;
    const finish = (error, expected) => { if (settled) return; settled = true; clearTimeout(timer); child.kill('SIGKILL'); error ? reject(error) : resolvePromise(expected); };
    const timer = setTimeout(() => finish(new Error('source observation timeout')), timeout);
    child.on('message', message => message.error ? finish(new Error(message.error)) : finish(null, message.expected));
    child.on('error', finish);
    child.on('exit', code => finish(new Error(`source observer exited ${code}`)));
    child.send({ record, args });
  });
}

async function executeSourceCase(record, args) {
  const vm = await import('node:vm');
  const parameters = record.function.parameters.map(parameter => parameter.name);
  const fn = `function ${record.function.name || 'candidate'}(${parameters.join(',')}) ${record.function.body}`;
  const source = `${(record.function.helpers ?? []).join('\n')}\n(${fn})(...__args)`;
  const context = vm.createContext({ __args: structuredClone(args) }, { codeGeneration: { strings: false, wasm: false } });
  const output = vm.runInContext(source, context, { timeout: CASE_TIMEOUT_MS });
  if (!portable(output)) throw new Error('source result is not a finite primitive/array value');
  return structuredClone(output);
}

if (process.argv.includes('--worker')) process.once('message', async ({ record, args }) => {
  try { process.send({ expected: await executeSourceCase(record, args) }); }
  catch (error) { process.send({ error: String(error) }); }
});

async function atomicWrite(path, value) {
  const target = resolve(path); await mkdir(dirname(target), { recursive: true }); const staging = `${target}.building-${randomUUID()}`;
  try { await writeFile(staging, value, { flag: 'wx' }); await rename(staging, target); }
  finally { await unlink(staging).catch(() => {}); }
}
async function atomicCacheWrite(path, value) {
  const target=resolve(path), staging=`${target}.building-${randomUUID()}`;
  try { await writeFile(staging,value,{flag:'wx'}); try { await link(staging,target); } catch(error) { if(error.code!=='EEXIST') throw error; } }
  finally { await unlink(staging).catch(()=>{}); }
}

async function main() {
  const argv = process.argv.slice(2), get = key => { const i=argv.indexOf(key); return i<0 ? undefined : argv[i+1]; };
  if (argv.includes('--help') || argv.includes('-h')) { process.stdout.write('Usage: node scripts/code-corpus/source-cases.mjs --input TASKS.jsonl [--input MORE.jsonl ...] --output OBSERVED.jsonl [--limit N] [--execute]\nDefault cap is 500 functions across inputs. --execute evaluates only the audited pure primitive/array subset in disposable timeout children; outputs are source observations, not upstream test labels. An unchanged command resumes from per-case cache.\n'); return; }
  const inputs=argv.flatMap((arg,index)=>arg==='--input'?[argv[index+1]]:[]).filter(Boolean), output=get('--output'); if(!inputs.length||!output) throw new Error('--input and --output are required');
  const limit=Number(get('--limit')??500); if(!Number.isSafeInteger(limit)||limit<1) throw new Error('--limit must be a positive integer');
  const batches=await Promise.all(inputs.map(path=>readJsonl(path,{limit}))),tasks=[];
  for(let index=0;tasks.length<limit;index++) {
    let added=false;
    for(const batch of batches) { if(batch[index] && tasks.length<limit) { tasks.push(batch[index]); added=true; } }
    if(!added) break;
  }
  const execute=argv.includes('--execute'), inputHashes=await Promise.all(inputs.map(async path=>[resolve(path),digest(await readFile(path))]));
  const sourceHash=digest(await readFile(new URL('./source-cases.mjs',import.meta.url)));
  const config={generator:'natlang.code_source_observations/1',inputs:inputHashes,limit,execute,timeout_ms:CASE_TIMEOUT_MS,source_cases_sha256:sourceHash,node:process.version,typescript:ts.version};
  const manifestPath=`${output}.manifest.json`,cacheDir=`${output}.cache`;
  try { const prior=JSON.parse(await readFile(manifestPath,'utf8')); if(JSON.stringify(prior.config)!==JSON.stringify(config)) throw new Error('output path belongs to a different immutable source-observation configuration'); }
  catch(error) { if(error.code!=='ENOENT') throw error; }
  await mkdir(cacheDir,{recursive:true});
  await atomicWrite(manifestPath,`${JSON.stringify({version:'natlang.code_source_observations_manifest/1',config,status:'in_progress'},null,2)}\n`);
  let interrupted=false; const checkpoint=signal=>{interrupted=true;process.stderr.write(`\n${signal}: checkpointing after the current source case\n`);};
  process.once('SIGINT',()=>checkpoint('SIGINT'));process.once('SIGTERM',()=>checkpoint('SIGTERM'));
  const out = [], reasons = {}, counters = { input: tasks.length, eligible: 0, observed_tasks: 0, observed_cases: 0, rejected: 0, execution_enabled: execute };
  for (const task of tasks) {
    if (task.source?.name === 'xlam-function-calling-60k' || task.kind === 'tool_calls') { counters.rejected++; reasons.never_execute_external_api = (reasons.never_execute_external_api ?? 0) + 1; continue; }
    const candidate = sourceCases(task);
    if (!candidate.eligible) { counters.rejected++; reasons[candidate.reason] = (reasons[candidate.reason] ?? 0) + 1; continue; }
    counters.eligible++;
    const row = { ...task, cases: [], observation: { kind:'source_execution_candidate', execution_enabled:counters.execution_enabled,
      expected_source:'original_function_body_and_captured_sibling_helpers', not_upstream_tests:true, verified:false } };
    if (counters.execution_enabled) {
      for (const item of candidate.cases) {
        if(interrupted) break;
        const cachePath=resolve(cacheDir,`${digest([config,task.id,task.function,item.args]).slice(0,40)}.json`);
        let observed;
        try { observed=JSON.parse(await readFile(cachePath,'utf8')); }
        catch(error) {
          if(error.code!=='ENOENT') throw error;
          try { observed={expected:await observeSourceCase(task,item.args)}; }
          catch(observationError) { observed={error:String(observationError)}; }
          await atomicCacheWrite(cachePath,`${JSON.stringify(observed)}\n`);
        }
        if(observed.error) { reasons.execution_error=(reasons.execution_error??0)+1;row.cases=[];row.observation.rejection=observed.error;break; }
        row.cases.push({...item,expected:observed.expected,outcome:'return',portable:true,provenance:'observed_from_original_source'});
      }
      if(interrupted) break;
      if (row.cases.length) { row.observation.verified = false; row.observation.kind = 'source_derived_observations'; counters.observed_tasks++; counters.observed_cases += row.cases.length; }
      else { counters.rejected++; continue; }
    }
    out.push(row);
  }
  const data = out.map(row => JSON.stringify(row)).join('\n') + (out.length ? '\n' : '');
  await atomicWrite(output, data);
  const report = { version:'natlang.code_source_observations/1', inputs:inputHashes.map(([path])=>path), input_sha256:digest(tasks), output:resolve(output), ...counters,
    by_source:Object.fromEntries([...new Set(tasks.map(task=>task.source?.name??'unknown'))].sort().map(name=>[name,tasks.filter(task=>task.source?.name===name).length])), rejected_by_reason:reasons,
    interpretation:counters.execution_enabled ? 'Outputs were observed by calling inspected original source in a disposable timeout process. They are not upstream test labels and require separate native replay before training admission.' : 'Dry-run eligibility report only.' };
  await atomicWrite(`${output}.report.json`, `${JSON.stringify(report, null, 2)}\n`);
  await atomicWrite(manifestPath,`${JSON.stringify({version:'natlang.code_source_observations_manifest/1',config,status:interrupted?'interrupted':'complete',...report},null,2)}\n`);
  if(interrupted) process.exitCode=75;
  process.stdout.write(JSON.stringify(report)+'\n');
}
if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href && !process.argv.includes('--worker'))
  main().catch(error => { process.stderr.write(`${error}\n`); process.exitCode = 1; });
