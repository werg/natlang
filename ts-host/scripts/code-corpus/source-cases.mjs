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
function valueProfiles(shape) {
  if (shape.depth === 0) {
    if (shape.primitive === 'number') return [3, 0, -1, 2, 2, Number.MAX_SAFE_INTEGER];
    if (shape.primitive === 'boolean') return [true, false, true, false, true, false];
    return ['sample text', '', '🙂', 'aa', 'aa', 'e\u0301'];
  }
  const child = { primitive: shape.primitive, depth: shape.depth - 1 };
  if (child.depth) return [[], [valueProfiles(child)[1]], [valueProfiles(child)[2], valueProfiles(child)[2]], [valueProfiles(child)[3], valueProfiles(child)[0], valueProfiles(child)[3]], [valueProfiles(child)[0], valueProfiles(child)[1], valueProfiles(child)[0]]];
  if (shape.primitive === 'number') return [[], [0], [2, 2, -1], [-1, 0, 1], [1, 1, 1], [Number.MAX_SAFE_INTEGER, 0, -Number.MAX_SAFE_INTEGER]];
  if (shape.primitive === 'boolean') return [[], [false], [true, true, false], [false, true], [true, true, true], [false, false]];
  return [[], [''], ['aa', 'aa', '🙂'], ['🙂', '', '🙂'], ['e\u0301', 'é', 'e\u0301'], ['x', 'x']];
}

function generatedArguments(parameters) {
  const profiles = parameters.map(parameter => valueProfiles(typeShape(parameter.type)));
  const count = Math.max(6, ...profiles.map(profile => profile.length));
  const seen = new Set();
  const correlated = Array.from({ length: count }, (_, index) => profiles.map(profile => profile[index % profile.length]));
  // Equal-valued parameters alone miss subtraction, ordering, and tie branches.
  // Vary one parameter at a time, bounded independently of unusual arities.
  const independent = profiles.slice(0, 8).map((_, varied) => profiles.map((profile, index) => profile[index === varied ? 2 : 0]));
  return [...correlated, ...independent]
    .filter(args => { const key = JSON.stringify(args); if (seen.has(key)) return false; seen.add(key); return true; })
    .map(args => ({ args, input_source: 'deterministic_boundary_generator' }));
}

function generatedOrObserved(record) {
  return Boolean(record.generation || record.observation || record.cases?.some(item => item.provenance === 'observed_from_original_source' || item.input_source === 'deterministic_boundary_generator'));
}
function upstreamArguments(record) {
  return (record.cases ?? []).map((item, index) => {
    const args = item?.args;
    const explicitlyAsserted = item?.provenance === 'upstream_test' || item?.provenance === 'upstream_assertion'
      || (record.case_provenance === 'upstream_tests' && !generatedOrObserved(record));
    const captured = !explicitlyAsserted && (Array.isArray(item?.input_after) || item?.provenance === 'runtime_capture'
      || (record.verification?.status === 'captured' && !generatedOrObserved(record)));
    const inputSource = explicitlyAsserted ? 'upstream_case' : captured ? 'runtime_capture' : 'existing_task_case';
    if (!Array.isArray(args) || args.length !== record.function.parameters.length || item.portable === false || !['return', undefined].includes(item.outcome))
      return { rejected: true, input_source: inputSource, upstream_case_index: index, ...(item.capture_key ? { capture_key:item.capture_key } : {}),
        ...(item.reasons ? { capture_reasons:item.reasons } : {}), rejection: inputSource === 'runtime_capture' ? 'unsupported_runtime_capture' : 'unsupported_upstream_case_shape_or_outcome' };
    if (!args.every((value, i) => portable(value) && matchesType(value, typeShape(record.function.parameters[i].type))))
      return { rejected:true, input_source:inputSource, upstream_case_index:index, ...(item.capture_key ? { capture_key:item.capture_key } : {}),
        rejection:inputSource === 'runtime_capture' ? 'runtime_capture_argument_type_mismatch' : 'upstream_argument_type_mismatch' };
    return { args: structuredClone(args), input_source: inputSource, upstream_case_index: index,
      ...(item.capture_key ? { capture_key:item.capture_key } : {}),
      ...(Object.hasOwn(item, 'expected') ? { ...(explicitlyAsserted ? { upstream_asserted_expected: structuredClone(item.expected), upstream_asserted_expected_portable: portable(item.expected) }
        : captured ? { captured_expected:structuredClone(item.expected), captured_input_after:item.input_after === undefined ? null : structuredClone(item.input_after) }
          : { existing_expected: structuredClone(item.expected) }) } : {}) };
  });
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
function samePortable(a,b) { return JSON.stringify(a)===JSON.stringify(b); }
function matchesType(value, shape) {
  if (shape.depth) return Array.isArray(value) && value.every(item => matchesType(item, { primitive: shape.primitive, depth: shape.depth - 1 }));
  return shape.primitive === 'number' ? typeof value === 'number' && Number.isFinite(value)
    : shape.primitive === 'boolean' ? typeof value === 'boolean' : typeof value === 'string';
}

export function sourceCases(record) {
  const reason = safeRecord(record);
  if (reason) return { eligible: false, reason, cases: [] };
  return { eligible: true, reason: null, cases: [...upstreamArguments(record), ...generatedArguments(record.function.parameters)] };
}

export function observeSourceCase(record, args, timeout = CASE_TIMEOUT_MS) {
  return new Promise((resolvePromise, reject) => {
    const child = fork(new URL('./source-cases.mjs', import.meta.url), ['--worker'], { stdio: ['ignore','ignore','pipe','ipc'], execArgv: [] });
    let settled = false;
    const finish = (error, expected) => { if (settled) return; settled = true; clearTimeout(timer); child.kill('SIGKILL'); error ? reject(error) : resolvePromise(expected); };
    const timer = setTimeout(() => finish(new Error('source observation timeout')), timeout);
    child.on('message', message => message.error ? finish(new Error(message.error)) : finish(null, message.observation));
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
  const before=JSON.stringify(context.__args);
  const output = vm.runInContext(source, context, { timeout: CASE_TIMEOUT_MS });
  if (!portable(output)) throw new Error('source result is not a finite primitive/array value');
  const inputAfter=structuredClone(context.__args);
  if(before!==JSON.stringify(inputAfter)) throw new Error('source function mutated invocation inputs');
  return {expected:structuredClone(output),input_after:inputAfter};
}

if (process.argv.includes('--worker')) process.once('message', async ({ record, args }) => {
  try { process.send({ observation: await executeSourceCase(record, args) }); }
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
async function outputHashes(paths) {
  return Object.fromEntries(await Promise.all(paths.map(async path=>[path,digest(await readFile(path))])));
}
async function validateOutputHashes(hashes) {
  for(const [path,expected] of Object.entries(hashes??{})) {
    const actual=digest(await readFile(path));
    if(actual!==expected) throw new Error(`completed source-observation output integrity check failed for ${path}; refusing silent repair`);
  }
}

async function main() {
  const argv = process.argv.slice(2), get = key => { const i=argv.indexOf(key); return i<0 ? undefined : argv[i+1]; };
  if (argv.includes('--help') || argv.includes('-h')) { process.stdout.write('Usage: node scripts/code-corpus/source-cases.mjs --input TASKS.jsonl [--input MORE.jsonl ...] [--captures CAPTURES.jsonl ...] --output OBSERVED.jsonl [--limit N] [--execute]\nDefault cap is 500 functions across inputs. Captures attach by capture.key === task.id and remain source-runtime observations, not upstream test assertions. --execute evaluates only the audited pure primitive/array subset in disposable timeout children; outputs are source observations, not upstream test labels. An unchanged command resumes from per-case cache.\n'); return; }
  const inputs=argv.flatMap((arg,index)=>arg==='--input'?[argv[index+1]]:[]).filter(Boolean);
  const captureInputs=argv.flatMap((arg,index)=>arg==='--captures'?[argv[index+1]]:[]).filter(Boolean);
  const output=get('--output'); if(!inputs.length||!output) throw new Error('--input and --output are required');
  const limit=Number(get('--limit')??500); if(!Number.isSafeInteger(limit)||limit<1) throw new Error('--limit must be a positive integer');
  const batches=await Promise.all(inputs.map(path=>readJsonl(path,{limit}))),tasks=[];
  for(let index=0;tasks.length<limit;index++) {
    let added=false;
    for(const batch of batches) { if(batch[index] && tasks.length<limit) { tasks.push(batch[index]); added=true; } }
    if(!added) break;
  }
  const captureBatches=await Promise.all(captureInputs.map(path=>readJsonl(path)));
  const captures=captureBatches.flat(), captureByKey=new Map();
  for(const capture of captures) { const list=captureByKey.get(capture.key)??[]; list.push(capture); captureByKey.set(capture.key,list); }
  const selectedTaskIds=new Set(tasks.map(task=>task.id));
  const matchedCaptureCases=captures.filter(capture=>selectedTaskIds.has(capture.key)).length;
  const tasksWithCaptures=tasks.map(task=>{
    const matched=captureByKey.get(task.id)??[];
    const appended=matched.map(capture=>({args:capture.args,expected:capture.expected,outcome:capture.outcome,input_after:capture.input_after,
      portable:capture.portable,reasons:capture.reasons,provenance:'runtime_capture',capture_key:capture.key}));
    return appended.length?{...task,cases:[...(task.cases??[]),...appended]}:task;
  });
  const unmatchedCaptureCases=captures.length-matchedCaptureCases;
  const execute=argv.includes('--execute'), inputHashes=await Promise.all([...inputs,...captureInputs].map(async path=>[resolve(path),digest(await readFile(path))]));
  const sourceHash=digest(await readFile(new URL('./source-cases.mjs',import.meta.url)));
  const config={generator:'natlang.code_source_observations/1',inputs:inputHashes,
    task_inputs:inputs.map(path=>resolve(path)),capture_inputs:captureInputs.map(path=>resolve(path)),
    limit,execute,timeout_ms:CASE_TIMEOUT_MS,source_cases_sha256:sourceHash,node:process.version,typescript:ts.version};
  const manifestPath=`${output}.manifest.json`,cacheDir=`${output}.cache`;
  const outputPath=resolve(output),reportPath=resolve(`${output}.report.json`);
  try {
    const prior=JSON.parse(await readFile(manifestPath,'utf8'));
    if(JSON.stringify(prior.config)!==JSON.stringify(config)) throw new Error('output path belongs to a different immutable source-observation configuration');
    if(['complete','interrupted'].includes(prior.status)&&prior.output_sha256) await validateOutputHashes(prior.output_sha256);
  }
  catch(error) { if(error.code!=='ENOENT') throw error; }
  await mkdir(cacheDir,{recursive:true});
  await atomicWrite(manifestPath,`${JSON.stringify({version:'natlang.code_source_observations_manifest/1',config,status:'in_progress'},null,2)}\n`);
  let interrupted=false; const checkpoint=signal=>{if(interrupted)return;interrupted=true;process.stderr.write(`\n${signal}: checkpointing after the current source case\n`);};
  process.on('SIGINT',()=>checkpoint('SIGINT'));process.on('SIGTERM',()=>checkpoint('SIGTERM'));
  const out = [], reasons = {}, counters = { input: tasks.length, eligible: 0, generated_cases_considered: 0, upstream_cases_considered: 0,
    observed_tasks: 0, observed_cases: 0, observed_upstream_cases: 0, observed_generated_cases: 0, diagnostic_cases:0, quarantined_tasks:0,
    case_rejections: 0, rejected: 0, execution_enabled: execute };
  Object.assign(counters, { upstream_assertions_checked:0, upstream_assertions_matched:0, upstream_assertions_mismatched:0,
    upstream_assertion_execution_failures:0, conflicting_duplicate_input_groups:0 });
  for (const task of tasksWithCaptures) {
    if (task.source?.name === 'xlam-function-calling-60k' || task.kind === 'tool_calls') { counters.rejected++; reasons.never_execute_external_api = (reasons.never_execute_external_api ?? 0) + 1; continue; }
    const candidate = sourceCases(task);
    if (!candidate.eligible) { counters.rejected++; reasons[candidate.reason] = (reasons[candidate.reason] ?? 0) + 1; continue; }
    counters.eligible++;
    counters.upstream_cases_considered += candidate.cases.filter(item => item.input_source === 'upstream_case').length;
    counters.generated_cases_considered += candidate.cases.filter(item => item.input_source === 'deterministic_boundary_generator').length;
    const row = { ...task, cases: [], observation: { kind:'source_execution_candidate', execution_enabled:counters.execution_enabled,
      expected_source:'original_function_body_and_captured_sibling_helpers', not_upstream_tests:true, verified:false } };
    if (counters.execution_enabled) {
      const caseRejections = [];
      const upstreamExecutionFailures = [];
      let upstreamAssertionsChecked = 0, upstreamAssertionsMatched = 0;
      for (const item of candidate.cases) {
        if(interrupted) break;
        if (item.rejected) { caseRejections.push({ input_source:item.input_source, upstream_case_index:item.upstream_case_index, rejection:item.rejection }); reasons[item.rejection] = (reasons[item.rejection] ?? 0) + 1; continue; }
        if (Object.hasOwn(item, 'upstream_asserted_expected') && !item.upstream_asserted_expected_portable) {
          caseRejections.push({ input_source:item.input_source, upstream_case_index:item.upstream_case_index, args:item.args, rejection:'upstream_expected_not_portable' }); reasons.upstream_expected_not_portable = (reasons.upstream_expected_not_portable ?? 0) + 1; continue;
        }
        const cacheKey=digest([config,task.id,task.function,item.args]);
        const cachePath=resolve(cacheDir,`${cacheKey.slice(0,40)}.json`);
        let cached;
        try { cached=JSON.parse(await readFile(cachePath,'utf8')); }
        catch(error) {
          if(error.code!=='ENOENT') throw error;
          let observation;
          try { observation=await observeSourceCase(task,item.args); }
          catch(observationError) { observation={error:String(observationError)}; }
          cached={cache_key:cacheKey,observation,payload_sha256:digest(observation)};
          await atomicCacheWrite(cachePath,`${JSON.stringify(cached)}\n`);
        }
        if(cached.cache_key!==cacheKey||cached.payload_sha256!==digest(cached.observation))
          throw new Error(`source-observation cache integrity check failed for ${task.id}; refusing silent repair`);
        if(cached.observation.error) {
          reasons.execution_error=(reasons.execution_error??0)+1;
          const rejection={input_source:item.input_source,upstream_case_index:item.upstream_case_index,args:item.args,rejection:cached.observation.error};
          caseRejections.push(rejection);
          if (item.input_source === 'upstream_case' && Object.hasOwn(item, 'upstream_asserted_expected')) upstreamExecutionFailures.push(rejection);
          continue;
        }
        if(!samePortable(item.args,cached.observation.input_after)) {
          reasons.input_mutation=(reasons.input_mutation??0)+1;
          const rejection={input_source:item.input_source,upstream_case_index:item.upstream_case_index,args:item.args,rejection:'source function mutated invocation inputs'};
          caseRejections.push(rejection);
          if (item.input_source === 'upstream_case' && Object.hasOwn(item, 'upstream_asserted_expected')) upstreamExecutionFailures.push(rejection);
          continue;
        }
        const upstreamMatch = Object.hasOwn(item, 'upstream_asserted_expected')
          ? samePortable(item.upstream_asserted_expected, cached.observation.expected) : null;
        if (upstreamMatch !== null) { upstreamAssertionsChecked++; if (upstreamMatch) upstreamAssertionsMatched++; }
        row.cases.push({...item,expected:cached.observation.expected,input_after:cached.observation.input_after,outcome:'return',portable:true,provenance:'observed_from_original_source',
          ...(item.input_source === 'upstream_case' ? { upstream_assertion_status: upstreamMatch === null ? 'no_expected_value' : upstreamMatch ? 'matched_source_observation' : 'mismatched_source_observation' } : {}),
          ...(item.input_source === 'runtime_capture' ? { captured_output_status: !portable(item.captured_expected) ? 'not_comparable' : samePortable(item.captured_expected, cached.observation.expected) ? 'matched_source_observation' : 'different_source_observation' } : {})});
      }
      if(interrupted) break;
      row.observation.case_rejections = caseRejections;
      counters.case_rejections = (counters.case_rejections ?? 0) + caseRejections.length;
      const assertionsByArgs = new Map();
      for (const item of row.cases.filter(item => item.input_source === 'upstream_case' && Object.hasOwn(item, 'upstream_asserted_expected'))) {
        const key = JSON.stringify(item.args), values = assertionsByArgs.get(key) ?? new Set();
        values.add(JSON.stringify(item.upstream_asserted_expected)); assertionsByArgs.set(key, values);
      }
      const conflictingArgs = new Set([...assertionsByArgs].filter(([, values]) => values.size > 1).map(([key]) => key));
      const mismatches = row.cases.filter(item => item.upstream_assertion_status === 'mismatched_source_observation');
      const captureMismatches = row.cases.filter(item => item.captured_output_status === 'different_source_observation');
      const duplicateConflicts = row.cases.filter(item => item.input_source === 'upstream_case' && conflictingArgs.has(JSON.stringify(item.args)));
      row.observation.behavioral_evidence = { kind:'source_observed', scope:'observed_inputs_only_not_specification_verification',
        upstream_assertions_checked:upstreamAssertionsChecked, upstream_assertions_matched:upstreamAssertionsMatched,
        upstream_assertions_mismatched:mismatches.length, upstream_assertion_execution_failures:upstreamExecutionFailures.length,
        capture_fidelity_mismatches:captureMismatches.length,
        conflicting_duplicate_input_groups:conflictingArgs.size };
      row.behavioral_evidence = row.observation.behavioral_evidence;
      counters.upstream_assertions_checked += upstreamAssertionsChecked;
      counters.upstream_assertions_matched += upstreamAssertionsMatched;
      counters.upstream_assertions_mismatched += mismatches.length;
      counters.upstream_assertion_execution_failures += upstreamExecutionFailures.length;
      counters.conflicting_duplicate_input_groups += conflictingArgs.size;
      if (mismatches.length || duplicateConflicts.length || upstreamExecutionFailures.length || captureMismatches.length) {
        counters.diagnostic_cases += row.cases.length;
        counters.quarantined_tasks++;
        row.observation.kind = (mismatches.length || duplicateConflicts.length || upstreamExecutionFailures.length)
          ? 'upstream_assertion_conflict' : 'capture_fidelity_conflict';
        row.observation.rejection = mismatches.length ? 'upstream_assertion_mismatch' : duplicateConflicts.length ? 'conflicting_upstream_assertions' : upstreamExecutionFailures.length ? 'upstream_assertion_execution_failure' : 'capture_fidelity_mismatch';
        row.observation.diagnostic_cases = row.cases;
        row.observation.case_rejections.push(...duplicateConflicts.map(item => ({ input_source:'upstream_case', upstream_case_index:item.upstream_case_index,
          args:item.args, rejection:'conflicting_upstream_assertions' })));
        row.cases = [];
        const rejectionReasons = [...new Set([...(task.verification?.reasons ?? []), ...(mismatches.length ? ['upstream_assertion_mismatch'] : []),
          ...(duplicateConflicts.length ? ['conflicting_upstream_assertions'] : []), ...(upstreamExecutionFailures.length ? ['upstream_assertion_execution_failure'] : []),
          ...(captureMismatches.length ? ['capture_fidelity_mismatch'] : [])])];
        row.verification = { ...(task.verification ?? {}), status:'rejected', reasons:rejectionReasons };
        counters.case_rejections += duplicateConflicts.length;
        counters.rejected++;
        reasons[row.observation.rejection] = (reasons[row.observation.rejection] ?? 0) + 1;
        if (duplicateConflicts.length) reasons.conflicting_upstream_assertions = (reasons.conflicting_upstream_assertions ?? 0) + 1;
        if (upstreamExecutionFailures.length) reasons.upstream_assertion_execution_failure = (reasons.upstream_assertion_execution_failure ?? 0) + 1;
        out.push(row);
        continue;
      }
      if (row.cases.length) {
        row.observation.verified = false; row.observation.kind = 'source_derived_observations'; counters.observed_tasks++; counters.observed_cases += row.cases.length;
        counters.observed_upstream_cases += row.cases.filter(item => item.input_source === 'upstream_case').length;
        counters.observed_generated_cases += row.cases.filter(item => item.input_source === 'deterministic_boundary_generator').length;
      }
      else { counters.rejected++; continue; }
    }
    out.push(row);
  }
  const data = out.map(row => JSON.stringify(row)).join('\n') + (out.length ? '\n' : '');
  await atomicWrite(outputPath, data);
  counters.matched_capture_cases=matchedCaptureCases;
  counters.unmatched_capture_cases=unmatchedCaptureCases;
  const report = { version:'natlang.code_source_observations/1', inputs:inputHashes.map(([path])=>path), task_inputs:inputs.map(path=>resolve(path)), capture_inputs:captureInputs.map(path=>resolve(path)),
    input_sha256:digest([tasks,captures]), output:resolve(output), ...counters,
    by_source:Object.fromEntries([...new Set(tasksWithCaptures.map(task=>task.source?.name??'unknown'))].sort().map(name=>[name,tasksWithCaptures.filter(task=>task.source?.name===name).length])), rejected_by_reason:reasons,
    interpretation:counters.execution_enabled ? 'Outputs were observed by calling inspected original source in a disposable timeout process. They are not upstream test labels and require separate native replay before training admission.' : 'Dry-run eligibility report only.' };
  await atomicWrite(reportPath, `${JSON.stringify(report, null, 2)}\n`);
  const status=interrupted?'interrupted':'complete';
  await atomicWrite(manifestPath,`${JSON.stringify({version:'natlang.code_source_observations_manifest/1',config,status,...report,
    output_sha256:await outputHashes([outputPath,reportPath])},null,2)}\n`);
  if(interrupted) process.exitCode=75;
  process.stdout.write(JSON.stringify(report)+'\n');
}
if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href && !process.argv.includes('--worker'))
  main().catch(error => { process.stderr.write(`${error}\n`); process.exitCode = 1; });
