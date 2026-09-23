import { isDeepStrictEqual } from 'node:util';
import { resolve, dirname, relative, sep } from 'node:path';
import ts from 'typescript';
import { pathToFileURL } from 'node:url';
import { fork } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { digest, readJsonl, writeJsonl } from './common.mjs';

async function workspaceSnapshot(workspace) {
  if (!workspace) return null;
  const hashes = {};
  for (const name of ['package.json', 'package-lock.json']) {
    try { hashes[name] = createHash('sha256').update(await readFile(resolve(workspace, name))).digest('hex'); }
    catch (error) { if (error.code !== 'ENOENT') throw error; hashes[name] = null; }
  }
  return { path: resolve(workspace), hashes, node: process.version };
}

// Deliberately small portable subset. Types are inferred across cases, never literal answers.
function infer(values) {
  let arrayType;
  const types = [...new Set(values.map(value => {
    if (value === null) return 'null';
    if (['string', 'boolean'].includes(typeof value)) return typeof value;
    if (typeof value === 'number' && Number.isFinite(value)) return 'number';
    if (Array.isArray(value)) {
      if (arrayType) return arrayType;
      const items = values.filter(Array.isArray).flat();
      if (!items.length) throw new Error('empty-only array requires a declared type adapter');
      return arrayType = `(${infer(items)})[]`;
    }
    throw new Error('portable replay currently supports primitives and arrays only');
  }))];
  return types.join(' | ');
}
export function attachCaptures(records, captures) {
  return records.map(record => ({ ...record, cases: captures.filter(c => c.key === record.id).map(c => ({
    args: c.args, expected: c.expected, outcome: c.outcome, input_after: c.input_after,
    portable: c.portable, reasons: c.reasons,
  })) }));
}
export async function materializeCorpus(rows) {
  const { materializeNativeRows } = await import('../../dist/teacher/native-materializer.js');
  const result = materializeNativeRows(rows);
  result.turns = result.turns.map(turn => {
    const program = turn.task.program_ir;
    return { ...turn, source_groups:program.source_groups, source_program_ids:program.source_ids,
      license:program.license, split:program.split, family:program.family ?? 'code_corpus', source:program.source,
      generation:program.generation, behavioral_evidence:program.behavioral_evidence,
      implementation_sha256:program.implementation_sha256,
      execution_verified:true };
  });
  return result;
}
export function project(record, index = 0) {
  if (record.verification?.status === 'rejected') throw new Error('rejected source cannot be projected');
  if (record.kind !== 'function' || !record.instruction?.trim()) throw new Error('requires a documented function');
  const cases = record.cases.filter(c => c.outcome === 'return' && c.portable !== false);
  const observations = new Map();
  for (const c of cases) {
    const key = digest(c.args);
    if (observations.has(key) && !isDeepStrictEqual(observations.get(key), c.expected)) throw new Error('conflicting outputs for identical arguments');
    observations.set(key,c.expected);
  }
  const item = cases[index];
  if (!item) throw new Error('no portable return case');
  const params = record.function.parameters;
  if (params.some(p => !/^[A-Za-z_$][\w$]*$/.test(p.name))) throw new Error('unsupported parameter binding');
  if (cases.some(c => c.args.length !== params.length)) throw new Error('argument arity mismatch');
  if (item.input_after && !isDeepStrictEqual(item.args, item.input_after)) throw new Error('mutating invocation deferred');
  const boundaryType = (values, declared) => {
    try { return infer(values); }
    catch (error) {
      if (/empty-only array/.test(error.message) && typeof declared === 'string'
          && /^(?:number|string|boolean|null)(?:\[\])+$/.test(declared.trim())) return declared.trim();
      throw error;
    }
  };
  const fields = params.map((p, i) => `${p.name}: ${boundaryType(cases.map(c => c.args[i]), p.type)}`);
  const body = record.function.body.trim().replace(/^\{/, '').replace(/\}$/, '');
  if (!body.trim()) throw new Error('missing function body');
  const id = `code:${digest([record.id, item.args]).slice(0, 24)}`;
  return { code: body, expected: item.expected, program: {
    version: 'natlang.program/1', id, kind: 'lambda_source', source: record.source.name,
    source_ids: [record.id], source_groups: [record.group_id], license: record.source.license,
    family: record.generation?.family ?? record.family ?? 'code_corpus', generation:record.generation,
    implementation_sha256: digest(record.function),
    behavioral_evidence: record.behavioral_evidence ?? {kind:record.observation ? 'source_observed' : record.generation ? 'generated_reference' : 'captured_return',
      note:'Native replay checks fidelity to supplied expected values, not full specification correctness.'},
    split: Number.parseInt(digest(record.group_id).slice(0, 8), 16) % 100 < 5 ? 'test' : 'train',
    semantics: { root: { $lambda: { type: `(${fields.join(', ')}) => ${boundaryType(cases.map(c => c.expected), record.function.return_type)}`,
      instructions: record.instruction.replace(/\s+/g, ' ').trim() } }, inputs: Object.fromEntries(params.map((p, i) => [p.name, item.args[i]])),
      expected: item.expected, operation: 'exact' },
  } };
}

/** Executes trusted code. CLI isolates each invocation in a time-limited child, not a security sandbox. */
export async function replayCase(record, index = 0, options = {}) {
  const { TypeScriptEnvironment } = await import('../../dist/environment.js');
  const { applicationCapabilityPrompt } = await import('../../dist/application-packages.js');
  const { NativeToolAgent } = await import('../../dist/native/agent.js');
  const { EXPLICIT_TOOLS_PROMPT } = await import('../../dist/native/prompt.js');
  const { NativeRuntime } = await import('../../dist/native/runtime.js');
  const { TypeEnv } = await import('../../dist/native/types.js');
  const { buildPending, coerce, dump } = await import('../../dist/native/values.js');
  const projection = project(record, index);
  const { program, expected } = projection;
  let code = projection.code;
  if (record.function.helpers?.length) {
    const helpers = record.function.helpers.join('\n');
    if (record.function.helpers.length > 32 || helpers.length > 32000) throw new Error('sibling helper context exceeds replay budget');
    // Block-local declarations preserve hoisting without persisting function handles.
    code = '{\n'+helpers+'\n'+code+'\n}';
  }
  if (record.function.imports?.length) {
    if (!options.workspace) throw new Error('dependency-bearing replay requires --workspace');
    const base = dirname(resolve(options.workspace,record.source.path));
    const imports=record.function.imports.map(item=>{
      const parsed=ts.createSourceFile('import.ts',item.source,ts.ScriptTarget.Latest,true);
      const declaration=parsed.statements[0];
      if(!declaration||!ts.isImportDeclaration(declaration)) throw new Error('invalid captured import');
      const specifier=item.specifier.startsWith('.')
        ? './'+relative(resolve(options.workspace),resolve(base,item.specifier)).split(sep).join('/') : item.specifier;
      return ts.createPrinter().printNode(ts.EmitHint.Unspecified,ts.factory.updateImportDeclaration(declaration,declaration.modifiers,
        declaration.importClause,ts.factory.createStringLiteral(specifier),declaration.attributes),parsed);
    });
    code=imports.join('\n')+'\n'+code;
  }
  const root = buildPending(program.semantics.root);
  const env = new TypeEnv().child(root.types);
  for (const [name, value] of Object.entries(program.semantics.inputs)) {
    const field = root.type.params.fields.find(f => f.name === name);
    root.args[name] = coerce(structuredClone(value), field.type, env, `args/${name}`);
  }
  const trajectory = [];
  const driver = async request => {
    if (trajectory.length >= 2) throw new Error('replay exceeded two model turns');
    const lastAction = runtime.trace.events.filter(e => e.kind === 'action').at(-1);
    const calls = trajectory.length ? (lastAction && lastAction.outcome !== 'ok'
      ? [['report_error', { message: `Captured implementation failed during replay: ${lastAction.result_text}` }]]
      : [['mark_lines', { start: 1 }]]) : [['eval', { code }]];
    trajectory.push({ phase: 'action', context: structuredClone(request.messages), tools_offered: request.tools,
      assistant: { content: '', reasoning: null, calls: calls.map(([tool, args]) => ({tool, source_tool: tool, arguments: args, call_id: null})) }, raw_response_sha256: null });
    return { calls };
  };
  const workspaceBefore = await workspaceSnapshot(options.workspace);
  const hostEvents = [];
  const environment = new TypeScriptEnvironment({ mode: 'fresh', workspace: options.workspace, network: options.network,
    observe: event => hostEvents.push(event) });
  const agent = new NativeToolAgent(driver, { systemPrompt: EXPLICIT_TOOLS_PROMPT + applicationCapabilityPrompt(environment.scopeCapabilities), segmentTurns: 3, segmentMessages: 12, validationFeedback: 'caller' });
  const runtime = new NativeRuntime({ environment, agent: session => agent.run(session), seedPolicy: {mode:'derived', root: 42}, runId: program.id });
  try {
    const result = await runtime.runRoot(root);
    const actual = dump(result.value);
    return { version: 'natlang.teacher_trajectory.native/1', id: program.id,
      task: {kind:'whole_program', program_ir:program, source_program_ids:[record.id]},
      provenance: { source:record.source, code_task_sha256:digest(record), runtime:'typescript-native', tool_schema:'scope-eval-v1', projection:'captured-bound-arguments/body-v1',
        workspace_before: workspaceBefore, workspace_after: await workspaceSnapshot(options.workspace),
        capabilities: environment.scopeCapabilities, verification: 'observed-output-equality' },
      outcome: { status:result.outcome.kind, detail:result.outcome.detail, value:actual,
        effects:{ host_events:hostEvents, complete:false, replayable:false },
        accepted:result.outcome.kind === 'done' && isDeepStrictEqual(actual, expected),
        action_ledger:runtime.trace.events.filter(e => e.kind === 'action') }, trajectory, capture_limits:[] };
  } finally { runtime.close(); environment.close(); }
}

export function replayIsolated(record, index, timeout = 10000, options = {}) {
  return new Promise((accept, reject) => {
    const child = fork(new URL('./replay.mjs', import.meta.url), ['--worker'], {stdio:['ignore','ignore','pipe','ipc'], execArgv:[]});
    let settled = false;
    const finish = (error, row) => { if (settled) return; settled = true; clearTimeout(timer); child.kill('SIGKILL'); error ? reject(error) : accept(row); };
    const timer = setTimeout(() => finish(new Error('replay timeout')), timeout);
    child.on('message', message => finish(message.error ? new Error(message.error) : null, message.row));
    child.on('error', error => finish(error));
    child.on('exit', code => finish(new Error(`replay worker exited ${code}`)));
    child.send({record,index,options});
  });
}
if (process.argv.includes('--worker')) process.once('message', async ({record,index,options}) => {
  try { process.send({row:await replayCase(record,index,options)}); } catch (error) { process.send({error:String(error)}); }
});
else if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const args = process.argv.slice(2);
  const get = key => args[args.indexOf(key)+1];
  if (!args.includes('--execute') || !args.includes('--input') || !args.includes('--output')) throw new Error('Usage: replay.mjs --execute --input TASKS --output ROWS [--captures JSONL] [--limit N] [--cases N] [--workspace APP_ROOT]. Trusted code only; no security sandbox. Workspace enables imports, installation and network access.');
  let tasks = await readJsonl(get('--input'), {limit:args.includes('--limit') ? Number(get('--limit')) : 100});
  if (args.includes('--captures')) tasks = attachCaptures(tasks, await readJsonl(get('--captures')));
  const rows = [], rejected = [];
  const cap = args.includes('--cases') ? Number(get('--cases')) : 20;
  if (!Number.isInteger(cap) || cap < 1) throw new Error('--cases must be a positive integer');
  for (const task of tasks) {
    const cases=task.cases.filter(c => c.outcome === 'return' && c.portable !== false);
    const seen=new Set();
    if (!cases.length) rejected.push({id:task.id,error:'no portable successful return cases'});
    for (let index=0; index<cases.length && seen.size<cap; index++) {
      const key=digest(cases[index].args); if (seen.has(key)) continue; seen.add(key);
      try { rows.push(await replayIsolated(task,index,10000,{workspace:args.includes('--workspace')?resolve(get('--workspace')):undefined})); } catch (error) { rejected.push({id:task.id,index,error:String(error)}); }
    }
  }
  await writeJsonl(get('--output'), rows);
  await writeJsonl(`${get('--output')}.rejected.jsonl`, rejected);
  const result = await materializeCorpus(rows);
  await writeJsonl(`${get('--output')}.turns.jsonl`, result.turns);
  console.log(JSON.stringify({tasks:tasks.length, trajectories:rows.length, accepted:rows.filter(r=>r.outcome.accepted).length, turns:result.turns.length, rejected:rejected.length}));
}
