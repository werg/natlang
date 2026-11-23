/**
 * Conformance programs: small projects of `.nl` and callable-folder `.ts` files, each run through
 * the runtime with a scripted reference agent in place of a model.
 */
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { resolve } from 'node:path';
import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { pathToFileURL } from 'node:url';
import YAML from 'yaml';
import { NatlangCallError, createNatlangRuntime, loadVirtualNatlang } from '../dist/index.js';
import { dump } from '../dist/native/values.js';

function valueKey(value) {
  if (typeof value === 'string') return value;
  if (Array.isArray(value)) return `[${value.map(x => JSON.stringify(x)).join(', ')}]`;
  return JSON.stringify(value);
}

/** The scripted reference for each invoked function, keyed by function name. */
function referenceAgent(specs) {
  return async session => {
    const lam = session.lam, spec = specs[lam.functionName];
    if (!spec) throw new Error(`no reference for ${lam.functionName}`);
    const act = async (tool, args) => {
      const result = ['eval', 'read_code', 'edit_code', 'diff_code'].includes(tool) ?
        await session.applyAsync(tool, args) : session.apply(tool, args);
      if (['error', 'rejected', 'refused'].includes(result.kind))
        throw new Error(`${tool} ${JSON.stringify(args)} -> ${result.kind}: ${result.text}`);
      return result;
    };
    const finish = async () => {
      if (!session.finish()) throw new Error(`reference did not finish: ${JSON.stringify(dump(lam.return))}`);
    };
    // A reference eval ends with an expression; its value is the function's result.
    const evaluate = code => act('eval', { code: `return await (async () => { ${code.replace(/;?\s*$/, '').replace(/([^;\n]*)$/, 'return $1;')} })()` });
    if (spec.steps) {
      for (const step of spec.steps) {
        const [tool, args] = Object.entries(step)[0] ?? [];
        if (!tool) throw new Error(`empty reference step for ${lam.functionName}`);
        const result = tool === 'eval' ? await evaluate(args) : await act(tool, args ?? {});
        if (result.kind === 'blocked') return result.text;
      }
      return finish();
    }
    if (spec.eval !== undefined) {
      const result = await evaluate(spec.eval);
      if (result.kind === 'blocked') return result.text;
      return finish();
    }
    let entry = spec;
    if (spec.variants) entry = spec.variants.find(v => !v.if_body_contains || lam.body.includes(v.if_body_contains)) ?? spec.variants.at(-1);
    if (entry.blocker) return (await act('return_result', { status: 'blocked', reason: entry.blocker })).text;
    let value = entry.answer;
    if (spec.answer_by) {
      const key = Object.hasOwn(lam.args, 'item') ? lam.args.item : Object.values(lam.args)[0] ?? null;
      value = spec.answer_by[valueKey(key)];
    }
    if (value === undefined) throw new Error(`no answer for ${lam.functionName}`);
    const result = await act('eval', { code: `return ${JSON.stringify(value)}` });
    if (result.kind === 'blocked') return result.text;
    return finish();
  };
}

const root = resolve(import.meta.dirname, '../..');
const folder = resolve(root, 'conformance/programs');
let passed = 0, failed = 0, skipped = 0;
for (const name of readdirSync(folder).filter(x => x.endsWith('.yaml')).sort()) {
  const doc = YAML.parse(readFileSync(resolve(folder, name), 'utf8'));
  if (!doc.reference || doc.reference.known_issue) { skipped++; continue; }
  try {
    const emitted = [];
    const runtime = createNatlangRuntime({ agent: referenceAgent(doc.reference),
      services: { out: { emit: record => { emitted.push(record); } } } });
    const fn = loadVirtualNatlang(doc.files, doc.root);
    const params = fn[Symbol.for('natlang.callable')].definition.params.map(param => param.name);
    let value, status = 'done';
    try { value = await runtime.run(() => fn(...params.map(param => doc.inputs?.[param]))); }
    catch (error) { if (!(error instanceof NatlangCallError)) throw error; status = error.outcome; value = error.detail; }
    const expected = doc.expect ?? {};
    if (expected.status && status !== expected.status) throw new Error(`wrong outcome ${status}; expected ${expected.status}`);
    if (!expected.status && status !== 'done') throw new Error(`${status}: ${value}`);
    if (expected.value !== undefined && JSON.stringify(value) !== JSON.stringify(expected.value))
      throw new Error(`wrong value ${JSON.stringify(value)}; expected ${JSON.stringify(expected.value)}`);
    if (expected.emitted && JSON.stringify(emitted) !== JSON.stringify(expected.emitted))
      throw new Error(`wrong emitted records ${JSON.stringify(emitted)}; expected ${JSON.stringify(expected.emitted)}`);
    for (const check of expected.checks ?? []) if (check?.kind === 'crisp' && !new Function('value', `return (${check.code});`)(value))
      throw new Error(`check failed: ${check.code}`);
    console.log(`PASS ${name}`); passed++;
  } catch (error) { console.log(`FAIL ${name}: ${error.message}`); failed++; }
}

// Behavioural Neuralese cases (conformance/neuralese/cases): an implemented case names its runnable form. A case whose
// form is a test file runs with the ts-host tests; a case whose form is a conformance module (test/conformance/) runs
// here, against a real Python native engine when its `engines` name one.
const caseFolder = resolve(root, 'conformance/neuralese/cases');
const only = process.env.NATLANG_CONFORMANCE_CASES?.split(',').filter(Boolean);
let pending = 0, server;
const environments = new Map();
/** One learning environment per case module, over one shared engine. */
const environment = (by, module) => {
  if (!environments.has(by)) environments.set(by, (async () => module.learningEnvironment(await (server ??= startNativeServer())))());
  return environments.get(by);
};
try {
  for (const name of readdirSync(caseFolder).filter(x => x.endsWith('.yaml')).sort()) {
    for (const entry of YAML.parse(readFileSync(resolve(caseFolder, name), 'utf8')).cases ?? []) {
      if (only && !only.includes(entry.id)) continue;
      if (entry.status !== 'implemented') { pending++; continue; }
      const label = `${name}#${entry.id}`;
      try {
        const by = entry.implemented_by;
        const paths = typeof by === 'string' ? by.split(',').map(path => path.trim().split('#')[0]) : [];
        if (!paths.length || paths.some(path => !existsSync(resolve(root, path)))) throw new Error(`implemented_by ${by} does not exist`);
        if (!by.startsWith('ts-host/test/conformance/')) { skipped++; continue; }
        const module = await import(pathToFileURL(resolve(root, by.split('#')[0])).href);
        const run = module.cases?.[entry.id];
        if (typeof run !== 'function') throw new Error(`${by} exports no case ${entry.id}`);
        const env = (entry.engines ?? []).includes('python-native') ? await environment(by, module) : {};
        const started = Date.now();
        await run(env);
        console.log(`PASS ${label} (${((Date.now() - started) / 1000).toFixed(1)} s)`); passed++;
      } catch (error) { console.log(`FAIL ${label}: ${error.stack ?? error.message}`); failed++; }
    }
  }
} finally { (await server?.catch(() => undefined))?.stop(); }

/**
 * The conformance engine (tests/neuralese/learning_conformance_server.py) on a tiny real backbone. Python comes from
 * NATLANG_PYTHON, else the repository's .venv-neuralese. A missing interpreter fails the cases that need it.
 */
async function startNativeServer() {
  const python = process.env.NATLANG_PYTHON || resolve(root, '.venv-neuralese/bin/python');
  if (!existsSync(python)) throw new Error(`the python-native conformance engine needs ${python} (set NATLANG_PYTHON)`);
  const child = spawn(python, [resolve(root, 'tests/neuralese/learning_conformance_server.py')], {
    cwd: root, stdio: ['pipe', 'pipe', 'inherit'],
    env: { ...process.env, PYTHONPATH: [resolve(root, 'training/neuralese'), root, process.env.PYTHONPATH].filter(Boolean).join(':'),
      OMP_NUM_THREADS: process.env.OMP_NUM_THREADS ?? '1', MKL_NUM_THREADS: process.env.MKL_NUM_THREADS ?? '1' } });
  const lines = createInterface({ input: child.stdout });
  const waiting = [];
  let exited;
  lines.on('line', line => waiting.shift()?.resolve(line));
  child.on('exit', code => { exited = new Error(`the conformance engine exited (${code})`); for (const item of waiting.splice(0)) item.reject(exited); });
  const next = () => new Promise((resolve, reject) => exited ? reject(exited) : waiting.push({ resolve, reject }));
  const ready = JSON.parse(await next());
  const command = async name => {
    child.stdin.write(name + '\n');
    const reply = JSON.parse(await next());
    if (reply.error) throw new Error(reply.error);
    return Object.values(reply)[0];
  };
  return { ...ready, command, stop: () => { child.stdin.end(); } };
}

console.log(JSON.stringify({ passed, failed, skipped, pending }));
if (failed) process.exitCode = 1;
