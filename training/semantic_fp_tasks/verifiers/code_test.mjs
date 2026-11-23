// Trusted artifact-execution verifier (ts-host/src/teacher/trusted-exact-verifier.ts):
// Executes a candidate JavaScript/TypeScript transform or module returned by the model (either a code string,
// `{ code: string }`, or a named file in `files[oracle.file]`) in a fresh vm context and verifies it against hidden
// test cases and invariants. Unlocks semantic-etl, codemod-studio, and code-synth tasks.
//
// Isolation: the verifier runs inside the trusted host process, so no host object ever reaches candidate code. The
// context has its own intrinsics, no string code generation, arguments enter as JSON text parsed inside it, and every
// result and error leaves as text produced inside it (only vm errors such as timeouts reach the host's catch). It runs
// in a worker thread with a memory limit: each synchronous step has the vm timeout, and the main thread terminates the
// worker when a case (a promise-chain loop, say) outlasts its timeout.
//
// oracle: {
//   file?: string,                 // read candidate code from folder `files[file]` instead of `value`
//   export?: string,               // named function to call (default: `transform`, or default export / bare function)
//   cases: [{ name?: string, args: unknown[], expected?: unknown,      // nonempty
//             assert?: { equals?: unknown, minLength?: number, keys?: string[], noNulls?: boolean } }],
//   forbidden_patterns?: string[], // regex strings the candidate source must not contain (e.g. hardcoded test outputs)
//   timeout_ms?: number,           // per-case execution timeout (default 500)
// }
import { stripTypeScriptTypes } from 'node:module';
import vm from 'node:vm';
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads';

function canon(value) {
  if (Array.isArray(value)) return value.map(canon);
  if (value && typeof value === 'object') return Object.fromEntries(Object.keys(value).sort().map(k => [k, canon(value[k])]));
  return value === undefined ? null : value;
}
const same = (a, b) => JSON.stringify(canon(a)) === JSON.stringify(canon(b));

function extractCode(value, files, oracle) {
  if (oracle.file) return typeof files?.[oracle.file] === 'string' ? files[oracle.file] : null;
  if (typeof value === 'string') return value;
  if (value && typeof value === 'object' && typeof value.code === 'string') return value.code;
  return null;
}

/** TypeScript annotations removed (JavaScript passes through unchanged). */
function stripTypes(source) {
  const emit = process.emitWarning;
  process.emitWarning = () => {};
  try { return stripTypeScriptTypes(source); } finally { process.emitWarning = emit; }
}

/** The candidate's module evaluated in a fresh context, with the called function bound to `__fn` inside it. */
function compileCandidate(source, exportName, timeout) {
  const exportedFns = [];
  const cleaned = stripTypes(source.replace(/^\s*```[a-zA-Z]*\n/, '').replace(/\n```\s*$/, ''))
    .replace(/\bexport\s+default\s+/g, 'module.exports.default = ')
    .replace(/\bexport\s+(async\s+)?function\s+([A-Za-z_$][\w$]*)/g, (_, asyncKw, name) => {
      exportedFns.push(name);
      return `${asyncKw ?? ''}function ${name}`;
    })
    .replace(/\bexport\s+(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=/g, 'const $1 = module.exports.$1 =');
  const exportFooter = exportedFns.map(name => `\nif (typeof ${name} === 'function') module.exports.${name} = ${name};`).join('');
  const ctx = vm.createContext(Object.create(null), { codeGeneration: { strings: false, wasm: false }, });
  const body = !exportedFns.length && /^\s*(?:\(|[A-Za-z_$][\w$]*\s*=>|function\b|async\s+function\b)/.test(cleaned)
    ? `module.exports.default = (${cleaned});` : `${cleaned}${exportFooter}`;
  vm.runInContext(`var module = { exports: {} }; var exports = module.exports; var __err;
try {\n${body}\n} catch (error) { try { __err = String(error && error.message || error); } catch { __err = 'the module threw'; } }`, ctx, { timeout });
  const failed = vm.runInContext(`typeof __err === 'string' ? __err : null`, ctx, { timeout });
  if (typeof failed === 'string') throw new Error(failed);
  const name = JSON.stringify(exportName ?? '');
  const found = vm.runInContext(`(() => {
    const mod = module.exports, name = ${name};
    globalThis.__fn = (name && typeof mod[name] === 'function' ? mod[name] : null)
      ?? (typeof mod.default === 'function' ? mod.default : null)
      ?? (typeof mod.transform === 'function' ? mod.transform : null)
      ?? (name && typeof globalThis[name] === 'function' ? globalThis[name] : null);
    return typeof globalThis.__fn === 'function';
  })()`, ctx, { timeout });
  return found === true ? ctx : null;
}

/** Call `__fn` on JSON arguments inside the context: the result as JSON text, or the error as text. */
async function runCase(ctx, args, timeout) {
  const json = JSON.stringify(JSON.stringify(args ?? []));
  vm.runInContext(`globalThis.__out = undefined; globalThis.__err = undefined;
    try {
      Promise.resolve(__fn(...JSON.parse(${json}))).then(
        value => { globalThis.__out = JSON.stringify(value === undefined ? null : value); },
        error => { try { globalThis.__err = String(error && error.message || error); } catch { globalThis.__err = 'threw'; } });
    } catch (error) { try { globalThis.__err = String(error && error.message || error); } catch { globalThis.__err = 'threw'; } }`, ctx, { timeout });
  const read = expression => {
    const value = vm.runInContext(expression, ctx, { timeout });
    return typeof value === 'string' ? value : null;
  };
  const deadline = Date.now() + timeout;
  for (;;) {
    const error = read(`typeof __err === 'string' ? __err : null`);
    if (error !== null) throw new Error(error);
    const out = read(`typeof __out === 'string' ? __out : null`);
    if (out !== null) return JSON.parse(out);
    if (Date.now() > deadline) throw new Error(`no result within ${timeout}ms (the promise did not settle)`);
    await new Promise(resolve => setImmediate(resolve));
  }
}

/** Worker side: compile, then run every case, reporting each as it finishes. */
async function workerMain({ source, exportName, cases, timeout }) {
  let ctx;
  try {
    ctx = compileCandidate(source, exportName, timeout);
  } catch (error) {
    parentPort.postMessage({ type: 'compile', error: String(error?.message ?? error) });
    return;
  }
  parentPort.postMessage({ type: 'compile', missing: !ctx });
  if (!ctx) return;
  for (let index = 0; index < cases.length; index++) {
    try {
      parentPort.postMessage({ type: 'case', index, got: await runCase(ctx, cases[index].args, timeout) });
    } catch (error) {
      parentPort.postMessage({ type: 'case', index, error: String(error?.message ?? error) });
    }
  }
}

if (!isMainThread && workerData?.codeTest) await workerMain(workerData.codeTest);

/** Run the candidate in a worker: `{ compileError?, missing?, results }` with one `{ got } | { error }` per case. A step
 * that outlasts its timeout (plus a second for scheduling) terminates the worker; later cases count as timed out. */
function execute(source, exportName, cases, timeout) {
  return new Promise(resolve => {
    const outcome = { results: [] };
    const worker = new Worker(new URL(import.meta.url), {
      workerData: { codeTest: { source, exportName, cases: cases.map(tc => ({ args: tc.args ?? [] })), timeout } },
      resourceLimits: { maxOldGenerationSizeMb: 256, maxYoungGenerationSizeMb: 32 },
    });
    let timer;
    const finish = () => { clearTimeout(timer); worker.terminate(); resolve(outcome); };
    const arm = () => { clearTimeout(timer); timer = setTimeout(() => {
      for (let index = outcome.results.length; index < cases.length; index++) outcome.results.push({ error: `timed out after ${timeout}ms` });
      if (outcome.compileError === undefined && outcome.missing === undefined) outcome.compileError = `timed out after ${timeout}ms`;
      finish();
    }, timeout * 2 + 1000); };
    worker.on('message', message => {
      if (message.type === 'compile') {
        if (message.error !== undefined) { outcome.compileError = message.error; return finish(); }
        outcome.missing = message.missing;
        if (message.missing) return finish();
      } else {
        outcome.results[message.index] = 'got' in message ? { got: message.got } : { error: message.error };
        if (outcome.results.length === cases.length) return finish();
      }
      arm();
    });
    worker.on('error', error => { outcome.compileError ??= `the candidate crashed its worker: ${error?.message ?? error}`; finish(); });
    worker.on('exit', () => {
      for (let index = outcome.results.length; index < cases.length; index++) outcome.results.push({ error: 'the worker exited' });
      clearTimeout(timer); resolve(outcome);
    });
    arm();
  });
}

export async function verifyCandidate({ value, files, oracle }) {
  const cases = oracle.cases ?? [];
  if (!Array.isArray(cases) || !cases.length) throw new Error('code_test verifier: oracle.cases must be nonempty');
  const issues = [];
  const timeout = oracle.timeout_ms ?? 500;
  const source = extractCode(value, files, oracle);
  if (!source || !source.trim()) return { ok: false, issues: ['candidate provided no executable code'], score: 0 };
  for (const pattern of oracle.forbidden_patterns ?? []) {
    if (new RegExp(pattern).test(source)) issues.push(`forbidden pattern matched: ${pattern}`);
  }
  const outcome = await execute(source, oracle.export, cases, timeout);
  if (outcome.compileError !== undefined) return { ok: false, issues: [...issues, `compile error: ${outcome.compileError}`], score: 0 };
  if (outcome.missing) return { ok: false, issues: [...issues, `no callable ${oracle.export ?? 'default'} function exported`], score: 0 };

  let passed = 0;
  for (let i = 0; i < cases.length; i++) {
    const tc = cases[i];
    const label = tc.name ?? `case ${i + 1}`;
    try {
      const result = outcome.results[i];
      if (!result || 'error' in result) throw new Error(result?.error ?? 'no result');
      const got = result.got;
      if ('expected' in tc) {
        if (same(got, tc.expected)) passed++;
        else issues.push(`${label}: expected ${JSON.stringify(tc.expected).slice(0, 120)}, got ${JSON.stringify(got)?.slice(0, 120)}`);
      } else if (tc.assert) {
        let ok = true;
        if ('equals' in tc.assert && !same(got, tc.assert.equals)) { ok = false; issues.push(`${label}: value mismatch`); }
        if (tc.assert.minLength !== undefined && (!Array.isArray(got) || got.length < tc.assert.minLength)) { ok = false; issues.push(`${label}: length < ${tc.assert.minLength}`); }
        if (tc.assert.keys && (!got || typeof got !== 'object' || !tc.assert.keys.every(k => Object.hasOwn(got, k)))) { ok = false; issues.push(`${label}: missing keys`); }
        if (tc.assert.noNulls && Array.isArray(got) && got.some(v => v === null || v === undefined)) { ok = false; issues.push(`${label}: contains null/undefined`); }
        if (ok) passed++;
      } else {
        passed++;
      }
    } catch (error) {
      issues.push(`${label}: threw ${error.message}`);
    }
  }
  const total = cases.length + (oracle.forbidden_patterns?.length ? 1 : 0);
  const patternCredit = (oracle.forbidden_patterns?.length && issues.every(s => !s.startsWith('forbidden'))) ? 1 : 0;
  const score = total ? (passed + patternCredit) / total : (issues.length ? 0 : 1);
  return { ok: issues.length === 0, issues, score: issues.length ? Math.min(score, 0.999) : 1 };
}
