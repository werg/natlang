// Trusted artifact-execution verifier (ts-host/src/teacher/trusted-exact-verifier.ts):
// Executes a candidate JavaScript/TypeScript transform or module returned by the model (either a code string,
// `{ code: string }`, or a named file in `files[oracle.file]`) in a fresh vm context per case and verifies it against
// hidden test cases and invariants. Unlocks semantic-etl, codemod-studio, and code-synth tasks.
//
// Isolation: the verifier runs inside the trusted host process, so no host object ever reaches candidate code. Each
// test case gets its own fresh vm context with untouched intrinsics closed over before candidate code executes, no
// string code generation, arguments entering as JSON text parsed inside it, and every result and error leaving as text
// produced inside it (only vm errors such as timeouts reach the host's catch). It runs in a worker thread with a memory
// limit: each synchronous step has the vm timeout, and the main thread terminates the worker when a case (a
// promise-chain loop, say) outlasts its timeout.
//
// oracle: {
//   file?: string,                 // read candidate code from folder `files[file]` instead of `value`
//   export?: string,               // named function to call (default: `transform`, or default export / bare function)
//   cases: [{ name?: string, args: unknown[], expected?: unknown,      // nonempty
//             assert?: { equals?: unknown, minLength?: number, keys?: string[], noNulls?: boolean } }],
//   forbidden_patterns?: string[], // regex strings the candidate source must not contain (e.g. hardcoded test outputs)
//   timeout_ms?: number,           // per-case execution timeout (default 500)
// }
import crypto from 'node:crypto';
import { stripTypeScriptTypes } from 'node:module';
import vm from 'node:vm';
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads';

function canon(value) {
  if (Array.isArray(value)) return value.map(canon);
  if (value && typeof value === 'object') return Object.fromEntries(Object.keys(value).sort().map(k => [k, canon(value[k])]));
  return value === undefined ? null : value;
}
const same = (a, b) => JSON.stringify(canon(a)) === JSON.stringify(canon(b));

function hasNullish(value) {
  if (value === null || value === undefined) return true;
  if (Array.isArray(value)) return value.some(hasNullish);
  if (typeof value === 'object') return Object.values(value).some(hasNullish);
  return false;
}

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

function prepareBody(source) {
  const exportedBindings = [];
  let sawExport = false;
  const cleaned = stripTypes(source.replace(/^\s*```[a-zA-Z]*\n/, '').replace(/\n```\s*$/, ''))
    .replace(/\bexport\s+default\s+/g, () => { sawExport = true; return 'module.exports.default = '; })
    .replace(/\bexport\s+(async\s+)?function\s+([A-Za-z_$][\w$]*)/g, (_, asyncKw, name) => {
      sawExport = true;
      exportedBindings.push([name, name]);
      return `${asyncKw ?? ''}function ${name}`;
    })
    .replace(/\bexport\s+(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=/g, (_, name) => {
      sawExport = true;
      return `const ${name} = module.exports.${name} =`;
    })
    .replace(/\bexport\s*\{([^}]+)\}\s*;?/g, (_, list) => {
      sawExport = true;
      for (const part of list.split(',')) {
        const trimmed = part.trim();
        if (!trimmed) continue;
        const m = /^([A-Za-z_$][\w$]*)(?:\s+as\s+([A-Za-z_$][\w$]*))?$/.exec(trimmed);
        if (m) exportedBindings.push([m[1], m[2] ?? m[1]]);
      }
      return '';
    });
  const exportFooter = exportedBindings
    .map(([local, exported]) => `\nif (typeof ${local} !== 'undefined') module.exports[${JSON.stringify(exported)}] = ${local};`)
    .join('');
  if (!sawExport && !/\b(?:module\.)?exports\b/.test(cleaned) &&
      /^\s*(?:\(|[A-Za-z_$][\w$]*\s*=>|async\s*(?:\(|[A-Za-z_$][\w$]*\s*=>)|(?:async\s+)?function\b)/.test(cleaned)) {
    const expr = cleaned.trim().replace(/;+\s*$/, '');
    const wrapped = `module.exports.default = (${expr});`;
    try {
      new vm.Script(wrapped);
      return wrapped;
    } catch {
      // Multi-function or multi-statement script: run unwrapped.
    }
  }
  return `${cleaned}${exportFooter}`;
}

/** Compile the candidate inside a fresh vm context with closed-over intrinsics bound to an unenumerable lexical handle. */
function compileCandidate(body, exportName, timeout) {
  const ctx = vm.createContext(Object.create(null), { codeGeneration: { strings: false, wasm: false } });
  const handle = `__h_${crypto.randomBytes(16).toString('hex')}`;
  // Capture untouched intrinsics in a declarative lexical binding (not on globalThis) before candidate code runs.
  vm.runInContext(`const ${handle} = (() => {
    const _Promise = Promise;
    const _Promise_then = Promise.prototype.then;
    const _JSON_parse = JSON.parse;
    const _JSON_stringify = JSON.stringify;
    const _String = String;
    let fn = null;
    let compileErr = null;
    let done = false;
    let outStr = null;
    let errStr = null;
    const errText = e => {
      try { return _String(e && typeof e === 'object' && 'message' in e && e.message ? e.message : e); }
      catch { return 'threw'; }
    };
    return {
      failCompile(e) { compileErr = errText(e); },
      compileError() { return compileErr; },
      bind(mod, exportName) {
        const candidate = (exportName && typeof mod[exportName] === 'function' ? mod[exportName] : null)
          ?? (typeof mod.default === 'function' ? mod.default : null)
          ?? (typeof mod.transform === 'function' ? mod.transform : null)
          ?? (exportName && typeof globalThis[exportName] === 'function' ? globalThis[exportName] : null)
          ?? (!exportName && typeof globalThis.transform === 'function' ? globalThis.transform : null);
        if (typeof candidate === 'function') { fn = candidate; return true; }
        return false;
      },
      start(argsJson) {
        done = false; outStr = null; errStr = null;
        const finishValue = value => {
          if (done) return;
          try {
            const serialized = _JSON_stringify(value === undefined ? null : value);
            if (typeof serialized !== 'string') {
              errStr = 'result is not JSON-serializable';
            } else {
              outStr = serialized;
            }
          } catch (e) {
            errStr = 'result is not JSON-serializable: ' + errText(e);
          }
          done = true;
        };
        const finishError = e => {
          if (done) return;
          errStr = errText(e);
          done = true;
        };
        try {
          const ret = fn(..._JSON_parse(argsJson));
          const settled = new _Promise((resolve, reject) => {
            try { resolve(ret); } catch (e) { reject(e); }
          });
          _Promise_then.call(settled, finishValue, finishError);
        } catch (e) {
          finishError(e);
        }
      },
      poll() {
        if (!done) return 0;
        return errStr !== null ? 'E:' + errStr : 'O:' + outStr;
      },
    };
  })();`, ctx, { timeout });
  vm.runInContext(`var module = { exports: {} }; var exports = module.exports;
try {\n${body}\n} catch (error) { ${handle}.failCompile(error); }`, ctx, { timeout });
  const failed = vm.runInContext(`${handle}.compileError()`, ctx, { timeout });
  if (typeof failed === 'string') throw new Error(failed);
  const found = vm.runInContext(`${handle}.bind(module.exports, ${JSON.stringify(exportName ?? '')})`, ctx, { timeout });
  return found === true ? { ctx, handle } : null;
}

/** Call the bound candidate function inside its isolated context: returns the parsed result or throws its error text. */
async function runCase({ ctx, handle }, args, timeout) {
  const json = JSON.stringify(JSON.stringify(args ?? []));
  vm.runInContext(`${handle}.start(${json})`, ctx, { timeout });
  const deadline = Date.now() + timeout;
  for (;;) {
    const status = vm.runInContext(`${handle}.poll()`, ctx, { timeout });
    if (typeof status === 'string') {
      if (status.startsWith('E:')) throw new Error(status.slice(2));
      if (status.startsWith('O:')) return JSON.parse(status.slice(2));
    }
    if (Date.now() > deadline) throw new Error(`no result within ${timeout}ms (the promise did not settle)`);
    await new Promise(resolve => setImmediate(resolve));
  }
}

/** Worker side: compile, then run every case in its own fresh VM context, reporting each as it finishes. */
async function workerMain({ source, exportName, cases, timeout }) {
  let body;
  let first;
  try {
    body = prepareBody(source);
    first = compileCandidate(body, exportName, timeout);
  } catch (error) {
    parentPort.postMessage({ type: 'compile', error: String(error?.message ?? error) });
    return;
  }
  parentPort.postMessage({ type: 'compile', missing: !first });
  if (!first) return;
  for (let index = 0; index < cases.length; index++) {
    try {
      const instance = index === 0 ? first : compileCandidate(body, exportName, timeout);
      if (!instance) throw new Error(`no callable ${exportName ?? 'default'} function exported`);
      parentPort.postMessage({ type: 'case', index, got: await runCase(instance, cases[index].args, timeout) });
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
    let compiled = false;
    let settled = false;
    const worker = new Worker(new URL(import.meta.url), {
      workerData: { codeTest: { source, exportName, cases: cases.map(tc => ({ args: tc.args ?? [] })), timeout } },
      resourceLimits: { maxOldGenerationSizeMb: 256, maxYoungGenerationSizeMb: 32 },
    });
    let timer;
    const finish = () => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      worker.terminate();
      resolve(outcome);
    };
    const arm = () => { clearTimeout(timer); timer = setTimeout(() => {
      for (let index = outcome.results.length; index < cases.length; index++) outcome.results[index] ??= { error: `timed out after ${timeout}ms` };
      if (!compiled && outcome.compileError === undefined && outcome.missing === undefined) outcome.compileError = `timed out after ${timeout}ms`;
      finish();
    }, timeout * 2 + 1000); };
    worker.on('message', message => {
      if (message.type === 'compile') {
        compiled = true;
        if (message.error !== undefined) { outcome.compileError = message.error; return finish(); }
        outcome.missing = message.missing;
        if (message.missing) return finish();
      } else {
        outcome.results[message.index] = 'got' in message ? { got: message.got } : { error: message.error };
        if (outcome.results.length === cases.length && outcome.results.every(Boolean)) return finish();
      }
      arm();
    });
    worker.on('error', error => {
      const msg = `the candidate crashed its worker: ${error?.message ?? error}`;
      if (!compiled) outcome.compileError ??= msg;
      else {
        for (let index = 0; index < cases.length; index++) outcome.results[index] ??= { error: msg };
      }
      finish();
    });
    worker.on('exit', () => {
      for (let index = 0; index < cases.length; index++) outcome.results[index] ??= { error: 'the worker exited' };
      if (!settled) { settled = true; clearTimeout(timer); resolve(outcome); }
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
        if (tc.assert.noNulls && hasNullish(got)) { ok = false; issues.push(`${label}: contains null/undefined`); }
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
