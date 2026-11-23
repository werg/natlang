// Trusted artifact-execution verifier (ts-host/src/teacher/trusted-exact-verifier.ts):
// Executes a candidate JavaScript/TypeScript transform or module returned by the model (either a code string,
// `{ code: string }`, or a named file in `files[oracle.file]`) inside an isolated vm context (no Node built-ins)
// and verifies it against hidden test cases and invariants. Unlocks semantic-etl, codemod-studio, and code-synth tasks.
//
// oracle: {
//   file?: string,                 // read candidate code from folder `files[file]` instead of `value`
//   export?: string,               // named function to call (default: `transform`, or default export / bare function)
//   cases: [{ name?: string, args: unknown[], expected?: unknown,
//             assert?: { equals?: unknown, minLength?: number, keys?: string[], noNulls?: boolean } }],
//   forbidden_patterns?: string[], // regex strings the candidate source must not contain (e.g. hardcoded test outputs)
//   timeout_ms?: number,           // per-case execution timeout (default 500)
// }
import vm from 'node:vm';

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

function compileCandidate(source, exportName, timeout) {
  const exportedFns = [];
  const cleaned = source
    .replace(/^\s*```[a-zA-Z]*\n/, '').replace(/\n```\s*$/, '')
    .replace(/\bexport\s+default\s+/g, 'module.exports.default = ')
    .replace(/\bexport\s+(async\s+)?function\s+([A-Za-z_$][\w$]*)/g, (_, asyncKw, name) => {
      exportedFns.push(name);
      return `${asyncKw ?? ''}function ${name}`;
    })
    .replace(/\bexport\s+(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=/g, 'const $1 = module.exports.$1 =');
  const exportFooter = exportedFns.map(name => `\nif (typeof ${name} === 'function') module.exports.${name} = ${name};`).join('');
  const sandbox = Object.create(null);
  sandbox.module = { exports: {} };
  sandbox.exports = sandbox.module.exports;
  sandbox.JSON = JSON;
  sandbox.Math = Math;
  sandbox.Number = Number;
  sandbox.String = String;
  sandbox.Boolean = Boolean;
  sandbox.Array = Array;
  sandbox.Object = Object;
  sandbox.Map = Map;
  sandbox.Set = Set;
  sandbox.Date = Date;
  sandbox.RegExp = RegExp;
  const ctx = vm.createContext(sandbox);
  const expr = !exportedFns.length && /^\s*(?:\(|[A-Za-z_$][\w$]*\s*=>|function\b|async\s+function\b)/.test(cleaned)
    ? `module.exports.default = (${cleaned});` : `${cleaned}${exportFooter}`;
  vm.runInContext(expr, ctx, { timeout });
  const mod = sandbox.module.exports;
  const fn = (exportName && typeof mod[exportName] === 'function' ? mod[exportName] : null)
    ?? (typeof mod.default === 'function' ? mod.default : null)
    ?? (typeof mod.transform === 'function' ? mod.transform : null)
    ?? (exportName && typeof sandbox[exportName] === 'function' ? sandbox[exportName] : null);
  return { fn, ctx };
}

export async function verifyCandidate({ value, files, oracle }) {
  const issues = [];
  const timeout = oracle.timeout_ms ?? 500;
  const source = extractCode(value, files, oracle);
  if (!source || !source.trim()) return { ok: false, issues: ['candidate provided no executable code'], score: 0 };
  for (const pattern of oracle.forbidden_patterns ?? []) {
    if (new RegExp(pattern).test(source)) issues.push(`forbidden pattern matched: ${pattern}`);
  }
  let compiled;
  try {
    compiled = compileCandidate(source, oracle.export, timeout);
  } catch (error) {
    return { ok: false, issues: [...issues, `compile error: ${error.message}`], score: 0 };
  }
  if (typeof compiled.fn !== 'function')
    return { ok: false, issues: [...issues, `no callable ${oracle.export ?? 'default'} function exported`], score: 0 };

  const cases = oracle.cases ?? [];
  let passed = 0;
  for (let i = 0; i < cases.length; i++) {
    const tc = cases[i];
    const label = tc.name ?? `case ${i + 1}`;
    try {
      const args = structuredClone(tc.args ?? []);
      const got = await Promise.race([
        Promise.resolve(compiled.fn(...args)),
        new Promise((_, reject) => setTimeout(() => reject(new Error('timeout')), timeout)),
      ]);
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
