/**
 * Conformance programs: small projects of `.nl` and callable-folder `.ts` files, each run through
 * the runtime with a scripted reference agent in place of a model.
 */
import { readFileSync, readdirSync } from 'node:fs';
import { resolve } from 'node:path';
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
      const result = ['eval', 'read_function', 'edit_function', 'diff_functions'].includes(tool) ?
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
    if (entry.blocker) return (await act('blocked', { missing: entry.blocker })).text;
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
console.log(JSON.stringify({ passed, failed, skipped }));
if (failed) process.exitCode = 1;
