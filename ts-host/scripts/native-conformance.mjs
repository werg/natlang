import { readFileSync, readdirSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import YAML from 'yaml';
import { NativeRuntime } from '../dist/index.js';
import { NativeSession } from '../dist/native/runtime.js';
import { loadFunctionFile } from '../dist/native/source.js';
import { buildPending, coerce, dump } from '../dist/native/values.js';
import { TypeEnv } from '../dist/native/types.js';

function valueKey(value) {
  if (typeof value === 'string') return value;
  if (Array.isArray(value)) return `[${value.map(x => JSON.stringify(x)).join(', ')}]`;
  return JSON.stringify(value);
}

const root = resolve(import.meta.dirname, '../..');
const folder = resolve(root, 'conformance/programs');
const baseline = JSON.parse(readFileSync(resolve(root, 'conformance/infrastructure_baseline.json'), 'utf8'));
let passed = 0, failed = 0, skipped = 0;
for (const name of readdirSync(folder).filter(x => x.endsWith('.yaml')).sort()) {
  const file = resolve(folder, name), doc = YAML.parse(readFileSync(file, 'utf8'));
  const specs = doc.reference;
  if (!specs || specs.known_issue) { skipped++; continue; }
  try {
    const pending = doc.program_file ? loadFunctionFile(resolve(dirname(file), doc.program_file)) :
      buildPending(doc.program);
    if (doc.inputs && pending.nodeKind === 'lambda' && pending.type.kind === 'lambda') {
      const env = new TypeEnv().child(pending.types);
      for (const [key, value] of Object.entries(doc.inputs)) {
        const field = pending.type.params.fields.find(f => f.name === key);
        pending.args[key] = coerce(value, field.type, env, `args/${key}`);
      }
    }
    const rt = new NativeRuntime({ agent: async session => {
      const lam = session.lam, spec = specs[lam.functionName];
      if (!spec) throw new Error(`no reference for ${lam.functionName}`);
      const doAction = async (tool, args) => {
        const result = ['eval', 'read_function', 'edit_function', 'diff_functions'].includes(tool) ?
          await session.applyAsync(tool, args) : session.apply(tool, args);
        if (['error', 'rejected', 'refused'].includes(result.kind))
          throw new Error(`${tool} ${JSON.stringify(args)} -> ${result.kind}: ${result.text}`);
        return result;
      };
      if (spec.steps) {
        for (const step of spec.steps) {
          const [tool, args] = Object.entries(step)[0] ?? [];
          if (!tool) throw new Error(`empty reference step for ${lam.functionName}`);
          const result = await doAction(tool, tool === 'eval' ? { code: args } : args ?? {});
          if (result.kind === 'blocked') return result.text;
          if (result.kind !== 'ok') throw new Error(`${tool} -> ${result.kind}: ${result.text}`);
        }
        const body = (lam.originalBody ?? lam.body).replace(/^\n+|\n+$/g, '').split('\n');
        if (body.some(line => line.trim() && !line.trim().startsWith('#') && !line.trim().startsWith('function ')))
          await doAction('mark_lines', { start: 1, end: body.length });
        if (!session.finish()) throw new Error(`reference did not finish: ${JSON.stringify(dump(lam.return))}`);
        return;
      }
      if (spec.eval !== undefined) {
        const result = await doAction('eval', { code: spec.eval });
        if (result.kind === 'blocked') return result.text;
        if (result.kind !== 'ok') throw new Error(`eval -> ${result.kind}: ${result.text}`);
        const body = (lam.originalBody ?? lam.body).replace(/^\n+|\n+$/g, '').split('\n');
        if (body.some(line => line.trim() && !line.trim().startsWith('#') && !line.trim().startsWith('function ')))
          await doAction('mark_lines', { start: 1, end: body.length });
        if (!session.finish()) throw new Error(`reference did not finish: ${JSON.stringify(dump(lam.return))}`);
        return;
      }
      let entry = spec;
      if (spec.variants) entry = spec.variants.find(v => !v.if_body_contains || lam.body.includes(v.if_body_contains)) ?? spec.variants.at(-1);
      if (entry.blocker) return (await doAction('report_blocker', { missing: entry.blocker })).text;
      let value = entry.answer;
      if (spec.answer_by) {
        const key = Object.hasOwn(lam.args, 'item') ? lam.args.item : Object.values(lam.args)[0] ?? null;
        value = spec.answer_by[valueKey(key)];
      }
      if (value === undefined) throw new Error(`no answer for ${lam.functionName}`);
      const encoded = JSON.stringify(value);
      const result = await doAction('eval', { code: `const answer = ${encoded}; answer` });
      if (result.kind === 'blocked') return result.text;
      if (result.kind !== 'ok') throw new Error(`eval answer -> ${result.kind}: ${result.text}`);
      const body = (lam.originalBody ?? lam.body).replace(/^\n+|\n+$/g, '').split('\n');
      if (body.some(line => line.trim() && !line.trim().startsWith('#') && !line.trim().startsWith('function ')))
        await doAction('mark_lines', { start: 1, end: body.length });
      if (!session.finish()) throw new Error('leaf did not finish');
    } });
    const result = await rt.runRoot(pending);
    const expected = doc.expect ?? {};
    if (expected.value !== undefined && JSON.stringify(dump(result.value)) !== JSON.stringify(expected.value))
      throw new Error(`wrong value ${JSON.stringify(dump(result.value))}; expected ${JSON.stringify(expected.value)}`);
    if (expected.status && result.outcome.kind !== expected.status)
      throw new Error(`wrong outcome ${result.outcome.kind}; expected ${expected.status}`);
    if (expected.emitted && JSON.stringify(result.emitted) !== JSON.stringify(expected.emitted))
      throw new Error(`wrong emitted effects ${JSON.stringify(result.emitted)}; expected ${JSON.stringify(expected.emitted)}`);
    const frozen = baseline.fixtures.find(item => item.file === name);
    if (frozen && (result.outcome.kind !== frozen.outcome ||
      (frozen.value !== undefined && JSON.stringify(dump(result.value)) !== JSON.stringify(frozen.value)) ||
      JSON.stringify(result.emitted) !== JSON.stringify(frozen.effects)))
      throw new Error(`infrastructure baseline drift: ${JSON.stringify({ outcome: result.outcome.kind,
        value: dump(result.value), effects: result.emitted })}`);
    console.log(`PASS ${name}`); passed++;
  } catch (error) { console.log(`FAIL ${name}: ${error.message}`); failed++; }
}
console.log(JSON.stringify({ passed, failed, skipped }));
if (failed) process.exitCode = 1;
