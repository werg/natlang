import { readFileSync, readdirSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import YAML from 'yaml';
import { NativeRuntime } from '../dist/native/runtime.js';
import { NativeSession } from '../dist/native/runtime.js';
import { loadFunctionFile } from '../dist/native/source.js';
import { buildPending, coerce, dump } from '../dist/native/values.js';
import { TypeEnv, formatType } from '../dist/native/types.js';

function pyKey(value) {
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
      buildPending(doc.start_state ?? doc.program);
    if (doc.inputs && pending.nodeKind === 'lambda' && pending.type.kind === 'lambda') {
      const env = new TypeEnv().child(pending.types);
      for (const [key, value] of Object.entries(doc.inputs)) {
        const field = pending.type.params.fields.find(f => f.name === key);
        pending.args[key] = coerce(value, field.type, env, `args/${key}`);
      }
    }
    const attempts = new WeakSet();
    const streamItems = doc.streams?.over?.[Symbol.iterator]();
    const rt = new NativeRuntime({ stream: streamItems && { poll() {
      const item = streamItems.next();
      return item.done || item.value === '$close' ? { kind: 'closed' } : { kind: 'item', value: item.value };
    } }, agent: async session => {
      const lam = session.lam, spec = specs[lam.functionName];
      if (!spec) throw new Error(`no reference for ${lam.functionName}`);
      const doAction = async (tool, args) => {
        const result = await session.applyAsync(tool, args);
        if (['error', 'rejected', 'refused'].includes(result.kind))
          throw new Error(`${tool} ${JSON.stringify(args)} -> ${result.kind}: ${result.text}`);
        return result;
      };
      if (spec.calls) {
        for (const call of spec.calls) {
          if (call.tool === 'glue') {
            const result = await doAction('run_code', { code: call.code, engine: 'typescript-host' });
            await doAction('write', { path: call.path, type: call.type, value: result.value });
          } else {
            const result = await doAction(call.tool, call.args ?? {});
            if (call.tool === 'report_blocker') return result.text;
          }
        }
        if (!session.finish()) throw new Error(`reference did not finish: ${JSON.stringify(dump(lam.return))}`);
        return;
      }
      let entry = spec;
      if (spec.variants) entry = spec.variants.find(v => !v.if_body_contains || lam.body.includes(v.if_body_contains)) ?? spec.variants.at(-1);
      if (entry.blocker) return (await doAction('report_blocker', { missing: entry.blocker })).text;
      let value = entry.answer;
      if (spec.answer_by) {
        const key = Object.hasOwn(lam.args, 'item') ? lam.args.item : Object.values(lam.args)[0] ?? null;
        value = spec.answer_by[pyKey(key)];
      }
      if (value && typeof value === 'object' && 'blocker' in value) {
        if (!attempts.has(lam)) { attempts.add(lam); return (await doAction('report_blocker', { missing: value.blocker })).text; }
        value = value.then;
      }
      if (value === undefined) throw new Error(`no answer for ${lam.functionName}`);
      await doAction('write', { path: 'return', type: formatType(lam.type.returns), value });
      if (!session.finish()) throw new Error('leaf did not finish');
    } });
    const result = await rt.runRoot(pending);
    const expected = doc.expect ?? {};
    if (expected.value !== undefined && JSON.stringify(dump(result.value)) !== JSON.stringify(expected.value))
      throw new Error(`wrong value ${JSON.stringify(dump(result.value))}; expected ${JSON.stringify(expected.value)}`);
    if (expected.status && result.outcome.kind !== expected.status)
      throw new Error(`wrong outcome ${result.outcome.kind}; expected ${expected.status}`);
    const frozen = baseline.fixtures.find(item => item.file === name);
    if (frozen && (result.outcome.kind !== frozen.outcome ||
      (frozen.value !== undefined && JSON.stringify(dump(result.value)) !== JSON.stringify(frozen.value)) ||
      JSON.stringify(result.emitted) !== JSON.stringify(frozen.effects)))
      throw new Error(`infrastructure baseline drift: ${JSON.stringify({ outcome: result.outcome.kind,
        value: dump(result.value), effects: result.emitted })}`);
    console.log(`PASS ${name}`); passed++;
  } catch (error) { console.log(`FAIL ${name}: ${error.message}`); failed++; }
}
for (const fixture of baseline.action_fixtures) {
  const doc = YAML.parse(readFileSync(resolve(folder, fixture.file), 'utf8'));
  const pending = buildPending(doc.program);
  const env = new TypeEnv().child(pending.types);
  for (const [key, value] of Object.entries(doc.inputs ?? {})) {
    const field = pending.type.params.fields.find(item => item.name === key);
    pending.args[key] = coerce(value, field.type, env, `args/${key}`);
  }
  const session = new NativeSession(new NativeRuntime(), pending, new TypeEnv());
  for (const call of fixture.calls) {
    const result = session.apply(call.name, call.arguments);
    if (result.kind !== call.outcome || JSON.stringify(result.codes ?? []) !== JSON.stringify(call.codes)) {
      console.log(`FAIL baseline ${fixture.file}: ${call.name} ${result.kind} ${JSON.stringify(result.codes)}`);
      failed++;
    }
  }
  if (session.finish() !== fixture.finish) { console.log(`FAIL baseline ${fixture.file}: finish`); failed++; }
  else { console.log(`PASS baseline ${fixture.file}`); passed++; }
}
console.log(JSON.stringify({ passed, failed, skipped }));
if (failed) process.exitCode = 1;
