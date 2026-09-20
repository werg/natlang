import { readFileSync, readdirSync } from 'node:fs';
import { resolve } from 'node:path';
import YAML from 'yaml';
import { NativeRuntime, NativeSession } from '../dist/native/runtime.js';
import { buildPending, dump, problems } from '../dist/native/values.js';
import { TypeEnv } from '../dist/native/types.js';

const folder = resolve(import.meta.dirname, '../../conformance/harness');
let passed = 0, failed = 0;
for (const name of readdirSync(folder).filter(x => x.endsWith('.yaml')).sort()) {
  const doc = YAML.parse(readFileSync(resolve(folder, name), 'utf8'));
  try {
    const lam = buildPending(doc.setup);
    const rt = new NativeRuntime({ agent: session => {
      if (session.lam.body.includes('Echo')) {
        if (session.lam.args.item === 'bad') return 'cannot echo this';
        const copied = session.apply('write', { path: 'return', type: 'Text', source: 'args/item' });
        if (copied.kind !== 'ok' || !session.finish()) throw new Error(`Echo: ${copied.text}`);
        return;
      }
      if (session.lam.body.includes('Say hello')) {
        const refused = session.apply('write', { path: 'args/name', type: 'Text', value: 'Bob' });
        if (refused.kind !== 'rejected') throw new Error('own input was writable');
        return 'I cannot proceed: no name was given that I recognise.';
      }
      throw new Error('unknown stub agent');
    } });
    const session = new NativeSession(rt, lam, new TypeEnv());
    for (const [index, step] of doc.script.entries()) {
      const priorEpisodes = rt.episodesStarted;
      const before = JSON.stringify(dump(lam));
      const result = await session.act(step.action);
      if (result.kind !== step.expect.result) throw new Error(`step ${index + 1}: ${result.kind} ${result.text}; expected ${step.expect.result}`);
      if (step.expect.codes && step.expect.codes.some(code => !(result.codes ?? []).includes(code)))
        throw new Error(`step ${index + 1}: codes ${JSON.stringify(result.codes)}; expected ${JSON.stringify(step.expect.codes)}`);
      if (step.expect.value !== undefined && JSON.stringify(dump(result.value)) !== JSON.stringify(step.expect.value))
        throw new Error(`step ${index + 1}: value ${JSON.stringify(dump(result.value))}`);
      if (step.expect.tree_changed === false && JSON.stringify(dump(lam)) !== before)
        throw new Error(`step ${index + 1}: rejected action changed the tree`);
      if (step.expect.paths && !step.expect.paths.every(path => result.text.includes(path)))
        throw new Error(`step ${index + 1}: missing diagnostic paths: ${result.text}`);
      if (step.expect.note_contains && !result.text.includes(step.expect.note_contains))
        throw new Error(`step ${index + 1}: missing note: ${result.text}`);
      if (step.expect.problems) {
        const state = problems(lam.return, lam.type.returns, session.env, 'return');
        if (state.holes.length !== step.expect.problems.holes || state.pending.length !== step.expect.problems.blocking)
          throw new Error(`step ${index + 1}: problems ${state.holes.length} holes, ${state.pending.length} blocking`);
      }
      if (step.expect.read) {
        for (const check of Array.isArray(step.expect.read) ? step.expect.read : [step.expect.read]) {
          const got = session.apply('read', { path: check.path });
          if (JSON.stringify(dump(got.value)) !== JSON.stringify(check.value))
            throw new Error(`step ${index + 1}: read ${check.path} gave ${JSON.stringify(dump(got.value))}`);
        }
      }
      if (step.expect.reduced && !result.text.includes(step.expect.reduced))
        throw new Error(`step ${index + 1}: expected reduction ${step.expect.reduced}; got ${result.text}`);
      if (step.expect.episodes_started !== undefined && rt.episodesStarted - priorEpisodes !== step.expect.episodes_started)
        throw new Error(`step ${index + 1}: episodes ${rt.episodesStarted - priorEpisodes}; expected ${step.expect.episodes_started}`);
      if (step.expect.journal_entries !== undefined && lam.journal.length !== step.expect.journal_entries)
        throw new Error(`step ${index + 1}: journal entries ${lam.journal.length}; expected ${step.expect.journal_entries}`);
    }
    console.log(`PASS ${name}`); passed++;
  } catch (error) { console.log(`FAIL ${name}: ${error.message}`); failed++; }
}
console.log(JSON.stringify({ passed, failed }));
if (failed) process.exitCode = 1;
