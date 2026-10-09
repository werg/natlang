// Wiring evidence (scripted model, no live executor): generation/prepare reaches the shared harness items through
// `uses:`, and harness/context.ts selects pi-durable's derivation or the natural-language one by the host setting
// through pluggable(); shadow runs both, serves the natural-language view and records whether they agree.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { createNatlangRuntime, loadNatlang } from '../../../ts-host/dist/index.js';
import { scriptedModel } from '../../../ts-host/test/support/natlang.mjs';

const app = fileURLToPath(new URL('..', import.meta.url));
const VIEW = { head: null, entries: [], contributions: [], messages: [], sections: [], tools: [] };

function services(mode, calls) {
  return { durable: {
    implementation: () => mode,
    view: async at => { calls.push(['view', at]); return { ...VIEW, sections: [{ key: 'from', text: 'crisp' }] }; },
    scan: async at => { calls.push(['scan', at]); return { head: null, entries: [] }; },
  } };
}

for (const mode of ['crisp', 'nl', 'natural-language', 'shadow']) {
  test(`prepare calls context(), cut and estimate through uses: (${mode} context)`, async () => {
    const calls = [];
    const model = scriptedModel(opening => {
      if (opening.includes('Prepare one request of this generation')) return `
        const view = await context(7);
        return [typeof cut, typeof estimate, typeof planSystem, view.sections[0]?.text ?? 'derived'].join(' ');`;
      if (opening.includes("Derive a conversation's model context")) return `
        return { head, entries, contributions: [], messages: [], sections: [], tools: [] };`;
      return null;
    });
    const prepare = loadNatlang(`${app}/generation/prepare.nl`, app);
    const traces = [];
    const runtime = createNatlangRuntime({ model: model.driver, trace: trace => traces.push(trace) });
    const facts = { task: { id: 1, kind: 'pi.generation', conversationId: 1, input: {}, checkpoint: { phase: 'prepare', attempt: 1 } },
      mode: 'run', agent: { thinkingLevel: 'off', tools: [], sections: [] }, settings: { stream: {}, retry: { enabled: true, maxRetries: 3, baseDelayMs: 1 },
        compaction: { enabled: true, reserveTokens: 1, keepRecentTokens: 1, backgroundTokens: 0 }, toolExecution: 'parallel',
        steeringMode: 'one-at-a-time', followUpMode: 'one-at-a-time' }, sessionId: 's', now: 0 };
    const info = { provider: 'p', modelId: 'm', name: 'm', contextWindow: 0, maxTokens: 0, reasoning: false };
    const result = await runtime.run(() => prepare(facts, facts.task.checkpoint, info), { services: services(mode, calls) });
    assert.equal(result, `function function function ${mode === 'crisp' ? 'crisp' : 'derived'}`);
    const expected = { crisp: [['view', 7]], shadow: [['scan', 7], ['view', 7]] }[mode] ?? [['scan', 7]];
    assert.deepEqual(calls.sort(), expected);
    const shadows = traces.flatMap(trace => trace.events).filter(event => event.kind === 'pluggable_shadow');
    assert.deepEqual(shadows.map(event => [event.name, event.served, event.agree]), mode === 'shadow' ? [['pi.context', 'nl', false]] : []);
  });
}
