import assert from 'node:assert/strict';
import { test } from 'node:test';
import { MemoryResearchAdapter } from '../studio/shared/research-workspace.mjs';
import { ResearchRuntime, validateInteraction } from '../studio/shared/research-runtime.mjs';

const comparison = { tag: 'section', children: [
    { tag: 'h2', text: 'Compare cohort explanations' },
    { tag: 'input', id: 'cohort', label: 'Cohort' },
    { tag: 'button', id: 'compare', text: 'Compare', action: { kind: 'compare', from: 'cohort' } },
] };

test('a generated interaction is bound to a real natlang handler and a pinned manifest', async () => {
    const adapter = new MemoryResearchAdapter();
    const calls = [];
    const runtime = new ResearchRuntime({ adapter, runSource: async (files, root, inputs) => {
        calls.push({ files, root, inputs });
        return { value: { cohort: inputs.cohort, difference: -2 }, trace_id: 'trace-1' };
    } });
    const base = await runtime.commit('', {
        'methods/compare.nl': { kind: 'source', content: '---\nargs:\n  cohort: Text\nreturns: Text\n---\nCompare.' },
        'views/cohorts.json': { kind: 'view', content: { tree: comparison, bindings: { compare: { root: 'methods/compare.nl', from: 'cohort' } } } },
        'evidence/cohorts.json': { kind: 'evidence', content: [{ cohort: 'A', failures: 2 }] },
    });
    const view = await runtime.interaction(base.id, 'views/cohorts.json');
    assert.equal(view.revision, base.id);
    assert.equal(view.bindings.compare.root, 'methods/compare.nl');
    const first = await runtime.execute(base.id, 'methods/compare.nl', { cohort: 'A' }, 'event-1');
    const again = await runtime.execute(base.id, 'methods/compare.nl', { cohort: 'A' }, 'event-1');
    assert.deepEqual(again, first);
    assert.equal(calls.length, 1);
    await assert.rejects(runtime.execute(base.id, 'methods/compare.nl', { cohort: 'B' }, 'event-1'), /reused/);
    const newVersion = await runtime.commit(base.id, { 'methods/compare.nl': { kind: 'source', content: 'revised' } }, { effects: ['event-1'] });
    assert.equal((await runtime.diff(base.id, newVersion.id))[0].path, 'methods/compare.nl');
    assert.equal((await runtime.read(base.id, 'methods/compare.nl')).content.includes('Compare.'), true);
    const found = await runtime.search(newVersion.id, 'failures');
    assert.equal(found[0].path, 'evidence/cohorts.json');
});

test('view validation rejects broken bindings and duplicate controls', () => {
    validateInteraction(comparison, { compare: { root: 'methods/compare.nl', from: 'cohort' } });
    assert.throws(() => validateInteraction(comparison, { compare: { root: 'methods/compare.nl', from: 'missing' } }), /no input/);
    assert.throws(() => validateInteraction(comparison, { missing: { root: 'methods/compare.nl' } }), /no control/);
    assert.throws(() => validateInteraction({ tag: 'section', children: [{ tag: 'input', id: 'x' }, { tag: 'input', id: 'x' }] }, {}), /Duplicate/);
    assert.throws(() => validateInteraction({ tag: 'script', text: 'bad' }, {}), /Unsupported/);
});
