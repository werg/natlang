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
    let deep = { tag: 'p', text: 'leaf' };
    for (let i = 0; i < 80; i++) deep = { tag: 'section', children: [deep] };
    validateInteraction(deep, {});
});

test('an interrupted execution keeps an unknown receipt and cannot silently replay', async () => {
    const adapter = new MemoryResearchAdapter(); let runs = 0;
    const runtime = new ResearchRuntime({ adapter, runSource: async () => { runs++; return { value: 1 }; } });
    const manifest = await runtime.commit('', { 'methods/effect.ts': { kind: 'source', content: 'return 1;' } });
    await adapter.beginEffect({ id: 'interrupted', manifest: manifest.id, root: 'methods/effect.ts', inputs: {} });
    const outcome = await runtime.execute(manifest.id, 'methods/effect.ts', {}, 'interrupted');
    assert.equal(outcome.status, 'unknown');
    assert.equal(runs, 0);
    assert.equal((await runtime.execute(manifest.id, 'methods/effect.ts', {}, 'interrupted')).status, 'unknown');
    assert.equal(runs, 0);
});

test('new evidence identifies claims and assessments with recorded dependencies', async () => {
    const runtime = new ResearchRuntime({ adapter: new MemoryResearchAdapter(), runSource: async () => ({ value: null }) });
    const manifest = await runtime.commit('', {
        'evidence/batch.json': { kind: 'evidence', content: { text: 'New mobile batch' } },
        'claims/reliability.json': { kind: 'claim', content: { text: 'Mobile improved', status: 'supported', supports: ['evidence/batch.json'], opposes: ['evidence/missing.json'] } },
        'assessment/summary.json': { kind: 'assessment', content: { text: 'Deployment improved reliability for mobile', depends_on: ['claims/reliability.json'] } },
        'claims/unrelated.json': { kind: 'claim', content: { text: 'Documentation was updated', supports: 'evidence/batch.json' } },
    });
    const graph = await runtime.beliefGraph(manifest.id);
    assert.equal(graph.links.length, 2);
    assert.equal(graph.missing[0].to, 'evidence/missing.json');
    assert.deepEqual(graph.invalid, [{ path: 'claims/unrelated.json', relation: 'supports', reason: 'Expected a list of artifact paths' }]);
    assert.deepEqual(await runtime.affected(manifest.id, ['evidence/batch.json']), ['assessment/summary.json', 'claims/reliability.json']);
});

test('branch listing distinguishes live candidates from active history', async () => {
    const runtime = new ResearchRuntime({ adapter: new MemoryResearchAdapter(), runSource: async () => ({ value: null }) });
    const base = await runtime.commit('', { 'a.txt': { kind: 'data', content: 'base' } });
    const alternative = await runtime.propose(base.id, { 'a.txt': { kind: 'data', content: 'alternative' } }, { message: 'Another interpretation' });
    const current = await runtime.commit(base.id, { 'a.txt': { kind: 'data', content: 'current' } });
    const branches = await runtime.branches();
    assert.equal(branches.find(row => row.id === alternative.id).kind, 'candidate');
    assert.equal(branches.find(row => row.id === base.id).kind, 'history');
    assert.equal((await runtime.diff(base.id, current.id))[0].path, 'a.txt');
    const review = await runtime.reviewCandidate(alternative.id);
    assert.equal(review.can_activate, false);
    assert.deepEqual(review.overlapping_paths, ['a.txt']);
    assert.equal(review.changes[0].before.content, 'base');
    assert.equal(review.changes[0].proposed.content, 'alternative');
    assert.equal(review.changes[0].active.content, 'current');
});
