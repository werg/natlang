import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFile } from 'node:fs/promises';
import { MemoryResearchAdapter } from '../studio/shared/research-workspace.mjs';
import { ResearchRuntime } from '../studio/shared/research-runtime.mjs';
import { ResearchHost } from '../studio/research/host.mjs';
import { scriptedModel } from './support/natlang.mjs';

const root = new URL('../studio/research/programs/', import.meta.url);
const paths = ['types.ts', 'reduce.nl', 'view.ts', 'learn.nl', 'revise_schema.nl', 'invent_interaction.nl', 'preserve_intent.nl', 'investigate_beliefs.nl', 'reduce/list.ts', 'reduce/search.ts', 'reduce/workspace_read.ts', 'reduce/commit.ts', 'reduce/execute.ts', 'reduce/diff.ts', 'reduce/review_candidate.ts', 'reduce/review_reconciliation.ts', 'reduce/propose.ts', 'reduce/activate.ts', 'reduce/receipt.ts', 'reduce/branches.ts', 'reduce/belief_graph.ts', 'reduce/affected.ts', 'reduce/audit_migration.ts', 'reduce/native_read.ts', 'reduce/native_search.ts'];
const files = Object.fromEntries(await Promise.all(paths.map(async path => [path, await readFile(new URL(path, root), 'utf8')])));
async function api() { const process = globalThis.process; try {
    globalThis.process = undefined;
    return await import('../dist/browser/natlang.js');
} finally { globalThis.process = process; } }

test('research reducer owns semantic state and actual source loads in the interpreter', async () => {
    await api();
    const { natlangApplication } = await import('../studio/shared/natlang-app.mjs');
    const initial = { revision: 0, head: 'manifest', question: '', notice: '', active_view: '', selected: '', receipts: [] };
    const model = scriptedModel(() => 'return { ...state, revision: state.revision + 1, question: event.value, notice: "The question is open." }');
    let committed;
    const app = natlangApplication({ source: { files, reducer: 'reduce.nl', view: 'view.ts' }, initialState: initial, seedRoot: 17,
        model: model.driver, services: { research: {} }, onCommit: commit => { committed = commit; } });
    try {
        await app.start();
        const result = await app.dispatch({ id: 'event-1', kind: 'question', value: 'Which cohort improved?' });
        assert.equal(result.state.question, 'Which cohort improved?');
        assert.equal(result.state.revision, 1);
        assert.equal(result.view.heading, 'Which cohort improved?');
        assert.ok(committed.trace.length);
        assert.match(model.openings[0], /workspace_read\(head: string, path: string\)/);
    } finally { await app.close(); }
});

test('a learned TypeScript method executes from its committed source and records the result', async () => {
    const natlang = await api();
    const adapter = new MemoryResearchAdapter();
    const runtime = new ResearchRuntime({ adapter, runSource: async (entries, rootName, inputs) => {
        const run = await natlang.runPlaygroundProject(natlang.createNatlangRuntime(),
            natlang.newPlaygroundProject('method', rootName, Object.fromEntries(entries.map(row => [row.id, row.source])), inputs),
            { runtimeNamespace: natlang });
        if (run.outcome.kind !== 'done') throw new Error(run.outcome.detail);
        return { value: run.value, trace_id: 'actual-trace' };
    } });
    const manifest = await runtime.commit('', {
        'methods/compare.ts': { kind: 'source', content: 'export function main(input: { before: number, after: number }): number { return input.after - input.before; }\n' },
    });
    const receipt = await runtime.execute(manifest.id, 'methods/compare.ts', { before: 18, after: 7 }, 'trial-1');
    assert.equal(receipt.status, 'complete', receipt.error);
    assert.equal(receipt.value, -11);
    assert.equal((await adapter.readEffect('trial-1')).trace_id, 'actual-trace');
});

test('research host permits semantic state changes but verifies heads and real receipts', async () => {
    const adapter = new MemoryResearchAdapter();
    const research = new ResearchHost({ store: adapter, runSource: async () => ({ value: 3 }) });
    const manifest = await research.runtime.commit('', { 'evidence/a.txt': { kind: 'evidence', content: 'A real observation' } });
    const previous = { revision: 0, head: manifest.id, question: '', notice: '', active_view: '', selected: '', receipts: [] };
    await research.verifyCommit(previous, { ...previous, revision: 1, question: 'What follows?' });
    await assert.rejects(research.verifyCommit(previous, { ...previous, revision: 1, head: 'invented' }), /disagrees/);
    await assert.rejects(research.verifyCommit(previous, { ...previous, revision: 1, receipts: ['invented'] }), /Unknown execution receipt/);
});

test('native evidence can be searched and read without copying full content into state', async () => {
    const adapter = new MemoryResearchAdapter(), full = 'x'.repeat(300_000) + 'critical observation' + 'y'.repeat(300_000);
    adapter.get = async (store, id) => store === 'native_values' && id === 'large' ? { value: full } : undefined;
    const api = new ResearchHost({ store: adapter, runSource: async () => ({ value: null }) }).api();
    const found = JSON.parse(await api.nativeSearch('large', 'critical observation'));
    assert.equal(found.total, 1);
    assert.equal(await api.nativeRead('large', found.hits[0].offset, 20), 'critical observation');
    await assert.rejects(api.nativeRead('large', 0, 100_000), /65536/);
});
