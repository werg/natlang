import assert from 'node:assert/strict';
import { test } from 'node:test';
import { MemoryResearchAdapter, ResearchWorkspace } from '../studio/shared/research-workspace.mjs';

const source = content => ({ kind: 'source', content });

test('source, evidence and generated view are immutable across workspace revisions', async () => {
    const adapter = new MemoryResearchAdapter(), workspace = new ResearchWorkspace(adapter);
    const first = await workspace.commitEdits('', {
        'methods/cohorts.nl': source('compare cohorts'),
        'evidence/batch.json': { kind: 'evidence', content: [{ group: 'A', failures: 3 }] },
    });
    const oldSource = await workspace.at(first.id, 'methods/cohorts.nl');
    const second = await workspace.commitEdits(first.id, {
        'methods/cohorts.nl': source('compare matched cohorts'),
        'views/comparison.json': { kind: 'view', content: { tag: 'button', text: 'Compare' } },
    });
    assert.equal((await workspace.at(first.id, 'methods/cohorts.nl')).content, oldSource.content);
    assert.equal((await workspace.at(second.id, 'methods/cohorts.nl')).content, 'compare matched cohorts');
    assert.equal((await workspace.at(second.id, 'evidence/batch.json')).content[0].failures, 3);
    assert.equal((await workspace.head()).id, second.id);
    assert.equal((await workspace.export()).manifests[first.id].id, first.id);
});

test('stale activation, fabricated effects and artifact corruption are rejected', async () => {
    const adapter = new MemoryResearchAdapter(), one = new ResearchWorkspace(adapter), two = new ResearchWorkspace(adapter);
    const first = await one.commitEdits('', { 'methods/one.nl': source('one') });
    const stale = await two.stage(first.id, { 'methods/two.nl': source('two') });
    await one.commitEdits(first.id, { 'data/new.json': { kind: 'data', content: 2 } });
    await assert.rejects(two.commit(first.id, stale), /rebase/);
    await assert.rejects(one.commitEdits((await one.head()).id, {}, { effects: ['invented'] }), /missing effect/);
    const current = (await one.head()).id;
    await adapter.recordEffect({ id: 'job-1', status: 'complete', observation: 42 });
    const linked = await one.commitEdits(current, { 'claims/result.json': { kind: 'claim', content: '42 observed' } }, { effects: ['job-1'] });
    assert.deepEqual(linked.effects, ['job-1']);
    const snapshot = await one.snapshot();
    snapshot.artifacts[linked.files['claims/result.json']].content = 'fabricated';
    await adapter.compareAndSwapWorkspace('research', linked.id, snapshot);
    await assert.rejects(one.at(linked.id, 'claims/result.json'), /Corrupt/);
});

test('candidates do not affect active state and clean bundles import', async () => {
    const adapter = new MemoryResearchAdapter(), workspace = new ResearchWorkspace(adapter);
    const first = await workspace.commitEdits('', { 'schema/types.ts': { kind: 'schema', content: 'export type Observation = { value: Num };' } });
    const candidate = await workspace.branch(first.id, { 'schema/types.ts': { kind: 'schema', content: 'export type Observation = { value: Num; unit: Text };' } });
    assert.equal((await workspace.head()).id, first.id);
    await workspace.commit(first.id, candidate);
    const imported = new ResearchWorkspace(new MemoryResearchAdapter(), 'copy');
    await imported.import(await workspace.export());
    assert.equal((await imported.head()).id, candidate.manifest.id);
    const broken = await workspace.export();
    broken.manifests[broken.head].message = 'changed';
    await assert.rejects(new ResearchWorkspace(new MemoryResearchAdapter()).import(broken), /Corrupt/);
});
