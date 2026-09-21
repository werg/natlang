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

test('export carries receipts as historical imported observations', async () => {
    const source = new MemoryResearchAdapter(), workspace = new ResearchWorkspace(source);
    await source.recordEffect({ id: 'effect-1', status: 'complete', value: 7 });
    const original = await workspace.commitEdits('', { 'claims/answer.json': { kind: 'claim', content: { value: 7 } } }, { effects: ['effect-1'] });
    const target = new MemoryResearchAdapter(), imported = new ResearchWorkspace(target, 'copy');
    await imported.import(await workspace.export());
    assert.equal((await imported.head()).id, original.id);
    assert.equal((await target.readEffect('effect-1')).imported, true);
});

test('two durable candidates preserve both histories for semantic reconciliation', async () => {
    const workspace = new ResearchWorkspace(new MemoryResearchAdapter());
    const base = await workspace.commitEdits('', { 'analysis/answer.txt': { kind: 'claim', content: 'One global estimate' } });
    const units = await workspace.propose(base.id, {
        'analysis/answer.txt': { kind: 'claim', content: 'Rates are per 100 requests' },
        'intent/units.json': { kind: 'intent', content: { request: 'Correct units' } },
    }, { message: 'Units correction' });
    const cohorts = await workspace.propose(base.id, {
        'analysis/answer.txt': { kind: 'claim', content: 'Compare each cohort separately' },
        'intent/cohorts.json': { kind: 'intent', content: { request: 'Expose cohorts' } },
    }, { message: 'Cohort correction' });
    assert.equal((await workspace.head()).id, base.id);
    assert.equal((await workspace.at(units.id, 'intent/units.json')).content.request, 'Correct units');
    assert.equal((await workspace.at(cohorts.id, 'intent/cohorts.json')).content.request, 'Expose cohorts');
    const merged = await workspace.propose(base.id, {
        'analysis/answer.txt': { kind: 'claim', content: 'Rates per 100 requests, compared by cohort' },
        'intent/units.json': { kind: 'intent', content: (await workspace.at(units.id, 'intent/units.json')).content },
        'intent/cohorts.json': { kind: 'intent', content: (await workspace.at(cohorts.id, 'intent/cohorts.json')).content },
    }, { message: 'Semantic reconciliation of both purposes' });
    await workspace.activate(base.id, merged.id);
    assert.equal((await workspace.head()).id, merged.id);
    await assert.rejects(workspace.activate(merged.id, units.id), /rebase/);
    assert.equal((await workspace.at(units.id, 'analysis/answer.txt')).content, 'Rates are per 100 requests');
});
