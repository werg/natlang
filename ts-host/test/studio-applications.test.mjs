import assert from 'node:assert/strict';
import { scriptedModel } from './support/natlang.mjs';
import { test } from 'node:test';
import { readFile } from 'node:fs/promises';
import { simulateInventory } from '../studio/apps/worlds.mjs';
import { apps, appById } from '../studio/apps/index.mjs';
import { loadProgram, fixtureTurn } from '../studio/shared/program.mjs';
import { applyOperation } from '../studio/shared/host.mjs';
async function api() { const process = globalThis.process; try {
    globalThis.process = undefined;
    return await import('../dist/browser/natlang.js');
}
finally {
    globalThis.process = process;
} }
const services = { trial:async config=>simulateInventory(config), service: async (operation) => {
        if (operation === 'media.sample')
            return { asset: 'asset-fixture.mp4' };
        if (operation === 'build.run')
            return { status: 'ok', output: 'FIXTURE' };
        if (operation === 'terminal.run')
            return { status: 'succeeded', output: 'fixture stdout' };
        if (operation === 'packages.resolve')
            return { locks: [{ id: 'fixture-lock' }] };
        if (operation === 'repository.check')
            return { status: 'reviewable', checks: [] };
        throw new Error('Unexpected fixture operation ' + operation);
    } };
function cellHost(executed=[]){const values=new Map();return {
 readValue:async id=>{assert.ok(values.has(id));return values.get(id);},
 cell:async(cell,deps)=>{executed.push(cell.id);const value=cell.id==='numbers'?[2,3,5,7,11]:deps.numbers.reduce((a,b)=>a+b,0);const id='value-'+values.size;values.set(id,value);return {id,preview:JSON.stringify(value)};}
};}
const sourceFor = spec => loadProgram(spec, path => readFile(new URL('../studio/' + path.replace('./', ''), import.meta.url), 'utf8'));
const { natlangApplication } = await (async () => { await api(); return import('../studio/shared/natlang-app.mjs'); })();
for (const spec of apps)
    test(`${spec.project} ${spec.id}: real natlang source, typed operation and view`, async () => {
        const event = { id: 'event-1', kind: spec.smoke.action, value: JSON.stringify(spec.smoke) };
        const source = await sourceFor(spec);
        let app, committed;
        app = natlangApplication({ source, initialState: spec.initial(), seedRoot: 17,
            services: { studio: { apply: (state, event, decision) => applyOperation(spec, state, decision, services) } },
            model: fixtureTurn(spec, () => app.state, () => event), onCommit: commit => { committed = commit; } });
        try {
            await app.start();
            const result = await app.dispatch(event);
            assert.equal(result.state.revision, 1);
            assert.deepEqual(result.view.focus, spec.panelIds);
            assert.ok(committed.trace.length);
            assert.equal(spec.panels(result.state).length, spec.panelIds.length);
            assert.equal(new Set(spec.panels(result.state).map(p => p.id)).size, spec.panelIds.length);
        }
        finally {
            await app.close();
        }
    });
test('cash, inventory, schedule, citations and workflow invariants reject invalid operations', async () => {
    for (const [id, decision, pattern] of [
        ['economy', { action: 'buy', target: 'mira', secondary: 'you', amount: 100 }, /stock|cash/],
        ['evidence', { action: 'claim', target: 'garden', text: 'A fabricated claim', secondary: 'not in this passage' }, /Quotation/],
        ['workflows', { action: 'execute', target: 'ship' }, /Dependencies/],
        ['spells', { action: 'cast', target: 'unknown', text: 'Boom', amount: 3 }, /element/],
    ]) {
        const spec = appById.get(id), before = spec.initial();
        const result = await applyOperation(spec, before, decision, {});
        assert.equal(result.ok, false);
        assert.match(result.detail, pattern);
        assert.equal(before.revision, 0);
    }
    const economy = appById.get('economy');
    const before = economy.initial(), after = await economy.apply(structuredClone(before), economy.smoke, {});
    for (const key of ['cash', 'apples'])
        assert.equal(after.merchants.reduce((n, m) => n + m[key], 0), before.merchants.reduce((n, m) => n + m[key], 0));
    const schedule = appById.get('scheduling');
    const scheduled = await schedule.apply(schedule.initial(), schedule.smoke, {});
    const collision = await applyOperation(schedule, scheduled, { action: 'schedule', target: 'walk', text: '2026-09-21T09:30:00Z' }, {});
    assert.equal(collision.ok, false);
    assert.match(collision.detail, /overlaps/);
});
test('notebook host executes exactly one cell and rejects unmet dependencies', async () => {
    const spec = appById.get('notebook'), executed = [];
    const host=cellHost(executed);
    const failed = await applyOperation(spec, spec.initial(), { action: 'execute', target: 'total' }, host);
    assert.equal(failed.ok, false);
    assert.deepEqual(executed, []);
    const one = await applyOperation(spec, spec.initial(), { action: 'execute', target: 'numbers' }, host);
    assert.deepEqual(executed, ['numbers']);
    const two = await applyOperation(spec, one.state, { action: 'execute', target: 'total' }, host);
    assert.equal(two.state.cells[1].result, '28');
    const edited = await applyOperation(spec, two.state, { action: 'save', target: 'numbers', text: 'return [1];' }, host);
    assert.equal(edited.state.cells[1].status, 'stale');
    assert.equal(edited.state.cells[1].result, '');
});
test('natlang can drive multiple inspected operations within a single UI interaction', async () => {
    const spec = appById.get('notebook');
    const source = await sourceFor(spec);
    const calls = [], cells = cellHost();
    const model = scriptedModel(opening => opening.includes(`Drive the ${spec.title} interaction to completion.`) ?
        'const first = await perform(state, event, { action: "execute", target: "numbers" }); const next = await perform(first.state, event, { action: "execute", target: "total" }); result = await finish(next)' :
        `result = ${JSON.stringify({ heading: spec.title, summary: 'Two cells complete', focus: spec.panelIds, suggestions: [] })}`);
    const app = natlangApplication({ source, initialState: spec.initial(), seedRoot: 17, model: model.driver,
        services: { studio: { apply: (state, event, d) => { calls.push(d.target); return applyOperation(spec, state, d, cells); } } } });
    try {
        await app.start();
        const result = await app.dispatch({ id: 'multi', kind: 'run', value: '{"target":"total"}' });
        assert.deepEqual(calls, ['numbers', 'total']);
        assert.equal(result.state.cells[1].result, '28');
    }
    finally {
        await app.close();
    }
});
