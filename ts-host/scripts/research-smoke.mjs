import assert from 'node:assert/strict';
import { mkdtemp, rm, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { chromium } from 'playwright-core';
import { startStudio } from './serve-studio.mjs';

const temporary = await mkdtemp(join(tmpdir(), 'natlang-research-smoke-'));
const studio = await startStudio({ port: 0, dataRoot: temporary });
let browser;
try {
    browser = await chromium.launch({ headless: true, executablePath: process.env.NATLANG_CHROMIUM ?? chromium.executablePath(),
        args: ['--no-sandbox', '--disable-gpu', '--disable-gpu-compositing', '--disable-features=Vulkan,WebGPU', '--use-gl=swiftshader'] });
    const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
    const page = await context.newPage(), errors = [];
    page.on('pageerror', error => errors.push(String(error)));
    await page.goto(studio.url + 'research/');
    await page.getByText('reliability-observations.json').waitFor();
    assert.match(await page.locator('#artifact-count').textContent(), /28 artifacts/);
    await page.getByRole('button', { name: /reliability-observations/ }).click();
    await page.locator('#detail-body').getByText(/failures/).waitFor();
    assert.match(await page.locator('#detail-body').textContent(), /"failures": 72/);
    await page.getByRole('button', { name: 'Close' }).click();
    await page.locator('#evidence-file').setInputFiles({ name: 'later.json', mimeType: 'application/json', buffer: Buffer.from('[{"period":"later","requests":10,"failures":1}]') });
    await page.locator('#artifacts strong').filter({ hasText: 'later.json' }).waitFor();
    assert.match(await page.locator('#revision').textContent(), /Event 1/);
    const large = 'start\n' + 'x'.repeat(270_000) + '\ncritical observation\n';
    await page.locator('#evidence-file').setInputFiles({ name: 'large.txt', mimeType: 'text/plain', buffer: Buffer.from(large) });
    await page.locator('#artifacts strong').filter({ hasText: 'large.txt' }).waitFor();
    await page.getByRole('button', { name: /large.txt/ }).click();
    await page.getByRole('button', { name: 'Download full evidence' }).waitFor();
    const fullDownload = page.waitForEvent('download');
    await page.getByRole('button', { name: 'Download full evidence' }).click();
    const full = await fullDownload;
    const fullPath = join(temporary, 'large.txt');
    await full.saveAs(fullPath);
    assert.equal((await readFile(fullPath, 'utf8')).length, large.length);
    await page.getByRole('button', { name: 'Close' }).click();
    const nativeExecution = await page.evaluate(async () => {
        const { StudioStore } = await import('../shared/store.mjs');
        const { runChild } = await import('../shared/child-runner.mjs');
        const store = await StudioStore.open();
        const state = (await store.get('states', 'research')).state;
        const workspace = await store.get('research_workspaces', 'research');
        const artifactPath = Object.keys(workspace.manifests[state.head].files).find(path => path.endsWith('-large.txt'));
        const artifactId = workspace.manifests[state.head].files[artifactPath];
        const id = workspace.artifacts[artifactId].content.native_id;
        store.close();
        const cell = { id: 'native-length', source: 'return (await host.research.readNative(args.deps.id)).length;' };
        const allowed = await runChild({ cell, deps: { id }, researchNative: [id] });
        const authored = await runChild({ request: {
            source: { kind: 'files', root: 'methods/native_length.ts', files: {
                'methods/native_length.ts': '/*---\nengine: typescript-host\nargs:\n  id: Text\nreturns: Num\n---*/\nreturn (await host.research.readNative(args.id)).length;',
            } }, inputs: { id }, options: { seed: { mode: 'derived', root: 7 } },
        }, researchNative: [id] });
        let rejected = false;
        try { await runChild({ cell, deps: { id }, researchNative: ['another-id'] }); }
        catch (error) { rejected = String(error).includes('outside this manifest'); }
        return { length: allowed.value, authoredLength: authored.value, rejected };
    });
    assert.deepEqual(nativeExecution, { length: large.length, authoredLength: large.length, rejected: true });
    await page.evaluate(async () => {
        const { StudioStore } = await import('../shared/store.mjs');
        const { ResearchWorkspace } = await import('../shared/research-workspace.mjs');
        const store = await StudioStore.open(), workspace = new ResearchWorkspace(store);
        const current = (await store.get('states', 'research')).state;
        const manifest = await workspace.commitEdits(current.head, {
            'methods/cohort.nl': { kind: 'source', content: '---\nargs:\n  value: Text\n  context: Text\nreturns: Text\n---\nCompare this cohort against the current evidence.' },
            'methods/add.ts': { kind: 'source', content: '/*---\nengine: typescript-host\nargs:\n  left: Num\n  right: Num\nreturns: Num\n---*/\nreturn args.left + args.right;' },
            'views/cohort.json': { kind: 'view', content: { tree: { tag: 'section', children: [
                { tag: 'h2', text: 'Explore cohorts' }, { tag: 'input', id: 'cohort', label: 'Cohort' },
                { tag: 'select', id: 'period', label: 'Period', value: 'after', children: [
                    { tag: 'option', text: 'Before', value: 'before' }, { tag: 'option', text: 'After', value: 'after' },
                ] },
                { tag: 'table', children: [{ tag: 'tbody', children: [
                    { tag: 'tr', children: [{ tag: 'th', text: 'Cohort' }, { tag: 'td', text: 'Mobile' }] },
                ] }] },
                { tag: 'meter', value: '0.8', min: 0, max: 1 },
                { tag: 'button', id: 'compare', text: 'Compare cohort', action: { kind: 'compare', from: 'cohort' } },
            ] }, bindings: { compare: { root: 'methods/cohort.nl', from: 'cohort' } } } },
        }, { message: 'Generated comparison interaction' });
        const state = { ...current, head: manifest.id, revision: current.revision + 1, active_view: 'views/cohort.json' };
        await store.commit('research', { state, revision: state.revision, event: { kind: 'fixture-generated-view' } });
        await workspace.propose(manifest.id, {
            'claims/reviewed.json': { kind: 'claim', content: { text: 'Candidate remains reviewable before activation', status: 'reviewed' } },
            'intent/custom-ui.json': { kind: 'intent', content: { request: 'Make cohort alternatives spatially explorable' } },
            'methods/choose.ts': { kind: 'source', content: '/*---\nengine: typescript-host\nargs:\n  value: Text\n  context: Text\nreturns: Text\n---*/\nreturn `Compared ${args.value}`;' },
            'views/cohort.json': { kind: 'view', content: { module: {
                title: 'Cohort constellation',
                html: '<main><h2>Cohort constellation</h2><p id="choice"></p><button id="mobile">Explore mobile</button></main>',
                style: 'main{padding:24px;background:linear-gradient(135deg,#edf7ef,#fff6df);border-radius:18px}button{padding:10px 14px;border:1px solid #39755c;border-radius:20px;background:white}',
                script: "const output=document.getElementById('choice');output.textContent=natlang.drafts.cohort||'No cohort selected';document.getElementById('mobile').onclick=()=>{output.textContent='mobile';natlang.draft('cohort','mobile');natlang.emit('choose','mobile')}",
            }, bindings: { choose: { root: 'methods/choose.ts' } } } },
        }, { message: 'Reviewed candidate finding' });
        await workspace.propose(manifest.id, {
            'intent/table-ui.json': { kind: 'intent', content: { request: 'Keep cohort alternatives in a conventional table' } },
            'views/cohort.json': { kind: 'view', content: (await workspace.at(manifest.id, 'views/cohort.json')).content },
        }, { message: 'Conventional table alternative' });
        store.close();
    });
    await page.reload();
    await page.getByRole('button', { name: 'Compare cohort' }).waitFor();
    assert.equal(await page.getByLabel('Period').inputValue(), 'after');
    assert.equal(await page.locator('.generated td').textContent(), 'Mobile');
    await page.getByLabel('Cohort').fill('mobile');
    await page.reload();
    await page.getByRole('button', { name: 'Compare cohort' }).waitFor();
    assert.equal(await page.getByLabel('Cohort').inputValue(), 'mobile');
    await page.locator('.method').filter({ hasText: 'methods/add.ts' }).getByRole('button', { name: 'Run' }).click();
    await page.locator('#method-input').fill('{"left":20,"right":22}');
    await page.getByRole('button', { name: 'Run method' }).click();
    await page.locator('#method-status').getByText('complete').waitFor();
    assert.match(await page.locator('#method-result').textContent(), /"value": 42/);
    await page.locator('#method-runner').getByRole('button', { name: 'Close' }).click();
    await page.getByLabel(/Select Reviewed candidate finding/).check();
    await page.getByLabel(/Select Conventional table alternative/).check();
    await page.getByRole('button', { name: 'Compare selected' }).click();
    await page.locator('#detail-body').getByText(/Make cohort alternatives spatially explorable/).waitFor();
    assert.match(await page.locator('#detail-body').textContent(), /Make cohort alternatives spatially explorable/);
    assert.match(await page.locator('#detail-body').textContent(), /Keep cohort alternatives in a conventional table/);
    await page.locator('#detail').getByRole('button', { name: 'Close' }).click();
    await page.locator('.branch').filter({ hasText: 'Reviewed candidate finding' }).locator('.branch-open').click();
    await page.getByRole('button', { name: 'Activate reviewed candidate' }).click();
    await page.locator('#artifacts strong').filter({ hasText: 'claims/reviewed.json' }).waitFor();
    await page.frameLocator('.generated-module').getByRole('heading', { name: 'Cohort constellation' }).waitFor();
    await page.frameLocator('.generated-module').getByRole('button', { name: 'Explore mobile' }).click();
    await page.getByText('Load the interpreter to investigate').waitFor();
    const exported = page.waitForEvent('download');
    await page.getByRole('button', { name: 'Export', exact: true }).click();
    const exportPath = join(temporary, 'research.json');
    await (await exported).saveAs(exportPath);
    const bundle = JSON.parse(await readFile(exportPath, 'utf8'));
    assert.equal(Object.keys(bundle.native_values).length, 1);
    await page.reload();
    await page.locator('#artifacts strong').filter({ hasText: 'later.json' }).waitFor();
    assert.match(await page.locator('#revision').textContent(), /Event 5/);
    await page.frameLocator('.generated-module').getByRole('heading', { name: 'Cohort constellation' }).waitFor();
    assert.equal(await page.frameLocator('.generated-module').locator('#choice').textContent(), 'mobile');
    const fresh = await browser.newContext({ viewport: { width: 390, height: 780 } });
    const imported = await fresh.newPage();
    imported.on('pageerror', error => errors.push(String(error)));
    await imported.goto(studio.url + 'research/');
    await imported.getByText('reliability-observations.json').waitFor();
    await imported.locator('#import-file').setInputFiles({ name: 'workspace.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(bundle)) });
    await imported.locator('#artifacts strong').filter({ hasText: 'later.json' }).waitFor();
    assert.match(await imported.locator('#revision').textContent(), /Event 5/);
    await imported.frameLocator('.generated-module').getByRole('heading', { name: 'Cohort constellation' }).waitFor();
    const overflow = await imported.evaluate(() => ({width:innerWidth,scroll:document.documentElement.scrollWidth,
        offenders:[...document.querySelectorAll('*')].filter(el=>el.getBoundingClientRect().right>innerWidth+2).slice(0,8).map(el=>[el.tagName,el.className,Math.round(el.getBoundingClientRect().right)])}));
    assert.equal(overflow.scroll <= overflow.width + 2, true, JSON.stringify(overflow));
    assert.deepEqual(errors, []);
    console.log('PASS research browser: evidence, persistence, export/import, mobile');
} finally {
    await browser?.close();
    await studio.close();
    await rm(temporary, { recursive: true, force: true });
}
