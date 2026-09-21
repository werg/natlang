import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
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
    assert.match(await page.locator('#artifact-count').textContent(), /19 artifacts/);
    await page.getByRole('button', { name: /reliability-observations/ }).click();
    await page.locator('#detail-body').getByText(/failures/).waitFor();
    assert.match(await page.locator('#detail-body').textContent(), /"failures": 72/);
    await page.getByRole('button', { name: 'Close' }).click();
    await page.locator('#evidence-file').setInputFiles({ name: 'later.json', mimeType: 'application/json', buffer: Buffer.from('[{"period":"later","requests":10,"failures":1}]') });
    await page.locator('#artifacts strong').filter({ hasText: 'later.json' }).waitFor();
    assert.match(await page.locator('#revision').textContent(), /Event 1/);
    await page.evaluate(async () => {
        const { StudioStore } = await import('../shared/store.mjs');
        const { ResearchWorkspace } = await import('../shared/research-workspace.mjs');
        const store = await StudioStore.open(), workspace = new ResearchWorkspace(store);
        const current = (await store.get('states', 'research')).state;
        const manifest = await workspace.commitEdits(current.head, {
            'methods/cohort.nl': { kind: 'source', content: '---\nargs:\n  value: Text\n  context: Text\nreturns: Text\n---\nCompare this cohort against the current evidence.' },
            'views/cohort.json': { kind: 'view', content: { tree: { tag: 'section', children: [
                { tag: 'h2', text: 'Explore cohorts' }, { tag: 'input', id: 'cohort', label: 'Cohort' },
                { tag: 'button', id: 'compare', text: 'Compare cohort', action: { kind: 'compare', from: 'cohort' } },
            ] }, bindings: { compare: { root: 'methods/cohort.nl', from: 'cohort' } } } },
        }, { message: 'Generated comparison interaction' });
        const state = { ...current, head: manifest.id, revision: current.revision + 1, active_view: 'views/cohort.json' };
        await store.commit('research', { state, revision: state.revision, event: { kind: 'fixture-generated-view' } });
        store.close();
    });
    await page.reload();
    await page.getByRole('button', { name: 'Compare cohort' }).waitFor();
    await page.getByLabel('Cohort').fill('mobile');
    await page.reload();
    await page.getByRole('button', { name: 'Compare cohort' }).waitFor();
    assert.equal(await page.getByLabel('Cohort').inputValue(), 'mobile');
    const bundle = await page.evaluate(async () => {
        const { StudioStore } = await import('../shared/store.mjs');
        const { ResearchWorkspace } = await import('../shared/research-workspace.mjs');
        const store = await StudioStore.open();
        const result = { workspace: await new ResearchWorkspace(store).export(), state: (await store.get('states', 'research')).state };
        store.close(); return result;
    });
    await page.reload();
    await page.locator('#artifacts strong').filter({ hasText: 'later.json' }).waitFor();
    assert.match(await page.locator('#revision').textContent(), /Event 2/);
    const fresh = await browser.newContext({ viewport: { width: 390, height: 780 } });
    const imported = await fresh.newPage();
    imported.on('pageerror', error => errors.push(String(error)));
    await imported.goto(studio.url + 'research/');
    await imported.getByText('reliability-observations.json').waitFor();
    await imported.locator('#import-file').setInputFiles({ name: 'workspace.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(bundle)) });
    await imported.locator('#artifacts strong').filter({ hasText: 'later.json' }).waitFor();
    assert.match(await imported.locator('#revision').textContent(), /Event 2/);
    await imported.getByRole('button', { name: 'Compare cohort' }).waitFor();
    await imported.getByLabel('Cohort').fill('desktop');
    await imported.getByRole('button', { name: 'Compare cohort' }).click();
    await imported.getByText('Load the interpreter to investigate').waitFor();
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
