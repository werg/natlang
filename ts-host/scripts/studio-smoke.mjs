import { execFileSync } from 'node:child_process';
import assert from 'node:assert/strict';
import { mkdir, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { chromium } from 'playwright-core';
import { apps } from '../studio/apps/index.mjs';
import { startStudio } from './serve-studio.mjs';
const temporary = await mkdtemp(join(tmpdir(), 'natlang-studio-browser-'));
const studio = await startStudio({ port: 0, dataRoot: temporary });
let browser;
const artifacts = resolve(import.meta.dirname, '../../runs/studio-ui-smoke');
await mkdir(artifacts, { recursive: true });
try {
    browser = await chromium.launch({ headless: true, executablePath: process.env.NATLANG_CHROMIUM ?? chromium.executablePath(), args: ['--no-sandbox', '--disable-gpu', '--disable-gpu-compositing', '--disable-features=Vulkan,WebGPU', '--use-gl=swiftshader'] });
    const context=await browser.newContext({viewport:{width:1440,height:1100}});
    const page = await context.newPage(), errors = [];
    page.on('pageerror', error => { errors.push(String(error)); console.error('PAGE', String(error)); });
    page.on('console', message => { if (message.type() === 'error')
        console.error('CONSOLE', message.text()); });
    await page.goto(studio.url + '?fixture');
    await page.waitForFunction(() => !!window.natlangStudio);
    assert.equal(await page.locator('.app-card').count(), 22);
    await page.screenshot({ path: join(artifacts, 'overview.png'), fullPage: true });
    for (const spec of apps) {
        await page.goto(studio.url + '?fixture#' + spec.id);
        await page.waitForFunction(id => window.natlangStudio?.spec?.id === id && window.natlangStudio.app?.view !== null, spec.id);
        await page.waitForFunction(() => !document.querySelector('.statusbar').classList.contains('busy'));
        const result = await page.evaluate(async (action) => { try {
            await window.natlangStudio.dispatch(action.action, action);
        }
        catch (error) {
            console.error(JSON.stringify(window.natlangStudio.lastFailure?.run?.trace?.filter(row => JSON.stringify(row).includes('code error')).slice(0, 1)));
            throw error;
        } return window.natlangStudio.app.state; }, spec.smoke);
        assert.equal(result.revision, 1, `${spec.id}: ${result.notice}`);
        assert.equal(await page.locator('[data-panel]').count(), spec.panelIds.length);
        console.log(`PASS browser ${spec.id}`);
        if (['economy', 'notebook', 'spells', 'scheduling'].includes(spec.id))
            await page.screenshot({ path: join(artifacts, spec.id + '.png'), fullPage: true });
    }
    await page.goto(studio.url + '?fixture#economy');
    await page.waitForFunction(() => window.natlangStudio?.app?.view !== null && window.natlangStudio?.spec?.id === 'economy');
    await page.waitForFunction(() => !document.querySelector('.statusbar').classList.contains('busy'));
    await page.getByRole('button', { name: 'Trade apples', exact: true }).click();
    await page.waitForFunction(() => window.natlangStudio.app.state.revision === 2);
    assert.equal(await page.evaluate(() => window.natlangStudio.app.state.merchants[0].apples), 8);
    // Unsaved fields survive rendering and reload. Committed state and trace survive reload.
    await page.getByLabel('Apples', { exact: true }).fill('3');
    await page.getByRole('button', { name: 'Refresh view', exact: true }).click();
    await page.waitForFunction(() => !document.querySelector('.statusbar').classList.contains('busy'));
    assert.equal(await page.getByLabel('Apples', { exact: true }).inputValue(), '3');
    await page.reload();
    await page.waitForFunction(() => window.natlangStudio?.app?.state?.revision === 2);
    assert.equal(await page.getByLabel('Apples', { exact: true }).inputValue(), '3');
    await page.getByRole('button', { name: 'History & traces', exact: true }).click();
    await page.waitForSelector('#history[open]');
    assert.ok(await page.locator('#history .trace-row').count() >= 2);
    await page.getByRole('button', { name: 'Close history' }).click();
    // A real browser crisp notebook cell, with natlang owning the operation invocation.
    await page.goto(studio.url + '?fixture#notebook');
    await page.waitForFunction(() => window.natlangStudio?.spec?.id === 'notebook' && window.natlangStudio.app?.view !== null);
    await page.waitForFunction(() => !document.querySelector('.statusbar').classList.contains('busy'));
    await page.getByRole('button', { name: 'Run this cell', exact: true }).first().click();
    await page.waitForFunction(() => window.natlangStudio.app.state.cells[0].status === 'complete');
    assert.deepEqual(await page.evaluate(() => JSON.parse(window.natlangStudio.app.state.cells[0].result)), [2, 3, 5, 7, 11]);
    await page.getByRole('button', { name: 'Run this cell', exact: true }).nth(1).click();
    await page.waitForFunction(() => window.natlangStudio.app.state.cells[1].status === 'complete');
    assert.equal(await page.evaluate(() => window.natlangStudio.app.state.cells[1].result), '28');
    await page.evaluate(()=>window.natlangStudio.dispatch('save',{target:'numbers',text:'return Array.from({length:20},(_,i)=>i+1);'}));
    await page.evaluate(()=>window.natlangStudio.dispatch('execute',{target:'numbers'}));
    assert.equal(await page.evaluate(()=>JSON.parse(window.natlangStudio.app.state.cells[0].result).length),8);
    await page.evaluate(()=>window.natlangStudio.dispatch('execute',{target:'total'}));
    assert.equal(await page.evaluate(()=>window.natlangStudio.app.state.cells[1].result),'210');
    await page.evaluate(() => window.natlangStudio.dispatch('add', { target: 'spin', secondary: 'typescript', text: 'while (true) {}', ids: [] }));
    await page.evaluate(() => { window.workerStopped = false; window.natlangStudio.dispatch('execute', { target: 'spin' }).then(() => { window.workerStopped = true; }, () => { window.workerStopped = true; }); });
    await page.waitForFunction(() => document.getElementById('status').textContent.includes('Executing child program'));
    await page.getByRole('button', { name: 'Stop this run', exact: true }).click();
    await page.waitForFunction(() => window.workerStopped && !document.querySelector('.statusbar').classList.contains('busy'));
    await page.goto(studio.url+'?fixture#ide');
    await page.waitForFunction(()=>window.natlangStudio?.spec?.id==='ide'&&window.natlangStudio.app?.view!==null&&!document.querySelector('.statusbar').classList.contains('busy'));
    await page.evaluate(()=>window.natlangStudio.dispatch('add',{target:'answer.ts',text:'export function main(): number { return 42; }'}));
    await page.evaluate(()=>window.natlangStudio.dispatch('run',{text:'{}'}));
    assert.equal(await page.evaluate(()=>window.natlangStudio.app.state.result),'42');
    // TypeScript without natural-language calls runs without a natlang invocation, so there is no trace to inspect.
    assert.equal(await page.evaluate(()=>window.natlangStudio.app.state.trace_count),0);
    const other=await page.context().newPage();
    await other.goto(studio.url+'?fixture#ide');
    await other.waitForFunction(()=>window.natlangStudio?.app?.view!==null&&window.natlangStudio?.spec?.id==='ide');
    await page.evaluate(()=>window.natlangStudio.dispatch('run',{text:'{}'}));
    const stale=await other.evaluate(()=>window.natlangStudio.dispatch('run',{text:'{}'}).then(()=>'',error=>error.message));
    assert.match(stale,/newer revision/);await other.close();
    await page.goto(studio.url+'?fixture#terminal');
    await page.waitForFunction(()=>window.natlangStudio?.spec?.id==='terminal'&&window.natlangStudio.app?.view!==null&&!document.querySelector('.statusbar').classList.contains('busy'));
    await page.getByLabel('Bash command',{exact:true}).fill('sleep 0.3; printf first-command');
    await page.getByRole('button',{name:'Run command',exact:true}).click();
    await page.waitForFunction(()=>document.querySelector('.statusbar').classList.contains('busy'));
    await page.getByLabel('Bash command',{exact:true}).fill('printf next-command');
    await page.waitForFunction(()=>!document.querySelector('.statusbar').classList.contains('busy'));
    assert.equal(await page.getByLabel('Bash command',{exact:true}).inputValue(),'printf next-command');
    await page.goto(studio.url+'#ide');
    await page.waitForFunction(()=>!!window.natlangStudio);
    await page.getByLabel('Tell natlang what you want to do').fill('Please inspect my program');
    await page.getByRole('button',{name:'Make it happen',exact:false}).click();
    await page.waitForSelector('#settings[open]');
    assert.equal(await page.getByLabel('Tell natlang what you want to do').inputValue(),'Please inspect my program');
    await page.getByRole('button',{name:'Close settings'}).click();
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto(studio.url + '?fixture');
    await page.waitForFunction(() => !!window.natlangStudio);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    await page.screenshot({ path: join(artifacts, 'mobile.png'), fullPage: true });
    assert.deepEqual(errors, []);
    console.log('PASS durable state, draft recovery, trace history, DOM controls, worker cancellation, notebook execution and mobile layout');
}
finally {
    await browser?.close();
    await studio.close();
    (execFileSync('chmod', ['-R', 'u+w', temporary]), await rm(temporary, { recursive: true, force: true }));
}
